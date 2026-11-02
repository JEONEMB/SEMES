"""Select existing line measurements; do not run extraction or statistical tests."""
from __future__ import annotations

import hashlib
import html
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent
SOURCE = OUT.parent / 'reference_line_methods_108_20261030'
REFERENCES = ['R01', 'R02', 'R03', 'R04']
TARGETS = {'external': ['upper_structure', 'lower_structure'],
           'qr': ['qr_L', 'qr_C', 'qr_R']}
KEEP_PER_FAMILY = {'external': 3, 'qr': 2}
QR_METHODS = ['Q02', 'Q04']  # User-selected methods; do not automatically replace them.
LABELS = {'upper_structure': '상단 구조물', 'lower_structure': '하단 구조물',
          'qr_L': '왼쪽 QR', 'qr_C': '중앙 QR', 'qr_R': '오른쪽 QR'}
source_files: dict[str, str] = {}
copied_files: list[dict] = []


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_source(relative: str):
    path = SOURCE / relative
    source_files[relative] = sha(path)
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write_json(relative: str, data):
    path = OUT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def copy_source(relative: str, destination: str | None = None):
    src, dst = SOURCE / relative, OUT / (destination or relative)
    digest = sha(src)
    source_files[relative] = digest
    if dst.exists() and sha(dst) != digest:
        raise ValueError(f'Existing copy differs from source: {dst}')
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copyfile(src, dst)
    if sha(dst) != digest:
        raise ValueError(f'Copy hash mismatch: {dst}')
    copied_files.append({'source': relative, 'destination': str(dst.relative_to(OUT)),
                         'sha256': digest})


def main():
    source_files.clear()
    copied_files.clear()
    measurements = read_source('results/line_measurements.json')
    original_combinations = read_source('results/combinations.json')
    original_methods = read_source('results/method_status.json')
    rows = measurements['records']
    assert len({r['measurement_id'] for r in rows}) == len(rows)
    rankings, selected = {}, {}
    for family, targets in TARGETS.items():
        prefix = 'E' if family == 'external' else 'Q'
        methods = []
        for method_id in sorted({r['method_id'] for r in rows if r['method_id'].startswith(prefix)}):
            rr = [r for r in rows if r['method_id'] == method_id]
            expected = {(ref, target) for ref in REFERENCES for target in targets}
            actual = {(r['reference_id'], r['target_id']) for r in rr}
            failed = [r['measurement_id'] for r in rr
                      if r['status'] != 'ok' or not isinstance(r.get('image_angle_deg'), (int, float))
                      or not math.isfinite(r['image_angle_deg'])]
            complete = actual == expected and len(rr) == len(expected) and not failed
            item = {'method_id': method_id, 'method_name': rr[0]['method_name'],
                    'eligible_all_required_slots_ok': complete,
                    'failed_measurement_ids': failed,
                    'missing_slots': sorted(expected - actual), 'targets': {}}
            for target in targets:
                tr = [r for r in rr if r['target_id'] == target and r['status'] == 'ok']
                angles = {r['reference_id']: r['image_angle_deg'] for r in tr}
                target_complete = len(angles) == 4 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in angles.values())
                item['targets'][target] = {
                    'image_angles_deg': angles,
                    'range_deg': max(angles.values()) - min(angles.values()) if target_complete else None}
            item['score_deg'] = max(t['range_deg'] for t in item['targets'].values()) if complete else None
            methods.append(item)
        methods.sort(key=lambda m: (m['score_deg'] if m['score_deg'] is not None else math.inf, m['method_id']))
        eligible = [m for m in methods if m['eligible_all_required_slots_ok']]
        selected[family] = ([m['method_id'] for m in eligible[:KEEP_PER_FAMILY[family]]]
                            if family == 'external' else QR_METHODS.copy())
        assert set(selected[family]) <= {m['method_id'] for m in eligible}
        for item in methods:
            item['rank_among_complete_methods'] = eligible.index(item) + 1 if item in eligible else None
            item['selected'] = item['method_id'] in selected[family]
            item['reason'] = ('상위 3개 후보로 선별' if item['selected'] else
                              '추출 누락 또는 실패가 있어 이번 선별에서 제외' if not item['eligible_all_required_slots_ok'] else
                              '완전 추출되었으나 이번 상위 3개 선별에서 보류; 사용 불가 판정 아님')
            if family == 'qr':
                item['reason'] = ('사용자 지정: Q02·Q04만 사용' if item['selected'] else
                                  '사용자 지정으로 제외: 초기 3위 후보였으나 Q02·Q04만 사용' if item['method_id'] == 'Q01' else
                                  '보류: 사용자 지정 Q02·Q04에 포함되지 않음; 사용 불가 판정 아님')
        rankings[family] = methods

    selected_ids = set(selected['external'] + selected['qr'])
    selected_rows = [r for r in rows if r['method_id'] in selected_ids]
    selected_combinations = [c for c in original_combinations['combinations']
                             if c['external_method'] in selected['external'] and c['qr_method'] in selected['qr']]
    measurement_ids = {r['measurement_id'] for r in selected_rows}
    assert len(selected_rows) == 48 and all(r['status'] == 'ok' for r in selected_rows)
    assert len(selected_combinations) == 24 and all(c['status'] == 'complete' for c in selected_combinations)
    assert all(set(c['measurement_refs'].values()) <= measurement_ids for c in selected_combinations)
    metadata = {'source_meta': measurements['meta'], 'selection_manifest': '../selection_manifest.json',
                'meaning': '기준 4장 기울기 범위에 따른 후속 검증 후보. 정확도 합격·통계 검정·θ 판별 검증 결과가 아님.'}
    write_json('results/line_measurements.json', {'meta': metadata, 'records': selected_rows})
    for family in TARGETS:
        subset = [r for r in selected_rows if r['method_id'] in selected[family]]
        write_json(f'results/{family}_lines.json', {'meta': metadata, 'records': subset})
    write_json('results/combinations.json', {'meta': {'count': 24, 'method_pairs': 6, 'reference_count': 4,
               'meaning': original_combinations['meta']['meaning']}, 'combinations': selected_combinations})
    write_json('results/method_status.json', {'meta': metadata,
               'methods': [m for m in original_methods['methods'] if m['method_id'] in selected_ids]})
    write_json('rankings.json', {'ranking_measure': 'max over targets(max over references(angle) - min over references(angle))',
                               'units': 'image angle degree; not robot theta', 'rankings': rankings})

    for row in selected_rows:
        copy_source('results/' + row['intermediate_json'])
        for image_info in row.get('intermediate_images', {}).values():
            copy_source('results/' + image_info['path'])
    for method_id in selected_ids:
        for reference_id in REFERENCES:
            copy_source(f'figures/methods/{method_id}_{reference_id}.png')
    for name in ['inputs_manifest.json', 'targets.json', 'methods.json', 'method_sources.json']:
        copy_source('configs/' + name, 'source_configs/' + name)

    esc = html.escape
    sections = []
    for family, methods in rankings.items():
        title = '외부 구조물' if family == 'external' else 'QR'
        table = []
        for m in methods:
            score = f"{m['score_deg']:.6f}°" if m['score_deg'] is not None else '추출 실패 포함'
            cells = ''.join('<td>' + (f"{m['targets'][t]['range_deg']:.6f}°" if m['targets'][t]['range_deg'] is not None else '—') + '</td>' for t in TARGETS[family])
            table.append(f"<tr class={'selected' if m['selected'] else 'held'}><td>{m['method_id']}</td><td>{esc(m['method_name'])}</td>{cells}<td>{score}</td><td>{esc(m['reason'])}</td></tr>")
        header = ''.join(f'<th>{LABELS[t]} 범위</th>' for t in TARGETS[family])
        sections.append(f'<h2>{title}</h2><div class="scroll"><table><thead><tr><th>ID</th><th>방법</th>{header}<th>선별 점수</th><th>결과</th></tr></thead><tbody>{"".join(table)}</tbody></table></div>')
    galleries = []
    for method_id in selected['external'] + selected['qr']:
        links = ''.join(f'<a href="figures/methods/{method_id}_{ref}.png" target="_blank">{ref} 추출 그림</a>' for ref in REFERENCES)
        galleries.append(f'<div class="gallery"><strong>{method_id}</strong> {links}</div>')
    page = '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>직선 추출 결과 후보 선별</title><style>
body{font-family:system-ui,"Malgun Gothic",sans-serif;margin:32px auto;max-width:1380px;padding:0 24px;color:#202936;line-height:1.65}
h1{font-size:27px}h2{font-size:21px;margin-top:30px}.note{background:#eff4fa;padding:16px 20px;border-left:4px solid #3267aa}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:10px;border:1px solid #d4dce6;text-align:left}th{background:#edf1f6}
.selected{background:#e9f6ed}.held{color:#4d5866}.scroll{overflow:auto}.gallery{padding:12px;border-bottom:1px solid #ddd}.gallery a{margin-left:22px}a{color:#165eae}
</style><h1>기준 4장 직선 추출 결과 — 후속 검증 후보</h1>
<div class="note">4장 모두 추출 성공 → 같은 대상의 영상 각도 최댓값−최솟값 계산 → 대상별 범위 중 가장 큰 값으로 순위 산정.<br>
외부 구조물은 상위 3개 유지, QR은 사용자 지정 Q02·Q04만 사용합니다. 선별 개수는 정확도 합격선이 아닙니다. 영상 각도는 로봇 θ와 다릅니다.</div>
<p><b>5개 방법 · 48개 직선 측정값 · 6개 방법 조합 × 기준 4장 = 24개 조합 항목</b></p>
<p>기존 측정값과 추출 그림을 그대로 복사했습니다. 원본 수정·영상 재측정·통계 검정은 수행하지 않았습니다.
이 선별은 수치 범위만으로 수행했으며 모든 경계의 정확성이나 θ 판별 성능을 보증하지 않습니다.
보류된 방법도 정확도가 부족하다고 확정한 것이 아닙니다. E06은 기존 보고서에 명시된 PAE 변형 구현입니다.</p>
<p><a href="results/external_lines.json">외부 구조물 24개</a> · <a href="results/qr_lines.json">QR 24개</a> ·
<a href="results/combinations.json">24개 조합</a> · <a href="rankings.json">전체 선별 근거</a> ·
<a href="selection_manifest.json">출처·검증 기록</a> · <a href="../reference_line_methods_108_20261030/report.html">기존 전체 보고서</a></p>
''' + ''.join(sections) + '<h2>선별한 방법의 기존 추출 그림</h2>' + ''.join(galleries) + '</html>'
    (OUT / 'selection.html').write_text(page, encoding='utf-8')
    notes = (
        '기준 4장 직선 추출 결과: 후속 검증 후보 분리\n\n'
        '열람: selection.html\n선별: 외부 구조물 ' + ', '.join(selected['external']) +
        ' / QR ' + ', '.join(selected['qr']) + '\n'
        '규칙: 4장 모든 대상 추출 성공, 대상별 영상 각도 최대-최소 범위 중 최댓값 순위. 외부 상위 3개 유지, QR은 사용자 지정 Q02·Q04만 사용.\n'
        '3개는 작업량을 줄이기 위한 선별 개수이며 통계적 합격 기준이 아니다.\n'
        '결과: 외부 24개, QR 24개, 전체 48개 직선. 방법 6조합 × 기준 4장 = 24개 조합 항목.\n'
        '기존 측정 레코드·중간 산출물·그림은 수정 없이 복사. 원본 데이터 및 분석 코드 변경 없음.\n'
        '새 영상 측정, 통계 검정, 각도-θ 환산, 정확도 검증은 수행하지 않음.\n'
        '수치 선별 결과이며 모든 경계가 정확하다는 판정은 아님. 보류 방법도 사용 불가 판정 아님.\n'
        'E06은 기존 PAE 변형 구현이다. 원 논문 재현 및 성능 보증과 구분할 것.\n'
        'source_configs는 출처 재현을 위한 원본 전체 설정 복사본이며, 선별 대상은 selection_manifest에 지정.\n'
        '원본 이미지 BMP는 복사하지 않았으며 각 records.input_path로 위치 확인 가능.\n')
    (OUT / 'README.txt').write_text(notes, encoding='utf-8')
    # Validate selection integrity without re-running any image-processing code.
    saved = json.loads((OUT / 'results/line_measurements.json').read_text(encoding='utf-8'))['records']
    assert saved == selected_rows
    for relative, digest in source_files.items():
        if sha(SOURCE / relative) != digest:
            raise ValueError(f'Source changed during selection: {relative}')
    write_json('selection_manifest.json', {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_directory': str(SOURCE), 'selection_state': 'screening_candidates_not_accuracy_validated',
        'rule': {'references': REFERENCES, 'require_all_slots_ok': True,
                 'measure': 'per-target max(image_angle_deg)-min(image_angle_deg), then max over targets',
                 'keep_per_family': KEEP_PER_FAMILY, 'qr_user_override': QR_METHODS,
                 'statistical_pass_threshold': None},
        'revision': 'QR은 Q02·Q04만 사용하도록 사용자 지시 반영. 이전 버전은 history/before_qr_q02_q04_only.zip에 보존.',
        'selected_methods': selected,
        'counts': {'external_lines': 24, 'qr_lines': 24, 'lines': 48, 'method_pairs': 6,
                   'combination_entries': 24, 'method_figures': 20, 'intermediate_json': 48},
        'source_sha256': source_files, 'copied_files': copied_files,
        'validation': {'selected_measurements_equal_original_records': True,
                       'all_selected_slots_ok': True, 'all_combination_references_resolve': True,
                       'copied_file_hashes_equal': True, 'source_hashes_unchanged': True,
                       'new_image_measurement_run': False, 'statistical_tests_run': False}})
    print(json.dumps({'selected': selected, 'line_count': len(saved),
                      'combination_count': len(selected_combinations), 'output': str(OUT)}, ensure_ascii=True))


if __name__ == '__main__':
    main()
