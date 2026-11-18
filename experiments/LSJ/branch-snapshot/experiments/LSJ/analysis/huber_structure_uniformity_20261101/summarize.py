"""Describe the eight saved ordinary-Huber reference lines; no refitting/filtering."""
from pathlib import Path
from datetime import datetime
import hashlib
import html
import json
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'huber_convergence_20261101'
LABELS = {'R01': '1차 center.bmp', 'R02': '2차 center1.bmp',
          'R03': '2차 center2.bmp', 'R04': '2차 center3.bmp'}
TARGETS = {'upper_structure': '상단', 'lower_structure': '하단'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(name):
    return json.loads((SOURCE / name).read_text(encoding='utf-8'))


def dump(name, obj):
    (HERE / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                                      allow_nan=False), encoding='utf-8')


def main():
    settings = read('settings.json')
    provenance = read('provenance.json')
    assert settings['fixed']['residual'] == 'vertical'
    assert settings['fixed']['gaussian'] is False
    assert sha(provenance['profile_file']) == provenance['profile_sha256']
    rows = [dict(r) for r in read('results.json') if r['config'] == 'A']
    assert len(rows) == 8
    point_sets = np.load(SOURCE / 'inputs.npz', allow_pickle=False)
    by_ref = {}
    for row in rows:
        ref, target = row['input'].split('_', 1)
        row.update(ref=ref, target=target, label=LABELS[ref])
        assert row['maxiter'] == 50 and row['tol'] == 1e-8
        assert row['tol_met'] and row['finite']
        points = point_sets[row['input']]
        assert len(points) == row['n'] == provenance['input_counts'][row['input']]
        assert np.isfinite(points).all()
        assert math.isclose(math.degrees(math.atan(row['slope'])),
                            row['image_angle_deg'], abs_tol=1e-14)
        row['x_min_px'], row['x_max_px'] = map(float, [points[:, 0].min(), points[:, 0].max()])
        by_ref.setdefault(ref, {})[target] = row

    summaries = {}
    for target, label in TARGETS.items():
        subset = [by_ref[ref][target] for ref in LABELS]
        lo = min(subset, key=lambda r: r['image_angle_deg'])
        hi = max(subset, key=lambda r: r['image_angle_deg'])
        summaries[target] = {
            'label': label, 'reference_images': 4,
            'minimum_ref': lo['ref'], 'maximum_ref': hi['ref'],
            'minimum_slope': lo['slope'], 'maximum_slope': hi['slope'],
            'slope_range': hi['slope'] - lo['slope'],
            'minimum_angle_deg': lo['image_angle_deg'],
            'maximum_angle_deg': hi['image_angle_deg'],
            'angle_range_deg': hi['image_angle_deg'] - lo['image_angle_deg'],
        }
    ratio = summaries['lower_structure']['angle_range_deg'] / summaries['upper_structure']['angle_range_deg']

    images = {item['id']: item for item in provenance['image_checks']}
    checks = []
    for ref, image_info in images.items():
        assert sha(image_info['path']) == image_info['sha256']
        checks.append({'ref': ref, 'path': image_info['path'],
                       'sha256': image_info['sha256'], 'matches_previous': True})

    # Both panels have the same degree axis. No inference, interval estimate or limit line.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, layout='constrained')
    colors = {'upper_structure': '#1464a5', 'lower_structure': '#b54a12'}
    all_angles = [r['image_angle_deg'] for r in rows]
    margin = (max(all_angles) - min(all_angles)) * .18
    for ax, target in zip(axes, TARGETS):
        vals = [by_ref[ref][target]['image_angle_deg'] for ref in LABELS]
        ax.scatter(range(1, 5), vals, s=55, color=colors[target], zorder=3)
        ax.axhspan(min(vals), max(vals), color=colors[target], alpha=.12,
                   label='Observed min-to-max (not a confidence interval)')
        for pos, val in enumerate(vals, 1):
            ax.annotate(f'{val:.6f}', (pos, val), xytext=(0, 9),
                        textcoords='offset points', ha='center', fontsize=10)
        title = 'Upper structure' if target == 'upper_structure' else 'Lower structure'
        ax.set_title(f'{title}\nRange = {summaries[target]["angle_range_deg"]:.6f} deg')
        ax.set_xticks(range(1, 5), list(LABELS))
        ax.set_xlim(.5, 4.5)
        ax.set_ylim(min(all_angles) - margin, max(all_angles) + margin)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax.set_xlabel('Reference image ID (not elapsed time)')
        ax.grid(axis='y', alpha=.25)
    axes[0].set_ylabel('Image line angle atan(m), degrees (image y increases down)')
    fig.suptitle('Ordinary Huber | all integer Sobel points | maxiter=50, tol=1e-8')
    fig.savefig(HERE / 'angle_comparison.png', dpi=160)
    plt.close(fig)

    # Full-width crops are for display only; points and coefficients are unchanged.
    for ref in LABELS:
        with Image.open(images[ref]['path']) as image:
            gray = np.asarray(image.convert('L'))
        fig, axes = plt.subplots(2, 1, figsize=(15, 5.2), layout='constrained')
        for ax, target in zip(axes, TARGETS):
            row = by_ref[ref][target]
            points = point_sets[row['input']]
            x = points[:, 0]
            y = points[:, 1]
            y0 = max(0, int(np.floor(y.min())) - 30)
            y1 = min(gray.shape[0], int(np.ceil(y.max())) + 31)
            ax.imshow(gray[y0:y1], cmap='gray', vmin=0, vmax=255,
                      extent=(-.5, gray.shape[1]-.5, y1-.5, y0-.5), aspect='auto')
            ax.scatter(x, y, s=2, color='#ff9d34', alpha=.55, label=f'All {len(x)} points', zorder=2)
            xx = np.array([x.min(), x.max()])
            ax.plot(xx, row['intercept_px'] + row['slope']*xx, color='#00ddff',
                    linewidth=1.8, label='One fitted line across visible sides', zorder=3)
            ax.set_ylim(y1-.5, y0-.5)
            ax.set_ylabel('Image y (px)')
            title = 'Upper' if target == 'upper_structure' else 'Lower'
            ax.set_title(f'{title}: m={row["slope"]:.9f}, angle={row["image_angle_deg"]:.6f} deg')
            ax.legend(loc='upper right', fontsize=8)
        axes[-1].set_xlabel('Original image x (px)')
        fig.suptitle(f'{ref}: original pixels + saved input points + saved Huber line | vertical display enlarged')
        fig.savefig(HERE / f'overlay_{ref}.png', dpi=160)
        plt.close(fig)

    rows.sort(key=lambda r: (r['ref'], list(TARGETS).index(r['target'])))
    dump('measurements.json', rows)
    dump('summary.json', {'targets': summaries, 'lower_to_upper_angle_range_ratio': ratio,
                          'metric': 'observed maximum minus minimum across four reference images',
                          'equivalence_test': 'not performed', 'pass_fail_threshold': None,
                          'mechanical_theta_conversion': 'not performed'})
    source_hashes = {name: sha(SOURCE / name) for name in
                     ['results.json', 'settings.json', 'inputs.npz', 'provenance.json', 'verification.json']}
    dump('provenance.json', {
        'created_at': datetime.now().astimezone().isoformat(),
        'source_directory': str(SOURCE), 'source_hashes': source_hashes,
        'source_config': 'A', 'fixed_settings': settings['fixed'],
        'maxiter': 50, 'tol': 1e-8, 'refitted': False,
        'method': 'Read already verified baseline results and calculate descriptive ranges.',
        'point_rejection': False, 'minimum_point_count_filter': False,
        'image_checks': checks, 'total_input_points': sum(r['n'] for r in rows),
        'illustration_only': 'Overlay vertical range is enlarged; 30px crop margin affects only display. '
                             'Lines connect across the occluded span for display; no points invented.',
    })

    table = ''.join(
        f'<tr><td>{LABELS[ref]}</td><td>{by_ref[ref]["upper_structure"]["slope"]:.9f}</td>'
        f'<td>{by_ref[ref]["upper_structure"]["image_angle_deg"]:.6f}°</td>'
        f'<td>{by_ref[ref]["lower_structure"]["slope"]:.9f}</td>'
        f'<td>{by_ref[ref]["lower_structure"]["image_angle_deg"]:.6f}°</td></tr>' for ref in LABELS)
    ranges = ''.join(f'<tr><th>{s["label"]}</th><td>{s["slope_range"]:.9f}</td>'
                     f'<td>{s["angle_range_deg"]:.6f}°</td><td>{LABELS[s["minimum_ref"]]}</td>'
                     f'<td>{LABELS[s["maximum_ref"]]}</td></tr>' for s in summaries.values())
    gallery = ''.join(f'<h3>{LABELS[ref]}</h3><a href="overlay_{ref}.png"><img src="overlay_{ref}.png" '
                       f'alt="{LABELS[ref]}의 상하단 전체 경계점과 단일 직선"></a>' for ref in LABELS)
    paths = ''.join(f'<li>{item["ref"]}: <code>{html.escape(item["path"])}</code></li>' for item in checks)
    upper, lower = summaries.values()
    document = f'''<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>외부 상·하단 구조물 기울기 비교</title>
<style>body{{font-family:"Malgun Gothic",sans-serif;color:#233044;background:#f4f6f9;margin:0;line-height:1.7}}
main{{max-width:1120px;margin:auto;padding:30px 24px 60px;background:white}}h1{{font-size:28px}}h2{{margin-top:36px}}
.lead{{background:#edf5ff;border-left:5px solid #1464a5;padding:17px 22px}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}
th,td{{padding:10px;border:1px solid #d7dee6;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#f0f4f8}}
img{{width:100%;height:auto}}code{{overflow-wrap:anywhere}}.muted{{color:#596679}}.scroll{{overflow-x:auto}}a{{color:#0059a7}}</style></head>
<body><main><h1>외부 상·하단 구조물의 기준 이미지 간 기울기 비교</h1>
<div class="lead"><b>관측된 범위는 상단이 더 작습니다.</b><br>
상단 각도 최댓값−최솟값: <b>{upper['angle_range_deg']:.6f}°</b><br>
하단 각도 최댓값−최솟값: <b>{lower['angle_range_deg']:.6f}°</b> — 상단의 약 <b>{ratio:.2f}배</b></div>
<p>대상: 1차 기준 영상 1장과 2차 기준 영상 3장, 이미지마다 상단·하단 각각 한 직선입니다.
상단끼리 4개, 하단끼리 4개를 비교했습니다. 상단과 하단을 하나의 값으로 평균내지 않았습니다.</p>
<h2>1. 확정 설정과 사용한 입력</h2>
<p>저장된 Sobel 정수 경계점 → 일반 Huber 세로 잔차 피팅 → 직선 y=mx+b.
Huber t=1.345, s=MAD(center=0), s 갱신, 최대 50회, 종료값 10⁻⁸, 종료 기준 dev를 적용한 기존 A 결과를 사용했습니다.
이 설정의 수렴성 검증은 <a href="../huber_convergence_20261101/report.html">기존 보고서</a>에 있습니다.</p>
<p>Gaussian, 추가 점 제거, 최소 점 개수 제한을 적용하지 않았습니다. 상단 2,712점·하단 3,052점을 이미지별로 모두 사용했습니다.
가려진 중앙 부분의 제외는 기존 입력과 동일하며, 좌우 가시 영역의 점을 모아 구조물당 한 직선을 계산한 결과입니다.</p>
<p>이번에는 이미 계산된 8개 직선의 계수를 재사용해 범위를 산출했습니다. 원본 이미지와 입력 캐시의 해시를 확인했습니다.</p>
<h2>2. 이미지별 기울기와 각도</h2><p>m은 px/px, 각도는 φ=atan(m)×180/π입니다.
영상 좌표의 y가 아래로 증가하므로 양수 기울기는 오른쪽으로 갈수록 아래로 내려감을 뜻합니다.
표의 각도는 <b>영상 직선 각도</b>이며 기계 θ값이 아닙니다.</p>
<div class="scroll"><table><thead><tr><th>기준 이미지</th><th>상단 m (px/px)</th><th>상단 φ</th><th>하단 m (px/px)</th><th>하단 φ</th></tr></thead><tbody>{table}</tbody></table></div>
<h2>3. 4장 사이 최댓값−최솟값</h2>
<div class="scroll"><table><thead><tr><th>대상</th><th>m 범위</th><th>φ 범위</th><th>최솟값 이미지</th><th>최댓값 이미지</th></tr></thead><tbody>{ranges}</tbody></table></div>
<p>현재 4장에서 상단이 하단보다 더 균일하게 측정됐습니다. 별도의 통과 기준이나 동등성 검정은 적용하지 않았습니다.
기계 θ 허용오차 ±0.005°를 영상 선 각도에 직접 대입하지 않았습니다.</p>
<a href="angle_comparison.png"><img src="angle_comparison.png" alt="동일한 세로축 척도로 비교한 상단과 하단의 이미지별 선 각도"></a>
<p class="muted">두 패널은 동일한 각도 축을 사용합니다. 색 영역은 관측된 최솟값부터 최댓값이며 신뢰구간이 아닙니다. R01~R04는 기준 이미지 ID입니다.</p>
<h2>4. 원본에서 사용한 점과 직선 확인</h2>
<p>주황 점: 입력 경계점 전체. 하늘색: 구조물 전체에 맞춘 단일 직선.
작은 기울기를 보기 위해 세로 방향을 확대했으므로 화면에 보이는 경사를 실제 각도로 읽으면 안 됩니다.
가려진 중앙 구간의 하늘색 선은 직선의 연장이며 그 구간에 측정점이 있다는 뜻은 아닙니다.</p>{gallery}
<h2>5. 근거 파일</h2><ul><li><a href="measurements.json">8개 직선 계수·각도·점 개수</a></li>
<li><a href="summary.json">최대−최소 범위 계산</a></li><li><a href="provenance.json">사용 설정·원본 경로·해시</a></li>
<li><a href="summarize.py">재현 스크립트</a></li></ul><details><summary>원본 이미지 경로</summary><ul>{paths}</ul></details>
</main></body></html>'''
    (HERE / 'report.html').write_text(document, encoding='utf-8')
    print(json.dumps({'targets': summaries, 'lower_upper_range_ratio': ratio,
                      'table': [{k: row[k] for k in ['ref', 'target', 'slope', 'image_angle_deg', 'n']} for row in rows]},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
