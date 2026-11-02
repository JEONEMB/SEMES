"""Write a report from saved screening results without changing the selection."""
import hashlib
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'reference_line_methods_108_20261030'
inputs = {}


def read(path):
    content = path.read_bytes()
    inputs[str(path)] = hashlib.sha256(content).hexdigest()
    return json.loads(content.decode('utf-8-sig'))


manifest = read(ROOT / 'selection_manifest.json')
ranking = read(ROOT / 'rankings.json')['rankings']
refs = read(ROOT / 'source_configs/inputs_manifest.json')['references']
original_rows = read(SOURCE / 'results/line_measurements.json')['records']
selected_rows = read(ROOT / 'results/line_measurements.json')['records']
parts, plain = [], []


def heading(text, level=2):
    parts.append(f'<h{level}>{html.escape(text)}</h{level}>')
    plain.extend(['', text, ''])


def paragraph(text, css=''):
    parts.append(f'<p class="{css}">{html.escape(text)}</p>')
    plain.extend([text, ''])


def table(headers, rows):
    parts.append('<div class="table-wrap"><table><thead><tr>' + ''.join(
        f'<th>{html.escape(str(v))}</th>' for v in headers) + '</tr></thead><tbody>')
    for row in rows:
        parts.append('<tr>' + ''.join(f'<td>{html.escape(str(v))}</td>' for v in row) + '</tr>')
    parts.append('</tbody></table></div>')
    plain.append(' | '.join(headers))
    plain.extend(' | '.join(str(v) for v in row) for row in rows)
    plain.append('')


def links(items):
    parts.append('<ul>' + ''.join(f'<li><a href="{html.escape(path)}">{html.escape(label)}</a></li>' for label, path in items) + '</ul>')
    plain.extend(f'{label}: {path}' for label, path in items)
    plain.append('')


def fmt(value):
    return '—' if value is None else f'{value:.6f}'


names = {
    '01': 'Gaussian 프로파일 피팅', '02': 'Canny–Devernay', '03': 'Canny–Hough 재피팅',
    '04': 'LSD', '05': 'Zernike 모멘트', '06': 'PAE 변형', '07': 'EDLines',
    '08': 'FLD', '09': 'ELSED', '10': '이진화·윤곽선 상단 피팅',
    '11': '외곽 코너·cornerSubPix', '12': '파인더 중심 연결선'}
labels = {'upper_structure': '상단', 'lower_structure': '하단', 'qr_L': '왼쪽', 'qr_C': '중앙', 'qr_R': '오른쪽'}
heading('직선 추출 방법 1차 분리 보고서', 1)
paragraph('작성일: 2026-10-30 | 대상: 1·2차 데이터의 기준 이미지 4장 | 기존 결과를 이용한 후보 선별')
paragraph('외부 구조물 E01·E06·E02는 유지하고, 사용자 지시에 따라 QR은 Q02·Q04만 사용한다. 현재 48개 직선 측정값, 6개 방법 조합을 보존했다. 이번 결과는 정확도 합격 판정이 아니다.', 'lead')

heading('1. 목적과 사용 데이터')
paragraph('동일한 기준으로 제공된 4장에서 같은 대상의 선 각도가 얼마나 달라지는지 비교하여 검토할 방법 수를 줄였다. 여기서 “1차 분리”는 첫 번째 선별 단계라는 뜻이며, 1차 촬영 데이터만 사용했다는 뜻이 아니다.')
table(['기준 ID', '데이터 구분', '파일명'], [[r['reference_id'], r['label'], r['file']] for r in refs])
paragraph('외부 구조물은 상단·하단 전체 경계를 각각 하나의 직선으로, QR은 왼쪽·중앙·오른쪽을 각각 별도로 비교했다. QR Q01~Q11은 바깥 밝은 테두리 상단, Q12는 위쪽 두 파인더 중심 연결선이다. Q12는 다른 특징이므로 절대 각도를 같은 경계의 정답과 직접 비교하지 않는다.')
table(['항목', '분리 전', '분리 후'], [
    ['외부 구조물 방법', '9개', '3개'], ['QR 방법', '12개', '2개'],
    ['방법 조합', '9 × 12 = 108개', '3 × 2 = 6개'],
    ['직선 측정 항목', '216개: 성공 209 / 실패 7', '48개: 외부 24 / QR 24'],
    ['기준 4장에 대한 조합 항목', '432개', '24개'],
    ['추출 과정 그림', '84개', '20개']])

heading('2. 적용한 선별 기준')
paragraph('① 한 방법이 담당하는 모든 대상에 대해 기준 4장 전부 추출에 성공해야 한다. 외부 구조물은 8개, QR은 12개 측정 항목을 모두 충족해야 한다.')
paragraph('② 같은 대상의 영상 선 각도 φ에 대해 R = max(φ_R01, φ_R02, φ_R03, φ_R04) − min(φ_R01, φ_R02, φ_R03, φ_R04)를 사용했다. 이 값은 표준편차나 정답 대비 오차가 아니라 최댓값과 최솟값의 차이인 범위다.')
paragraph('③ 외부 구조물 점수 = max(R_상단, R_하단), QR 점수 = max(R_왼쪽, R_중앙, R_오른쪽)으로 정했다. 대상 간 평균으로 불균일한 결과가 가려지지 않도록 가장 큰 범위를 사용했다.')
paragraph('④ 초기에는 외부 구조물과 QR에서 각각 상위 3개를 남겼다. 이후 사용자 지시에 따라 QR Q01을 제외하고 Q02·Q04만 사용하도록 변경했다. 외부 구조물은 상위 3개를 유지한다. 개수는 운영상 선택이며 통계적으로 도출한 합격 개수나 허용오차 기준이 아니다.', 'note')
paragraph('이하 모든 수치의 단위는 영상 속 선 각도의 도(°)다. 로봇 θ의 도(°)와 바로 같은 값으로 취급하거나 정렬 허용오차 ±0.005°에 직접 대입하지 않는다. 순위는 반올림 전 원본 값으로 정했고 표만 소수점 6자리로 표시했다.')

heading('3. 선별 결과')
summary = []
for family, title in [('external', '외부 구조물'), ('qr', 'QR')]:
    for m in ranking[family]:
        if m['selected']:
            worst = max(m['targets'], key=lambda key: m['targets'][key]['range_deg'])
            summary.append([title, m['method_id'], names[m['method_id'][1:]], m['rank_among_complete_methods'], fmt(m['score_deg']), labels[worst]])
table(['구분', 'ID', '방법', '순위', '최대 범위(°)', '점수 결정 대상'], summary)
paragraph('보존한 6개 조합: E01·E06·E02 각각에 Q02·Q04를 조합했다. 조합은 외부 선 2개와 QR 선 3개를 함께 열람하기 위한 묶음이다. 각도를 평균하거나 새로운 추정값을 만든 것이 아니다.')

heading('4. 전체 방법의 수치와 처리 결과')
for family, title, targets in [('external', '외부 구조물 9개', ['upper_structure', 'lower_structure']),
                               ('qr', 'QR 12개', ['qr_L', 'qr_C', 'qr_R'])]:
    heading(title, 3)
    rr = []
    for m in ranking[family]:
        reason = ('선별' if m['selected'] else '제외: 사용자 지정' if m['method_id'] == 'Q01' else
                  '보류: 선택 대상 밖' if m['eligible_all_required_slots_ok'] else '제외: 일부 추출 실패')
        rr.append([m['method_id'], names[m['method_id'][1:]], *[fmt(m['targets'][t]['range_deg']) for t in targets], fmt(m['score_deg']), reason])
    table(['ID', '방법', *[labels[t] + ' 범위(°)' for t in targets], '점수(°)', '처리'], rr)
paragraph('“보류”는 현재 선택 대상에 포함되지 않았다는 뜻이며 방법론 자체를 사용할 수 없다고 확정한 것이 아니다. Q01은 초기 3위였지만 사용자 지시로 제외했다. “—”는 4장 모두의 값이 없어 범위 또는 최종 점수를 산출하지 않은 항목이다.')
table(['추출 실패 방법', '실패 항목', '기존 실행 결과에 기록된 사유'], [
    ['E04 LSD', 'R01~R04 하단: 4개', '검출 선분 중 각도·탐색 영역·가림·길이·극성 선택 조건을 통과한 선분 없음'],
    ['E08 FLD', 'R01~R03 하단: 3개', '최종 피팅용 표본 26·24·24개로 설정 최소 50개에 미달']])
paragraph('E04·E08의 상단 결과는 존재하지만, 이번에는 방법 하나가 상·하단을 모두 담당하는 기준으로 선별하여 제외했다. E08의 표본 수는 선택 선분에서 생성한 피팅용 표본 수이며, 원본 경계 픽셀 수와 같다는 의미가 아니다.')

heading('5. 결과에서 확인할 점')
paragraph('외부 구조물에서 선별된 3개 방법은 모두 하단 구조물의 범위가 최종 점수를 결정했다. E05 Zernike는 상단 범위가 0.000729°로 가장 작지만 하단을 포함한 점수는 0.029247°로 4위여서 보류됐다. 따라서 상단만 사용할 경우의 순위와 이번 순위는 다를 수 있다.')
paragraph('현재 QR은 Q02(0.032376°)와 Q04(0.040357°)만 사용한다. Q01(0.108362°)은 초기 3위 후보였으나 사용자 지시에 따라 제외했다. 수치와 최초 결과는 기존 전체 추출 결과 및 변경 전 보관본에 남아 있다.')
paragraph('E06은 기존 보고서의 PAE 변형 구현이다. 이 선별 결과를 원 논문 PAE의 완전한 재현이나 일반적인 우수성에 대한 결론으로 해석하지 않는다.')
paragraph('기준 4장에서 출력이 비슷하다는 것만으로 정확한 경계를 잡았거나 θ 변화에 반응한다고 결론 내릴 수 없다. 이번에 확인된 것은 저장된 결과의 추출 성공 여부와 기준 4장 사이의 수치 범위다. 별도의 경계 정답 대조, 통계적 균일성 검정, Y·θ 상관관계 검정은 수행하지 않았다.')

heading('6. 선별 방법의 기준별 원본 각도')
for family, title in [('external', '외부 구조물'), ('qr', 'QR')]:
    heading(title, 3)
    values = []
    for m in ranking[family]:
        if m['selected']:
            for target, data in m['targets'].items():
                values.append([m['method_id'], labels[target], *[fmt(data['image_angles_deg'][ref]) for ref in ['R01','R02','R03','R04']], fmt(data['range_deg'])])
    table(['방법', '대상', 'R01(°)', 'R02(°)', 'R03(°)', 'R04(°)', '범위(°)'], values)

heading('7. 산출물과 원본 보존 확인')
paragraph('현재 직선 측정값 48개, 중간 결과 JSON 48개, 추출 과정 그림 20개를 별도 폴더에서 제공한다. 선별 시 레코드 동일성, 복사 파일 해시 일치, 조합 참조 연결을 확인했다. 영상 재측정이나 C++ 알고리즘 코드 변경은 없었다.')
links([
    ('선별 결과 열람 및 20개 추출 그림', 'selection.html'),
    ('전체 순위와 대상별 범위', 'rankings.json'),
    ('선별 직선 48개', 'results/line_measurements.json'),
    ('외부 구조물 직선 24개', 'results/external_lines.json'),
    ('QR 직선 24개', 'results/qr_lines.json'),
    ('6개 방법 조합 × 기준 4장', 'results/combinations.json'),
    ('선별 기준·파일 해시·보존 확인 기록', 'selection_manifest.json'),
    ('기준 이미지의 원본 경로·SHA-256', 'source_configs/inputs_manifest.json'),
    ('기존 전체 추출 보고서', '../reference_line_methods_108_20261030/report.html')])
paragraph('보고서의 수치는 rankings.json과 기존 측정 JSON에서 가져왔다. Q01을 선택 대상에서 제외했으며 남은 측정값은 변경하지 않았다. 변경 전 결과는 history/before_qr_q02_q04_only.zip에 보존했다.')

heading('8. E06(PAE)·Q04(LSD)의 레퍼런스 대응 확인')
table(['방법', '확인 결과', '변형 또는 추가 처리'], [
    ['E06 PAE', 'PAE 기본 원리에 기반한 자체 변형 구현. 원 논문 그대로의 재현이라고 확인할 수 없음.',
     'EdgeLocator Method.md의 2차 면적식 참고. 기본 창 3×5를 3×9로 일반화, A/B 밝기 추정을 모서리 3화소에서 위·아래 행 평균으로 변경. 평활 보정·가변 창·수직 방향 일반화는 미구현.'],
    ['Q04 LSD', '저자 공개 LSD 1.6 핵심 검출 소스를 사용함. lsd.c·lsd.h는 보관된 IPOL 배포 ZIP과 바이트 동일.',
     '공식 함수 LineSegmentDetection 호출. scale 기본값 0.8을 1.0으로 설정. ROI·극성·선분 선택·여러 선분의 공통 직선 피팅은 별도 자체 후처리.']])
paragraph('PAE 코드의 a,b,c 계산은 Method.md의 고정 창 2차 기본형을 창 높이 2m+1로 일반화한 식에 대응한다. 다만 A/B 추정 등 주변 절차가 달라 전체 결과가 원문 구현과 동등하다고 확인한 것은 아니다. 2013년 원 논문 전문과의 완전한 대조 및 출력 동등성 시험은 미수행이다.')
paragraph('LSD의 최종 QR 각도는 “공개 LSD 검출 + 프로젝트용 선분 선택·통합” 결과다. 공개 소스를 그대로 사용했다는 사실과 최종 측정 전체가 원 논문 절차 그대로라는 주장은 구분해야 한다.')
paragraph('LSD 검증 범위: IPOL 공식 페이지의 버전 1.6 배포 정보 확인, 로컬 배포 ZIP과 컴파일 대상 lsd.c·lsd.h의 바이트·SHA-256 대조, CMake 빌드 연결 및 실제 호출 코드 확인. 이번 세션의 ZIP 재다운로드는 연결 오류로 완료되지 않아 공식 서버와의 새 다운로드 해시 대조는 미수행이다.')
table(['파일', '로컬 배포 ZIP 및 실제 사용 소스 SHA-256'], [
    ['lsd.c', 'f4b0c6194724e9d2cec01503fc2f4b7120f2e7d1fa570de6e67b4193a6b6723f'],
    ['lsd.h', '02edcbb6e7b19dfd7117ae1a13b9b5810c74e1cf69ea5492eafe7ccc65026eaa']])
links([
    ('PAE 실제 구현: method_pae, 392행부터', '../reference_line_methods_108_20261030/cpp/src/methods_points.cpp'),
    ('LSD 호출 및 선분 후처리: method_lsd / select_and_fit', '../reference_line_methods_108_20261030/cpp/src/methods_segments.cpp'),
    ('방법별 출처·변경 내역', '../reference_line_methods_108_20261030/configs/method_sources.json'),
    ('PAE 참고 구현 문서: EdgeLocator Method.md', 'https://github.com/juanse77/EdgeLocator/blob/master/Method.md'),
    ('PAE 원 논문 서지 정보', 'https://doi.org/10.1016/j.imavis.2012.10.005'),
    ('LSD 공식 논문·소스 배포 페이지', 'https://www.ipol.im/pub/art/2012/gjmr-lsd/')])

style = '''body{max-width:1220px;margin:34px auto;padding:0 26px;font-family:system-ui,"Malgun Gothic",sans-serif;color:#1c2b3b;line-height:1.7}h1{font-size:30px}h2{font-size:22px;margin-top:36px;border-bottom:2px solid #dbe4ed;padding-bottom:7px}h3{font-size:18px}.lead{background:#eaf4ee;padding:18px;border-left:4px solid #2b7650}.note{background:#f2f5fa;padding:16px;border-left:4px solid #55749b}table{border-collapse:collapse;width:100%;font-size:14px;margin:16px 0}td,th{border:1px solid #d7dee6;padding:8px 10px;text-align:left;vertical-align:top}th{background:#eef3f8}tbody tr:nth-child(even){background:#fafbfd}.table-wrap{overflow:auto}a{color:#165ca0}@media print{body{margin:0;padding:0;font-size:10pt;max-width:none}h1{font-size:20pt}h2{font-size:15pt}table{font-size:8pt}thead{display:table-header-group}tr{break-inside:avoid}h2,h3{break-after:avoid}.table-wrap{overflow:visible}a{color:inherit;text-decoration:none}}'''
document = '<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>직선 추출 방법 1차 분리 보고서</title><style>' + style + '</style></head><body>' + '\n'.join(parts) + '</body></html>'
(ROOT / 'first_screening_report.html').write_text(document, encoding='utf-8')
(ROOT / 'first_screening_report.txt').write_text('\n'.join(plain), encoding='utf-8')
assert len(original_rows) == 216 and sum(r['status']=='ok' for r in original_rows) == 209
assert len(selected_rows) == 48 and len(summary) == 5
for path, digest in inputs.items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
print('Created first_screening_report.html and first_screening_report.txt. Source files unchanged.')
