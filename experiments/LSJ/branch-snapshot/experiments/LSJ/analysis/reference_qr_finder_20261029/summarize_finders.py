"""Format saved reference measurements without extracting more features."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
data = json.loads((HERE/'finder_results.json').read_text(encoding='utf-8'))
names = {'L': '왼쪽', 'C': '중앙', 'R': '오른쪽'}
lines = [
    '기준 이미지 4장의 QR 파인더 중심 연결선 측정',
    '',
    '범위: 1차 center.bmp, 2차 center1.bmp/center2.bmp/center3.bmp만 측정했다.',
    '각 사진의 좌·중·우 QR에서 위쪽 두 파인더 패턴 중심 A,B를 잇는 직선 하나씩, 총 12개 직선이다.',
    '아래쪽 파인더 C는 패턴 배치를 확인하는 데 사용했다. A,B,C 세 점에 직선을 맞추지 않았다.',
    '',
    '방법',
    '- 각 QR 영역에서 Otsu 이진화 및 중첩 윤곽으로 파인더 후보 3개를 찾았다.',
    '- 각 파인더 바깥 검정 정사각형의 네 모서리를 원본 그레이스케일 영상의 cornerSubPix로 정밀화했다.',
    '- 모서리 정밀화는 반경 3px(7×7 창), 최대 100회, 종료 이동량 0.00001px 설정이다. 이 값들은 계산 설정이며 정확도 보장이 아니다.',
    '- 네 모서리의 대각선 교점을 패턴 중심으로 정의했다.',
    '- A=(xA,yA), B=(xB,yB), m=(yB-yA)/(xB-xA), b=yA-m*xA, phi=atan2(yB-yA,xB-xA).',
    '- x,y,b는 원본 이미지 px, m은 px/px, phi는 영상 수평선에 대한 각도(°)다. 오른쪽으로 내려갈수록 양수이다.',
    '- 원본 영상을 워핑하거나 QR끼리/사진끼리 평균하지 않았다.',
    '',
    '각도 결과 (단위: 영상 °)',
    '이미지 | 왼쪽 QR | 중앙 QR | 오른쪽 QR',
]
for row in data['rows']:
    lines.append(row['file']+' | '+' | '.join(f"{row['qrs'][n]['image_angle_deg']:.6f}" for n in names))
lines += ['최대-최소 | '+' | '.join(f"{data['max_minus_min_image_deg_by_qr'][n]:.6f}" for n in names),
    '', '직선 방정식 y=m*x+b (원본 이미지 좌표)', '이미지 | QR | m (px/px) | b (px)']
for row in data['rows']:
    for name, qr in row['qrs'].items():
        lines.append(f"{row['file']} | {names[name]} | {qr['slope_dy_dx']:.10f} | {qr['intercept_y_px']:.6f}")
lines += [
    '', '시각 확인',
    '- finder_lines_4references.png의 12개 QR 확대 패널에서 파인더 위치를 확인했다.',
    '- 하늘색 사각형/점: 정밀화한 바깥 모서리. 노란 +: 대각선 교점. 주황 선: A-B 기준선.',
    '- 네 사진 모두 각 QR의 올바른 파인더 3개가 검출됐다. 이것은 서브픽셀 정확도 검증과 다르다.',
    '', '해석 범위',
    '- 이번 선은 QR 내부 파인더 중심 연결선이다. 기존 QR 바깥 윗변/아랫변 각도와 측정 기준선이 다르다.',
    '- 기준 이미지 사이의 각도 차이가 관측됐다. 이번 결과만으로 이전 방법보다 안정적이라고 결론 내리지 않는다.',
    '- 각도는 로봇 θ나 보정 오차가 아니다. θ 변화 사진의 반응, 회귀 및 최종 정렬 성능은 이번에 측정하지 않았다.',
    '- 두 중심은 항상 직선 하나를 결정한다. 두 점에서 잔차가 0이라는 사실은 검출 정확도를 검증하지 않는다.',
    '- 표시 소수점 자릿수는 계산 출력이며 그 자릿수까지 실제 정확하다는 의미가 아니다.',
    '- 원본 네 장의 실행 전후 SHA256은 이전 저장 해시와 동일하다. C++ 알고리즘/설정/기존 결과는 변경하지 않았다.',
    '', '산출물',
    '- measure_finders.py: 검출과 측정 및 그림 생성.',
    '- finder_results.json: 모든 패턴의 초기/정밀 모서리, 중심점, 직선, 각도, 원본 경로/해시, 계산 설정.',
    '- finder_lines_4references.png: 실제 추출 직선 및 검출 모서리 시각화.',
    '- summarize_finders.py: 저장된 결과를 이 문서로 정리.',
]
(HERE/'report.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
print('Saved reference finder-line report.')
