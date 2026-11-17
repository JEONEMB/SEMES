# 검증: center QR 기준 위치가 1~4px 틀려도 '이동량'이 바뀌는지
#   ECC 아핀(QR 패치 전체를 center 패치에 맞춤)으로 구한 변환 W 를 두 기준점에 각각 적용해서 이동량을 비교
#   (a) 기존 CSV 기준 중심 (OpenCV 검출 꼭짓점 평균)   (b) 사진에서 잰 QR 모듈 경계 중심
import cv2, numpy as np, pandas as pd, pathlib
from qr_bbox import qr_bbox
ROOT = pathlib.Path(r"C:\Users\SSAFY\Downloads\[기업연계_세메스] 교보재\[기업연계_세메스] 교보재\extracted\SSAFY Test Image Set")
Q = pd.read_csv(pathlib.Path(__file__).parent.parent / '14_데이터_QR마커위치.csv').set_index('file')
rd = lambda f: cv2.imdecode(np.fromfile(str(ROOT / f), np.uint8), cv2.IMREAD_GRAYSCALE)
cen = rd('center.bmp')
x0, x1, y0, y1 = 1815, 1995, 972, 1164                     # analysis_agent/measure_qr.py 의 가운데 QR 템플릿 창
tpl = cen[y0:y1, x0:x1]
A = np.array([Q.loc['center.bmp', 'Cx'], Q.loc['center.bmp', 'Cy']])
b = qr_bbox(cen, *A); B = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 150, 1e-7)
rows = []
for f in ['Theta/T+0.01.bmp', 'Theta/T-1.00.bmp', 'Theta/T+1.00.bmp', 'Y/Y+0.05.bmp', 'Y/Y-2.00.bmp', 'Y/Y+2.00.bmp']:
    im = rd(f); pad = 130
    m = cv2.matchTemplate(im[y0 - pad:y1 + pad, x0 - pad:x1 + pad], tpl, cv2.TM_CCOEFF_NORMED)
    _, _, _, loc = cv2.minMaxLoc(m); dx0, dy0 = loc[0] - pad, loc[1] - pad
    W = np.float32([[1, 0, 0], [0, 1, 0]])
    _, W = cv2.findTransformECC(tpl, im[y0 + dy0:y1 + dy0, x0 + dx0:x1 + dx0], W, cv2.MOTION_AFFINE, crit, None, 3)
    move = lambda P: (W[:, :2] @ (P - [x0, y0]) + W[:, 2] + [x0 + dx0, y0 + dy0]) - P   # 기준점 P 가 옮겨간 양
    ma, mb = move(A), move(B)
    rows.append(dict(file=f.split('/')[1], csv_dCx=Q.loc[f.split('/')[1], 'dCx'], dx_a=ma[0], dx_b=mb[0], diff_px=mb[0] - ma[0]))
r = pd.DataFrame(rows).round(4); print('기준 중심 차이 (b - a):', (B - A).round(2)); print(r.to_string(index=False))
r.to_csv(pathlib.Path(__file__).parent / 'img' / 'ref_bias_check.csv', index=False, encoding='utf-8-sig')
