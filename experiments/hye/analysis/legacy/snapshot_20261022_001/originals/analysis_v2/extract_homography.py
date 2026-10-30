"""QR 3개가 붙은 판 영역을 center 사진에 원근 변환(호모그래피, ECC)으로 맞춰서
판 네 귀퉁이와 각 QR 중심의 이동량을 구한다. QR 내부 픽셀 전부를 쓰는 측정."""
import sys, re, cv2, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(r"C:\Users\SSAFY\Downloads\[기업연계_세메스] 교보재\[기업연계_세메스] 교보재\extracted\SSAFY Test Image Set")
OUT = Path(__file__).parent
q = pd.read_csv(Path(__file__).parent.parent / '14_데이터_QR마커위치.csv')
c0 = q[q.file == 'center.bmp'].iloc[0]
X0, X1 = int(c0.Lx - 110), int(c0.Rx + 110)
Y0, Y1 = int(min(c0.Ly, c0.Ry) - 120), int(max(c0.Ly, c0.Ry) + 120)
def read(p): return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_GRAYSCALE).astype(np.float32)
cen = read(ROOT / 'center.bmp'); tpl = cen[Y0:Y1, X0:X1].copy()
pts = np.float32([[X0, Y0], [X1, Y0], [X1, Y1], [X0, Y1], [c0.Lx, c0.Ly], [c0.Cx, c0.Cy], [c0.Rx, c0.Ry]])
names = ['좌상', '우상', '우하', '좌하', 'QR좌', 'QR중', 'QR우']
crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 1e-7)
rows = []
files = list(q.file) if len(sys.argv) < 2 else sys.argv[1:]
for i, f in enumerate(files):
    r = q[q.file == f].iloc[0]
    p = ROOT / 'center.bmp' if f == 'center.bmp' else ROOT / ('Theta' if f[0] == 'T' else 'Y') / f
    im = read(p)
    sx, sy = int(round((r.dLx + r.dCx + r.dRx) / 3)), int(round((r.dLy + r.dCy + r.dRy) / 3))
    crop = im[Y0 + sy:Y1 + sy, X0 + sx:X1 + sx]
    W = np.eye(3, dtype=np.float32)
    cc, W = cv2.findTransformECC(tpl, crop, W, cv2.MOTION_HOMOGRAPHY, crit, None, 5)
    loc = pts - np.float32([X0, Y0])
    m = cv2.perspectiveTransform(loc[None], W)[0] + np.float32([X0 + sx, Y0 + sy])
    d = m - pts
    row = {'file': f, 'cc': cc, 'h20': W[2, 0], 'h21': W[2, 1]}
    for n, (dx, dy) in zip(names, d): row[f'{n}_dx'] = dx; row[f'{n}_dy'] = dy
    rows.append(row)
    if i % 20 == 0: print(i, len(files), f, round(cc, 5), flush=True)
pd.DataFrame(rows).to_csv(OUT / ('homography_features.csv' if len(sys.argv) < 2 else 'homography_test.csv'), index=False, encoding='utf-8-sig')
