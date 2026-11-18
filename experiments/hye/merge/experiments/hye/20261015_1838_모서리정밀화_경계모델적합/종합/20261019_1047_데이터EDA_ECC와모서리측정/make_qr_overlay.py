# 기초 정보 슬라이드용: 구간(trial)마다 최대 이동 사진 한 장에 QR 위치 세 개를 겹쳐 그림 (모든 사진이 같은 해상도, 같은 창이라 좌표를 그대로 겹칠 수 있음)
#   파란 가는 실선 = center의 QR 위치 (초록과 거의 겹쳐서 실선으로 깔아 둠), 초록 점선 = 첫 이동 사진의 QR 위치, 빨강 실선 = 최대 이동 사진의 QR 위치
import cv2, numpy as np, pandas as pd, pathlib
HERE = pathlib.Path(__file__).parent
ROOT = pathlib.Path(r"C:\Users\SSAFY\Downloads\[기업연계_세메스] 교보재\[기업연계_세메스] 교보재\extracted\SSAFY Test Image Set")
Q = pd.read_csv(HERE.parent / '14_데이터_QR마커위치.csv').set_index('file')
X0, X1, Y0, Y1 = 1720, 2080, 970, 1170   # make_qr_crops.py 와 같은 창
K = 4
BLUE, GREEN, RED = (227, 113, 0), (74, 163, 22), (38, 38, 220)   # BGR
# (구간 이름, 쪽, 첫 이동, 최대 이동)
PAIRS = [('Y01', '-', 'Y-0.10', 'Y-2.00'), ('Y01', '+', 'Y+0.10', 'Y+2.00'),
         ('Y005', '-', 'Y-0.05', 'Y-1.95'), ('Y005', '+', 'Y+0.05', 'Y+1.95'),
         ('T', '-', 'T-0.01', 'T-1.00'), ('T', '+', 'T+0.01', 'T+1.00')]

def box(f):
    r = Q.loc[f + '.bmp']
    return tuple(int(v * K) for v in (r.Cx - r.Cw / 2 - X0, r.Cy - r.Ch / 2 - Y0, r.Cx + r.Cw / 2 - X0, r.Cy + r.Ch / 2 - Y0))

def dashed(im, b, color, t, dash=44, gap=26):
    x0, y0, x1, y1 = b
    for a, c in [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))]:
        L = int(np.hypot(c[0] - a[0], c[1] - a[1])); d = ((c[0] - a[0]) / L, (c[1] - a[1]) / L)
        for s in range(0, L, dash + gap):
            e = min(s + dash, L)
            cv2.line(im, (int(a[0] + d[0] * s), int(a[1] + d[1] * s)), (int(a[0] + d[0] * e), int(a[1] + d[1] * e)), color, t)

rows = []
for trial, sign, first, last in PAIRS:
    p = ROOT / ('Theta' if last[0] == 'T' else 'Y') / (last + '.bmp')
    g = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_GRAYSCALE)[Y0:Y1, X0:X1]
    im = cv2.cvtColor(cv2.resize(g, None, fx=K, fy=K, interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2BGR)
    b = box('center'); cv2.rectangle(im, b[:2], b[2:], BLUE, 8)
    dashed(im, box(first), GREEN, 14, dash=30, gap=22)
    x0, y0, x1, y1 = box(last); cv2.rectangle(im, (x0, y0), (x1, y1), RED, 22)
    out = f'qrov_{trial}{"m" if sign == "-" else "p"}.png'
    cv2.imencode('.png', im)[1].tofile(str(HERE / 'img' / out))
    rows.append((out, first, last, Q.loc[first + '.bmp', 'dCx'], Q.loc[last + '.bmp', 'dCx']))
pd.DataFrame(rows, columns=['img', 'first', 'last', 'first_dx_px', 'last_dx_px']).round(1).to_csv(HERE / 'img' / 'qrov_index.csv', index=False)
print(pd.read_csv(HERE / 'img' / 'qrov_index.csv'))
