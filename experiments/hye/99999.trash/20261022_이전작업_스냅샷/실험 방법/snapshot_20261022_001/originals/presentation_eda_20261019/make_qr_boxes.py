# 기초 정보 슬라이드용: 가운데 QR 확대 이미지에 QR 위치 네모를 직접 그림
#   13장 모두: 파란 점선 = center QR 위치
#   최소 이동 6장: + 초록 실선 = 이 사진 QR 위치
#   최대 이동 6장: + 초록 점선 = 같은 구간 최소 이동 QR 위치, 빨강 실선 = 이 사진 QR 위치
import cv2, numpy as np, pandas as pd, pathlib
from qr_bbox import qr_bbox
HERE = pathlib.Path(__file__).parent
ROOT = pathlib.Path(r"C:\Users\SSAFY\Downloads\[기업연계_세메스] 교보재\[기업연계_세메스] 교보재\extracted\SSAFY Test Image Set")
Q = pd.read_csv(HERE.parent / '14_데이터_QR마커위치.csv').set_index('file')
X0, X1, Y0, Y1 = 1720, 2080, 970, 1170   # make_qr_crops.py 와 같은 창
K = 4                                    # 선이 선명하도록 4배로 키워서 그림
BLUE, GREEN, RED = (227, 113, 0), (74, 163, 22), (38, 38, 220)   # BGR
FIRST = ['Y-0.10', 'Y+0.10', 'Y-0.05', 'Y+0.05', 'T-0.01', 'T+0.01']
MAX = ['Y-2.00', 'Y+2.00', 'Y-1.95', 'Y+1.95', 'T-1.00', 'T+1.00']

_cache = {}
def box(f):
    # 가운데 QR의 실제 경계 (원본 좌표) -> 확대한 잘라낸 그림 좌표
    if f not in _cache:
        p = ROOT / 'center.bmp' if f == 'center' else ROOT / ('Theta' if f[0] == 'T' else 'Y') / (f + '.bmp')
        g = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_GRAYSCALE)
        r = Q.loc[f + '.bmp' if f != 'center' else 'center.bmp']
        _cache[f] = qr_bbox(g, r.Cx, r.Cy)
    x0, y0, x1, y1 = _cache[f]
    return ((x0 - X0) * K, (y0 - Y0) * K, (x1 - X0) * K, (y1 - Y0) * K)

def dashed(im, b, color, t, dash=28, gap=18):
    x0, y0, x1, y1 = map(int, b)
    for (a, c) in [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))]:
        L = int(np.hypot(c[0] - a[0], c[1] - a[1])); d = ((c[0] - a[0]) / L, (c[1] - a[1]) / L)
        for s in range(0, L, dash + gap):
            e = min(s + dash, L)
            cv2.line(im, (int(a[0] + d[0] * s), int(a[1] + d[1] * s)), (int(a[0] + d[0] * e), int(a[1] + d[1] * e)), color, t)

# 구간마다 (첫 이동, 최대 이동) 짝. 같은 구간의 두 사진에는 같은 네모 3개를 겹친다
PAIR = dict(zip(FIRST, MAX)); PAIR.update({v: v for v in MAX}); FIRST_OF = {v: k for k, v in zip(FIRST, MAX)}; FIRST_OF.update({k: k for k in FIRST})
shift = {}
for f in ['center'] + FIRST + MAX:
    p = ROOT / 'center.bmp' if f == 'center' else ROOT / ('Theta' if f[0] == 'T' else 'Y') / (f + '.bmp')
    g = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_GRAYSCALE)[Y0:Y1, X0:X1]
    im = cv2.cvtColor(cv2.resize(g, None, fx=K, fy=K, interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2BGR)
    if f in FIRST:                                                       # 최소 이동 6장: + 초록 실선 = 이 사진 QR 위치
        x0, y0, x1, y1 = map(int, box(f)); cv2.rectangle(im, (x0, y0), (x1, y1), GREEN, 18)
    if f in MAX:                                                         # 최대 이동 6장: + 초록 점선 = 같은 구간 최소 이동 위치, 빨강 실선 = 이 사진 위치
        dashed(im, tuple(map(int, box(FIRST_OF[f]))), GREEN, 14, dash=30, gap=22)
        x0, y0, x1, y1 = map(int, box(f)); cv2.rectangle(im, (x0, y0), (x1, y1), RED, 18)
    dashed(im, tuple(map(int, box('center'))), BLUE, 14)                 # 13장 모두: 파란 점선 = center QR 위치 (맨 위에 그려서 항상 보이게)
    if f != 'center':
        shift[f] = Q.loc[f + '.bmp', 'dCx']
    cv2.imencode('.png', im)[1].tofile(str(HERE / 'img' / f'qrbox_{f}.png'))
pd.Series(shift).round(1).to_csv(HERE / 'img' / 'qrbox_shift_px.csv', header=['dCx_px'])
print(pd.Series(shift).round(1).to_dict())
