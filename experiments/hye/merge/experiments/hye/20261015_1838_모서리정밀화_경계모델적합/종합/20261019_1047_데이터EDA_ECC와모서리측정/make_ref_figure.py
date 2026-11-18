# '기준 위치' 슬라이드 그림: center 사진의 QR 판에 QR 3개 네모, 중심점, 크기와 간격을 표시
import cv2, numpy as np, pandas as pd, pathlib
from PIL import Image, ImageDraw, ImageFont
from qr_bbox import qr_bbox
HERE = pathlib.Path(__file__).parent
ROOT = pathlib.Path(r"C:\Users\SSAFY\Downloads\[기업연계_세메스] 교보재\[기업연계_세메스] 교보재\extracted\SSAFY Test Image Set")
c = pd.read_csv(HERE.parent / '14_데이터_QR마커위치.csv').set_index('file').loc['center.bmp']
X0, X1, Y0, Y1, K = 1460, 2440, 905, 1290, 2          # QR 3개 + 위, 아래, 오른쪽 치수 표시 여백
full = cv2.imdecode(np.fromfile(str(ROOT / 'center.bmp'), np.uint8), cv2.IMREAD_GRAYSCALE)
B = {q: qr_bbox(full, c[q + 'x'], c[q + 'y']) for q in 'LCR'}                 # 실제 QR 경계 (x0, y0, x1, y1)
G = {q: ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[2] - b[0], b[3] - b[1]) for q, b in B.items()}   # 중심 x, y, 폭, 높이
g = full[Y0:Y1, X0:X1]
im = Image.fromarray(cv2.resize(g, None, fx=K, fy=K, interpolation=cv2.INTER_CUBIC)).convert('RGB')
d = ImageDraw.Draw(im)
F = ImageFont.truetype('C:/Windows/Fonts/NotoSansKR-Medium.ttf', 44)
BLUE, WHITE = (0, 113, 227), (255, 255, 255)
P = lambda x, y: ((x - X0) * K, (y - Y0) * K)

def label(xy, text, anchor='mm'):
    x, y = xy; b = d.textbbox((x, y), text, font=F, anchor=anchor)
    d.rounded_rectangle((b[0] - 10, b[1] - 6, b[2] + 10, b[3] + 6), 8, fill=WHITE)
    d.text((x, y), text, font=F, fill=BLUE, anchor=anchor)

def arrow(a, b):
    d.line([a, b], fill=BLUE, width=6)
    for p, q in ((a, b), (b, a)):
        v = np.subtract(q, p); v = v / np.linalg.norm(v) * 24; n = np.array([-v[1], v[0]]) * 0.5
        d.polygon([tuple(p), tuple(p + v + n), tuple(p + v - n)], fill=BLUE)

for q in 'LCR':
    x, y, w, h = G[q]
    d.rectangle([P(x - w / 2, y - h / 2), P(x + w / 2, y + h / 2)], outline=BLUE, width=6)
    cx, cy = P(x, y); d.ellipse((cx - 12, cy - 12, cx + 12, cy + 12), fill=BLUE)
    label(P(x, y - h / 2 - 30), f'{q}  ({x:.0f}, {y:.0f})')
# 크기 (가운데 QR 아래), 간격 (QR 중심끼리, 맨 아래)
x, y, w, h = G['C']
yb = y + h / 2 + 22; arrow(P(x - w / 2, yb), P(x + w / 2, yb)); label(P(x, yb + 20), f'폭 {w:.0f}px', 'mt')
xr = G['R'][0] + G['R'][2] / 2 + 22; arrow(P(xr, G['R'][1] - G['R'][3] / 2), P(xr, G['R'][1] + G['R'][3] / 2))
label(P(xr + 10, G['R'][1]), f'높이 {G["R"][3]:.0f}px', 'lm')
yg = Y1 - 28
for a, b in (('L', 'C'), ('C', 'R')):
    xa, xb = G[a][0], G[b][0]
    arrow(P(xa, yg), P(xb, yg)); label(P((xa + xb) / 2, yg - 6), f'간격 {xb - xa:.0f}px', 'mb')
im.save(HERE / 'img' / 'ref_center.png')
print({q: (round(v[0], 1), round(v[1], 1), v[2], v[3]) for q, v in G.items()})
