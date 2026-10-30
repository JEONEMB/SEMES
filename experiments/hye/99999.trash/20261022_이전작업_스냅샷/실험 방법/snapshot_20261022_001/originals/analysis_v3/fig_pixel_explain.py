"""픽셀 직접 방법(G1) 설명 그림: QR 판 영역, 픽셀 크기, Y와 Θ가 QR 판 위 어디서 다르게 움직이는지."""
import numpy as np, pandas as pd, cv2, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.colors import TwoSlopeNorm
import gn_pixel as G
HERE = G.HERE; PROJ = G.PROJ; OUT = HERE/'deck_img'
plt.rcParams['font.family'] = 'Malgun Gothic'; plt.rcParams['axes.unicode_minus'] = False
TXT, SUB, BLUE, ORG, GRN, RED = '#1f1f1e', '#5f5e58', '#2a78d6', '#eb6834', '#1baf7a', '#c00000'
cen = cv2.imdecode(np.fromfile(str(G.ROOT/'center.bmp'), np.uint8), cv2.IMREAD_GRAYSCALE)
B = np.linalg.lstsq(np.c_[G.X, np.ones(len(G.X))], G.CD, rcond=None)[0]       # 전체 사진으로 학습한 귀퉁이 이동 모델
PXMM = B[1][0::2].mean()                                                          # Y 1mm당 판의 가로 이동 px = 판 위 1mm의 화면 크기
TEQ = 0.5 * PXMM / B[0][0::2].mean()                                              # Y 0.5mm와 평균 이동이 같은 Θ
h = pd.read_csv(PROJ/'analysis_v2/homography_features.csv').set_index('file'); CN = ['좌상', '우상', '우하', '좌하']
def corners(f): return np.array([[h.loc[f, f'{c}_dx'], h.loc[f, f'{c}_dy']] for c in CN])

fig = plt.figure(figsize=(16, 9.6)); gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1], height_ratios=[1, 1.05], hspace=0.32, wspace=0.2)
# (a) QR 판 영역 ---------------------------------------------------------------
a = fig.add_subplot(gs[0, 0]); x0, x1, y0, y1 = 1250, 2560, 700, 1450
a.imshow(cen[y0:y1, x0:x1], cmap='gray', extent=(x0, x1, y1, y0))
a.add_patch(Rectangle((G.X0, G.Y0), G.WW, G.HH, fill=False, ec=BLUE, lw=3))
for (cx, cy), lab in zip(G.SRC, CN):
    a.plot(cx, cy, 'o', color=BLUE, ms=9); a.text(cx + (12 if lab[0] == '좌' else -12), cy + (-14 if lab[1] == '상' else 30), lab, color=BLUE, fontsize=12, fontweight='bold',
                                                  ha='left' if lab[0] == '좌' else 'right')
a.annotate('', xy=(G.X1, G.Y1 + 70), xytext=(G.X0, G.Y1 + 70), arrowprops=dict(arrowstyle='<->', color=TXT))
a.text((G.X0 + G.X1) / 2, G.Y1 + 108, f'가로 {G.WW}px, 약 {G.WW / PXMM:.0f}mm', ha='center', fontsize=12, color=TXT, bbox=dict(fc='white', ec='none', alpha=0.85))
a.annotate('', xy=(G.X1 + 40, G.Y1), xytext=(G.X1 + 40, G.Y0), arrowprops=dict(arrowstyle='<->', color=TXT))
a.text(G.X1 + 55, (G.Y0 + G.Y1) / 2, f'세로 {G.HH}px\n약 {G.HH / PXMM:.0f}mm', va='center', fontsize=12, color=TXT, bbox=dict(fc='white', ec='none', alpha=0.85))
a.set_title(f'① QR 판 영역 (파란 상자): QR 3개와 주변 판. 픽셀 {G.WW * G.HH:,}개를 모두 비교에 씀', loc='left', fontsize=13, color=TXT); a.axis('off')
# (b) 픽셀 크기 -----------------------------------------------------------------
b = fig.add_subplot(gs[0, 1]); zx, zy, Z = int(G.X0 + 113), int(G.Y0 + 34), 22
b.imshow(cen[zy:zy + Z, zx:zx + Z], cmap='gray', interpolation='nearest', extent=(0, Z, Z, 0))
for k in range(Z + 1): b.axhline(k, color='#9ec5f4', lw=0.6); b.axvline(k, color='#9ec5f4', lw=0.6)
b.add_patch(Rectangle((1, 1), 1, 1, fill=False, ec=ORG, lw=3)); b.text(2.3, 1.75, f'한 칸 = 1px, 약 {1000 / PXMM:.0f}µm', color=ORG, fontsize=12, fontweight='bold',
                                                                      bbox=dict(fc='white', ec='none', alpha=0.9))
b.annotate('', xy=(1 + 0.5 * PXMM, Z - 2.2), xytext=(1, Z - 2.2), arrowprops=dict(arrowstyle='->', color=BLUE, lw=2.5))
b.text(1, Z - 3, f'Y 0.5mm 이동 = {0.5 * PXMM:.1f}칸', color=BLUE, fontsize=12, fontweight='bold', bbox=dict(fc='white', ec='none', alpha=0.9))
b.add_patch(Rectangle((14, 13), 1, 1, fill=False, ec=RED, lw=2)); b.add_patch(Rectangle((14, 13), 0.03, 1, color=RED))
b.text(15.3, 13.3, 'Y와 Θ를 가르는\n차이 약 0.03칸\n(빨간 실선 두께)', color=RED, fontsize=11, va='top', bbox=dict(fc='white', ec='none', alpha=0.9))
b.set_title('② 픽셀은 얼마나 촘촘한가 (판 왼쪽 위 22x22px 확대)', loc='left', fontsize=13, color=TXT); b.axis('off')
# (c) 모델: Θ와 Y의 세로 이동 차이 지도 ------------------------------------------------
c = fig.add_subplot(gs[1, 0])
d = (G.PHI @ (TEQ * B[0])) - (G.PHI @ (0.5 * B[1])); dy = d[:, 1].reshape(G.HH, G.WW); dx = d[:, 0].reshape(G.HH, G.WW)
dyc = dy - dy.mean()
plate = cen[G.Y0:G.Y1, G.X0:G.X1]
c.imshow(plate, cmap='gray', extent=(0, G.WW, G.HH, 0), alpha=0.45)
lim = np.abs(dyc).max(); im = c.imshow(dyc, cmap='RdBu_r', norm=TwoSlopeNorm(0, -lim, lim), extent=(0, G.WW, G.HH, 0), alpha=0.6)
cb = fig.colorbar(im, ax=c, orientation='horizontal', fraction=0.06, pad=0.03, aspect=40); cb.set_label('세로 이동 차이 (px, 빨강 = 아래로, 파랑 = 위로)', fontsize=10)
for (cx, cy), lab in zip(G.SRC - [G.X0, G.Y0], CN):
    v = dyc[min(int(cy), G.HH - 1), min(int(cx), G.WW - 1)]
    c.annotate('', xy=(cx, cy + v * 2500), xytext=(cx, cy), arrowprops=dict(arrowstyle='->', color=TXT, lw=2.2))
    c.text(cx + (10 if lab[0] == '좌' else -10), cy + (25 if lab[1] == '상' else -12), f'{lab} {v:+.3f}px', ha='left' if lab[0] == '좌' else 'right', fontsize=11, fontweight='bold', color=TXT,
           bbox=dict(fc='white', ec='none', alpha=0.85))
c.set_title(f'③ 모델: Θ {TEQ:.3f} 사진과 Y 0.50mm 사진의 세로 이동 차이 (평균 가로 이동은 둘 다 {0.5 * PXMM:.2f}px)\n'
            '왼쪽 변은 위아래가 좁아지고 오른쪽 변은 넓어짐 = 판이 비스듬해짐. 화살표는 2500배 과장', loc='left', fontsize=12, color=TXT)
c.set_xlim(-5, G.WW + 5); c.set_ylim(G.HH + 5, -5); c.axis('off')
# (d) 측정: 실제 두 사진의 귀퉁이 세로 이동 -------------------------------------------
e = fig.add_subplot(gs[1, 1])
mY, mT = corners('Y+0.50.bmp'), corners('T+0.09.bmp')
pY, pT = (0.5 * B[1]).reshape(4, 2), (0.09 * B[0]).reshape(4, 2)
R = lambda arr: (arr - arr.mean(0))[:, 1]
xs = np.arange(4); w = 0.2
noise = np.array([np.array([R(corners(f'T+0.{k:02d}.bmp')) for k in range(5, 14)]).std(0)])[0]
e.bar(xs - 1.5 * w, R(pY), w, color='#f6c4a6', label='모델 Y 0.50mm'); e.bar(xs - 0.5 * w, R(mY), w, color=ORG, label='측정 Y +0.50 사진')
e.bar(xs + 0.5 * w, R(pT), w, color='#b6e2cf', label='모델 Θ 0.09'); e.bar(xs + 1.5 * w, R(mT), w, color=GRN, label='측정 Θ +0.09 사진')
for i in range(4): e.add_patch(Rectangle((i - 2 * w, -noise[i]), 4 * w, 2 * noise[i], color='#c3c2b7', alpha=0.45, zorder=0))
e.axhline(0, color=SUB, lw=1); e.set_xticks(xs); e.set_xticklabels(CN, fontsize=12)
e.set_ylabel('판 귀퉁이 세로 이동 (평균 뺌, px)'); e.legend(fontsize=9.5, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
e.set_title('④ 측정: 실제 두 사진도 비슷한 모양\n회색 띠 = 귀퉁이 하나의 측정 잡음. 신호가 잡음의 약 2배라 아슬아슬', loc='left', fontsize=12, color=TXT)
for sp in ['top', 'right']: e.spines[sp].set_visible(False)
e.grid(axis='y', color='#e4e3dc'); e.set_axisbelow(True)
fig.text(0.01, 0.005, f'판 위 1mm = 화면 {PXMM:.1f}px (Y 1mm 이동량). 가로 방향은 두 사진의 평균 이동만 다르고 모양은 같아 구분 단서가 없다. 단서는 세로 방향의 원근(판 양끝 높이 변화)뿐. '
         'G1은 귀퉁이 4개가 아니라 판 픽셀 전부로 이 모양을 맞추기 때문에 귀퉁이 하나보다 잡음이 작다.', fontsize=10, color=SUB)
plt.savefig(PROJ/'31_픽셀직접방법_설명.png', dpi=120, facecolor='white', bbox_inches='tight')
print('TEQ', round(TEQ, 4), 'PXMM', round(PXMM, 2), 'noise', np.round(noise, 4))
