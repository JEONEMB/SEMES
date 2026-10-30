"""발표 자료용 새 그림 3개.
fig_measure.png  : 측정 기준 (a = center 사진의 QR 중심, b = Y+0.50 사진의 QR 중심, b - a)
fig_twin.png     : 헷갈리는 짝 예시 (Y +0.50mm vs Θ +0.09)
fig_theta0.png   : Θ 구간별 0점 어긋남 (18번 그림의 Θ 버전)"""
import numpy as np, pandas as pd, cv2, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
HERE = Path(__file__).parent; PROJ = HERE.parent; OUT = HERE/'deck_img'; OUT.mkdir(exist_ok=True)
ROOT = Path(r"C:\Users\SSAFY\Downloads\[기업연계_세메스] 교보재\[기업연계_세메스] 교보재\extracted\SSAFY Test Image Set")
plt.rcParams['font.family'] = 'Malgun Gothic'; plt.rcParams['axes.unicode_minus'] = False
TXT, SUB, GRID, BLUE, ORG, GRN = '#1f1f1e', '#5f5e58', '#e4e3dc', '#2a78d6', '#eb6834', '#1baf7a'
q = pd.read_csv(PROJ/'14_데이터_QR마커위치.csv').set_index('file'); c0 = q.loc['center.bmp']
def load(f):
    p = ROOT/'center.bmp' if f == 'center.bmp' else ROOT/('Theta' if f[0] == 'T' else 'Y')/f
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_GRAYSCALE)

# ---------- 1. 측정 기준 ----------
cen, img = load('center.bmp'), load('Y+0.50.bmp'); r = q.loc['Y+0.50.bmp']
fig = plt.figure(figsize=(15, 4.9)); gs = fig.add_gridspec(1, 3, width_ratios=[1.55, 1, 1])
a = fig.add_subplot(gs[0]); X0, X1, Y0, Y1 = int(c0.Lx - 260), int(c0.Rx + 260), int(c0.Cy - 330), int(c0.Cy + 330)
a.imshow(cen[Y0:Y1, X0:X1], cmap='gray', extent=(X0, X1, Y1, Y0))
for k, lab in [('L', '좌'), ('C', '중'), ('R', '우')]:
    cx, cy, w, h = c0[f'{k}x'], c0[f'{k}y'], c0[f'{k}w'], c0[f'{k}h']
    a.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, fill=False, ec=ORG, lw=2))
    a.plot(cx, cy, '+', color=ORG, ms=14, mew=2.5); a.text(cx, cy - h / 2 - 48, f'{lab} QR', ha='center', color=ORG, fontsize=12, fontweight='bold')
a.add_patch(Rectangle((c0.Lx - 110, c0.Cy - 120), c0.Rx - c0.Lx + 220, 240, fill=False, ec=BLUE, lw=1.8, ls='--'))
a.text(c0.Lx - 110, c0.Cy + 150, 'QR 판 영역 (판 원근 변환, 픽셀 직접 방법이 쓰는 범위)', color=BLUE, fontsize=11)
a.set_title('① 무엇을 재나: QR 3개의 네모 중심 (주황 +), QR 판 전체 (파란 점선)', loc='left', fontsize=13, color=TXT); a.axis('off')
Z = 46
for ax, im, ttl, mark in [(fig.add_subplot(gs[1]), cen, '② a = center 사진의 중 QR 중심', 'a'), (fig.add_subplot(gs[2]), img, '③ b = Y +0.50mm 사진의 중 QR 중심', 'b')]:
    cx, cy = c0.Cx, c0.Cy; ax.imshow(im[int(cy - Z):int(cy + Z), int(cx - Z):int(cx + Z)], cmap='gray', extent=(cx - Z, cx + Z, cy + Z, cy - Z), interpolation='nearest')
    ax.plot(c0.Cx, c0.Cy, 'o', mfc='none', mec=ORG, ms=16, mew=2.5)
    if mark == 'b':
        ax.plot(r.Cx, r.Cy, 'o', color=BLUE, ms=10)
        ax.annotate('', xy=(r.Cx, r.Cy - 14), xytext=(c0.Cx, c0.Cy - 14), arrowprops=dict(arrowstyle='->', color=BLUE, lw=2.5))
        ax.text((r.Cx + c0.Cx) / 2, c0.Cy - 20, f'b - a = {r.dCx:+.2f}px', ha='center', va='bottom', color=BLUE, fontsize=13, fontweight='bold',
                bbox=dict(fc='white', ec='none', alpha=0.85))
    ax.set_title(ttl, loc='left', fontsize=13, color=TXT); ax.axis('off')
fig.text(0.01, 0.01, 'a는 항상 center 사진에서, b는 각 사진에서 같은 방법(QR 네모 서브픽셀 정합)으로 잰다. 좌, 중, 우 세 값의 평균이 "QR 이동량". 확대 화면은 중 QR 중심 주변 92x92px (실제 픽셀 크기).',
         fontsize=10, color=SUB)
plt.tight_layout(rect=(0, 0.04, 1, 1)); plt.savefig(OUT/'fig_measure.png', dpi=130, facecolor='white'); plt.close()

# ---------- 2. 헷갈리는 짝 ----------
f15 = pd.read_csv(PROJ/'15_데이터_파인더_볼트이동량.csv').set_index('file')
A, B = load('Y+0.50.bmp'), load('T+0.09.bmp')
fig, axs = plt.subplots(1, 3, figsize=(15, 3.3), gridspec_kw={'width_ratios': [1, 1, 1]})
X0, X1, Y0, Y1 = int(c0.Lx - 130), int(c0.Rx + 130), int(c0.Cy - 130), int(c0.Cy + 130)
for ax, im, f, ttl, col in [(axs[0], A, 'Y+0.50.bmp', 'Y +0.50mm 사진 (Θ = 0)', ORG), (axs[1], B, 'T+0.09.bmp', 'Θ +0.09 사진 (Y = 0)', GRN)]:
    ax.imshow(im[Y0:Y1, X0:X1], cmap='gray'); ax.axis('off')
    rr = q.loc[f]; mx = rr[['dLx', 'dCx', 'dRx']].mean()
    ax.set_title(f'{ttl}\nQR 3개 평균 가로 이동 {mx:+.2f}px', loc='left', fontsize=13, color=TXT)
    for s in ax.spines.values(): s.set_visible(False)
    ax.add_patch(Rectangle((0, 0), X1 - X0 - 1, Y1 - Y0 - 1, fill=False, ec=col, lw=5))
d = A[Y0:Y1, X0:X1].astype(float) - B[Y0:Y1, X0:X1].astype(float)
axs[2].imshow(d, cmap='RdBu_r', vmin=-60, vmax=60); axs[2].axis('off')
axs[2].set_title('두 사진의 차이 (빨강/파랑 = 밝기 차이)\n거의 회색 = QR만 보면 거의 같은 사진', loc='left', fontsize=13, color=TXT)
fig.text(0.01, 0.02, f'뒤쪽 볼트(QR 밖)는 다르게 움직인다: Y +0.50은 {f15.loc["Y+0.50.bmp","bolt_dx"]:+.1f}px, Θ +0.09는 {f15.loc["T+0.09.bmp","bolt_dx"]:+.1f}px (거리가 다른 물체의 시차).',
         fontsize=11, color=SUB)
plt.tight_layout(rect=(0, 0.08, 1, 1)); plt.savefig(OUT/'fig_twin.png', dpi=130, facecolor='white'); plt.close()

# ---------- 3. Θ 0점 어긋남 ----------
q2 = q.reset_index(); q2['ax'] = q2.file.str[0]; q2['x'] = q2[['dLx', 'dCx', 'dRx']].mean(1)
t = q2[q2.ax == 'T'].copy(); t['seg'] = np.where(t.value > 0, 'Θ +쪽 (13~26분, + 방향)', 'Θ -쪽 (27~33분, - 방향)')
S = np.mean([np.polyfit(g.value, g.x, 1)[0] for _, g in t.groupby('seg')])
t['res'] = t.x - S * t.value
fig, (a, b) = plt.subplots(1, 2, figsize=(14.5, 5.4), gridspec_kw={'width_ratios': [2.2, 1]})
icp = {}
for (s, g), col in zip(sorted(t.groupby('seg'), key=lambda z: z[0], reverse=True), [BLUE, ORG]):
    g = g.sort_values('value'); p = np.polyfit(g.value, g.res, 1); icp[s] = p[1]
    a.scatter(g.value, g.res, s=12, color=col, label=s, zorder=3)
    xs = np.array([0, g.value.max() if g.value.max() > 0 else g.value.min()]); a.plot(xs, np.polyval(p, xs), color=col, lw=2)
    a.scatter([0], [p[1]], s=90, color=col, edgecolor='white', lw=2, zorder=4)
a.scatter([0], [0], marker='*', s=240, color=TXT, zorder=5, label='center 사진')
a.axvline(0, color=SUB, lw=1, ls=':'); a.axhline(0, color=GRID, lw=1)
a.set_xlabel('명령한 Θ', color=TXT); a.set_ylabel('예상 위치에서 벗어난 양 (px)', color=TXT)
a.set_title(f'① Θ 이동 2번: 구간마다 0점이 다르다 (공통 기울기 {S:.2f}px/Θ를 뺀 나머지)', loc='left', fontsize=13, color=TXT)
a.legend(frameon=False, fontsize=10, loc='lower left'); a.grid(color=GRID); a.set_axisbelow(True)
for sp in ['top', 'right']: a.spines[sp].set_visible(False)
names = list(icp) + ['center']; vals = [icp[n] for n in icp] + [0.0]; cols = [BLUE, ORG, TXT]
for i, (n, v, c) in enumerate(zip(names, vals, cols)):
    b.scatter([v], [len(names) - 1 - i], s=150 if n != 'center' else 260, color=c, marker='o' if n != 'center' else '*', zorder=3)
    b.text(v, len(names) - 1 - i + 0.25, f'{v:+.2f}px (Θ {v / S:+.4f})', ha='center', fontsize=10, color=TXT)
b.axvline(0, color=SUB, ls=':'); b.set_yticks(range(len(names))); b.set_yticklabels(names[::-1], fontsize=10)
b.set_xlim(-1.0, 0.4); b.set_ylim(-0.6, len(names) - 0.3); b.set_xlabel('0점 위치 (px)')
b.set_title('② 같은 "Θ=0"인데 실제로 멈춘 자리', loc='left', fontsize=13, color=TXT)
for sp in ['top', 'right', 'left']: b.spines[sp].set_visible(False)
plt.tight_layout(); plt.savefig(OUT/'fig_theta0.png', dpi=130, facecolor='white'); plt.close()
print('ok', {k: round(v, 3) for k, v in icp.items()}, round(S, 2))
