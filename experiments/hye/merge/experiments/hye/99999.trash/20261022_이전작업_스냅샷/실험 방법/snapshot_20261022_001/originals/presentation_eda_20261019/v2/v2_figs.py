# 정리본 v2 그림 (모두 모서리 630점 = 대각선 모서리 제외 기준, 숫자는 v2_data.py 결과)
#   ref_corners.png   4번  center 모서리 기준 위치 (초록 = 사용 630점, 주황 = 대각선 125점 제외, 파랑 = 모서리 범위로 잡은 QR 네모)
#   zero.png          5번  trial마다 0점 위치 (평균 이동에서 공통 기울기를 빼고 남은 양 vs 명령값, 직선이 0을 지나는 높이 = 0점)
#   path.png          6번  이동 경로 (평균 가로 이동 vs 세로 이동, 명령값 크기 순 연결)
#   n1~n8.png         노이즈 4분할 2장
import pathlib, numpy as np, pandas as pd, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from PIL import Image, ImageDraw
HERE = pathlib.Path(__file__).parent; IMG = HERE / 'img'; PROJ = HERE.parent.parent; CORN = PROJ / 'analysis_grid_corners_20261019' / 'out'
ROOT = pathlib.Path(r"C:\Users\SSAFY\Downloads\[기업연계_세메스] 교보재\[기업연계_세메스] 교보재\extracted\SSAFY Test Image Set")
for f in ['NotoSansKR-Regular.ttf', 'NotoSansKR-Medium.ttf', 'NotoSansKR-Bold.ttf']: fm.fontManager.addfont(f'C:/Windows/Fonts/{f}')
plt.rcParams.update({'font.family': 'Noto Sans KR', 'font.size': 8, 'axes.unicode_minus': False})
INK, SUB, FAINT, GRID, BG = '#111827', '#4B5563', '#9CA3AF', '#E5E7EB', '#F7F7F8'
TC = {'Y 0.1 간격 +': '#0071E3', 'Y 0.1 간격 -': '#D9480F', 'Y 0.05 홀수 +': '#0F766E', 'Y 0.05 홀수 -': '#B45309', 'Θ +': '#0071E3', 'Θ -': '#D9480F'}
TL = {t: t.replace(' +', ' +쪽').replace(' -', ' -쪽') for t in TC}
M = pd.read_csv(IMG / 'photo_motion.csv'); Z = pd.read_csv(IMG / 'zero_offsets.csv').set_index('trial')

def style(ax):
    ax.set_facecolor(BG); ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True); ax.tick_params(colors=SUB, length=2, labelsize=7)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
    for sp in ['left', 'bottom']: ax.spines[sp].set_color(GRID)

def save(fig, name): fig.patch.set_facecolor(BG); fig.tight_layout(pad=0.4); fig.savefig(IMG / name, dpi=250, facecolor=BG); plt.close(fig)

# ---- 4번: 모서리 기준 위치
C = pd.read_csv(CORN / 'corners_280.csv'); cen = C[C.file == 'center.bmp']
X0, Y0, X1, Y1, K = 1545, 962, 2262, 1180, 3
g = cv2.imdecode(np.fromfile(str(ROOT / 'center.bmp'), np.uint8), cv2.IMREAD_GRAYSCALE)[Y0:Y1, X0:X1]
im = Image.fromarray(cv2.resize(g, None, fx=K, fy=K, interpolation=cv2.INTER_CUBIC)).convert('RGB'); d = ImageDraw.Draw(im)
for q, gq in cen.groupby('qr'):
    u = gq[gq.black_neighbors != 2]
    d.rectangle(((u.x.min() - X0) * K, (u.y.min() - Y0) * K, (u.x.max() - X0) * K, (u.y.max() - Y0) * K), outline=(0, 113, 227), width=4)
    cx, cy = ((u.x.min() + u.x.max()) / 2 - X0) * K, ((u.y.min() + u.y.max()) / 2 - Y0) * K; d.ellipse((cx - 10, cy - 10, cx + 10, cy + 10), fill=(0, 113, 227))
for r in cen.itertuples():
    x, y = (r.x - X0) * K, (r.y - Y0) * K; col = (22, 163, 74) if r.black_neighbors != 2 else (234, 88, 12)
    d.ellipse((x - 5, y - 5, x + 5, y + 5), fill=col, outline=(15, 23, 42), width=1)
im.save(IMG / 'ref_corners.png')

# ---- 5번: 0점 위치
fig, axs = plt.subplots(1, 2, figsize=(6.0, 3.0), dpi=250)
for ax, axis, xl in [(axs[0], 'Y', '명령한 Y (mm)'), (axs[1], 'Θ', '명령한 Θ')]:
    style(ax); ts = [t for t in Z.index if t[0] == axis]; k = Z.loc[ts].slope_px.mean()
    for t in ts:
        gg = M[M.trial == t]; r = gg.dx - k * gg.value
        ax.plot(gg.value, r, 'o', ms=2.2, color=TC[t], alpha=0.85, label=TL[t])
        vv = np.linspace(min(0, gg.value.min()), max(0, gg.value.max()), 10); kk, bb = np.polyfit(gg.value, r, 1)
        ax.plot(vv, kk * vv + bb, '-', color=TC[t], lw=1)
        ax.plot(0, bb, 'D', ms=5, color=TC[t], mec='white', mew=0.6, zorder=5)
    ax.plot(0, 0, '*', ms=9, color=INK, zorder=6, label='center')
    ax.axvline(0, color=FAINT, lw=0.6, ls=':'); ax.set_xlabel(xl, color=SUB, fontsize=7.5)
    ax.set_title(f'{axis}: 직선에서 남은 가로 이동 (px)', loc='left', fontsize=8, color=INK)
    ax.legend(fontsize=6, frameon=False, loc='best', handletextpad=0.2, borderaxespad=0.2)
save(fig, 'zero.png')

# ---- 6번: 이동 경로
fig, axs = plt.subplots(1, 2, figsize=(6.0, 3.0), dpi=250)
for ax, axis in [(axs[0], 'Y'), (axs[1], 'Θ')]:
    style(ax)
    for t in [t for t in Z.index if t[0] == axis]:
        gg = pd.concat([M[M.file == 'center.bmp'], M[M.trial == t].sort_values('value', key=abs)])
        ax.plot(gg.dx, gg.dy, '-', color=TC[t], lw=0.9, alpha=0.9); ax.plot(gg.dx[1:], gg.dy[1:], 'o', ms=2, color=TC[t], label=TL[t])
    ax.plot(0, 0, '*', ms=9, color=INK, zorder=6, label='center')
    ax.set_xlabel('가로 이동 (px)', color=SUB, fontsize=7.5); ax.set_ylabel('세로 이동 (px)', color=SUB, fontsize=7.5)
    ax.set_title(f'{axis} trial: 630점 평균 이동 경로', loc='left', fontsize=8, color=INK)
    ax.legend(fontsize=6, frameon=False, loc='best', handletextpad=0.2)
save(fig, 'path.png')

# ---- 노이즈 패널 (모두 같은 크기)
PW, PH = 5.6, 2.3
def panel(name, fn):
    fig, ax = plt.subplots(figsize=(PW, PH), dpi=250); style(ax); fn(ax); save(fig, name)

def p1(ax):   # 촬영 순서
    for t in Z.index:
        gg = M[M.trial == t]; ax.plot(gg.minutes, gg.value / (2 if t[0] == 'Y' else 1), 'o', ms=1.8, color=TC[t] if t[0] == 'Y' else ('#6D28D9' if t.endswith('+') else '#BE185D'), label=TL[t])
    ax.plot(0, 0, '*', ms=8, color=INK, label='center'); ax.set_xlabel('촬영 시작 후 분', color=SUB, fontsize=7)
    ax.set_ylabel('명령값 (Y는 ÷2)', color=SUB, fontsize=7); ax.axvspan(33.2, 48.3, color='#E5E7EB', lw=0)
    ax.text(40.7, 0.85, '15분 공백', ha='center', fontsize=7, color=SUB); ax.legend(fontsize=5.5, frameon=False, ncol=4, loc='lower center', bbox_to_anchor=(0.5, -0.02))
def p2(ax):   # 0점 어긋남
    ts = list(Z.index); y = np.arange(len(ts))[::-1]
    ax.barh(y, Z.zero_px, color=[TC[t] for t in ts], height=0.55); ax.axvline(0, color=INK, lw=0.8)
    for yy, t in zip(y, ts):
        r = Z.loc[t]; lab = f'{r.zero_px:+.2f}px  ({r.zero_um:+.0f}µm)' if t[0] == 'Y' else f'{r.zero_px:+.2f}px  (Θ {r.zero_cmd:+.4f})'
        ax.text(r.zero_px + (0.03 if r.zero_px >= 0 else -0.03), yy, lab, va='center', ha='left' if r.zero_px >= 0 else 'right', fontsize=6.5, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([TL[t] for t in ts], fontsize=7); ax.set_xlim(-1.35, 0.6); ax.set_xlabel('0점 위치 (px, center = 0)', color=SUB, fontsize=7)
def p3(ax):   # 방향 바뀐 직후
    S = pd.read_csv(CORN / 'nondiag' / 'corner_step_px.csv'); st = S[(S.kind == 'step') & (S.axis == 'Y')]; m = st.px.mean()
    g2 = st[st.trial == 'Y 0.05 홀수 -'].copy(); g2['end'] = g2.step_to.str[1:].astype(float)
    g2 = g2.sort_values('end')   # 촬영은 -1.95 부터 0 쪽으로
    ax.plot(range(len(g2)), g2.px, 'o-', ms=3, lw=0.8, color=TC['Y 0.05 홀수 -']); ax.axhline(m, color=SUB, ls='--', lw=0.8)
    ax.plot(0, g2.px.iloc[0], 'o', ms=9, mfc='none', mec=INK); ax.annotate(f'방향 바꾼 첫 칸 -1.85→-1.95: {g2.px.iloc[0]:.2f}px', (0, g2.px.iloc[0]), xytext=(12, -4), textcoords='offset points', fontsize=7, color=INK)
    ax.text(len(g2) - 1, g2.px.min() + 0.005, f'점선 = Y 한 칸 평균 {m:.3f}px', ha='right', va='bottom', fontsize=6.5, color=SUB)
    ax.set_xticks(range(0, len(g2), 3)); ax.set_xticklabels([f'{v:.2f}' for v in g2.end.values[::3]], fontsize=6.5)
    ax.set_xlabel('Y 0.05 홀수 -쪽, 촬영 순서 (끝난 위치)', color=SUB, fontsize=7); ax.set_ylabel('한 칸 이동 (px)', color=SUB, fontsize=7)
def p4(ax):   # 밝기
    B = pd.read_csv(IMG / 'brightness.csv')
    for t, gg in B.groupby('trial'):
        ax.plot(gg.minutes, gg.plate_mean, 'o', ms=1.8, color=INK if t == 'center' else (TC.get(t, SUB) if t[0] == 'Y' else ('#6D28D9' if t.endswith('+') else '#BE185D')))
    c = B[B.file == 'center.bmp'].plate_mean.iloc[0]; o = B[B.file != 'center.bmp'].plate_mean
    ax.annotate(f'center {c:.1f} (나머지 평균보다 {100 * (c / o.mean() - 1):+.1f}%)', (0, c), xytext=(8, -3), textcoords='offset points', fontsize=7, color=INK, va='top')
    ax.set_xlabel('촬영 시작 후 분', color=SUB, fontsize=7); ax.set_ylabel('QR 판 평균 밝기', color=SUB, fontsize=7)
def p5(ax):   # 세로 이동: 가로 이동에 비례 (이동 방향이 화면에서 조금 기울어짐) + trial마다 작은 위아래 차이
    m = M[M.trial != 'center']; k = np.polyfit(m.dx, m.dy, 1)
    for t in Z.index:
        gg = M[M.trial == t]; ax.plot(gg.dx, gg.dy, 'o', ms=1.8, color=TC[t] if t[0] == 'Y' else ('#6D28D9' if t.endswith('+') else '#BE185D'), label=TL[t])
    xx = np.linspace(m.dx.min(), m.dx.max(), 10); ax.plot(xx, np.polyval(k, xx), '--', color=INK, lw=0.9)
    ax.text(xx[0], np.polyval(k, xx[-1]), f'점선 기울기 {k[0]:.4f} (이동 방향 {np.degrees(np.arctan(k[0])):.2f}° 기울어짐)', ha='left', va='top', fontsize=6.5, color=INK)
    ax.set_xlabel('가로 이동 (px)', color=SUB, fontsize=7); ax.set_ylabel('세로 이동 (px)', color=SUB, fontsize=7)
    ax.legend(fontsize=5.5, frameon=False, ncol=3, loc='lower center', bbox_to_anchor=(0.5, 1.0), borderaxespad=0)
def p6(ax):   # 사진 공통 흔들림
    Rr = pd.read_csv(IMG / 'common_resid.csv')
    for t, gg in Rr.groupby('trial'):
        col = TC[t] if t[0] == 'Y' else ('#6D28D9' if t.endswith('+') else '#BE185D')
        ax.plot(gg.minutes, gg.common_px, 'o', ms=1.8, color=col, label=f'{TL[t]}  σ {gg.common_px.std():.3f}')
    ax.axhline(0, color=INK, lw=0.6); ax.set_xlabel('촬영 시작 후 분', color=SUB, fontsize=7); ax.set_ylabel('사진 공통 남은 이동 (px)', color=SUB, fontsize=7)
    ax.legend(fontsize=5.5, frameon=False, ncol=3, loc='lower center', bbox_to_anchor=(0.5, 1.0), borderaxespad=0)
def p7(ax):   # 점 하나 흔들림
    R = pd.read_csv(CORN / 'nondiag' / 'corner_point_sigma.csv')
    for comp, col, lab in [('dx', '#0071E3', '가로'), ('dy', '#D9480F', '세로')]:
        v = R[R.comp == comp].sigma_point; ax.hist(v, bins=np.linspace(0, 0.12, 49), color=col, alpha=0.55, label=f'{lab}  중앙값 {v.median():.3f}px')
        ax.axvline(v.median(), color=col, lw=1)
    ax.set_xlabel('점 하나의 흔들림 σ (px), 630점 x 6 trial', color=SUB, fontsize=7); ax.set_ylabel('점 수', color=SUB, fontsize=7); ax.legend(fontsize=6.5, frameon=False)
def p8(ax):   # 대각선 모서리
    R = pd.read_csv(CORN / 'all' / 'corner_point_sigma.csv'); bn = cen.set_index('id').black_neighbors
    V = C.pivot(index='file', columns='id', values='valid').astype(bool)
    x = R[R.comp == 'dx'].copy(); x['diag'] = x.id.map(bn) == 2
    groups = [('대각선 아닌 점', x[~x['diag']].sigma_point.median(), '#9CA3AF'), ('대각선 모서리 125점', x[x['diag']].sigma_point.median(), '#EA580C')]
    ax.barh([1, 0], [g_[1] for g_ in groups], color=[g_[2] for g_ in groups], height=0.5)
    for yy, (lab, v, _) in zip([1, 0], groups): ax.text(v + 0.003, yy, f'{v:.3f}px', va='center', fontsize=8, color=INK, fontweight='bold')
    ax.set_yticks([1, 0]); ax.set_yticklabels([g_[0] for g_ in groups], fontsize=7.5); ax.set_xlim(0, 0.2)
    ax.set_xlabel('점 하나의 가로 흔들림 σ (중앙값, px)', color=SUB, fontsize=7)
for i, fn in enumerate([p1, p2, p3, p4, p5, p6, p7, p8], 1): panel(f'n{i}.png', fn)
print('ok')
