# 점별 흔들림 그림
#   (1) noise_box.png: trial별 점 고유 흔들림 σ 분포(상자), 가로 x / 세로 y, 마름모 = 사진 공통 흔들림 σ
#   (2) noise_map_Y.png, noise_map_T.png: 점마다 가로 흔들림 σ (Y trial 4개 평균, Θ trial 2개 평균), 진할수록 큼 (한 색 단계)
import numpy as np, pandas as pd, pathlib, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from PIL import Image, ImageDraw
import os
HERE = pathlib.Path(__file__).parent; OUT = HERE / 'out' / ('' if os.environ.get('RULE', 'valid') == 'valid' else os.environ['RULE'])
ROOT = pathlib.Path(r"C:\Users\SSAFY\Downloads\[기업연계_세메스] 교보재\[기업연계_세메스] 교보재\extracted\SSAFY Test Image Set")
for f in ['NotoSansKR-Regular.ttf', 'NotoSansKR-Medium.ttf']: fm.fontManager.addfont(f'C:/Windows/Fonts/{f}')
plt.rcParams.update({'font.family': 'Noto Sans KR', 'font.size': 8, 'axes.unicode_minus': False})
INK, SUB, GRID, BG = '#111827', '#4B5563', '#E5E7EB', '#F7F7F8'
COL = {'Y': '#0071E3', 'Θ': '#D9480F'}
R = pd.read_csv(OUT / 'corner_point_sigma.csv'); PH = pd.read_csv(OUT / 'photo_common_sigma.csv')
TR = ['Y 0.1 간격 +', 'Y 0.1 간격 -', 'Y 0.05 홀수 +', 'Y 0.05 홀수 -', 'Θ +', 'Θ -']
LAB = [t.replace(' +', '\n+쪽').replace(' -', '\n-쪽') for t in TR]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.1), dpi=300, sharey=True); fig.patch.set_facecolor(BG)
for ax, comp, ttl in [(axs[0], 'dx', '가로 (x)'), (axs[1], 'dy', '세로 (y)')]:
    ax.set_facecolor(BG)
    data = [R[(R.trial == t) & (R.comp == comp)].sigma_point.values for t in TR]
    bp = ax.boxplot(data, widths=0.55, showfliers=False, patch_artist=True, medianprops=dict(color=INK, lw=1.2), whiskerprops=dict(color=SUB, lw=0.8), capprops=dict(color=SUB, lw=0.8), whis=(5, 95))
    for b, t in zip(bp['boxes'], TR): b.set(facecolor=COL[t[0]], alpha=0.35, edgecolor=COL[t[0]], lw=1)
    com = [PH[(PH.trial == t) & (PH.comp == comp)].sigma_common.iloc[0] for t in TR]
    ax.plot(range(1, 7), com, ls='none', marker='D', ms=5, mfc=INK, mec='white', mew=0.6, label='사진 공통 흔들림 σ')
    for i, d in enumerate(data, 1): ax.text(i, np.percentile(d, 95) + 0.003, f'{np.median(d):.3f}', ha='center', va='bottom', fontsize=6.5, color=SUB)
    ax.set_xticks(range(1, 7)); ax.set_xticklabels(LAB, fontsize=6.5, color=SUB); ax.set_title(ttl, loc='left', fontsize=8.5, color=INK)
    ax.grid(axis='y', color=GRID, lw=0.6); ax.set_axisbelow(True); ax.tick_params(colors=SUB, length=2)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
    for sp in ['left', 'bottom']: ax.spines[sp].set_color(GRID)
axs[0].set_ylabel('점 하나의 흔들림 σ (px)', color=SUB); axs[0].set_ylim(0, 0.105)
axs[1].legend(frameon=False, fontsize=7, loc='upper right')
fig.tight_layout(); fig.savefig(OUT / 'noise_box.png', facecolor=BG); plt.close(fig)

# 지도
D0 = pd.read_csv(HERE / 'out' / 'corners_280.csv'); C = D0[D0.file == 'center.bmp'].set_index('id')
X0, Y0, X1, Y1, K = 1545, 962, 2262, 1180, 3
base = cv2.imdecode(np.fromfile(str(ROOT / 'center.bmp'), np.uint8), cv2.IMREAD_GRAYSCALE)[Y0:Y1, X0:X1]
base = cv2.resize(base, None, fx=K, fy=K, interpolation=cv2.INTER_CUBIC)
base = Image.fromarray((255 - (255 - base) * 0.45).astype(np.uint8)).convert('RGB')   # 바탕을 옅게 해서 점 색이 보이게
cmap = matplotlib.colormaps['Blues']; lo, hi = 0.03, 0.08
for ax_name, key in [('Y', 'Y'), ('T', 'Θ')]:
    m = R[(R.axis == key) & (R.comp == 'dx')].groupby('id').sigma_point.mean()
    im = base.copy(); dr = ImageDraw.Draw(im)
    for cid, sg in m.items():
        x, y = (C.loc[cid, 'x'] - X0) * K, (C.loc[cid, 'y'] - Y0) * K
        col = tuple(int(255 * c) for c in cmap(0.25 + 0.75 * np.clip((sg - lo) / (hi - lo), 0, 1))[:3])
        dr.ellipse((x - 9, y - 9, x + 9, y + 9), fill=col, outline=(15, 23, 42), width=1)
    im.save(OUT / f'noise_map_{ax_name}.png')
# 색 막대
fig, ax = plt.subplots(figsize=(3.2, 0.42), dpi=300); fig.patch.set_facecolor(BG)
g = np.linspace(0.25, 1, 256)[None]; ax.imshow(g, aspect='auto', cmap='Blues', vmin=0, vmax=1, extent=(lo, hi, 0, 1))
ax.set_yticks([]); ax.set_xticks([lo, 0.04, 0.05, 0.06, 0.07, hi]); ax.set_xticklabels([f'≤{lo:.2f}', '0.04', '0.05', '0.06', '0.07', f'≥{hi:.2f}px'], fontsize=6, color=SUB)
for sp in ax.spines.values(): sp.set_visible(False)
ax.tick_params(length=0); fig.tight_layout(pad=0.1); fig.savefig(OUT / 'noise_colorbar.png', facecolor=BG); plt.close(fig)
print('ok')
