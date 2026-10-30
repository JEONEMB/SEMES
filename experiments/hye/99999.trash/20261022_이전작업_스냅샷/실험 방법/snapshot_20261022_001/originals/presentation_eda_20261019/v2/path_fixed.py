# 이동 경로 그림 (가로 세로 배율 고정): 모서리 630점 / ECC 두 가지, 같은 축 범위와 같은 배율
#   모든 패널: 가로 -85 ~ 85px, 세로 -0.2 ~ 0.35px, 세로를 가로의 100배로 확대해서 그림 (배율을 패널마다 명시)
#   ECC = 14_데이터_QR마커위치.csv 의 QR 3개 이동(dLx, dCx, dRx / dLy, dCy, dRy) 평균 (QR 패치 전체를 center에 맞춘 값)
#   출력: img/path_corner.png, img/path_ecc.png, img/path_ecc_stats.csv
import pathlib, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
HERE = pathlib.Path(__file__).parent; IMG = HERE / 'img'; PROJ = HERE.parent.parent
for f in ['NotoSansKR-Regular.ttf', 'NotoSansKR-Medium.ttf']: fm.fontManager.addfont(f'C:/Windows/Fonts/{f}')
plt.rcParams.update({'font.family': 'Noto Sans KR', 'font.size': 8, 'axes.unicode_minus': False})
INK, SUB, GRID, BG = '#111827', '#4B5563', '#E5E7EB', '#F7F7F8'
TC = {'Y 0.1 간격 +': '#0071E3', 'Y 0.1 간격 -': '#D9480F', 'Y 0.05 홀수 +': '#0F766E', 'Y 0.05 홀수 -': '#B45309', 'Θ +': '#0071E3', 'Θ -': '#D9480F'}
XL, YL, AMP = (-85, 85), (-0.2, 0.35), 100

def trial(f):
    if f == 'center.bmp': return 'center'
    v = float(f[1:-4]); s = '+' if v > 0 else '-'
    return ('Θ ' + s) if f[0] == 'T' else (('Y 0.1 간격 ' if round(abs(v) * 100) % 10 == 0 else 'Y 0.05 홀수 ') + s)

corner = pd.read_csv(IMG / 'photo_motion.csv')
Q = pd.read_csv(PROJ / '14_데이터_QR마커위치.csv')
ecc = pd.DataFrame({'file': Q.file, 'dx': Q[['dLx', 'dCx', 'dRx']].mean(1), 'dy': Q[['dLy', 'dCy', 'dRy']].mean(1)})
ecc['trial'] = ecc.file.map(trial); ecc['value'] = ecc.file.map(lambda f: 0.0 if f == 'center.bmp' else float(f[1:-4]))

def draw(M, name, label):
    fig, axs = plt.subplots(2, 1, figsize=(6.0, 4.3), dpi=250); fig.patch.set_facecolor(BG)
    for ax, axis in zip(axs, ['Y', 'Θ']):
        ax.set_facecolor(BG); ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True); ax.tick_params(colors=SUB, length=2, labelsize=7)
        for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
        for sp in ['left', 'bottom']: ax.spines[sp].set_color(GRID)
        for t in [t for t in TC if t[0] == axis]:
            g = pd.concat([M[M.file == 'center.bmp'], M[M.trial == t].sort_values('value', key=abs)])
            ax.plot(g.dx, g.dy, '-', color=TC[t], lw=0.8); ax.plot(g.dx[1:], g.dy[1:], 'o', ms=1.6, color=TC[t], label=t.replace(' +', ' +쪽').replace(' -', ' -쪽'))
        ax.plot(0, 0, '*', ms=8, color=INK, zorder=6, label='center')
        ax.set_xlim(*XL); ax.set_ylim(*YL); ax.set_aspect(AMP)
        ax.set_title(f'{axis} trial  |  {label}  |  가로 ×1, 세로 ×{AMP} 확대 (모든 패널 같은 축)', loc='left', fontsize=7.5, color=INK)
        ax.set_ylabel('세로 이동 (px)', color=SUB, fontsize=7); ax.legend(fontsize=5.5, frameon=False, loc='upper left', ncol=5, handletextpad=0.2, columnspacing=0.8)
    axs[1].set_xlabel('가로 이동 (px)', color=SUB, fontsize=7)
    fig.tight_layout(pad=0.4); fig.savefig(IMG / name, facecolor=BG); plt.close(fig)

draw(corner, 'path_corner.png', '모서리 630점 평균')
draw(ecc, 'path_ecc.png', 'ECC QR 3개 평균')

# ECC 표 숫자 (모서리 표와 같은 방식)
rows = []
for t, g in ecc[ecc.trial != 'center'].groupby('trial'):
    k, b = np.polyfit(g.value, g.dx, 1)
    rows.append(dict(trial=t, axis=t[0], slope_px=k, zero_px=b, resid_px=np.std(g.dx - (k * g.value + b), ddof=2)))
S = pd.DataFrame(rows); m = ecc[ecc.trial != 'center']; kd = np.polyfit(m.dx, m.dy, 1)
S['dy_slope'] = kd[0]; S.to_csv(IMG / 'path_ecc_stats.csv', index=False, encoding='utf-8-sig')
print(S.round(4).to_string(index=False), 'dy/dx', kd)
mc = corner[corner.trial != 'center']; print('corner vs ecc dx diff max', (corner.set_index('file').dx - ecc.set_index('file').dx).abs().max(), 'dy diff max', (corner.set_index('file').dy - ecc.set_index('file').dy).abs().max())
