# '한 칸 이동량 (모서리)' 그림: presentation_eda_20261019/step_figure.py 와 같은 그림, 데이터만 모서리 기반 (out/corner_step_px.csv)
import numpy as np, pandas as pd, pathlib, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D
import os
HERE = pathlib.Path(__file__).parent; SUBDIR = HERE / 'out' / ('' if os.environ.get('RULE', 'valid') == 'valid' else os.environ['RULE'])
DATA = SUBDIR / 'corner_step_px.csv'; FIG = SUBDIR / 'corner_step_px.png'
for f in ['NotoSansKR-Regular.ttf', 'NotoSansKR-Medium.ttf']: fm.fontManager.addfont(f'C:/Windows/Fonts/{f}')
plt.rcParams.update({'font.family': 'Noto Sans KR', 'font.size': 8, 'axes.unicode_minus': False})
INK, SUB, FAINT, GRID = '#111827', '#4B5563', '#9CA3AF', '#E5E7EB'
COL = {'+': '#0071E3', '-': '#D9480F'}                        # 검증 통과 (dataviz validate_palette.js, light)
S = pd.read_csv(DATA)
S['end'] = S.step_to.str[1:].astype(float).abs(); S['sign'] = S.trial.str[-1]
fig, axs = plt.subplots(2, 1, figsize=(5.8, 4.4), dpi=300)
for ax, axis, lab, xmax in [(axs[0], 'Y', 'Y  (한 칸 = 0.1mm)', 2.05), (axs[1], 'Θ', 'Θ  (한 칸 = 0.01)', 1.02)]:
    A = S[S.axis == axis]; st = A[A.kind == 'step']; m, sd = st.px.mean(), st.px.std()
    ax.axhspan(m - 3 * sd, m + 3 * sd, color='#F3F4F6', lw=0, zorder=0)
    ax.axhline(m, color=SUB, lw=1, ls='--', zorder=1)
    ax.text(xmax * 0.98, m + 3 * sd + 0.01, f'평균 {m:.3f}px  (±3σ = ±{3 * sd:.3f}px)', ha='right', va='bottom', color=SUB, fontsize=7)
    for t, g in st.groupby('trial'):
        ax.plot(g.end, g.px, ls='none', marker='^' if '0.05' in t else 'o', ms=3.2 if axis == 'Θ' else 4,
                mfc=COL[t[-1]], mec='white', mew=0.4, zorder=3)
    for _, r in A[A.kind == 'first'].iterrows():               # center -> 첫 사진 (0점 영향)
        ax.plot(r.end, r.px, ls='none', marker='^' if '0.05' in r.trial else 'o', ms=5, mfc='white', mec=COL[r.sign], mew=1.2, zorder=4)
    for _, r in st[(st.px - m).abs() >= 3 * sd].iterrows():    # 평균 ±3σ 밖 단계
        ax.plot(r.end, r.px, ls='none', marker='o', ms=9, mfc='none', mec=INK, mew=1, zorder=5)
        name = ('Θ' if r.step_from[0] == 'T' else 'Y') + f' {r.step_from[1:]} → {r.step_to[1:]}:  {r.px:.2f}px'
        up, right = r.px > m, r.end < xmax * 0.75
        ax.annotate(name, (r.end, r.px), xytext=(10 if right else -10, 16 if up else -16), textcoords='offset points',
                    fontsize=6.5, color=INK, ha='left' if right else 'right', va='bottom' if up else 'top',
                    bbox=dict(boxstyle='round,pad=0.2', fc='#F7F7F8', ec='none'), arrowprops=dict(arrowstyle='-', color=INK, lw=0.6))
    ax.set_title(lab, loc='left', fontsize=8.5, color=INK, pad=3)
    ax.set_ylabel('한 칸 이동 (px)', color=SUB)
    ax.grid(axis='y', color=GRID, lw=0.6); ax.set_axisbelow(True)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
    for sp in ['left', 'bottom']: ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=SUB, length=2)
    ax.set_xlim(0, xmax); ax.set_ylim(min(A.px.min(), m - 3 * sd) - 0.08, max(A.px.max(), m + 3 * sd) + 0.18)
axs[1].set_xlabel('명령값 크기 (한 칸 이동이 끝난 위치)', color=SUB)
L = lambda **k: Line2D([], [], ls='none', **k)
h = [L(marker='o', ms=5, mfc=COL['+'], mec='white'), L(marker='o', ms=5, mfc=COL['-'], mec='white'),
     L(marker='o', ms=5, mfc=FAINT, mec='white'), L(marker='^', ms=5, mfc=FAINT, mec='white'),
     L(marker='o', ms=5, mfc='white', mec=FAINT, mew=1.2), L(marker='o', ms=8, mfc='none', mec=INK, mew=1)]
l = ['+쪽 trial', '-쪽 trial', 'Y 0.1 간격 trial', 'Y 0.05 홀수 trial', 'center → 첫 사진 (0점 영향)', '평균 ±3σ 밖']
axs[0].legend(h, l, ncol=3, fontsize=6.5, frameon=False, loc='lower right', handletextpad=0.3, columnspacing=1.0)
fig.tight_layout(h_pad=1.2)
fig.savefig(FIG, facecolor='#F7F7F8')   # 슬라이드 회색 바탕과 같게
print('ok')
