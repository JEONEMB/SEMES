"""56: 단계별 실험 요약 (축 틀린 사진 수), 57: 최종 방법 사진별 판정과 값 오차."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pathlib import Path
HERE = Path(__file__).parent; PROJ = HERE.parent
plt.rcParams['font.family'] = 'Malgun Gothic'; plt.rcParams['axes.unicode_minus'] = False
TXT, SUB, GRID = '#1f1f1e', '#5f5e58', '#e4e3dc'; BLUE, ORG, GRN, RED = '#4a72d0', '#d2703f', '#5aa776', '#c8423b'
def clean(ax):
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
    ax.grid(color=GRID, lw=0.8, axis='x'); ax.set_axisbelow(True)

def act(csv): T = pd.read_csv(HERE/csv); return T[T.정답 == '실제위치']
items = []
S1 = act('S1.csv'); items += [('S1 특징점', f'{r.템플릿}px {r.세트} ({r.점수}점)', r.틀림_Y, r.틀림_Θ) for r in S1.itertuples()]
S3 = act('S3.csv'); items += [('S3 0점 (M1)', f'라벨 {r.학습라벨}, 절편 {r.절편}', r.틀림_Y, r.틀림_Θ) for r in S3.itertuples()]
S4 = act('S4.csv'); items += [('S4 계산 방법', f'{r.특징} {r.매핑} {r.잡음}', r.틀림_Y, r.틀림_Θ) for r in S4.itertuples() if not r.특징.startswith('M2')]
S4c = act('S4c.csv'); items += [('S4 계산 방법', f'{r.특징} {r.매핑} {r.잡음}'.replace('M2 동시', 'M2 동시 추정'), r.틀림_Y, r.틀림_Θ) for r in S4c.itertuples() if r.특징 != 'P12']
S4b = act('S4b.csv'); items += [('S4b 점 넣는 법', f'{r.방법} {r.가중} {r.사용점}', r.틀림_Y, r.틀림_Θ) for r in S4b.itertuples()]
S5 = act('S5.csv'); items += [('S5 문턱/크기 (M3)', f'{r.템플릿}px {r.세트} 문턱 {r.문턱}', r.틀림_Y, r.틀림_Θ) for r in S5.itertuples()]
fig, ax = plt.subplots(figsize=(15, 0.27 * len(items) + 1.5)); yy = np.arange(len(items))[::-1]
ty = np.array([i[2] for i in items]); tt = np.array([i[3] for i in items])
ax.barh(yy, ty, color=BLUE, height=0.7, label='Y 사진을 Θ로'); ax.barh(yy, tt, left=ty, color=ORG, height=0.7, label='Θ 사진을 Y로')
for y_, a, b in zip(yy, ty, tt): ax.text(a + b + 0.2, y_, f'{a + b}', va='center', fontsize=9, color=TXT)
ax.set_yticks(yy); ax.set_yticklabels([f'[{i[0]}] {i[1]}' for i in items], fontsize=9); ax.set_xlabel('축을 틀린 사진 수 (280장 중, 짝/홀 교차검증, 정답 = 실제 위치)')
clean(ax); ax.legend(frameon=False, loc='lower right')
ax.set_title('단계별 실험: 축(Y/Θ) 판정을 틀린 사진 수. 0이 목표', loc='left', fontsize=13)
plt.tight_layout(); plt.savefig(OUT/'56_v7_단계별실험.png', dpi=100, facecolor='white'); plt.close()

P = pd.read_csv(OUT/'55_v7_사진별예측.csv'); y = P.axis == 'Y'; ok = P.axis == P.pred_axis
ye = np.where(y, P.cmd, P.cmd * 80.66 / 14.39)
fig, axs = plt.subplots(1, 3, figsize=(22, 6.3))
ax = axs[0]
for m, c, l in [(y & ok, BLUE, 'Y 사진 (맞음)'), (~y & ok, ORG, 'Θ 사진 (맞음)')]: ax.scatter(ye[m], P.llr[m], s=10, color=c, label=l)
ax.scatter(ye[~ok], P.llr[~ok], s=70, marker='x', color=RED, lw=2, label=f'틀림 {(~ok).sum()}장'); ax.axhline(0, color=TXT, lw=1)
ax.set_xlabel('정답 크기 (Y mm, Θ는 같은 QR 이동의 Y mm로 환산)'); ax.set_ylabel('log(Y 가설 비용 / Θ 가설 비용). 0 아래 = Y로 판정')
ax.set_title('(a) 축 판정: 0 근처 (|Y| < 0.3mm)에서 두 가설이 붙음', loc='left', fontsize=12); ax.legend(frameon=False); ax.grid(color=GRID)
for ax, a, tol, u, nm in [(axs[1], 'Y', 0.05, 1000, 'Y 오차 (µm)'), (axs[2], 'Θ', 0.01, 1000, 'Θ 오차 (mΘ = 0.001)')]:
    g = P[(P.axis == a) & ok]
    ax.scatter(g.act, (g.pred - g.act) * u, s=10, color=BLUE if a == 'Y' else ORG, label='정답 = 실제 위치')
    ax.scatter(g.cmd, (g.pred - g.cmd) * u, s=8, color=SUB, alpha=0.5, label='정답 = 명령값 (참고)')
    ax.axhspan(-tol * u, tol * u, color=GRN, alpha=0.08); ax.axhline(0, color=TXT, lw=0.8)
    ax.text(0.01, 0.97, f'초록 띠 = 한 스텝 ±{tol}', transform=ax.transAxes, va='top', fontsize=10, color=SUB)
    ax.set_xlabel(f'정답 {a}' + (' (mm)' if a == 'Y' else '')); ax.set_ylabel(nm); ax.grid(color=GRID); ax.legend(frameon=False, loc='lower right')
    ax.set_title(f'({"b" if a == "Y" else "c"}) {a} 값 오차 (축 맞은 사진). 0점 보정본으로 학습', loc='left', fontsize=12)
fig.suptitle('최종 방법 (QR만, 17px 점 232개, 2차 판 변형 12개, 1차 매핑, 전체 공분산): 값은 한 스텝 안, 축 판정은 0 근처에서 3장 틀림', x=0.01, ha='left', fontsize=14)
plt.tight_layout(rect=(0, 0, 1, 0.93)); plt.savefig(OUT/'57_v7_최종방법_사진별.png', dpi=100, facecolor='white'); plt.close()
print('saved')

