"""58: 틀린 3장 (사진, 측정한 판 변형, Y 가설 / Θ 가설로 맞춘 뒤 남는 차이, 비용).
59: 왜 틀리는 방향이 바뀌나. 같은 사진, 같은 점으로 예전 방법(M1 점별 2차)과 이번 방법(M3 판 변형 12개 1차)의 비용 비를 비교."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from exp import *
from common import read_raw
plt.rcParams['font.family'] = 'Malgun Gothic'; plt.rcParams['axes.unicode_minus'] = False
TXT, SUB, GRID = '#1f1f1e', '#5f5e58', '#e4e3dc'; BLUE, ORG, GRN, RED = '#4a72d0', '#d2703f', '#5aa776', '#c8423b'
K = 80.66 / 14.39

F, D = load(8); Fe = feats(F, D, 'P12')
x = (F.x.values - 1906) / 486; y = (F.y.values - 1070) / 122; B = np.c_[np.ones_like(x), x, y, x * x, x * y, y * y]
def field(f12):                      # 12개 값 → 점마다 (dx, dy)
    return B @ f12.reshape(6, 2)

P3 = run(Fe, M(kind='P12', quad=False, noise='full', F=F))
P1 = run(D.reshape(len(D), -1), M(kind='pts', quad=True, noise='diag', F=F))
bad = P3[P3.axis != P3.pred_axis].sort_values('cmd').reset_index(drop=True)

# ---------- 58 ----------
fig, axs = plt.subplots(len(bad), 4, figsize=(24, 3.3 * len(bad)), gridspec_kw={'width_ratios': [1.25, 1, 1, 1]})
for r, b in bad.iterrows():
    i = NAMES.index(b.file); k = 0 if V5.FOLD[b.file] == 0 else 1
    mdl = M(kind='P12', quad=False, noise='full', F=F)().fit(Fe[~np.array([V5.FOLD.get(n, -1) == V5.FOLD[b.file] for n in NAMES])],
                                                           CORR[V5.FOLD[b.file]][~np.array([V5.FOLD.get(n, -1) == V5.FOLD[b.file] for n in NAMES])],
                                                           AX[~np.array([V5.FOLD.get(n, -1) == V5.FOLD[b.file] for n in NAMES])])
    img = read_raw(b.file); ax = axs[r, 0]; ax.imshow(img, cmap='gray', vmin=0, vmax=255); ax.set_xlim(1420, 2392); ax.set_ylim(1192, 948); ax.axis('off')
    ax.set_title(f'{b.file}  정답 {b.axis} {b.cmd:+.2f}\n판정 {b.pred_axis} {b.pred:+.3f}  (Y로 환산하면 정답 {b.cmd * K:+.3f}mm)', loc='left', fontsize=13, color=RED)
    f = Fe[i]; U = U_for(F, 'P12'); fits = {}
    for a in ['Y', 'Θ']:
        v, cost = mdl.solve(f, a); s, c, bb = mdl.co[a]; pred = s * v + bb
        Wp = mdl.W @ U; u, *_ = np.linalg.lstsq(Wp, mdl.W @ (f - pred), rcond=None); fits[a] = (v, cost, f - pred - U @ u)
    meas = field(f); meas = meas - meas.mean(0)
    for c_, (ttl, vec, col) in enumerate([(f'측정한 판 변형 (평균 이동 {field(f).mean(0)[0]:+.2f}px 뺀 나머지)', meas, TXT),
                                          (f'Y 가설로 맞춘 뒤 남는 차이\nY = {fits["Y"][0]:+.3f}mm, 비용 {fits["Y"][1]:.1f}', field(fits['Y'][2]), BLUE),
                                          (f'Θ 가설로 맞춘 뒤 남는 차이\nΘ = {fits["Θ"][0]:+.4f}, 비용 {fits["Θ"][1]:.1f}', field(fits['Θ'][2]), ORG)]):
        ax = axs[r, c_ + 1]; ax.imshow(img, cmap='gray', vmin=0, vmax=255, alpha=0.35); ax.set_xlim(1420, 2392); ax.set_ylim(1192, 948); ax.axis('off')
        sc = 1 / 1500 if c_ else 1 / 300
        Q = ax.quiver(F.x, F.y, vec[:, 0], -vec[:, 1], color=col, angles='xy', scale_units='xy', scale=sc, width=0.003)
        ax.quiverkey(Q, 0.82, -0.04, 0.01 if c_ else 0.05, '0.01px' if c_ else '0.05px', labelpos='E', coordinates='axes')
        ax.set_title(ttl + ('\n(x300 확대)' if not c_ else '  (x1500 확대)'), loc='left', fontsize=12)
    axs[r, 0].text(0.0, -0.10, f'비용 비 Y/Θ = {fits["Y"][1] / fits["Θ"][1]:.2f} → 1보다 작아서 Y로 판정 (정답은 Θ)', transform=axs[r, 0].transAxes, fontsize=12, color=RED, va='top')
fig.suptitle('틀린 3장: 정답은 Θ인데 Y 가설이 조금 더 잘 맞음. 남는 차이 크기가 둘 다 0.01px 수준이라 거의 비김 (점 하나 흔들림 약 0.02px)', x=0.01, ha='left', fontsize=15)
plt.tight_layout(rect=(0, 0, 1, 0.93), h_pad=3); plt.savefig(OUT/'58_v7_틀린3장.png', dpi=90, facecolor='white'); plt.close()

# ---------- 59 ----------
fig, axs = plt.subplots(1, 2, figsize=(22, 7.5), sharey=False)
stats = []
for ax, P, ttl in [(axs[0], P1, '예전 방법 (M1 점별 모델, 2차): Y 사진이 틀림'), (axs[1], P3, '이번 방법 (M3 판 변형 12개, 1차, 전체 공분산): Θ 사진이 틀림')]:
    ye = np.where(P.axis == 'Y', P.cmd, P.cmd * K); ok = P.axis == P.pred_axis; z = np.abs(ye) <= 0.6
    for m, c, l in [((P.axis == 'Y') & ok, BLUE, 'Y 사진'), ((P.axis == 'Θ') & ok, ORG, 'Θ 사진')]: ax.scatter(ye[m & z], P.llr[m & z], s=26, color=c, label=l)
    ax.scatter(ye[~ok & z], P.llr[~ok & z], s=120, marker='x', color=RED, lw=2.5, label=f'틀림 {(~ok).sum()}장')
    for r_ in P[~ok & z].itertuples():
        ax.annotate(f'{r_.axis}{r_.cmd:+.2f}', (r_.cmd * (1 if r_.axis == 'Y' else K), r_.llr), textcoords='offset points', xytext=(6, 6), fontsize=10, color=RED)
    ax.axhline(0, color=TXT, lw=1.2); ax.text(-0.59, 0.02, '위 = Θ로 판정', fontsize=11, color=SUB, va='bottom'); ax.text(-0.59, -0.02, '아래 = Y로 판정', fontsize=11, color=SUB, va='top')
    near = np.abs(ye) <= 0.15; my, mt = np.median(P.llr[near & (P.axis == 'Y')]), np.median(P.llr[near & (P.axis == 'Θ')])
    ax.axhline((my + mt) / 2, color=GRN, lw=1.5, ls='--'); ax.text(0.35, (my + mt) / 2, f'0 근처 사진들의 가운데\n{(my + mt) / 2:+.2f}', color=GRN, fontsize=11, va='bottom')
    ax.set_xlim(-0.6, 0.6); ax.set_xlabel('정답 크기 (Y mm, Θ는 Y mm로 환산)'); ax.set_ylabel('log(Y 가설 비용 / Θ 가설 비용)'); ax.grid(color=GRID); ax.legend(frameon=False, loc='lower right')
    ax.set_title(ttl, loc='left', fontsize=13); stats.append((my, mt))
    if ax is axs[0]: ax.set_ylim(-1.6, 1.6)
    else: ax.set_ylim(-2.5, 4)
fig.suptitle('왜 틀리는 쪽이 바뀌나: 0 근처에서는 Y 사진과 Θ 사진이 한 덩어리로 붙어 있고, 그 덩어리가 문턱(0)의 어느 쪽에 걸치느냐는 모델이 정함', x=0.01, ha='left', fontsize=14)
plt.tight_layout(rect=(0, 0, 1, 0.93)); plt.savefig(OUT/'59_v7_틀리는방향.png', dpi=95, facecolor='white'); plt.close()
print(bad[['file', 'cmd', 'pred_axis', 'pred', 'llr']]); print('0 근처 llr 중앙값 (Y 사진, Θ 사진): M1', stats[0], 'M3', stats[1])


