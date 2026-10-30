"""2단계 방법 설명 PPT용 그림 (out/ppt_fig/). 판 위 실제 사진 + 점, 이동 화살표, 12개 요약, 판정, 값 모델, 왜 나눴나, 결과."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from exp import *
from common import read_raw
from twostage import TwoStage
plt.rcParams['font.family'] = 'Malgun Gothic'; plt.rcParams['axes.unicode_minus'] = False
TXT, SUB, GRID = '#1f1f1e', '#5f5e58', '#e4e3dc'; YC, TC, RED, GRN = '#2f6fb0', '#d2703f', '#c8423b', '#4f9a6a'
FD = OUT/'ppt_fig'; FD.mkdir(exist_ok=True)
K = 80.66 / 14.39
F, D = load(8); Fe = feats(F, D, 'P12'); n = len(F)
x = (F.x.values - 1906) / 486; y = (F.y.values - 1070) / 122; B = np.c_[np.ones_like(x), x, y, x * x, x * y, y * y]
EXT = (1420, 2392, 1192, 948)
cen = read_raw('center.bmp')
def plate(ax, img, alpha=1.0):
    ax.imshow(img, cmap='gray', vmin=0, vmax=255, alpha=alpha); ax.set_xlim(1420, 2392); ax.set_ylim(1192, 948); ax.axis('off')
def save(fig, name): fig.savefig(FD/name, dpi=150, facecolor='white', bbox_inches='tight'); plt.close(fig)
def field(f12): return B @ f12.reshape(6, 2)

# ---- 1. 특징점 232개 ----
fig, ax = plt.subplots(figsize=(14, 3.9)); plate(ax, cen)
col = {'QR좌': YC, 'QR중': GRN, 'QR우': TC}
for r, c in col.items():
    m = F.region == r; ax.scatter(F.x[m], F.y[m], s=26, facecolors='none', edgecolors=c, lw=1.4, label=f'{r} {m.sum()}개')
ax.legend(loc='lower center', ncol=3, fontsize=11, frameon=True, bbox_to_anchor=(0.5, -0.16))
save(fig, 'f1_points.png')

# ---- 2. 현재 사진의 실제 이동 (그대로) ----
nY, nT = 'Y+1.00.bmp', 'T+0.18.bmp'; iY, iT = NAMES.index(nY), NAMES.index(nT)
fig, axs = plt.subplots(1, 2, figsize=(16, 2.9))
for ax, i, nm, c in [(axs[0], iY, 'Y +1.00mm', YC), (axs[1], iT, 'Θ +0.18', TC)]:
    plate(ax, read_raw(NAMES[i])); d = D[i]
    ax.quiver(F.x, F.y, d[:, 0], -d[:, 1], color=c, angles='xy', scale_units='xy', scale=1 / 2, width=0.0018)
    ax.set_title(f'{nm}: 평균 이동 ({d[:, 0].mean():+.2f}, {d[:, 1].mean():+.2f}) px   (화살표 x2)', loc='left', fontsize=12)
plt.tight_layout(); save(fig, 'f2_raw_arrows.png')

# ---- 3. 평균을 뺀 나머지 (이동량 + 방향) ----
def dir_color(v):
    ang = (np.arctan2(-v[:, 1], v[:, 0]) / (2 * np.pi)) % 1; mag = np.hypot(*v.T); s = np.clip(mag / np.quantile(mag, 0.95), 0, 1)
    return hsv_to_rgb(np.c_[ang, 0.25 + 0.75 * s, 0.85 - 0.25 * s])
fig, axs = plt.subplots(1, 2, figsize=(16, 3.2))
for ax, i, nm in [(axs[0], iY, 'Y +1.00mm'), (axs[1], iT, 'Θ +0.18')]:
    plate(ax, read_raw(NAMES[i]), 0.45); dv = D[i] - D[i].mean(0)
    ax.quiver(F.x, F.y, dv[:, 0], -dv[:, 1], color=dir_color(dv), angles='xy', scale_units='xy', scale=1 / 250, width=0.0026)
    ax.set_title(f'{nm}: 점마다 이동 - 평균 이동 (x250). 색 = 방향, 진하기 = 크기', loc='left', fontsize=12)
plt.tight_layout(); save(fig, 'f3_dev_arrows.png')

# ---- 4. 12개 요약의 기본 무늬 ----
terms = ['1 (통째로 밀림)', 'x (가로로 늘어남)', 'y (기울어짐)', 'x² (양 끝이 더)', 'x·y (사다리꼴)', 'y² (위아래 휨)']
gx, gy = np.meshgrid(np.linspace(-0.75, 0.75, 13), np.linspace(-0.8, 0.8, 5)); G = np.c_[np.ones(gx.size), gx.ravel(), gy.ravel(), gx.ravel()**2, (gx * gy).ravel(), gy.ravel()**2]
fig, axs = plt.subplots(2, 6, figsize=(20, 4.6))
for r, (comp, nm) in enumerate([(0, 'a (가로 이동 dx)'), (1, 'b (세로 이동 dy)')]):
    for j in range(6):
        ax = axs[r, j]; u = G[:, j] * (comp == 0); v = G[:, j] * (comp == 1)
        ax.add_patch(plt.Rectangle((-0.85, -0.95), 1.7, 1.9, fc='#f2f1ec', ec='#bbb'))
        ax.quiver(gx.ravel(), gy.ravel(), u, -v if comp else -v, color=YC if comp == 0 else TC, angles='xy', scale_units='xy', scale=6, width=0.012)
        ax.set_xlim(-1, 1); ax.set_ylim(1.05, -1.05); ax.set_aspect(0.5); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f'{"ab"[comp]}{j}: {terms[j]}', fontsize=10.5)
plt.tight_layout(); save(fig, 'f4_basis.png')

# ---- 5. 맞추기: 측정 / 12개로 표현한 것 / 버려진 흔들림 ----
fig, axs = plt.subplots(1, 3, figsize=(21, 3.2))
d = D[iT]; fit = field(Fe[iT]); m0 = d.mean(0)
for ax, v, ttl, c in [(axs[0], d - m0, '측정한 이동 (평균 뺌, x250)', TXT), (axs[1], fit - m0, '12개 값으로 다시 그린 이동 (x250)', TC),
                      (axs[2], d - fit, f'버려진 점별 흔들림 (x250), RMS {np.sqrt(((d - fit)**2).mean()):.3f}px', RED)]:
    plate(ax, cen, 0.4); ax.quiver(F.x, F.y, v[:, 0], -v[:, 1], color=c, angles='xy', scale_units='xy', scale=1 / 250, width=0.0026); ax.set_title(ttl, loc='left', fontsize=12)
plt.tight_layout(); save(fig, 'f5_fit.png')

# ---- 6. 12개 값 막대 비교 ----
lab12 = [f'{"ab"[c]}{j}' for j in range(6) for c in (0, 1)]
order = [2 * j for j in range(6)] + [2 * j + 1 for j in range(6)]
fig, axs = plt.subplots(1, 2, figsize=(17, 3.8))
for ax, (nA, nB) in zip(axs, [('Y+1.00.bmp', 'T+0.18.bmp'), ('Y+0.10.bmp', 'T+0.02.bmp')]):
    fa, fb = Fe[NAMES.index(nA)], Fe[NAMES.index(nB)]; xx = np.arange(11)
    ax.bar(xx - 0.2, fa[order][1:], 0.4, color=YC, label=nA.replace('.bmp', '').replace('T', 'Θ'))
    ax.bar(xx + 0.2, fb[order][1:], 0.4, color=TC, label=nB.replace('.bmp', '').replace('T', 'Θ'))
    ax.set_xticks(xx); ax.set_xticklabels([lab12[k] for k in order][1:]); ax.axhline(0, color=SUB, lw=0.8); ax.grid(color=GRID, axis='y'); ax.legend(frameon=False)
    ax.set_title(f'a0 (통째로 밀림): {fa[0]:+.2f} vs {fb[0]:+.2f} px (거의 같음, 그림에서 뺌)', loc='left', fontsize=11); ax.set_ylabel('px')
    for k, nm in [(3, 'a3'), (4 + 5, 'b4')]: pass
plt.tight_layout(); save(fig, 'f6_coef_bars.png')

# ---- 7. 정답 모양 두 개: a0 vs a3, a0 vs b4 ----
fig, axs = plt.subplots(1, 3, figsize=(20, 5))
for ax, k, nm, lim in [(axs[0], 6, 'a3 (양 끝이 더 움직임)', None), (axs[1], 9, 'b4 (사다리꼴)', None), (axs[2], 6, 'a3, 0 근처 확대', 6)]:
    for a, c in [('Y', YC), ('Θ', TC)]:
        m = AX == a; ax.scatter(Fe[m, 0], Fe[m, k], s=10, color=c, label=f'{a} 사진')
    ax.set_xlabel('a0: 통째로 밀린 양 (px)'); ax.set_ylabel(f'{nm} (px)'); ax.grid(color=GRID); ax.legend(frameon=False)
    if lim: ax.set_xlim(-lim, lim); m = np.abs(Fe[:, 0]) < lim; ax.set_ylim(Fe[m, k].min() - 0.02, Fe[m, k].max() + 0.02)
    ax.set_title(nm, loc='left', fontsize=12)
plt.tight_layout(); save(fig, 'f7_two_shapes.png')

# ---- 8. 판정 공간: 화이트닝 후 Θ 방향 vs 구분 방향 ----
P2 = pd.read_csv(OUT/'63_v7_2단계_사진별예측.csv')
mA = TwoStage(F).fit(Fe, CORR[0], AX).A
W = mA.W; sY, sT = mA.co['Y'][0], mA.co['Θ'][0]
e1 = W @ sT; e1 /= np.linalg.norm(e1); g = W @ sY - (W @ sY @ e1) * e1; e2 = g / np.linalg.norm(g)
Z = (W @ Fe.T).T
fig, axs = plt.subplots(1, 2, figsize=(17, 5.6))
for ax, lim in [(axs[0], None), (axs[1], 0.5)]:
    for a, c in [('Y', YC), ('Θ', TC)]:
        m = AX == a; ax.scatter(Z[m] @ e1, Z[m] @ e2, s=12, color=c, label=f'{a} 사진')
    vv = np.linspace(-2, 2, 50); ax.plot(np.outer(vv, W @ sY) @ e1, np.outer(vv, W @ sY) @ e2, color=YC, lw=1.2, label='Y 정답 모양 (선)')
    vt = np.linspace(-1, 1, 50); ax.plot(np.outer(vt, W @ sT) @ e1, np.outer(vt, W @ sT) @ e2, color=TC, lw=1.2, label='Θ 정답 모양 (선)')
    bad = P2[P2.axis != P2.pred_axis]
    for r in bad.itertuples():
        i = NAMES.index(r.file); ax.scatter(Z[i] @ e1, Z[i] @ e2, s=140, marker='x', color=RED, lw=2.5)
    ax.set_xlabel('통째로 밀린 방향 (잡음 σ 단위)'); ax.set_ylabel('Y와 Θ를 가르는 방향 (잡음 σ 단위)'); ax.grid(color=GRID)
    if lim:
        sc = np.abs(Z @ e1); s = sc[np.abs(CMD * np.where(AX == 'Y', 1, K)) <= lim].max(); ax.set_xlim(-s, s)
        m = (np.abs(Z @ e1) <= s); ax.set_ylim((Z[m] @ e2).min() - 1, (Z[m] @ e2).max() + 1); ax.set_title('0 근처 확대 (|Y| ≤ 0.5mm). 빨강 x = 틀린 2장', loc='left', fontsize=12)
    else: ax.set_title('전체: 두 선(Y, Θ)이 0에서 만나고 멀어질수록 벌어짐', loc='left', fontsize=12); ax.legend(frameon=False, loc='best')
plt.tight_layout(); save(fig, 'f8_decision_space.png')

# ---- 9. 공분산: 12개 값이 같이 흔들리는 정도 ----
C = np.linalg.inv(W.T @ W); sd = np.sqrt(np.diag(C)); R = C / np.outer(sd, sd)
fig, axs = plt.subplots(1, 2, figsize=(15, 5.4), gridspec_kw={'width_ratios': [1.1, 1]})
im = axs[0].imshow(R, cmap='RdBu_r', vmin=-1, vmax=1); axs[0].set_xticks(range(12)); axs[0].set_yticks(range(12))
axs[0].set_xticklabels(lab12, fontsize=9); axs[0].set_yticklabels(lab12, fontsize=9); plt.colorbar(im, ax=axs[0], fraction=0.046, label='같이 흔들리는 정도 (상관)')
axs[0].set_title('12개 값의 잡음 상관 (빨강 = 같이 커짐, 파랑 = 반대로)', loc='left', fontsize=12)
axs[1].bar(range(12), sd, color=[YC if k % 2 == 0 else TC for k in range(12)]); axs[1].set_xticks(range(12)); axs[1].set_xticklabels(lab12, fontsize=9)
axs[1].set_ylabel('흔들림 크기 (px)'); axs[1].grid(color=GRID, axis='y'); axs[1].set_title('12개 값 각각의 흔들림 (파랑 = 가로, 주황 = 세로)', loc='left', fontsize=12)
plt.tight_layout(); save(fig, 'f9_covariance.png')

# ---- 10. 값 모델: Θ는 휘어 있음 (휨이 가장 큰 값을 직선으로 맞춘 뒤 남는 차이) ----
fig, axs = plt.subplots(1, 2, figsize=(17, 5))
sdv = np.sqrt(np.diag(np.linalg.inv(W.T @ W)))
for ax, a, c in [(axs[0], 'Θ', TC), (axs[1], 'Y', YC)]:
    m = AX == a; v = CORR[0][m]; curv = [abs(np.polyfit(v, Fe[m, k], 2)[0]) / sdv[k] for k in range(12)]; k = int(np.argmax(curv)) if a == 'Θ' else kT
    if a == 'Θ': kT = k
    f = Fe[m, k]; p1 = np.polyfit(v, f, 1); p2 = np.polyfit(v, f, 2); g_ = np.linspace(v.min(), v.max(), 100)
    ax.scatter(v, (f - np.polyval(p1, v)) / sdv[k], s=9, color=c, label=f'{a} 사진: 직선으로 맞춘 뒤 남는 차이')
    ax.plot(g_, (np.polyval(p2, g_) - np.polyval(p1, g_)) / sdv[k], color=TXT, lw=1.8, label='2차 곡선 - 1차 직선')
    loc = {'Y': 0.6, 'Θ': 0.12}[a]; ax.axvspan(-loc, loc, color=GRN, alpha=0.12, label=f'판정 모델 학습 범위 (|{a}| ≤ {loc})')
    ax.axhline(0, color=SUB, lw=0.8); ax.grid(color=GRID); ax.legend(frameon=False, fontsize=10, loc='upper center')
    gap = (p2[-1] - p1[-1]) / sdv[k]
    ax.set_xlabel(f'{a}' + (' (mm)' if a == 'Y' else '')); ax.set_ylabel(f'{lab12[k]} 값의 남는 차이 (잡음 σ 단위)')
    ax.set_title(f'{a}, 값 {lab12[k]}: 0에서 직선이 곡선보다 {gap:+.1f}σ 어긋남', loc='left', fontsize=12)
plt.tight_layout(); save(fig, 'f10_curvature.png')
# ---- 11. 왜 나눴나: 0 근처 판정 점수 3가지 ----
runs = {'한 모델, 1차 (직선)': run(Fe, M(kind='P12', quad=False, noise='full', F=F)), '한 모델, 2차 (곡선)': run(Fe, M(kind='P12', quad=True, noise='full', F=F)),
        '2단계 (판정 = 0 근처 직선, 값 = 곡선)': P2}
fig, axs = plt.subplots(1, 3, figsize=(21, 5.4), sharey=True); stats = {}
for ax, (nm, P) in zip(axs, runs.items()):
    ye = np.where(P.axis == 'Y', P.cmd, P.cmd * K); ok = P.axis == P.pred_axis; z = np.abs(ye) <= 0.6
    ax.scatter(ye[(P.axis == 'Y') & ok & z], P.llr[(P.axis == 'Y') & ok & z], s=24, color=YC, label='Y 사진')
    ax.scatter(ye[(P.axis == 'Θ') & ok & z], P.llr[(P.axis == 'Θ') & ok & z], s=24, color=TC, label='Θ 사진')
    ax.scatter(ye[~ok & z], P.llr[~ok & z], s=130, marker='x', color=RED, lw=2.5, label=f'틀림 {(~ok).sum()}장')
    ax.axhline(0, color=TXT, lw=1.2); ax.set_xlim(-0.6, 0.6); ax.set_ylim(-2.5, 3); ax.grid(color=GRID); ax.legend(frameon=False, loc='lower right')
    ax.set_xlabel('정답 크기 (Y mm, Θ는 Y로 환산)'); ax.set_title(nm, loc='left', fontsize=12)
    stats[nm] = int((~ok).sum())
axs[0].set_ylabel('판정 점수 = log(Y 거리 / Θ 거리)\n0 아래 → Y, 0 위 → Θ')
plt.tight_layout(); save(fig, 'f11_why_split.png')

# ---- 12. 결과: 사진별 오차 ----
fig, axs = plt.subplots(1, 2, figsize=(17, 4.6))
for ax, a, c, tol, u in [(axs[0], 'Y', YC, 0.05, 'µm'), (axs[1], 'Θ', TC, 0.01, 'mΘ (0.001)')]:
    g_ = P2[(P2.axis == a) & (P2.pred_axis == a)]
    ax.scatter(g_.act, (g_.pred - g_.act) * 1000, s=10, color=c); ax.axhspan(-tol * 1000, tol * 1000, color=GRN, alpha=0.1)
    ax.axhline(0, color=TXT, lw=0.8); ax.grid(color=GRID); ax.set_xlabel(f'정답 {a}' + (' (mm)' if a == 'Y' else '')); ax.set_ylabel(f'오차 ({u})')
    ax.set_title(f'{a} 값 오차 (축 맞은 사진). 초록 띠 = 한 스텝 ±{tol}', loc='left', fontsize=12)
plt.tight_layout(); save(fig, 'f12_result_errors.png')
print('saved', stats)

