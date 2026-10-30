"""방법 설명 그림 48~53: 특징점 고르기 → 위치 재기 → 이동장 → 이동 모델링 → 구분 단서 → 판정. (평가는 46번)"""
import numpy as np, pandas as pd, matplotlib, cv2
matplotlib.use('Agg'); import matplotlib.pyplot as plt, matplotlib.ticker
from matplotlib.patches import Rectangle
from scipy import ndimage
from common import *
from t6_F1 import fit_axis, leftover, nuis
plt.rcParams['font.family'] = 'Malgun Gothic'; plt.rcParams['axes.unicode_minus'] = False
TXT, SUB, GRID = '#1f1f1e', '#5f5e58', '#e4e3dc'
BLUE, ORG, GRN, RED = '#4a72d0', '#d2703f', '#5aa776', '#c8423b'
HP = 8; THETA_TO_Y = 80.66 / 14.39
cen = read_raw('center.bmp'); bcen = blur(cen)
F = pd.read_csv(HERE/'features_hp8_ratio.csv'); Z = np.load(HERE/'tracks_hp8.npz'); NM = list(Z['names']); D = Z['D_gn']
onp = (F.region.str.startswith('QR') | F.region.isin(['판', '판나사'])).values
XL, YL = (150, 3690), (2110, 50)

def clean(ax):
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
    ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
def bg(ax, alpha=0.55):
    ax.imshow(cen, cmap='gray', vmin=0, vmax=255, alpha=alpha); ax.set_xlim(*XL); ax.set_ylim(*YL); ax.axis('off')

# ======================= 48: 특징점 고르기 =======================
cand = cv2.goodFeaturesToTrack(bcen, maxCorners=3000, qualityLevel=0.02, minDistance=2 * HP + 1, blockSize=7).reshape(-1, 2)
why = []
for x, y in cand:
    xi, yi = int(round(x)), int(round(y))
    if not (160 < xi < 3680 and 60 < yi < 2100): why.append('가장자리'); continue
    r = region_of(xi, yi)
    if r in EXCL: why.append(r); continue
    if (cen[yi - HP:yi + HP + 1, xi - HP:xi + HP + 1] >= 250).any(): why.append('포화'); continue
    why.append('사용')
why = np.array(why)
fig = plt.figure(figsize=(22, 10)); gs = fig.add_gridspec(2, 3, width_ratios=[2.2, 1, 1])
ax = fig.add_subplot(gs[:, 0]); bg(ax)
ax.scatter(F.x[onp], F.y[onp], s=10, color=BLUE, label=f'사용: 판 위 (QR) {onp.sum()}개')
ax.scatter(F.x[~onp], F.y[~onp], s=34, color=ORG, edgecolors='white', lw=0.6, label=f'사용: 판밖 {(~onp).sum()}개')
for w, mk in [('포화', 'x'), ('라벨', 'x'), ('케이블', 'x'), ('가장자리', '+')]:
    m = why == w
    if m.any(): ax.scatter(cand[m, 0], cand[m, 1], s=30, marker=mk, color=RED, lw=1.3, label=f'뺌: {w} {m.sum()}개' if mk == 'x' else f'뺌: {w} {m.sum()}개')
for nm, (a, b, c, d) in EXCL.items(): ax.add_patch(Rectangle((a, b), c - a, d - b, fill=False, ec=RED, lw=1.2, ls='--'))
ax.legend(loc='lower left', fontsize=11, framealpha=0.9)
ax.set_title(f'① 특징점 고르기: center 사진(흐림 σ1)에서 모서리 후보 {len(cand)}개 (Shi-Tomasi, 서로 17px 이상 떨어지게) → 조건에 안 맞는 점을 빼고 {len(F)}개 사용',
             loc='left', fontsize=13)
for (r, c), (x0, x1, y0, y1), ttl in [((0, 1), (1560, 1760, 960, 1170), 'QR 좌 확대: 모듈 모서리'), ((0, 2), (2400, 2560, 290, 450), '벽 볼트 확대 (판밖)'),
                                       ((1, 1), (1320, 1560, 1700, 1930), '브래킷 아래 손잡이 (판밖)'), ((1, 2), (2235, 2390, 1000, 1150), '판 나사: 와셔 포화로 뺌')]:
    ax = fig.add_subplot(gs[r, c]); ax.imshow(cen, cmap='gray', vmin=0, vmax=255); ax.set_xlim(x0, x1); ax.set_ylim(y1, y0); ax.axis('off')
    for (x, y), ok in zip(cand, why):
        if x0 < x < x1 and y0 < y < y1:
            col = (BLUE if region_of(int(x), int(y)).startswith(('QR', '판')) else ORG) if ok == '사용' else RED
            ax.add_patch(Rectangle((x - HP - 0.5, y - HP - 0.5), 2 * HP + 1, 2 * HP + 1, fill=False, ec=col, lw=1.6, ls='-' if ok == '사용' else ':'))
    ax.set_title(ttl + '\n네모 = 17x17 템플릿 (실선 사용, 점선 뺌)', loc='left', fontsize=11)
plt.tight_layout(); plt.savefig(PROJ/'48_방법1_특징점고르기.png', dpi=95, facecolor='white'); plt.close()

# ======================= 49: 위치 재기 =======================
k = int(np.argmin(np.hypot(F.x - 1600, F.y - 1040) + 1e6 * ~F.region.str.startswith('QR')))
x, y = int(F.x[k]), int(F.y[k]); n = 'T+0.50.bmp'; img = read_raw(n); i = NM.index(n)
T = cen[y - HP:y + HP + 1, x - HP:x + HP + 1]
s = img[y - HP - SY:y + HP + SY + 1, x - HP - SX:x + HP + SX + 1]
Rs = cv2.matchTemplate(s, T, cv2.TM_CCOEFF_NORMED)
BIGT = cen[y - HPC:y + HPC + 1, x - HPC:x + HPC + 1]
sb = img[y - HPC - SY:y + HPC + SY + 1, x - HPC - SX:x + HPC + SX + 1]
Rb = cv2.matchTemplate(sb, BIGT, cv2.TM_CCOEFF_NORMED)
# 가우스-뉴턴 반복 기록
bimg = blur(img); Tb = bcen[y - HP:y + HP + 1, x - HP:x + HP + 1].astype(np.float64).ravel()
_, _, _, (bx, by) = cv2.minMaxLoc(Rb); c0 = np.array([bx - SX, by - SY], float)
L = HP + 6; cx, cy = x + int(c0[0]), y + int(c0[1]); win = bimg[cy - L:cy + L + 1, cx - L:cx + L + 1].astype(np.float64)
gxw = (np.roll(win, -1, 1) - np.roll(win, 1, 1)) / 2; gyw = (np.roll(win, -1, 0) - np.roll(win, 1, 0)) / 2
cI, cX, cY = (ndimage.spline_filter(w_, order=3) for w_ in (win, gxw, gyw))
jj, ii = np.meshgrid(np.arange(-HP, HP + 1), np.arange(-HP, HP + 1)); d = np.zeros(2); a_, b_ = 1.0, 0.0; hist = [(0, *d, np.nan)]
for it in range(1, 9):
    yy = (ii + L + d[1]).ravel(); xx = (jj + L + d[0]).ravel()
    smp = lambda c: ndimage.map_coordinates(c, [yy, xx], order=3, prefilter=False, mode='nearest')
    I, Ix, Iy = smp(cI), smp(cX), smp(cY); r = a_ * I + b_ - Tb
    if it == 1: r_before = r.copy()
    J = np.c_[a_ * Ix, a_ * Iy, I, np.ones_like(I)]; dl = -np.linalg.solve(J.T @ J, J.T @ r)
    d += np.clip(dl[:2], -0.5, 0.5); a_ += dl[2]; b_ += dl[3]; hist.append((it, *d, np.sqrt((r**2).mean())))
yy = (ii + L + d[1]).ravel(); xx = (jj + L + d[0]).ravel()
r_after = a_ * ndimage.map_coordinates(cI, [yy, xx], order=3, prefilter=False) + b_ - Tb
H = np.array(hist)
fig, axs = plt.subplots(2, 3, figsize=(21, 10))
ax = axs[0, 0]; ax.imshow(cen, cmap='gray', vmin=0, vmax=255); ax.set_xlim(x - 120, x + 120); ax.set_ylim(y + 70, y - 70)
ax.add_patch(Rectangle((x - HP - .5, y - HP - .5), 2 * HP + 1, 2 * HP + 1, fill=False, ec=BLUE, lw=2)); ax.add_patch(Rectangle((x - HPC - .5, y - HPC - .5), 2 * HPC + 1, 2 * HPC + 1, fill=False, ec=ORG, lw=2))
ax.set_title(f'(a) center에서 자른 템플릿. 파랑 17x17 (정밀용), 주황 81x81 (대략 위치용)\n특징점 #{k} ({x}, {y}), QR 좌', loc='left', fontsize=12); ax.axis('off')
for ax, R_, ttl in [(axs[0, 1], Rs, '(b) 17x17로 넓게 찾기 (사진 T+0.50, 가로 ±110, 세로 ±16px)\nQR 무늬가 반복돼서 비슷한 봉우리가 여러 개'),
                    (axs[0, 2], Rb, '(c) 81x81로 넓게 찾기: 봉우리 하나 → 정수 위치 확정')]:
    im_ = ax.imshow(R_, cmap='viridis', extent=(-SX - .5, SX + .5, SY + .5, -SY - .5), aspect='auto', vmin=-0.5, vmax=1)
    _, mx, _, (px, py) = cv2.minMaxLoc(R_); ax.plot(px - SX, py - SY, 'r+', ms=16, mew=2)
    srt = np.sort(R_.ravel()); peaks = (ndimage.maximum_filter(R_, 9) == R_) & (R_ > 0.6)
    ax.scatter(np.where(peaks)[1] - SX, np.where(peaks)[0] - SY, s=40, facecolors='none', edgecolors='white', lw=1.2)
    ax.set_xlabel('가로 이동 (px)'); ax.set_ylabel('세로 이동 (px)'); plt.colorbar(im_, ax=ax, fraction=0.03, label='정규화 상관')
    ax.set_title(ttl + f'\n흰 원 = 상관 0.6 넘는 봉우리 {peaks.sum()}개, 빨강 + = 최고 ({px - SX:+d}, {py - SY:+d})', loc='left', fontsize=12)
ax = axs[1, 0]
ax.plot(H[:, 0], H[:, 1], 'o-', color=BLUE, label='가로 소수 이동 dx'); ax.plot(H[:, 0], H[:, 2], 's-', color=ORG, label='세로 소수 이동 dy')
ax.set_xlabel('가우스-뉴턴 반복 횟수'); ax.set_ylabel('정수 위치에서 더한 소수 이동 (px)'); clean(ax); ax.legend(frameon=False)
ax.set_title(f'(d) 가우스-뉴턴 정밀화: (dx, dy, 밝기 a, b) 4개를 반복해서 풂\n최종 이동 = 정수 ({int(c0[0])}, {int(c0[1])}) + 소수 ({d[0]:+.4f}, {d[1]:+.4f}) px', loc='left', fontsize=12)
for ax, rr, ttl in [(axs[1, 1], r_before, '(e) 정수 위치에서 템플릿과의 차이'), (axs[1, 2], r_after, '(f) 소수 이동까지 맞춘 뒤 차이')]:
    im_ = ax.imshow(rr.reshape(2 * HP + 1, 2 * HP + 1), cmap='RdBu_r', vmin=-30, vmax=30); plt.colorbar(im_, ax=ax, fraction=0.04, label='밝기 차이'); ax.axis('off')
    ax.set_title(ttl + f'\nRMS {np.sqrt((rr**2).mean()):.2f}', loc='left', fontsize=12)
fig.suptitle('② 특징점 위치 재기: 큰 템플릿으로 대략 (정수) → 작은 템플릿 + 가우스-뉴턴으로 소수점까지. 보간은 3차 스플라인 (OpenCV remap의 1/32px 잘림 없음)',
             x=0.01, ha='left', fontsize=14)
plt.tight_layout(rect=(0, 0, 1, 0.95)); plt.savefig(PROJ/'49_방법2_위치재기.png', dpi=95, facecolor='white'); plt.close()

# ======================= 50: 이동장 =======================
nY, nT = 'Y+1.00.bmp', 'T+0.18.bmp'; iY, iT = NM.index(nY), NM.index(nT)
qr = F.region.str.startswith('QR').values
dY, dT = D[iY], D[iT]; mY, mT = dY[qr].mean(0), dT[qr].mean(0)
fig, axs = plt.subplots(1, 3, figsize=(24, 6.2))
for ax, dd, mm, ttl, sc_ in [(axs[0], dY, mY, f'(a) {nY}: QR 평균 이동 ({mY[0]:+.2f}, {mY[1]:+.2f}) px', 1 / 120),
                             (axs[1], dT, mT, f'(b) {nT}: QR 평균 이동 ({mT[0]:+.2f}, {mT[1]:+.2f}) px', 1 / 120)]:
    bg(ax); dev = dd - mm
    Q = ax.quiver(F.x, F.y, dev[:, 0], -dev[:, 1], color=np.where(onp, BLUE, ORG), angles='xy', scale_units='xy', scale=sc_, width=0.0022)
    ax.quiverkey(Q, 0.80, 0.04, 1.0, '1px', labelpos='E', coordinates='axes')
    ax.set_title(ttl + '\n화살표 = 점 이동 - QR 평균 이동 (x120 확대)', loc='left', fontsize=12)
ax = axs[2]; bg(ax); df_ = (dY - mY) - (dT - mT)
Q = ax.quiver(F.x, F.y, df_[:, 0], -df_[:, 1], color=np.where(onp, BLUE, ORG), angles='xy', scale_units='xy', scale=1 / 120, width=0.0022)
ax.quiverkey(Q, 0.80, 0.04, 1.0, '1px', labelpos='E', coordinates='axes')
ax.set_title(f'(c) 두 사진의 차이 = (a) - (b)\n판 위 점(파랑)은 거의 0, 판밖 점(주황)은 최대 {np.abs(df_[~onp, 0]).max():.2f}px', loc='left', fontsize=12)
fig.suptitle('③ 이동장: 사진 1장마다 특징점 309개의 (dx, dy). Y 1mm와 Θ 0.18은 QR 평균 이동이 같아서 QR 위치만 보면 같은 사진처럼 보임', x=0.01, ha='left', fontsize=14)
plt.tight_layout(rect=(0, 0, 1, 0.93)); plt.savefig(PROJ/'50_방법3_이동장.png', dpi=95, facecolor='white'); plt.close()

# ======================= 51: 이동 모델링 =======================
grp = np.where(onp, '판위', '판밖'); sY, sT, v1, v2 = fit_axis(D, NM, grp)
from t6_F1 import LAB
lab = np.array([LAB[nn] for nn in NM]); ax_ = np.array([AXIS[nn] for nn in NM])
kq = k; kf = int(np.argmin(np.where(~onp, F.ratio.values, 9)))
fig = plt.figure(figsize=(22, 13.5)); gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.15], hspace=0.28)
for j, a in enumerate(['Y', 'Θ']):
    ax = fig.add_subplot(gs[0, j]); m = ax_ == a
    for kk, col, nm in [(kq, BLUE, f'QR 점 #{kq}'), (kf, ORG, f'먼 판밖 점 #{kf} ({F.x[kf]}, {F.y[kf]})')]:
        A = np.c_[np.ones(m.sum()), lab[m], lab[m]**2]; co, *_ = np.linalg.lstsq(A, D[m, kk, 0], rcond=None)
        ax.scatter(lab[m], D[m, kk, 0], s=8, color=col, label=f'{nm}: 기울기 {co[1]:.2f}px')
        g = np.linspace(lab[m].min(), lab[m].max(), 100); ax.plot(g, co[0] + co[1] * g + co[2] * g * g, color=TXT, lw=0.8)
    ax.set_xlabel('Y (mm, 보정본)' if a == 'Y' else 'Θ (보정본)'); ax.set_ylabel('가로 이동 dx (px)'); clean(ax); ax.legend(frameon=False, loc='upper left')
    ax.set_title(f'({"a" if a == "Y" else "b"}) {a} 사진 {m.sum()}장: 점마다 dx = b + s·{a} + c·{a}² 를 맞춤 (검정 선)', loc='left', fontsize=12)
ax = fig.add_subplot(gs[0, 2]); ax.axis('off')
ax.text(0, 0.95, '모델 (점 p, 성분 x/y마다)', fontsize=14, weight='bold', va='top')
ax.text(0, 0.80, 'Y 사진:   d_p = b_p + s_Y(p)·Y + c_Y(p)·Y²\nΘ 사진:   d_p = b_p + s_Θ(p)·Θ + c_Θ(p)·Θ²\n\n'
        '학습: 축마다 그 축 사진만으로 최소제곱\n(점 309개 x 2성분 = 618개 식, 각 3개 계수)\n\n'
        f'두 점의 기울기 비 (Y / Θ)\n  QR 점      {F.ratio[kq]:.4f}\n  먼 판밖 점  {F.ratio[kf]:.4f}\n'
        f'→ 같은 QR 이동이어도 먼 점은\n   Y일 때 {(1 - F.ratio[kf] / F.ratio[kq]) * 100:.0f}% 덜 움직임 (시차)', fontsize=13, va='top', family='Malgun Gothic')
bot = gs[1, :].subgridspec(1, 2)
for j, (s_, a) in enumerate([(sY, 'Y'), (sT, 'Θ')]):
    ax = fig.add_subplot(bot[j])
    bg(ax); v = s_.reshape(-1, 2); dev = v - v[qr].mean(0)
    Q = ax.quiver(F.x, F.y, dev[:, 0], -dev[:, 1], np.hypot(*dev.T), cmap='viridis', angles='xy', scale_units='xy', scale=1 / 60, width=0.0022)
    ax.quiverkey(Q, 0.80, 0.04, 1.0, '1px', labelpos='E', coordinates='axes')
    ax.set_title(f'({"c" if a == "Y" else "d"}) s_{a}: {a} {"1mm" if a == "Y" else "1"}가 만드는 이동 무늬 (QR 평균 ({v[qr].mean(0)[0]:+.2f}, {v[qr].mean(0)[1]:+.2f})px 을 뺀 나머지, x60)',
                 loc='left', fontsize=12)
fig.suptitle('④ 이동 모델링: 점마다 "Y가 1 바뀌면 얼마나, Θ가 1 바뀌면 얼마나 움직이나"를 학습 (v5 그림 39 위 두 칸과 같은 것, 패치 대신 특징점)', x=0.01, ha='left', fontsize=14)
fig.subplots_adjust(left=0.04, right=0.99, top=0.93, bottom=0.03, wspace=0.18); plt.savefig(PROJ/'51_방법4_이동모델링.png', dpi=95, facecolor='white'); plt.close()

# ======================= 52: 구분 단서 =======================
x_, y_ = F.x.values.astype(float), F.y.values.astype(float)
nA, aA, rA = leftover(sY, sT, v1, x_, y_, 1); nQ, aQ, rQ = leftover(sY[np.repeat(qr, 2)], sT[np.repeat(qr, 2)], v1[np.repeat(qr, 2)], x_[qr], y_[qr], 1)
w = 1 / np.maximum(v1, 0.003**2); con = (w * rA**2).reshape(-1, 2).sum(1)
fig = plt.figure(figsize=(22, 8.5)); gs = fig.add_gridspec(1, 2, width_ratios=[2.3, 1])
ax = fig.add_subplot(gs[0]); bg(ax); rv = rA.reshape(-1, 2)
Q = ax.quiver(F.x, F.y, rv[:, 0], -rv[:, 1], np.sqrt(con), cmap='magma_r', angles='xy', scale_units='xy', scale=1 / 100, width=0.0024)
plt.colorbar(Q, ax=ax, fraction=0.02, label='점 하나의 구분 단서 크기 (잡음 σ 단위)'); ax.quiverkey(Q, 0.80, 0.04, 1.0, '1px', labelpos='E', coordinates='axes')
ax.set_title(f'Y 1mm 무늬를 Θ {aA:+.4f} + 방해 무늬(세로 이동, 화면 회전)로 최대한 흉내 내고 남은 차이 (x100)', loc='left', fontsize=13)
ax = fig.add_subplot(gs[1]); parts = [('QR 점', qr), ('판밖 점', ~onp)]
tot = con.sum(); vals = [con[m].sum() / tot * 100 for _, m in parts]
ax.bar([p for p, _ in parts], vals, color=[BLUE, ORG], width=0.5); clean(ax)
for xi, vv in enumerate(vals): ax.text(xi, vv + 1, f'{vv:.1f}%', ha='center', fontsize=12)
ax.set_ylabel('구분 단서 (잡음 대비 제곱합) 중 비율 (%)'); ax.set_ylim(0, 110)
ax.set_title(f'단서 크기 (Y 1mm당): QR만 {nQ:.1f}σ, QR + 판밖 {nA:.0f}σ\n→ 3σ 구분 최소 |Y|: QR만 {3 / nQ:.2f}mm, QR + 판밖 {3 / nA:.3f}mm', loc='left', fontsize=12)
fig.suptitle('⑤ 구분 단서: Y와 Θ 무늬가 서로 흉내 낼 수 없는 부분. Y와 Θ를 가르는 단서의 87%가 판밖 점에서 나옴 (v5 그림 39 맨 아래 칸과 같은 계산)', x=0.01, ha='left', fontsize=14)
plt.tight_layout(rect=(0, 0, 1, 0.93)); plt.savefig(PROJ/'52_방법5_구분단서.png', dpi=95, facecolor='white'); plt.close()

# ======================= 53: 판정 =======================
P = pd.read_csv(HERE/'t7_predictions.csv')
fig, axs = plt.subplots(1, 2, figsize=(21, 7.2), sharey=True)
for ax, key, ttl in [(axs[0], '17px / QR만 / 보정본 / 2차 / 짝홀', 'QR 점만 (177개)'), (axs[1], '17px / QR + 판밖 / 보정본 / 2차 / 짝홀', 'QR + 판밖 점 (309개)')]:
    d = P[P.조건 == key]; ye = np.where(d.axis == 'Y', d.cmd, d.cmd * THETA_TO_Y); ok = d.pred_axis == d.axis
    for a, col in [('Y', BLUE), ('Θ', ORG)]:
        m = (d.axis == a).values & ok.values; ax.scatter(ye[m], d.cost_ratio[m], s=12, color=col, label=f'{a} 사진 (맞음)')
    ax.scatter(ye[~ok], d.cost_ratio[~ok], s=70, marker='x', color=RED, lw=2, label=f'판정 틀림 {(~ok).sum()}장')
    ax.axhline(1, color=TXT, lw=1); ax.set_yscale('log'); ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g')); ax.set_xlim(-2.1, 2.1); clean(ax)
    ax.text(-2.05, 1.3, '위: Θ로 판정', fontsize=11, color=SUB); ax.text(-2.05, 0.72, '아래: Y로 판정', fontsize=11, color=SUB, va='top')
    ax.set_xlabel('정답 크기 (Y는 mm, Θ는 같은 QR 이동의 Y mm로 환산)'); ax.set_title(ttl, loc='left', fontsize=13); ax.legend(frameon=True, framealpha=0.95, loc='upper center', ncol=3)
axs[0].set_ylabel('비용 비 = (Y 가설 비용) / (Θ 가설 비용)')
fig.suptitle('⑥ 판정: 새 사진의 이동장을 "Y만 움직였다" 모델과 "Θ만 움직였다" 모델에 각각 맞춤 (값 v + 방해 무늬, 잡음 가중, Huber) → 비용이 작은 쪽 축, 그 v가 답.  짝/홀 교차검증',
             x=0.01, ha='left', fontsize=14)
plt.tight_layout(rect=(0, 0, 1, 0.93)); plt.savefig(PROJ/'53_방법6_판정.png', dpi=95, facecolor='white'); plt.close()
print('saved', k, kf)



