"""G2 그림: 44 특징점 지도 + 깊이 비, 45 구분 가능 최소 |Y| (이론), 46 평가 결과 (평가 기준), 47 사진별 오차."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt, matplotlib.ticker
from common import *
plt.rcParams['font.family'] = 'Malgun Gothic'; plt.rcParams['axes.unicode_minus'] = False
TXT, SUB, GRID = '#1f1f1e', '#5f5e58', '#e4e3dc'
BLUE, ORG, GRN, RED = '#4a72d0', '#d2703f', '#5aa776', '#c8423b'
THETA_TO_Y = 80.66 / 14.39

def clean(ax):
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
    ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)

# ---------- 44: 특징점 지도 + 깊이 비 ----------
F = pd.read_csv(HERE/'features_hp8_ratio.csv'); cen = read_raw('center.bmp')
onp = F.region.str.startswith('QR') | F.region.isin(['판', '판나사'])
fig = plt.figure(figsize=(20, 8.2)); gs = fig.add_gridspec(1, 2, width_ratios=[2.1, 1])
ax = fig.add_subplot(gs[0]); ax.imshow(cen, cmap='gray', vmin=0, vmax=255, alpha=0.6)
sc = ax.scatter(F.x, F.y, c=F.ratio, cmap='viridis_r', vmin=0.140, vmax=0.180, s=np.where(onp, 14, 46),
                marker='o', edgecolors='white', linewidths=0.6)
for nm, (a, b, c, d) in EXCL.items():
    ax.add_patch(plt.Rectangle((a, b), c - a, d - b, fill=False, ec=RED, lw=1.5, ls='--')); ax.text(a + 8, d - 15, f'제외: {nm}', color=RED, fontsize=11)
ax.set_xlim(150, 3690); ax.set_ylim(2110, 50); ax.axis('off')
plt.colorbar(sc, ax=ax, fraction=0.025, label='Y 기울기 / Θ 기울기 (작을수록 멀다)')
ax.set_title(f'특징점 {len(F)}개 (17x17px, 겹침 없음). 판 위 {onp.sum()}개 (작은 점), 판밖 {(~onp).sum()}개 (큰 점). 색 = 깊이 비', loc='left', fontsize=13)
ax = fig.add_subplot(gs[1]); bins = np.linspace(0.135, 0.182, 48)
ax.hist(F.ratio[onp], bins=bins, color=BLUE, label=f'판 위 (QR) {onp.sum()}개'); ax.hist(F.ratio[~onp], bins=bins, color=ORG, alpha=0.85, label=f'판밖 {(~onp).sum()}개')
ax.set_xlabel('Y 기울기 / Θ 기울기 (가로 이동)'); ax.set_ylabel('특징점 수'); clean(ax); ax.legend(frameon=False)
ax.set_title('판 위 점은 모두 0.178 한 값 (깊이 같음)\n판밖 점은 0.142~0.178로 퍼짐 = 구분 단서', loc='left', fontsize=13)
plt.tight_layout(); plt.savefig(PROJ/'44_특징점지도_깊이비.png', dpi=100, facecolor='white'); plt.close()

# ---------- 45: 이론 한계 ----------
T = pd.read_csv(HERE/'t6_F1.csv'); T = T[T.방해무늬 == 'v5']
pick = ['특징점 17px / QR만', '특징점 25px / QR만', 'v5 패치 32px 겹침 / QR만', 'v5 패치 32px 안 겹침 / QR만',
        '특징점 17px / 판밖만', '특징점 17px / QR + 판밖']
lab = ['특징점 17px, QR만', '특징점 25px, QR만', 'v5 패치, QR만 (16px 겹침, v5 보고값)', 'v5 패치, QR만 (겹침 없음)',
       '특징점 17px, 판밖만', '특징점 17px, QR + 판밖']
fig, ax = plt.subplots(figsize=(14, 5.8)); yy = np.arange(len(pick))[::-1]
for j, (vn, col) in enumerate([('맞추기 잔차(v5식)', BLUE), ('점마다 독립', ORG)]):
    v = [T[(T.대상 == p) & (T.잡음 == vn)].최소Y_mm.iloc[0] for p in pick]
    ax.barh(yy + (0.18 if j == 0 else -0.18), v, height=0.34, color=col, label=f'잡음 = {vn}')
    for y_, vv in zip(yy, v): ax.text(vv * 1.06, y_ + (0.18 if j == 0 else -0.18), f'{vv:.3f}', va='center', fontsize=10, color=TXT)
ax.axvline(0.09, color=RED, lw=1.2, ls='--'); ax.text(0.092, yy[0] + 0.55, '허용 오차 0.09mm', color=RED, fontsize=10)
ax.set_xscale('log'); ax.set_xticks([0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3]); ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g')); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.set_yticks(yy); ax.set_yticklabels(lab, fontsize=11); ax.set_xlabel('3σ로 Y와 Θ를 구분하는 가장 작은 |Y| (mm, 로그 눈금)')
clean(ax); ax.legend(frameon=False, loc='lower right')
ax.set_title('이론 한계: Y 무늬를 Θ + 방해 무늬로 최대한 흉내 내고 남은 차이 (잡음 대비). 작을수록 작은 이동부터 구분. 겹친 패치는 잡음이 얽혀 낙관적',
             loc='left', fontsize=12)
plt.tight_layout(); plt.savefig(PROJ/'45_이론한계_구분가능최소Y.png', dpi=110, facecolor='white'); plt.close()

# ---------- 46: 평가 결과 ----------
E = pd.read_csv(PROJ/'42_평가표_특징점vs패치.csv'); E = E[E.대표.notna() & (E.대표 != '')]
meth = ['v6 특징점 QR만', 'v5 패치 QR만', 'v6 특징점 QR + 판밖']
ser = [('명령값 (보정 안 함)', ' (명령값 학습)', BLUE, '정답 = 명령값 (명령값으로 학습)'),
       ('실제 위치 (보정함)', '', ORG, '정답 = 실제 위치 (보정본으로 학습)')]
fig, axs = plt.subplots(1, 3, figsize=(20, 6))
for ax, col, yl, fmt in [(axs[0], '축정확도', 'Y/Θ 판정 정확도 (%)', '{:.1f}'), (axs[1], 'Y_평균오차', 'Y 평균 절대 오차 (µm, 축 맞은 사진)', '{:.1f}'),
                         (axs[2], 'Θ_평균오차', 'Θ 평균 절대 오차 (mΘ = 0.001, 축 맞은 사진)', '{:.2f}')]:
    x = np.arange(len(meth))
    for j, (basis, suf, c, ln) in enumerate(ser):
        v = [E[(E.대표 == m + suf) & (E.정답기준 == basis)][col].iloc[0] for m in meth]
        b = ax.bar(x + (j - 0.5) * 0.36, v, width=0.34, color=c, label=ln)
        for xi, vv in zip(x + (j - 0.5) * 0.36, v): ax.text(xi, vv, fmt.format(vv), ha='center', va='bottom', fontsize=10, color=TXT)
    ax.set_xticks(x); ax.set_xticklabels(meth, fontsize=11); ax.set_title(yl, loc='left', fontsize=12); clean(ax)
    if col == '축정확도': ax.set_ylim(85, 101); ax.set_title(yl + ' (세로축 85부터)', loc='left', fontsize=12)
fig.legend(*axs[0].get_legend_handles_labels(), frameon=False, loc='upper right', ncol=2, fontsize=11)
fig.suptitle('평가 (00_평가기준.md): 사진 280장, 짝/홀 교차검증, 2차 매핑, 특징점 17px. 정답 기준 2개 (학습 라벨과 같은 기준으로 채점)', x=0.01, ha='left', fontsize=14)
plt.tight_layout(rect=(0, 0, 1, 0.94)); plt.savefig(PROJ/'46_평가결과_정답기준2개.png', dpi=110, facecolor='white'); plt.close()

# ---------- 47: 사진별 오차 (실제 위치 기준, 보정본 학습) ----------
P = pd.read_csv(PROJ/'43_사진별_예측_대표방법.csv')
fig, axs = plt.subplots(2, 3, figsize=(21, 9.5), sharey='row')
for j, m in enumerate(meth):
    d = P[P.방법 == m]
    for i, a in enumerate(['Y', 'Θ']):
        ax = axs[i, j]; g = d[d.axis == a]; ok = g.축맞음
        ax.scatter(g.act[ok], g.오차_실제위치[ok] * 1000, s=10, color=BLUE if a == 'Y' else ORG)
        bad = g[~ok]; ax.scatter(bad.act, np.zeros(len(bad)), marker='x', s=55, color=RED, label=f'축 판정 틀림 {len(bad)}장')
        ax.axhline(0, color=SUB, lw=0.8); clean(ax)
        ax.set_xlabel(f'실제 위치 {a}' + (' (mm)' if a == 'Y' else '')); ax.set_ylabel('오차 (µm)' if a == 'Y' else '오차 (mΘ)')
        ax.set_title(f'{m}: {a} 사진', loc='left', fontsize=12)
        if len(bad): ax.legend(frameon=False, loc='lower right')
fig.suptitle('사진별 오차 (정답 = 실제 위치, 보정본으로 학습). 빨강 x = Y인지 Θ인지 틀린 사진 (값은 0에 표시)', x=0.01, ha='left', fontsize=14)
plt.tight_layout(rect=(0, 0, 1, 0.95)); plt.savefig(PROJ/'47_사진별오차_대표방법.png', dpi=100, facecolor='white'); plt.close()
print('saved')


