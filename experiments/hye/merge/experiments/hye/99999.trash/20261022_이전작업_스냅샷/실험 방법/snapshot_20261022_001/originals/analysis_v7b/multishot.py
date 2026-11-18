"""여러 장 촬영 시뮬레이션: 같은 자세에서 N장 찍어 판 변형 12개 값을 평균 내면 몇 장부터 Y/Θ 판정이 100%가 되나.
잡음 나누기 (12개 값 공간):
  Σ_tot  = 모델 학습 잔차의 공분산 (지금 판정에 쓰는 것, 1장 기준)
  Σ_shot = 점마다 독립 흔들림 (사진마다 새로 생김)을 12개 값으로 옮긴 것  → N장 평균이면 1/N
  Σ_sys  = Σ_tot - Σ_shot (스테이지 실제 위치 오차, 모델 오차: 여러 장 찍어도 그대로)
가정 A (낙관): Σ_tot 전부가 1/N.  가정 B (현실): Σ_shot만 1/N, Σ_sys는 그대로.
'정답 생성기'는 전체 사진으로 맞춘 2차 모델, '판정기'는 학습 폴드로 맞춘 최종 방법(1차) → 모델 불일치도 포함.
판정기의 공분산은 N장 평균에 맞게 Σ_sys + Σ_shot/N (가정 B) 또는 Σ_tot/N (가정 A)로 바꿔 쓴다."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from exp import *
plt.rcParams['font.family'] = 'Malgun Gothic'; plt.rcParams['axes.unicode_minus'] = False
rng = np.random.default_rng(0)

F, D = load(8); Fe = feats(F, D, 'P12'); n = len(F)
x = (F.x.values - 1906) / 486; y = (F.y.values - 1070) / 122; B = np.c_[np.ones_like(x), x, y, x * x, x * y, y * y]; Bp = np.linalg.pinv(B)   # (6, n)
lab = CORR[0]

# ---- Σ_shot: 점마다 독립 흔들림 (구간마다 2차 맞춤 잔차에서 사진별 공통 이동을 뺀 것) ----
R = np.full(D.shape, np.nan)
for s in ['Y1+', 'Y1-', 'Y2+', 'Y2-', 'Θ+', 'Θ-']:
    m = SES == s; A = np.c_[np.ones(m.sum()), CMD[m], CMD[m]**2]
    for c in (0, 1): co, *_ = np.linalg.lstsq(A, D[m, :, c], rcond=None); R[m, :, c] = D[m, :, c] - A @ co
Rr = R[~ISC]; Rr = Rr - np.median(Rr, 1, keepdims=True); var_pt = Rr.var(0)            # (n, 2)
S_shot = np.zeros((12, 12))
for c in (0, 1):
    Cc = (Bp * var_pt[:, c]) @ Bp.T                                                        # (6, 6)
    for j in range(6):
        for k in range(6): S_shot[2 * j + c, 2 * k + c] = Cc[j, k]

def model(quad, mask):
    return M(kind='P12', quad=quad, noise='full', F=F)().fit(Fe[mask], lab[mask], AX[mask])
gen = model(True, np.ones(len(NAMES), bool))                                                # 정답 생성기
Rg = []
for a in ['Y', 'Θ']:
    m = AX == a; s, c, b = gen.co[a]; Rg.append(Fe[m] - (np.outer(lab[m], s) + np.outer(lab[m]**2, c) + b))
S_tot = np.cov(np.vstack(Rg).T)
w, V = np.linalg.eigh(S_tot - S_shot); S_sys = (V * np.maximum(w, 0)) @ V.T

# 구분 방향에서 잡음 몫 (판정에 실제로 영향을 주는 방향)
def disc_share():
    W = np.linalg.cholesky(np.linalg.inv(S_tot)).T; sY, sT = gen.co['Y'][0], gen.co['Θ'][0]; U = U_for(F, 'P12')
    A = W @ np.c_[sT, U]; p, *_ = np.linalg.lstsq(A, W @ sY, rcond=None); g = sY - np.c_[sT, U] @ p   # Y 무늬 중 Θ로 흉내 못 내는 부분
    gw = np.linalg.solve(S_tot, g); return float(gw @ S_shot @ gw / (gw @ S_tot @ gw))
share = disc_share()
print(f'구분 방향의 잡음 중 사진마다 새로 생기는 몫 (Σ_shot): {share * 100:.1f}%  → 여러 장 평균으로 줄일 수 있는 부분')

from twostage import TwoStage
trm = np.array([V5.FOLD.get(nm, -1) != 0 for nm in NAMES])
dec = TwoStage(F).fit(Fe[trm], lab[trm], AX[trm]).A                                         # 판정기: 2단계 방법의 축 판정 모델 (폴드0 시험 기준 학습)
C_A = np.linalg.inv(dec.W.T @ dec.W)                                                          # 판정기 1장 공분산
C_A_sys = (lambda w, V: (V * np.maximum(w, 0)) @ V.T)(*np.linalg.eigh(C_A - S_shot))
def decide(f, Sig):
    return 'Y' if dec.predict(f)['llr'] < 0 else 'Θ'

GRID = [('Y', v) for v in np.round(np.arange(0.05, 0.55, 0.05), 2)] + [('Θ', v) for v in np.round(np.arange(0.01, 0.11, 0.01), 2)]
GRID += [(a, -v) for a, v in GRID]
NS = [1, 2, 4, 8, 16, 32, 64]; TRIALS = 300
rows = []
for scen in ['A 낙관 (잡음 전부 줄어듦)', 'B 현실 (점마다 흔들림만 줄어듦)']:
    for N in NS:
        Sig = S_tot / N if scen[0] == 'A' else S_sys + S_shot / N
        L = np.linalg.cholesky(Sig + 1e-12 * np.eye(12))
        CA = C_A / N if scen[0] == 'A' else C_A_sys + S_shot / N; dec.W = np.linalg.cholesky(np.linalg.inv(CA + 1e-12 * np.eye(12))).T
        for a, v in GRID:
            s, c, b = gen.co[a]; mu = s * v + c * v * v + b
            wrong = sum(decide(mu + L @ rng.standard_normal(12), Sig) != a for _ in range(TRIALS))
            rows.append(dict(가정=scen, N=N, 축=a, 값=v, 틀림확률=wrong / TRIALS))
        T = pd.DataFrame(rows); t = T[(T.가정 == scen) & (T.N == N)]
        exp_wrong = sum(t[(t.축 == a) & (t.값 == v)].틀림확률.iloc[0] for a, v in GRID)          # 이 데이터와 같은 간격 사진 1장씩일 때 기대 틀림 수
        print(f'{scen} N={N:3d}: 최악 틀림확률 {t.틀림확률.max() * 100:5.1f}% ({t.loc[t.틀림확률.idxmax(), "축"]}{t.loc[t.틀림확률.idxmax(), "값"]:+.2f}), '
              f'280장 기준 기대 틀림 {exp_wrong:.2f}장', flush=True)
T = pd.DataFrame(rows); T.to_csv(OUT/'64_v7_여러장_시뮬레이션.csv', index=False, encoding='utf-8-sig')

# 그림
fig, axs = plt.subplots(1, 2, figsize=(21, 7), sharey=True)
for ax, scen in zip(axs, T.가정.unique()):
    for a, col in [('Y', '#4a72d0'), ('Θ', '#d2703f')]:
        for v in sorted({abs(v) for aa, v in GRID if aa == a}):
            if (a == 'Y' and v > 0.3) or (a == 'Θ' and v > 0.06): continue
            t = T[(T.가정 == scen) & (T.축 == a) & (T.값.abs() == v)].groupby('N').틀림확률.max()
            ax.plot(t.index, t.values * 100, 'o-' if a == 'Y' else 's--', color=col, alpha=0.35 + 0.65 * (1 - v / (0.3 if a == 'Y' else 0.06)), label=f'{a} {v:.2f}')
    ax.set_xscale('log'); ax.set_xticks(NS); ax.set_xticklabels(NS); ax.set_xlabel('같은 자세에서 찍는 장 수 N'); ax.grid(color='#e4e3dc')
    ax.set_title(scen, loc='left', fontsize=13); ax.legend(ncol=2, fontsize=9, frameon=False, title='정답 (진할수록 작은 값)')
axs[0].set_ylabel('축 판정 틀릴 확률 (%, ±값 중 큰 쪽)')
fig.suptitle(f'여러 장 평균 시뮬레이션. 구분 방향 잡음 중 찍을 때마다 바뀌는 몫 = {share * 100:.0f}% (나머지는 여러 장 찍어도 안 줄어듦)', x=0.01, ha='left', fontsize=14)
plt.tight_layout(rect=(0, 0, 1, 0.93)); plt.savefig(OUT/'64_v7_여러장_시뮬레이션.png', dpi=100, facecolor='white'); plt.close()



