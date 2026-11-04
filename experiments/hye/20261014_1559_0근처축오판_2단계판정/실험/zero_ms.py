"""0점 맞추기(Z1, 구간 0점을 뺀 판 변형) + 여러 장 평균 시뮬레이션.
정답 생성기: 0점 뺀 특징으로 전체 사진 2차 모델 (절편 0, 정답 = 명령값). 판정기: 2단계 방법의 축 판정 모델 (폴드0 시험 기준 학습).
잡음: Σ_tot = 생성기 잔차 공분산 (구간 0점 차이가 빠진 잔차), Σ_shot = 점마다 독립 흔들림. 가정 A: 전부 1/N, 가정 B: Σ_shot만 1/N.
0점 사진 K장의 잡음도 넣는다: 0점 추정 오차 ~ Σ_tot / K (가정 B면 Σ_sys + Σ_shot/K), 구간마다 한 번 뽑혀 그 구간 사진 전체에 공통."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from zero_cal import *
from twostage import TwoStage
plt.rcParams['font.family'] = 'Malgun Gothic'; plt.rcParams['axes.unicode_minus'] = False
rng = np.random.default_rng(2)

Fc = corrected(~ISC)                                                        # 0점 뺀 특징 (전체 사진으로 구한 구간 0점)
def shot_cov():
    """점마다 독립 흔들림 (구간마다 2차 맞춤 잔차 - 사진별 공통 이동)을 판 변형 12개 값으로 옮긴 공분산."""
    x = (F.x.values - 1906) / 486; y = (F.y.values - 1070) / 122
    Bp = np.linalg.pinv(np.c_[np.ones_like(x), x, y, x * x, x * y, y * y]); R = np.full(D.shape, np.nan)
    for s in SESS:
        m = SES == s; A = np.c_[np.ones(m.sum()), CMD[m], CMD[m]**2]
        for c in (0, 1): co, *_ = np.linalg.lstsq(A, D[m, :, c], rcond=None); R[m, :, c] = D[m, :, c] - A @ co
    Rr = R[~ISC]; Rr = Rr - np.median(Rr, 1, keepdims=True); vp = Rr.var(0); S = np.zeros((12, 12))
    for c in (0, 1):
        Cc = (Bp * vp[:, c]) @ Bp.T
        for j in range(6):
            for k in range(6): S[2 * j + c, 2 * k + c] = Cc[j, k]
    return S
S_shot = shot_cov()
gen = M(kind='P12', quad=True, noise='full', F=F, intercept='zero')().fit(Fc, CMD, AX)
Rg = np.vstack([Fc[AX == a] - (np.outer(CMD[AX == a], gen.co[a][0]) + np.outer(CMD[AX == a]**2, gen.co[a][1])) for a in ['Y', 'Θ']])
S_tot = np.cov(Rg.T); w, V = np.linalg.eigh(S_tot - S_shot); S_sys = (V * np.maximum(w, 0)) @ V.T
trm = np.array([V5.FOLD.get(nm, -1) != 0 for nm in NAMES])
dec = TwoStage(F, 'zero').fit(Fc[trm], CMD[trm], AX[trm]).A; C_A = np.linalg.inv(dec.W.T @ dec.W)
C_A_sys = (lambda w, V: (V * np.maximum(w, 0)) @ V.T)(*np.linalg.eigh(C_A - S_shot))

GRID = [('Y', v) for v in np.round(np.arange(0.05, 0.55, 0.05), 2)] + [('Θ', v) for v in np.round(np.arange(0.01, 0.11, 0.01), 2)]
GRID += [(a, -v) for a, v in GRID]
NS = [1, 4, 16, 64]; KS = [1, 4, 16, None]; TR = 300; rows = []
for scen in ['A', 'B']:
    for K in KS:
        for N in NS:
            Sig = S_tot / N if scen == 'A' else S_sys + S_shot / N
            Sz = np.zeros((12, 12)) if K is None else (S_tot / K if scen == 'A' else S_sys + S_shot / K)
            L = np.linalg.cholesky(Sig + Sz + 1e-12 * np.eye(12))
            CA = (C_A / N if scen == 'A' else C_A_sys + S_shot / N) + Sz; dec.W = np.linalg.cholesky(np.linalg.inv(CA + 1e-12 * np.eye(12))).T
            for a, v in GRID:
                s, c, _ = gen.co[a]; mu = s * v + c * v * v
                wrong = sum((dec.predict(mu + L @ rng.standard_normal(12))['llr'] < 0) != (a == 'Y') for _ in range(TR))
                rows.append(dict(가정=scen, 영점사진=('정확' if K is None else K), N=N, 축=a, 값=v, 틀림확률=wrong / TR))
            t = pd.DataFrame(rows); t = t[(t.가정 == scen) & (t.영점사진 == ('정확' if K is None else K)) & (t.N == N)]
            small = ((t.축 == 'Y') & (t.값.abs() <= 0.05)) | ((t.축 == 'Θ') & (t.값.abs() <= 0.01))
            print(f'{scen} 0점사진 {"정확" if K is None else K:>3} N={N:3d}: 기대 틀림 전체 {t.틀림확률.sum():.2f}장, 한 스텝 제외 {t[~small].틀림확률.sum():.2f}장, '
                  f'최악 {t.틀림확률.max() * 100:.0f}% ({t.loc[t.틀림확률.idxmax(), "축"]}{t.loc[t.틀림확률.idxmax(), "값"]:+.2f})', flush=True)
T = pd.DataFrame(rows); T.to_csv(OUT/'66_v7_0점맞추기_여러장.csv', index=False, encoding='utf-8-sig')
