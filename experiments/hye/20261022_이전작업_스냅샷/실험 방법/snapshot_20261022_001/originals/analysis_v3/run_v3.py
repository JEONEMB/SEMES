"""v3: 어제 조합 13개 + Q5b(판 원근 변환, 오늘 v2-a2) x 알고리즘 C1~C6, 그리고 G1(픽셀 직접 가우스-뉴턴)을
같은 교차검증(짝/홀 2겹)과 같은 판정 기준으로 비교한다.
판정(규칙3): 추정 Θ가 짝의 Θ에서 3σ 넘게 떨어지면 구분 가능. σ = 그 방법의 Θ 추정 오차 표준편차(교차검증)."""
import numpy as np, pandas as pd, json
from pathlib import Path
from gn_features import *          # FEAT, ALG(C1~C6), X, folds, names
HERE = Path(__file__).parent; PROJ = HERE.parent
q = pd.read_csv(PROJ/'14_데이터_QR마커위치.csv').set_index('file')
hom = pd.read_csv(PROJ/'analysis_v2/homography_features.csv').set_index('file').drop(columns=['cc', 'h20', 'h21'])
FEAT = dict(FEAT)
FEAT['Q5b 판 전체 원근 변환, 귀퉁이+QR중심 (14개)'] = ('QR만', hom.loc[names].values)
order = list(FEAT)[:5] + ['Q5b 판 전체 원근 변환, 귀퉁이+QR중심 (14개)'] + list(FEAT)[5:-1]
FEAT = {k: FEAT[k] for k in order}
# 짝: QR 3개 가로 평균 px가 같은 상대 축 값 (v1, v2와 같은 보간)
m = q.loc[names, ['dLx', 'dCx', 'dRx']].mean(1).values
isY = (X[:, 0] == 0) & (X[:, 1] != 0); isT = (X[:, 1] == 0) & (X[:, 0] != 0); isC = (X == 0).all(1)
Tm = np.r_[m[isT], m[isC]]; Tv = np.r_[X[isT, 0], 0]; o = np.argsort(Tm); Tm, Tv = Tm[o], Tv[o]
Ym = np.r_[m[isY], m[isC]]; o = np.argsort(Ym); Ym = Ym[o]
twinT = np.full(len(X), np.nan)                                   # 짝의 Θ (Y 사진은 θ짝, Θ 사진은 0)
twinT[isY] = np.interp(m[isY], Tm, Tv)
inrange = isT & (m >= Ym.min()) & (m <= Ym.max()); twinT[inrange] = 0.0
def run(F, fa):
    est = np.full(X.shape, np.nan)
    for tr, te in folds:
        r = fa(F, tr, te)
        if r is None: return None
        est[te] = r
    return est
def judge(est):
    e = est - X; sT = e[:, 0].std()
    can = np.abs(est[:, 0] - twinT) > 3 * sT                      # 짝 가설을 3σ로 기각하면 구분 가능
    return dict(Y불가=int((isY & ~can).sum()), Θ불가=int((inrange & ~can).sum()),
                오판율=float((np.abs(e[:, 0]) > 3 * sT).mean()), σΘ=sT), can
def metrics(est):
    e = est - X; eT, eY = np.abs(e[:, 0]), np.abs(e[:, 1])
    return dict(T_RMS=np.sqrt((eT**2).mean()), T_최대=eT.max(), Y_평균=eY.mean(), Y_RMS=np.sqrt((eY**2).mean()), Y_최대=eY.max(),
                Y_009안=np.mean(eY <= 0.09) * 100, Y_005안=np.mean(eY <= 0.05) * 100)
rows, per = [], {}
for fn, (g, F) in FEAT.items():
    for an, fa in ALG.items():
        est = run(F, fa)
        if est is None: continue
        j, can = judge(est); rows.append(dict(영역=g, 특징점=fn, 알고리즘=an, **metrics(est), **j)); per[(fn, an)] = (est, can)
g1 = np.load(HERE/'g1_est.npy'); j, can = judge(g1)
rows.append(dict(영역='QR만', 특징점='G1 QR 판 픽셀 전체 (특징점 없음)', 알고리즘='G1 픽셀 직접 가우스-뉴턴 (Θ, Y + 방해 3)', **metrics(g1), **j))
per[('G1 QR 판 픽셀 전체 (특징점 없음)', 'G1 픽셀 직접 가우스-뉴턴 (Θ, Y + 방해 3)')] = (g1, can)
R = pd.DataFrame(rows); R.to_csv(HERE/'v3_results.csv', index=False, encoding='utf-8-sig')
pd.to_pickle(dict(R=R, per=per, names=names, X=X, twinT=twinT, isY=isY, isT=isT, inrange=inrange, m=m), HERE/'v3.pkl')
pd.set_option('display.width', 250)
print(R[['특징점', '알고리즘', 'T_RMS', 'Y_RMS', 'Y_최대', 'Y_009안', 'Y불가', 'Θ불가', '오판율']].round(4).to_string(index=False))
