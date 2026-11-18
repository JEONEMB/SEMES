"""D4, D8: 실제 사진에서 특징점 위치 재기의 정밀도.
촬영 구간(6개)마다, 특징점 성분마다 d = α + β v + γ v² 를 맞추고 남은 흔들림을 본다.
흔들림 = (모든 점이 같이 움직이는 성분: 스테이지 위치 오차 등) + (점마다 따로: 측정 잡음).
같이 움직이는 성분은 영역(QR / 판밖)마다 사진별 중앙값으로 뺀다. 판밖은 깊이가 달라 같이 움직이는 양이 점마다 조금 다를 수 있음."""
import numpy as np, pandas as pd
from common import *

def analyze(hp):
    Z = np.load(HERE/f'tracks_hp{hp}.npz'); F = pd.read_csv(HERE/f'features_hp{hp}.csv')
    grp = np.where(F.region.str.startswith('QR'), 'QR', np.where(F.region == '판밖', '판밖', '기타'))
    ses = np.array([session_of(n) for n in NAMES]); v = np.array([VAL[n] for n in NAMES])
    rows, spec_all = [], {}
    for meth in ['tm', 'gn']:
        D = Z[f'D_{meth}']
        R = np.full(D.shape, np.nan)
        for s in ['Y1+', 'Y1-', 'Y2+', 'Y2-', 'Θ+', 'Θ-']:
            m = ses == s; A = np.c_[np.ones(m.sum()), v[m], v[m]**2]
            for c in (0, 1):
                coef, *_ = np.linalg.lstsq(A, D[m, :, c], rcond=None); R[m, :, c] = D[m, :, c] - A @ coef
        ok = ses != 'C'
        for g in ['QR', '판밖']:
            k = grp == g
            for c, cn in [(0, 'dx'), (1, 'dy')]:
                r = R[ok][:, k, c]; cm = np.median(r, 1, keepdims=True); sp = r - cm
                spec_all[(meth, g, cn)] = sp
                th = np.isin(ses[ok], ['Θ+', 'Θ-']); av = np.abs(v[ok])
                rows.append(dict(hp=hp, 측정법=meth.upper(), 영역=g, 성분=cn, 점수=int(k.sum()),
                                 전체흔들림=np.median(r.std(0)), 같이움직임=cm.std(),
                                 점마다=np.median(sp.std(0)), 점마다_상위10pct=np.quantile(sp.std(0), 0.9),
                                 점마다_Θ작음=np.median(sp[th & (av < 0.3)].std(0)), 점마다_Θ큼=np.median(sp[th & (av > 0.7)].std(0)),
                                 점마다_Y=np.median(sp[~th].std(0)), 튀는값05=int((np.abs(sp) > 0.05).sum())))
    return pd.DataFrame(rows), Z, F, grp

if __name__ == '__main__':
    out = []
    for hp in (8, 12):
        T, Z, F, grp = analyze(hp); out.append(T)
        # D8: QR 영역 특징점 평균 가로 이동 vs 14번 CSV (QR 3개 평균)
        qr = np.array([q.loc[n, ['dLx', 'dCx', 'dRx']].mean() for n in NAMES])
        for meth in ['tm', 'gn']:
            mine = Z[f'D_{meth}'][:, grp == 'QR', 0].mean(1); d = mine - qr; d -= np.median(d)
            print(f'hp{hp} {meth.upper()} QR 평균 dx - 14번CSV: 표준편차 {d.std():.4f}px, 최대 {np.abs(d).max():.4f}px')
        sc = Z['score']; print(f'hp{hp} 정합 점수 최소 {sc.min():.3f}, 0.9 미만 비율 {(sc < 0.9).mean() * 100:.2f}%')
    T = pd.concat(out); T.round(4).to_csv(HERE/'t3_real_accuracy.csv', index=False, encoding='utf-8-sig')
    print(T.round(4).to_string())
