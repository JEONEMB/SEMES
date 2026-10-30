"""F1~F3: 구분 가능성의 이론 한계. v5 그림 39와 같은 공식.
  점(또는 패치) 성분마다 d = b + s·v + c·v² 를 Y 사진, Θ 사진으로 따로 맞춘다 (라벨 = v5 보정본, 폴드0 학습 기준, center 제외).
  s_Y(Y 1mm 무늬)를 Θ 무늬 + 방해 무늬로 최대한 흉내 내고 남은 차이 r.
  n = sqrt(Σ w r²), w = 1/잡음 분산  →  3σ로 구분되는 최소 |Y| = 3/n (mm).
잡음 두 가지: (i) v5와 같음: 맞추기 잔차 분산 (라벨 오차 포함), (ii) 점마다 독립 잡음: 사진마다 같이 움직이는 성분(영역별 중앙값)을 뺀 잔차.
방해 무늬: 없음 / v5 (세로 평행 이동, 화면 내 회전) / v5 + 크기 변화."""
import sys, io, contextlib
import numpy as np, pandas as pd
from common import *
sys.path.insert(0, str(PROJ/'analysis_v5'))
with contextlib.redirect_stdout(io.StringIO()):
    import step4_8 as V5
LAB = {n: V5.CORR[0][V5.NAMES.index(n)] for n in V5.NAMES}       # 보정본 라벨 (폴드0 학습으로 맞춘 ①)

def nuis(x, y, kind):
    """방해 무늬 (2n x m). 성분 순서 = 점마다 (dx, dy)."""
    xc, yc = (x - x.mean()) / 100, (y - y.mean()) / 100; z = np.zeros_like(xc); o = np.ones_like(xc)
    cols = []
    if kind >= 1: cols += [np.c_[z, o].ravel(), np.c_[-yc, xc].ravel()]
    if kind >= 2: cols += [np.c_[xc, yc].ravel()]
    return np.stack(cols, 1) if cols else np.zeros((2 * len(x), 0))

def fit_axis(D, names, group_of_pt):
    """D: (사진, 점, 2). 반환 s(Y), s(Θ) (2n), 잡음 분산 두 종류 (2n)."""
    ax = np.array([AXIS[n] for n in names]); lab = np.array([LAB[n] for n in names])
    S, V1, R = {}, [], np.full(D.shape, np.nan)
    for a in ['Y', 'Θ']:
        m = ax == a; A = np.c_[np.ones(m.sum()), lab[m], lab[m]**2]
        Dm = D[m].reshape(m.sum(), -1); co, *_ = np.linalg.lstsq(A, Dm, rcond=None)
        S[a] = co[1]; R[m] = (Dm - A @ co).reshape(m.sum(), *D.shape[1:])
    ok = ax != 'C'; Rr = R[ok]
    var1 = Rr.reshape(len(Rr), -1).var(0)                       # (i) v5와 같음
    Rs = Rr.copy()
    for g in np.unique(group_of_pt):
        k = group_of_pt == g; Rs[:, k] -= np.median(Rr[:, k], 1, keepdims=True)
    var2 = Rs.reshape(len(Rs), -1).var(0)                       # (ii) 점마다 독립
    return S['Y'], S['Θ'], var1, var2

def leftover(sY, sT, var, x, y, kind, floor=0.003**2):
    w = 1 / np.maximum(var, floor); A = np.c_[sT, nuis(x, y, kind)]; sw = np.sqrt(w)
    p, *_ = np.linalg.lstsq(A * sw[:, None], sY * sw, rcond=None); r = sY - A @ p
    return np.sqrt(np.sum(w * r**2)), p[0], r

if __name__ == '__main__':
    rows = []
    def run(label, D, names, x, y, grp, n_pts):
        sY, sT, v1, v2 = fit_axis(D, names, grp)
        for kind, kn in [(0, '없음'), (1, 'v5'), (2, 'v5+크기')]:
            for vv, vn in [(v1, '맞추기 잔차(v5식)'), (v2, '점마다 독립')]:
                n, a, _ = leftover(sY, sT, vv, x, y, kind)
                rows.append(dict(대상=label, 점수=n_pts, 방해무늬=kn, 잡음=vn, n_sigma_per_mm=n, Θ흉내=a, 최소Y_mm=3 / n))
    # ---- 특징점 ----
    for hp in (8, 12):
        Z = np.load(HERE/f'tracks_hp{hp}.npz'); F = pd.read_csv(HERE/f'features_hp{hp}.csv'); names = list(Z['names'])
        D = Z['D_gn']; onplate = F.region.str.startswith('QR') | F.region.isin(['판', '판나사'])
        grp = np.where(onplate, '판위', '판밖')
        # 판밖 점의 깊이 비 (Y 기울기 / Θ 기울기, 가로)
        sY, sT, _, _ = fit_axis(D, names, grp); ratio = (sY.reshape(-1, 2)[:, 0] / sT.reshape(-1, 2)[:, 0])
        F['ratio'] = ratio; F.to_csv(HERE/f'features_hp{hp}_ratio.csv', index=False, encoding='utf-8-sig')
        far = (~onplate) & (np.abs(ratio - np.median(ratio[onplate])) > 0.01)
        sets = {'QR만': F.region.str.startswith('QR').values, '판 위 전체': onplate.values, '판밖만': (~onplate).values,
                'QR + 판밖': np.ones(len(F), bool), 'QR + 깊이 다른 판밖 점': (F.region.str.startswith('QR') | far).values}
        for nm, k in sets.items():
            run(f'특징점 {2 * hp + 1}px / {nm}', D[:, k], names, F.x.values[k].astype(float), F.y.values[k].astype(float), grp[k], int(k.sum()))
        print(f'hp{hp}: 판위 깊이비 중앙값 {np.median(ratio[onplate]):.4f} (범위 {np.quantile(ratio[onplate], 0.05):.4f}~{np.quantile(ratio[onplate], 0.95):.4f}), '
              f'판밖 {np.quantile(ratio[~onplate], [0.05, 0.5, 0.95]).round(4)}, 깊이 다른 판밖 점 {far.sum()}개', flush=True)
    # ---- v5 패치 (같은 공식, 같은 라벨) ----
    D5 = V5.F[1, 0]                                           # PC+GN, 폴드0 기준 영상
    P = V5.P; names5 = V5.NAMES
    for nm, k in [('QR만', P.region.str.startswith('QR').values), ('QR+나사', np.ones(len(P), bool))]:
        nov = k & (P.px.values % 32 == 0) & (P.py.values % 32 == 0)
        for lab_, kk in [(f'v5 패치 32px 겹침 / {nm}', k), (f'v5 패치 32px 안 겹침 / {nm}', nov)]:
            run(lab_, D5[:, kk], names5, P.cx.values[kk], P.cy.values[kk], np.where(P.region.values[kk] == 'x', 'a', '판위'), int(kk.sum()))
    T = pd.DataFrame(rows); T.round(4).to_csv(HERE/'t6_F1.csv', index=False, encoding='utf-8-sig')
    print(T.pivot_table(index=['대상', '점수'], columns=['잡음', '방해무늬'], values='최소Y_mm', sort=False).round(3).to_string())
