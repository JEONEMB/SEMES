"""QR 판만 분석 (판밖 점은 판정에 쓰지 않음).
(1) 점마다 이동이 관찰되나 (추적 성공, 직선성) / 점마다 Y-Θ 구분 단서가 얼마나 있나 (점 하나로 구분되는 최소 |Y|)
(2) 단서가 몇 개 점에 몰려 있나 (상위 10%가 차지하는 비율)
(3) 점 전부 vs 단서 상위 k개만 (학습 폴드로만 고름) → 실제 판정, 평가 기준 2개
출력: t12_points_hp{hp}.csv, t12_topk.csv"""
import numpy as np, pandas as pd
from common import *
from t6_F1 import fit_axis, leftover, nuis, V5, LAB
from t7_F4 import fit_map, solve, FOLD
from t8_eval import evaluate

def per_point(D, names, F):
    grp = np.full(len(F), '판위'); sY, sT, v1, v2 = fit_axis(D, names, grp)
    x, y = F.x.values.astype(float), F.y.values.astype(float)
    n, a, r = leftover(sY, sT, v1, x, y, 1); w = 1 / np.maximum(v1, 0.003**2)
    c = (w * r**2).reshape(-1, 2).sum(1)                         # 점 하나의 단서 (σ², Y 1mm당)
    return n, c, sY.reshape(-1, 2), sT.reshape(-1, 2), np.sqrt(v1.reshape(-1, 2))

def classify(D, names, F, keep_fn, quad=True, label='보정본'):
    ax = np.array([AXIS[n] for n in names]); cmd = np.array([VAL[n] for n in names]); recs = []
    for k in (0, 1):
        te = np.array([FOLD.get(n, -1) == k for n in names]); tr = ~te & (ax != 'C')
        lab = cmd if label == 'raw' else np.array([V5.CORR[k][V5.NAMES.index(n)] for n in names])
        act = np.array([V5.CORR[k][V5.NAMES.index(n)] for n in names])
        keep = keep_fn(D[tr | (ax == 'C')], [n for n, t in zip(names, tr | (ax == 'C')) if t], F)   # 학습 사진으로만 점 고르기
        Dk = D[:, keep].reshape(len(names), -1); U = nuis(F.x.values[keep].astype(float), F.y.values[keep].astype(float), 1)
        maps = {a: fit_map(Dk[tr & (ax == a) | (ax == 'C')], lab[tr & (ax == a) | (ax == 'C')], quad) for a in ['Y', 'Θ']}
        w = 1 / np.maximum((maps['Y'][1] + maps['Θ'][1]) / 2, 0.003**2)
        for i in np.where(te)[0]:
            out = {a: solve(Dk[i], maps[a][0], w, U) for a in ['Y', 'Θ']}; pa = min(out, key=lambda a: out[a][1])
            recs.append(dict(file=names[i], axis=ax[i], cmd=cmd[i], act=act[i], pred_axis=pa, pred=out[pa][0], cost_ratio=out['Y'][1] / out['Θ'][1]))
    return pd.DataFrame(recs)

if __name__ == '__main__':
    rows, topk = [], []
    for hp in (8, 4):
        F = pd.read_csv(HERE/f'features_qr_hp{hp}.csv'); Z = np.load(HERE/f'tracks_qr_hp{hp}.npz'); names = list(Z['names']); D = Z['D_gn']; sc = Z['score']
        for setn, msk in [('QR만', F.region.str.startswith('QR').values), ('QR+나사', np.ones(len(F), bool))]:
            Fs = F[msk].reset_index(drop=True); Ds = D[:, msk]; S = sc[:, msk]
            n, c, sY, sT, sd = per_point(Ds, names, Fs)
            # 이동 관찰: 모든 사진에서 정합 점수 0.8 이상, Y 1mm당 이동이 잡음의 100배 이상
            track_ok = S.min(0) >= 0.8; moved = np.abs(sY[:, 0]) / sd[:, 0] > 100
            ymin_pt = 3 / np.sqrt(np.maximum(c, 1e-12))
            cs = np.sort(c)[::-1]; top10 = cs[:max(1, len(c) // 10)].sum() / cs.sum() * 100
            rows.append(dict(템플릿=2 * hp + 1, 세트=setn, 점수=len(Fs), 추적성공=int(track_ok.sum()), 이동관찰=int((track_ok & moved).sum()),
                             점하나로_Y2mm안에서_구분=int((ymin_pt <= 2).sum()), 점하나로_Y1mm=int((ymin_pt <= 1).sum()), 점하나로_Y05mm=int((ymin_pt <= 0.5).sum()),
                             상위10pct_단서비율=top10, 전체_단서_sigma_per_mm=n, 전체_최소Y_mm=3 / n,
                             점_흔들림_중앙_px=np.median(sd[:, 0])))
            Fs.assign(c=c, ymin_pt=ymin_pt, sY_x=sY[:, 0], sT_x=sT[:, 0], sd_x=sd[:, 0], track_ok=track_ok).to_csv(HERE/f't12_points_{setn}_hp{hp}.csv', index=False, encoding='utf-8-sig')
            print(rows[-1], flush=True)
            # (3) 전부 vs 상위 k
            for kk in [10, 25, 50, 100, 200, None]:
                if kk is not None and kk >= len(Fs): continue
                def keep_fn(Dtr, ntr, Ff, kk=kk):
                    if kk is None: return np.ones(len(Ff), bool)
                    _, cc, *_ = per_point(Dtr, ntr, Ff); m = np.zeros(len(Ff), bool); m[np.argsort(-cc)[:kk]] = True; return m
                df = classify(Ds, names, Fs, keep_fn)
                for basis, bn in [('cmd', '명령값'), ('act', '실제위치')]:
                    e = evaluate(df, basis); topk.append(dict(템플릿=2 * hp + 1, 세트=setn, 사용점='전부' if kk is None else f'상위 {kk}', 정답기준=bn, **e))
                print(' ', kk, topk[-1]['축틀림_Y'], topk[-1]['축틀림_Θ'], round(topk[-1]['축정확도'], 2), flush=True)
    pd.DataFrame(rows).round(4).to_csv(HERE/'t12_points_summary.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(topk).round(3).to_csv(HERE/'t12_topk.csv', index=False, encoding='utf-8-sig')
