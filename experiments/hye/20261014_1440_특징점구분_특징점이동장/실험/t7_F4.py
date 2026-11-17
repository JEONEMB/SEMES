"""F4, B4, B5: 특징점 이동장으로 실제 판정 (새 사진 1장 → Y인지 Θ인지 + 값). v5 step4_8과 같은 맞추기, 같은 채점.
교차검증 두 가지: (1) 촬영 순서 짝/홀 (v5와 같은 폴드), (2) 촬영 구간 하나 빼기 (6구간, 1차 매핑만: 한쪽 부호 구간을 빼면 2차는 외삽이 됨).
추가 지표: 구분 시작점 = 이 |값| 이상이면 축 판정이 전부 맞음 (Y는 mm, Θ는 같은 QR 이동의 Y mm로 환산)."""
import sys, io, contextlib
import numpy as np, pandas as pd
from common import *
from t6_F1 import V5, LAB, nuis

FOLD = {n: V5.FOLD[n] for n in V5.FOLD}; THETA_TO_Y = 80.66 / 14.39       # Θ 1 ≈ Y 5.6mm (QR 평균 이동 기준)

def fit_map(D, v, quad):
    A = np.c_[v, v**2, np.ones_like(v)] if quad else np.c_[v, np.ones_like(v)]
    co, *_ = np.linalg.lstsq(A, D, rcond=None); res = D - A @ co
    var = res.var(0) * len(v) / max(len(v) - A.shape[1], 1)
    if not quad: co = np.stack([co[0], np.zeros_like(co[0]), co[1]])
    return co, var

def solve(d, co, w, U, huber=3.0, iters=6):
    s, c, b = co; base = w; ww = base.copy(); v = 0.0
    for _ in range(iters):
        J = s + 2 * c * v; z = d - (s * v + c * v * v + b); A = np.c_[J, U]; sw = np.sqrt(ww)
        p, *_ = np.linalg.lstsq(A * sw[:, None], z * sw, rcond=None); v += p[0]
        r = (d - (s * v + c * v * v + b) - U @ p[1:]) * np.sqrt(base)
        ww = base * np.where(np.abs(r) <= huber, 1, huber / np.maximum(np.abs(r), 1e-12))
    z = d - (s * v + c * v * v + b); sw = np.sqrt(ww)
    u, *_ = np.linalg.lstsq(U * sw[:, None], z * sw, rcond=None)
    rr = (z - U @ u) * np.sqrt(base); return v, np.sum(np.where(np.abs(rr) <= huber, rr**2, 2 * huber * np.abs(rr) - huber**2))

def run(D, names, x, y, label, quad, cv, kind=1):
    ax = np.array([AXIS[n] for n in names]); cmd = np.array([VAL[n] for n in names])
    U = nuis(x, y, kind); Dm = D.reshape(len(names), -1); recs = []
    if cv == '짝홀':
        splits = [(k, np.array([FOLD.get(n, -1) == k for n in names])) for k in (0, 1)]
    else:
        ses = np.array([session_of(n) for n in names]); splits = [(s, ses == s) for s in ['Y1+', 'Y1-', 'Y2+', 'Y2-', 'Θ+', 'Θ-']]
    for key, te in splits:
        k = key if cv == '짝홀' else 0
        lab = cmd if label == 'raw' else np.array([V5.CORR[k][V5.NAMES.index(n)] for n in names])
        act = np.array([V5.CORR[k][V5.NAMES.index(n)] for n in names])
        tr = ~te
        maps = {a: fit_map(Dm[tr & ((ax == a) | (ax == 'C'))], lab[tr & ((ax == a) | (ax == 'C'))], quad) for a in ['Y', 'Θ']}
        w = 1 / np.maximum((maps['Y'][1] + maps['Θ'][1]) / 2, 0.003**2)
        for i in np.where(te)[0]:
            out = {a: solve(Dm[i], maps[a][0], w, U) for a in ['Y', 'Θ']}
            pa = min(out, key=lambda a: out[a][1])
            recs.append(dict(file=names[i], axis=ax[i], cmd=cmd[i], act=act[i], pred_axis=pa, pred=out[pa][0],
                             cost_ratio=out['Y'][1] / out['Θ'][1]))
    return pd.DataFrame(recs)

def score(df, basis):
    t = df[basis].values; ok = (df.pred_axis == df.axis).values
    err = np.where(ok, df.pred.values - t, np.nan)
    passed = np.where(ok, np.abs(df.pred.values - t) <= 0.09, (np.abs(t) <= 0.09) & (np.abs(df.pred.values) <= 0.09))
    y, th = (df.axis == 'Y').values, (df.axis == 'Θ').values
    yeq = np.abs(df.cmd.values) * np.where(y, 1, THETA_TO_Y)               # Y mm 환산 크기
    bad = ~ok
    start = yeq[bad].max() if bad.any() else 0.0                             # 이 크기 넘으면 전부 맞음
    return dict(판정오류_Y=int((bad & y).sum()), 판정오류_Θ=int((bad & th).sum()),
                판정오류_Θ_범위안=int((bad & th & (np.abs(df.cmd.values) * THETA_TO_Y <= 2.0)).sum()),
                구분시작_Yeq_mm=start, Y_RMS_um=np.sqrt(np.nanmean(err[y]**2)) * 1000, T_RMS_m=np.sqrt(np.nanmean(err[th]**2)) * 1000,
                pass09=passed.mean() * 100)

if __name__ == '__main__':
    rows, PRED = [], {}
    for hp in (8, 12):
        Z = np.load(HERE/f'tracks_hp{hp}.npz'); F = pd.read_csv(HERE/f'features_hp{hp}_ratio.csv'); names = list(Z['names'])
        onplate = (F.region.str.startswith('QR') | F.region.isin(['판', '판나사'])).values
        far = (~onplate) & (np.abs(F.ratio.values - np.median(F.ratio.values[onplate])) > 0.01)
        sets = {'QR만': F.region.str.startswith('QR').values, '판밖만': ~onplate, 'QR + 판밖': np.ones(len(F), bool)}
        for sn, k in sets.items():
            for label in ['raw', '보정본']:
                for quad in [False, True]:
                    for cv in ['짝홀', '구간빼기']:
                        if cv == '구간빼기' and quad: continue
                        df = run(Z['D_gn'][:, k], names, F.x.values[k].astype(float), F.y.values[k].astype(float), label, quad, cv)
                        key = f'{2 * hp + 1}px / {sn} / {label} / {"2차" if quad else "1차"} / {cv}'; PRED[key] = df
                        r = dict(조건=key, 템플릿=2 * hp + 1, 영역=sn, 점수=int(k.sum()), 라벨=label, 매핑='2차' if quad else '1차', 교차검증=cv)
                        for b, nm in [('cmd', '명령값'), ('act', '실제위치')]:
                            for kk, vv in score(df, b).items():
                                if nm == '명령값' and kk.startswith(('판정', '구분')): continue
                                r[(kk if kk.startswith(('판정', '구분')) else f'{nm}_{kk}')] = vv
                        rows.append(r); print(key, {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in r.items() if kk not in ('조건', '템플릿', '영역', '라벨', '매핑', '교차검증')}, flush=True)
    T = pd.DataFrame(rows); T.round(3).to_csv(HERE/'t7_F4.csv', index=False, encoding='utf-8-sig')
    pd.concat([d.assign(조건=k) for k, d in PRED.items()]).to_csv(HERE/'t7_predictions.csv', index=False, encoding='utf-8-sig')
