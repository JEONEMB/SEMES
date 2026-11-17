"""B 방식: center와 현재 사진에서 모서리를 각각 따로 검출(소수점, cv2.cornerSubPix)해서 점-점으로 짝지은 이동.
A 방식(템플릿 + 가우스-뉴턴)과 같은 점(QR 232개, 17px 세트), 같은 사진, 같은 평가로 비교.
짝짓기: 현재 사진에서 'center 점 + A의 정수 이동' 근처에서 출발해 가장 가까운 모서리로 수렴 (QR 반복 무늬 오짝 방지)."""
import numpy as np, pandas as pd
from multiprocessing import Pool
from common import *

F = pd.read_csv(HERE/'features_qr_hp8.csv'); Z = np.load(HERE/'tracks_qr_hp8.npz'); DA = Z['D_gn']
qr = F.region.str.startswith('QR').values
CRIT = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4)
P0 = np.c_[F.x.values, F.y.values].astype(np.float32)

def corners(img, start, win):
    return cv2.cornerSubPix(blur(img), start.reshape(-1, 1, 2).copy(), (win, win), (-1, -1), CRIT).reshape(-1, 2)

CEN = {w: corners(read_raw('center.bmp'), P0, w) for w in (3, 5, 8)}

def job(i):
    img = read_raw(NAMES[i]); st = (P0 + np.round(DA[i])).astype(np.float32)
    return np.stack([corners(img, st, w) - CEN[w] for w in (3, 5, 8)])

if __name__ == '__main__':
    with Pool(16) as p: R = np.stack(p.map(job, range(len(NAMES))), 1)   # (창, 사진, 점, 2)
    np.savez_compressed(HERE/'tracks_qr_p2p.npz', D=R.astype(np.float32), wins=np.array([3, 5, 8]), names=np.array(NAMES))
    from t3_real_accuracy import analyze
    from t12_qr_analysis import classify, per_point
    from t8_eval import evaluate
    ses = np.array([session_of(n) for n in NAMES]); v = np.array([VAL[n] for n in NAMES]); ok = ses != 'C'
    def scatter(D):
        Rr = np.full(D.shape, np.nan)
        for s in ['Y1+', 'Y1-', 'Y2+', 'Y2-', 'Θ+', 'Θ-']:
            m = ses == s; A = np.c_[np.ones(m.sum()), v[m], v[m]**2]
            for c in (0, 1): co, *_ = np.linalg.lstsq(A, D[m, :, c], rcond=None); Rr[m, :, c] = D[m, :, c] - A @ co
        Rr = Rr[ok]; Rr -= np.median(Rr, 1, keepdims=True)
        return np.median(Rr[:, :, 0].std(0)), np.median(Rr[:, :, 1].std(0)), int((np.abs(Rr) > 0.5).any(axis=(0, 2)).sum())
    rows = []
    cand = {'A 템플릿 + 가우스-뉴턴 (지금)': DA[:, qr]}
    for j, w in enumerate((3, 5, 8)): cand[f'B 점-점 (cornerSubPix 창 {2 * w + 1}px)'] = R[j][:, qr]
    Fq = F[qr].reset_index(drop=True)
    for nm, D in cand.items():
        sx, sy, bad = scatter(D)
        n, *_ = per_point(D, NAMES, Fq)
        df = classify(D, NAMES, Fq, lambda D_, n_, F_: np.ones(len(F_), bool), quad=True)
        r = dict(방법=nm, 점_흔들림_가로_px=sx, 점_흔들림_세로_px=sy, 크게튄점=bad, 구분단서_sigma_per_mm=n, 이론최소Y_mm=3 / n)
        for b, bn in [('cmd', '명령값'), ('act', '실제위치')]:
            e = evaluate(df, b); r.update({f'{bn}_축정확도': e['축정확도'], f'{bn}_틀림Y': e['축틀림_Y'], f'{bn}_틀림Θ': e['축틀림_Θ'],
                                          f'{bn}_Y평균오차um': e['Y_평균오차'], f'{bn}_Θ평균오차m': e['Θ_평균오차']})
        rows.append(r); print(nm, {k: round(vv, 4) if isinstance(vv, float) else vv for k, vv in r.items() if k != '방법'}, flush=True)
    pd.DataFrame(rows).round(4).to_csv(HERE/'t14_point_to_point.csv', index=False, encoding='utf-8-sig')
