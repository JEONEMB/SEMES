"""점마다 흔들림이 사진마다 다른가 (선명도, 밝기와 관계), 이웃 사진끼리 같은 점 흔들림이 이어지는가."""
import numpy as np, pandas as pd
from multiprocessing import Pool
from common import *

def sharp(n):
    im = read_raw(n)[Y0:Y1, X0:X1].astype(np.float64)
    g = np.hypot(np.diff(im, axis=1)[:-1], np.diff(im, axis=0)[:, :-1]); return np.quantile(g, 0.99), im.mean()

if __name__ == '__main__':
    with Pool(16) as p: S = np.array(p.map(sharp, NAMES))
    hp = 12; Z = np.load(HERE/f'tracks_hp{hp}.npz'); F = pd.read_csv(HERE/f'features_hp{hp}.csv')
    ses = np.array([session_of(n) for n in NAMES]); v = np.array([VAL[n] for n in NAMES]); ok = ses != 'C'
    k = F.region.str.startswith('QR').values; D = Z['D_gn'][:, k]; R = np.full(D.shape, np.nan)
    for s in ['Y1+', 'Y1-', 'Y2+', 'Y2-', 'Θ+', 'Θ-']:
        m = ses == s; A = np.c_[np.ones(m.sum()), v[m], v[m]**2]
        for c in (0, 1): co, *_ = np.linalg.lstsq(A, D[m, :, c], rcond=None); R[m, :, c] = D[m, :, c] - A @ co
    L = R - np.median(R, 1, keepdims=True); per = np.sqrt(np.nanmean(L[:, :, 0]**2, 1))
    sh = S[:, 0]
    print('선명도(상위 1%% 기울기) 범위 %.1f~%.1f, center %.1f' % (sh[ok].min(), sh[ok].max(), sh[0]))
    print('사진별 점마다 흔들림 RMS 범위 %.4f~%.4f, 중앙 %.4f' % (per[ok].min(), per[ok].max(), np.median(per[ok])))
    print('선명도와 흔들림 상관 %.3f, 밝기와 %.3f' % (np.corrcoef(sh[ok], per[ok])[0, 1], np.corrcoef(S[ok, 1], per[ok])[0, 1]))
    print('흔들림 상위 8장', [(NAMES[i], round(per[i], 4)) for i in np.argsort(-np.where(ok, per, 0))[:8]])
    for s in ['Θ+', 'Θ-', 'Y1+', 'Y2+']:
        m = np.where(ses == s)[0]; o = m[np.argsort(np.abs(v[m]))]; x = L[o, :, 0]
        print(s, '이웃 사진끼리 같은 점 흔들림 상관 %.3f' % np.mean([np.corrcoef(x[:-1, j], x[1:, j])[0, 1] for j in range(x.shape[1])]))
    pd.DataFrame(dict(file=NAMES, sharp=sh, bright=S[:, 1], scatter=per)).to_csv(HERE/'t5_per_image.csv', index=False, encoding='utf-8-sig')
