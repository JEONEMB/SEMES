"""실제 사진의 점마다 흔들림(0.015~0.02px)이 어디서 오는지 진단.
(a) 매끄러운 무늬인가: 사진마다 QR 점 흔들림에 평행 이동 / 아핀 / 원근(8) 무늬를 맞추고 남는 양. 많이 줄면 = 실제 자세 흔들림.
(b) 소수 위상 편향인가: 점마다 흔들림을 sin, cos(2π × 이동의 소수 부분)으로 설명되는 비율. 크면 = 앨리어싱 / 픽셀 고착.
(c) 이웃 점끼리 흔들림 상관."""
import numpy as np, pandas as pd
from common import *
from t3_real_accuracy import analyze

def resid(D, ses, v):
    R = np.full(D.shape, np.nan)
    for s in ['Y1+', 'Y1-', 'Y2+', 'Y2-', 'Θ+', 'Θ-']:
        m = ses == s; A = np.c_[np.ones(m.sum()), v[m], v[m]**2]
        for c in (0, 1):
            coef, *_ = np.linalg.lstsq(A, D[m, :, c], rcond=None); R[m, :, c] = D[m, :, c] - A @ coef
    return R

for hp in (8, 12):
    Z = np.load(HERE/f'tracks_hp{hp}.npz'); F = pd.read_csv(HERE/f'features_hp{hp}.csv')
    ses = np.array([session_of(n) for n in NAMES]); v = np.array([VAL[n] for n in NAMES]); ok = ses != 'C'
    for g in ['QR', '판밖']:
        k = (F.region.str.startswith('QR') if g == 'QR' else F.region == '판밖').values
        x = (F.x.values[k] - F.x.values[k].mean()) / 500; y = (F.y.values[k] - F.y.values[k].mean()) / 500
        D = Z['D_gn'][:, k]; R = resid(D, ses, v)[ok]                     # (사진, 점, 2)
        # (a) 공간 모델: [dx; dy] 를 쌓아 사진마다 최소제곱
        n = k.sum(); one, zer = np.ones(n), np.zeros(n)
        B = {'평행이동': [np.r_[one, zer], np.r_[zer, one]]}
        B['아핀'] = B['평행이동'] + [np.r_[x, zer], np.r_[y, zer], np.r_[zer, x], np.r_[zer, y]]
        B['원근'] = B['아핀'] + [np.r_[x * x, x * y], np.r_[x * y, y * y]]
        B['원근+2차'] = B['원근'] + [np.r_[y * y, zer], np.r_[zer, x * x], np.r_[x * x, zer], np.r_[zer, y * y]]
        out = {}
        for nm, cols in B.items():
            A = np.stack(cols, 1); r = np.c_[R[:, :, 0], R[:, :, 1]].T          # (2n, 사진)
            co, *_ = np.linalg.lstsq(A, r, rcond=None); left = (r - A @ co).T
            out[nm] = (np.median(left[:, :n].std(0)), np.median(left[:, n:].std(0)))
        print(f'hp{hp} {g} (점 {n}) 남는 흔들림 중앙값 (dx, dy): ' + ', '.join(f'{nm} ({a:.4f}, {b:.4f})' for nm, (a, b) in out.items()))
        # (b) 소수 위상 (원근 무늬를 뺀 나머지로)
        A = np.stack(B['원근'], 1); r = np.c_[R[:, :, 0], R[:, :, 1]].T; co, *_ = np.linalg.lstsq(A, r, rcond=None); left = (r - A @ co).T
        Lx = left[:, :n]; ph = 2 * np.pi * np.mod(D[ok][:, :, 0], 1.0)
        expl = []
        for j in range(n):
            M = np.c_[np.sin(ph[:, j]), np.cos(ph[:, j]), np.ones(len(ph))]; c, *_ = np.linalg.lstsq(M, Lx[:, j], rcond=None)
            expl.append(1 - (Lx[:, j] - M @ c).var() / Lx[:, j].var())
        amp = []
        for j in range(n):
            M = np.c_[np.sin(ph[:, j]), np.cos(ph[:, j]), np.ones(len(ph))]; c, *_ = np.linalg.lstsq(M, Lx[:, j], rcond=None); amp.append(np.hypot(c[0], c[1]))
        print(f'   소수 위상으로 설명되는 비율 중앙값 {np.median(expl) * 100:.1f}% (우연 기대치 약 {2 / ok.sum() * 100:.1f}%), 진폭 중앙값 {np.median(amp):.4f}px')
        # (c) 이웃 상관
        P = np.c_[F.x.values[k], F.y.values[k]]; dd = np.hypot(*(P[:, None] - P[None]).transpose(2, 0, 1))
        C = np.corrcoef(Lx.T); near = (dd > 0) & (dd < 40); far = dd > 200
        print(f'   원근 무늬 뺀 뒤 흔들림 상관: 이웃(<40px) {np.nanmean(C[near]):.3f}, 먼 점(>200px) {np.nanmean(C[far]):.3f}')
