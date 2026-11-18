"""3단계: 패치별 서브픽셀 이동 (dx, dy) 재기. 두 가지 측정법.
  PC  : 위상 상관 (skimage, 1/100px 업샘플, 정규화 없음, 48px Hann 창). 보간 없이 정수 위치에서 잘라 잰다.
        합성 검증에서 창/정규화 조합 8개 중 편향이 가장 작은 조합 (0.25px 이동에서 편향 -0.022px).
  GN  : PC 결과에서 시작해 가우스-뉴턴 정밀화. 패치마다 (dx, dy, 밝기 a, b) 4개를 푼다.
        야코비안 = a * 밝기 기울기(Ix, Iy), 보간은 3차 스플라인(정확한 소수 위치).
기준 영상: 폴드 k를 시험할 때의 평균 영상(refs.npz mean_test{k}). 모든 사진을 두 기준 모두에 대해 잰다
(폴드 k 시험 사진 + 폴드 k 학습 사진이 같은 기준을 써야 매핑 함수와 맞추기가 일관됨).
초기 위치: v2 원근 변환으로 패치 중심을 옮긴 위치 (출발점일 뿐, 최종 값은 측정으로 정해진다).
출력: field.npz  (method, ref, 사진, 패치, 2)"""
import sys, time
import cv2, numpy as np, pandas as pd
from scipy import ndimage
from skimage.registration import phase_cross_correlation
from step1_2 import (HERE, PROJ, X0, Y0, HH, WW, PATCH, SRC, read_raw, corner_shift, names)

P = pd.read_csv(HERE/'patches.csv'); P = P[P.use].reset_index(drop=True)
R = np.load(HERE/'refs.npz'); REFS = [R['mean_test0'], R['mean_test1']]
NP = len(P); ii, jj = np.mgrid[0:PATCH, 0:PATCH]
# 패치 픽셀의 center 좌표 (NP x 1024)
PXc = (X0 + P.px.values[:, None] + jj.ravel()[None]).astype(np.float64)
PYc = (Y0 + P.py.values[:, None] + ii.ravel()[None]).astype(np.float64)
CTR = np.c_[P.cx.values, P.cy.values].astype(np.float32)
REFPATCH = [np.stack([r[py:py + PATCH, px:px + PATCH] for px, py in zip(P.px, P.py)]) for r in REFS]  # NP x 32 x 32
PCW, PM = 48, 8; HANN = np.outer(np.hanning(PCW), np.hanning(PCW))
def _win(a): return (a - a.mean()) * HANN
REFPC = [np.stack([_win(r[py - PM:py + PATCH + PM, px - PM:px + PATCH + PM]) for px, py in zip(P.px, P.py)]) for r in REFS]
assert (P.px >= PM).all() and (P.py >= PM).all() and (P.px + PATCH + PM <= WW).all() and (P.py + PATCH + PM <= HH).all()
CX0, CY0 = X0 - 140, Y0 - 50          # 측정용 자르기 원점 (Θ ±81px, Y ±29px 여유)
CX1, CY1 = X0 + WW + 140, Y0 + HH + 50

def prep(im):
    b = cv2.GaussianBlur(im, (0, 0), 1.0)[CY0:CY1, CX0:CX1].astype(np.float64)
    gx = (np.roll(b, -1, 1) - np.roll(b, 1, 1)) / 2; gy = (np.roll(b, -1, 0) - np.roll(b, 1, 0)) / 2
    sp = lambda a: ndimage.spline_filter(a, order=3)
    return b, sp(b), sp(gx), sp(gy)

def coarse(cs):
    H = cv2.getPerspectiveTransform(SRC, SRC + cs)
    return cv2.perspectiveTransform(CTR[None], H)[0].astype(np.float64) - CTR

def measure_pc(b, refp, c):
    """b: 자르기 영상 (원점 CX0, CY0), refp: 창을 씌운 48px 기준 패치"""
    out = np.zeros((NP, 2)); ci = np.round(c).astype(int)
    for p in range(NP):
        x0 = int(P.px[p]) + X0 + ci[p, 0] - PM - CX0; y0 = int(P.py[p]) + Y0 + ci[p, 1] - PM - CY0
        mov = _win(b[y0:y0 + PCW, x0:x0 + PCW])
        sh, _, _ = phase_cross_correlation(refp[p], mov, upsample_factor=100, normalization=None)
        out[p] = ci[p] - sh[::-1]          # sh = (행, 열) 이동 → (dx, dy)
    return out

def measure_gn(coefs, refp, d0, iters=10):
    _, ci, cgx, cgy = coefs
    d = d0.copy(); T = refp.reshape(NP, -1); a = np.ones(NP); bb = np.zeros(NP)
    for _ in range(iters):
        X = PXc + d[:, :1] - CX0; Yy = PYc + d[:, 1:] - CY0
        smp = lambda c: ndimage.map_coordinates(c, [Yy.ravel(), X.ravel()], order=3, prefilter=False, mode='nearest').reshape(NP, -1)
        I, Ix, Iy = smp(ci), smp(cgx), smp(cgy)
        r = a[:, None] * I + bb[:, None] - T
        J = np.stack([a[:, None] * Ix, a[:, None] * Iy, I, np.ones_like(I)], -1)     # NP x 1024 x 4
        A = np.einsum('npi,npj->nij', J, J) + 1e-6 * np.eye(4); g = np.einsum('npi,np->ni', J, r)
        delta = -np.linalg.solve(A, g[..., None])[..., 0]
        d += np.clip(delta[:, :2], -0.5, 0.5); a += delta[:, 2]; bb += delta[:, 3]
        if np.abs(delta[:, :2]).max() < 1e-4: break
    return d

def run_one(im, cs):
    coefs = prep(im); c = coarse(cs); out = np.zeros((2, 2, NP, 2))
    for k in (0, 1):
        pc = measure_pc(coefs[0], REFPC[k], c); out[0, k] = pc
        out[1, k] = measure_gn(coefs, REFPATCH[k], pc)
    return out

def synthetic():
    """합성 검증: 기준 영상(폴드0용)을 알고 있는 양만큼 옮긴 가짜 사진을 만들어 두 측정법의 편향과 흩어짐을 잰다."""
    ref = REFS[0]; rows = []
    big = np.full((2160, 3840), ref.mean())
    rng = np.random.default_rng(0); coef = ndimage.spline_filter(ref.astype(np.float64), order=3)
    gy_, gx_ = np.mgrid[0:HH, 0:WW].astype(np.float64)
    for s in [0.0, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5]:
        for noise in [0.0, 1.0]:
            sh = ndimage.map_coordinates(coef, [gy_ - s * 0.6, gx_ - s], order=3, prefilter=False, mode='nearest')  # 장면이 (+s, +0.6s) 이동
            img = big.copy(); img[Y0:Y0 + HH, X0:X0 + WW] = sh + rng.normal(0, noise, sh.shape); img = img[CY0:CY1, CX0:CX1]
            # 이미 흐림이 들어간 기준을 옮긴 것이라 prep의 흐림을 건너뛰기 위해 직접 준비
            b = img; gxi = (np.roll(b, -1, 1) - np.roll(b, 1, 1)) / 2; gyi = (np.roll(b, -1, 0) - np.roll(b, 1, 0)) / 2
            coefs = (b, ndimage.spline_filter(b, order=3), ndimage.spline_filter(gxi, order=3), ndimage.spline_filter(gyi, order=3))
            c = np.zeros((NP, 2)); pc = measure_pc(b, REFPC[0], c); gn = measure_gn(coefs, REFPATCH[0], pc)
            for nm, m in [('PC', pc), ('GN', gn)]:
                for ax, true in [(0, s), (1, 0.6 * s)]:
                    e = m[:, ax] - true
                    rows.append(dict(이동=s, 잡음=noise, 측정법=nm, 방향='dx' if ax == 0 else 'dy', 편향=e.mean(), 흩어짐=e.std(), 최대=np.abs(e).max()))
    return pd.DataFrame(rows)

def job(n):
    cs = np.zeros((4, 2), np.float32) if n == 'center.bmp' else corner_shift(n)
    return run_one(read_raw(n), cs)

if __name__ == '__main__':
    if 'syn' in sys.argv:
        t = synthetic(); t.round(4).to_csv(HERE/'synthetic.csv', index=False, encoding='utf-8-sig'); print(t.round(4).to_string()); sys.exit()
    from multiprocessing import Pool
    allnames = ['center.bmp'] + names; t0 = time.time()
    with Pool(8) as pool:
        res = []
        for i, r in enumerate(pool.imap(job, allnames)):
            res.append(r)
            if i % 20 == 0: print(i, allnames[i], f'{time.time() - t0:.0f}s', flush=True)
    F = np.stack(res, 2).astype(np.float32)
    np.savez_compressed(HERE/'field.npz', F=F, names=np.array(allnames), methods=np.array(['PC', 'GN']))
    print('done', f'{time.time() - t0:.0f}s')
