"""물리 모델용 특징점: center 기준 영상에서 QR 안 코너(Shi-Tomasi)와 나사 안 코너를 찾고,
사진마다 기준 영상(폴드별 평균) 대비 이동을 가우스-뉴턴(21x21 창, 밝기 a, b 포함)으로 잰다.
출력: features.csv (점 위치, 영역), feat_field.npz (기준 2 x 사진 x 점 x 2)"""
import time
import cv2, numpy as np, pandas as pd
from scipy import ndimage
from step1_2 import HERE, X0, Y0, HH, WW, SRC, read_raw, corner_shift, names, region_of, QRBOX, SCREWS, R_SCREW

W = 10                                   # 창 반지름 (21x21)
R = np.load(HERE/'refs.npz'); REFS = [R['mean_test0'], R['mean_test1']]; CEN = R['center']
CX0, CY0, CX1, CY1 = X0 - 140, Y0 - 50, X0 + WW + 140, Y0 + HH + 50

def detect():
    mask = np.zeros((HH, WW), np.uint8)
    for x, y, hw, hh in QRBOX: mask[int(y - hh - Y0):int(y + hh - Y0), int(x - hw - X0):int(x + hw - X0)] = 255
    for x, y in SCREWS: cv2.circle(mask, (int(x - X0), int(y - Y0)), R_SCREW, 255, -1)
    img = CEN.astype(np.float32)
    pts = cv2.goodFeaturesToTrack(img, maxCorners=600, qualityLevel=0.03, minDistance=9, mask=mask, blockSize=7)
    pts = cv2.cornerSubPix(img, pts, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.001))[:, 0]
    pts = pts[(pts[:, 0] > W + 2) & (pts[:, 0] < WW - W - 3) & (pts[:, 1] > W + 2) & (pts[:, 1] < HH - W - 3)]
    df = pd.DataFrame(dict(cx=pts[:, 0] + X0, cy=pts[:, 1] + Y0))
    df['region'] = [region_of(x, y) for x, y in zip(df.cx, df.cy)]
    return df[df.region != '그 외'].reset_index(drop=True)

FT = detect(); NF = len(FT)
jj, ii = np.meshgrid(np.arange(-W, W + 1), np.arange(-W, W + 1))
PXc = FT.cx.values[:, None] + jj.ravel()[None]; PYc = FT.cy.values[:, None] + ii.ravel()[None]
_sp = lambda a: ndimage.spline_filter(a.astype(np.float64), order=3)
REFC = [_sp(r) for r in REFS]
def _samp(coef, X, Y, ox, oy):
    return ndimage.map_coordinates(coef, [(Y - oy).ravel(), (X - ox).ravel()], order=3, prefilter=False, mode='nearest').reshape(X.shape)
TPL = [_samp(c, PXc, PYc, X0, Y0) for c in REFC]           # 기준 창 (소수 위치 코너 중심)
CTR = np.c_[FT.cx, FT.cy].astype(np.float32)

def measure(im, cs, k, iters=15):
    b = cv2.GaussianBlur(im, (0, 0), 1.0)[CY0:CY1, CX0:CX1].astype(np.float64)
    gx = (np.roll(b, -1, 1) - np.roll(b, 1, 1)) / 2; gy = (np.roll(b, -1, 0) - np.roll(b, 1, 0)) / 2
    ci, cgx, cgy = _sp(b), _sp(gx), _sp(gy)
    H = cv2.getPerspectiveTransform(SRC, SRC + cs); d = cv2.perspectiveTransform(CTR[None], H)[0].astype(np.float64) - CTR
    T = TPL[k]; a = np.ones(NF); bb = np.zeros(NF)
    for _ in range(iters):
        X = PXc + d[:, :1]; Y = PYc + d[:, 1:]
        I, Ix, Iy = (_samp(c, X, Y, CX0, CY0) for c in (ci, cgx, cgy))
        r = a[:, None] * I + bb[:, None] - T
        J = np.stack([a[:, None] * Ix, a[:, None] * Iy, I, np.ones_like(I)], -1)
        A = np.einsum('npi,npj->nij', J, J) + 1e-6 * np.eye(4); g = np.einsum('npi,np->ni', J, r)
        dl = -np.linalg.solve(A, g[..., None])[..., 0]
        d += np.clip(dl[:, :2], -0.5, 0.5); a += dl[:, 2]; bb += dl[:, 3]
        if np.abs(dl[:, :2]).max() < 1e-4: break
    return d

def job(n):
    cs = np.zeros((4, 2), np.float32) if n == 'center.bmp' else corner_shift(n); im = read_raw(n)
    return np.stack([measure(im, cs, k) for k in (0, 1)])

if __name__ == '__main__':
    from multiprocessing import Pool
    FT.to_csv(HERE/'features.csv', index=False, encoding='utf-8-sig'); print(FT.region.value_counts().to_dict(), flush=True)
    allnames = ['center.bmp'] + names; t0 = time.time(); res = []
    with Pool(8) as pool:
        for i, r in enumerate(pool.imap(job, allnames)):
            res.append(r)
            if i % 40 == 0: print(i, f'{time.time() - t0:.0f}s', flush=True)
    np.savez_compressed(HERE/'feat_field.npz', F=np.stack(res, 1).astype(np.float32), names=np.array(allnames))
    print('done', f'{time.time() - t0:.0f}s')
