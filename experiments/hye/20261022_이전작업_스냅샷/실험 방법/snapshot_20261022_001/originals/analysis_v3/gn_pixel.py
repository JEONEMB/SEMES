"""G1: 픽셀 직접 가우스-뉴턴. 특징점 없이 QR 판 영역 픽셀 전부로 Θ, Y를 직접 푼다.

모델
- 판은 평평해서 사진 사이의 변형은 원근 변환(호모그래피)이고, 원근 변환은 판 귀퉁이 4개의 이동(8개 값)으로 정해진다.
- 학습 사진: 귀퉁이 이동(v2 호모그래피 추출값)을 [Θ, Y, 1]로 선형 회귀 → B (Θ 1, Y 1mm일 때 귀퉁이가 어떻게 움직이는지 + 0점)
- 방해 요소 3개: 세로 이동, 화면 내 회전, 크기. Θ와 Y를 구분하는 단서(판이 비스듬해지는 원근)와 섞이지 않는 변형이라 따로 풀어서
  모델에 담기지 않은 작은 차이가 Θ, Y로 새어 들어가지 않게 한다. 이게 없으면 크게 틀림 (예: Y+0.10을 Y 0.83으로).
- 기준 영상: 학습 사진들을 center 좌표로 되돌려 평균한 영상 (center 사진만 선명도와 밝기가 달라서).
- 새 사진: 기준 영상 픽셀을 (Θ, Y, 방해 3)가 만드는 변형으로 옮긴 위치에서 새 사진을 읽어 밝기 차이(밝기 a, b 포함) 제곱합을
  최소화하도록 가우스-뉴턴 반복. 오차가 줄 때만 이동하는 감쇠로 발산을 막는다.

계산 속도
- 픽셀 좌표는 귀퉁이 이동 d에 대해 map = P0 + Φ·d 로 계산한다. 평행 이동은 정확히 비례하고, 원근 성분은 1px 미만이라
  선형 근사 오차가 매우 작다 (check_linear()로 확인). Φ는 한 번만 계산한다.
"""
import cv2, numpy as np, pandas as pd, sys
from pathlib import Path
from common import names, X, folds
HERE = Path(__file__).parent; PROJ = HERE.parent
ROOT = Path(r"C:\Users\SSAFY\Downloads\[기업연계_세메스] 교보재\[기업연계_세메스] 교보재\extracted\SSAFY Test Image Set")
q = pd.read_csv(PROJ/'14_데이터_QR마커위치.csv').set_index('file')
hom = pd.read_csv(PROJ/'analysis_v2/homography_features.csv').set_index('file')
c0 = q.loc['center.bmp']
X0, X1 = int(c0.Lx - 110), int(c0.Rx + 110); Y0, Y1 = int(min(c0.Ly, c0.Ry) - 120), int(max(c0.Ly, c0.Ry) + 120)
HH, WW = Y1 - Y0, X1 - X0
from scipy import ndimage
def warp(coef, mx, my, order=3):
    """정확한 소수 위치 보간. OpenCV remap은 위치를 1/32px 단위로 잘라서(약 0.03px) 우리가 찾는 신호(0.02~0.06px)와
    크기가 같아 비용 곡면이 계단처럼 깨진다. scipy 3차 스플라인(미리 계수로 변환한 영상)으로 정확히 보간한다."""
    return ndimage.map_coordinates(coef, [my.ravel(), mx.ravel()], order=order, prefilter=False, mode='nearest').reshape(mx.shape).astype(np.float32)
MX, MY = 130, 40; BX0, BY0 = X0 - MX, Y0 - MY                               # 큰 자르기 영역의 시작점
CORN = ['좌상', '우상', '우하', '좌하']
SRC = np.float32([[X0, Y0], [X1, Y0], [X1, Y1], [X0, Y1]])
CD = np.array([[hom.loc[n, f'{c}_{a}'] for c in CORN for a in ('dx', 'dy')] for n in names])   # 사진별 귀퉁이 이동 (8)
_c = SRC - SRC.mean(0)
NUIS = np.stack([np.tile([0, 1], 4), np.c_[-_c[:, 1], _c[:, 0]].ravel() / 1000, _c.ravel() / 1000], 1)  # 8 x 3: 세로 이동, 회전, 크기
gy_, gx_ = np.mgrid[0:HH, 0:WW].astype(np.float64)
P0 = np.stack([gx_ + X0, gy_ + Y0], -1).reshape(-1, 2)                       # 판 영역 픽셀의 전체 사진 좌표

def exact_map(d):
    H = cv2.getPerspectiveTransform(SRC, (SRC + np.asarray(d).reshape(4, 2)).astype(np.float32))
    return cv2.perspectiveTransform(P0[None], H)[0]
PHI = np.stack([(exact_map(np.eye(8)[k] * 0.5) - exact_map(-np.eye(8)[k] * 0.5)) for k in range(8)], -1)  # Npix x 2 x 8

def lin_map(d):
    m = P0 + PHI @ d
    return (m[:, 0] - BX0).reshape(HH, WW).astype(np.float32), (m[:, 1] - BY0).reshape(HH, WW).astype(np.float32)

def check_linear():
    worst = 0
    for n in ['T-1.00.bmp', 'T+1.00.bmp', 'Y-2.00.bmp', 'Y+2.00.bmp', 'T+0.50.bmp']:
        d = CD[names.index(n)]; worst = max(worst, np.abs(exact_map(d) - (P0 + PHI @ d)).max())
    return worst

def read(n):
    p = ROOT/'center.bmp' if n == 'center.bmp' else ROOT/('Theta' if n[0] == 'T' else 'Y')/n
    im = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    big = cv2.GaussianBlur(im[BY0:Y1 + MY, BX0:X1 + MX], (0, 0), 1.0)
    sp = lambda a: ndimage.spline_filter(a.astype(np.float64), order=3)
    return sp(big), sp(cv2.Sobel(big, cv2.CV_32F, 1, 0, ksize=3) / 8), sp(cv2.Sobel(big, cv2.CV_32F, 0, 1, ksize=3) / 8)

def mean_template(train_names):
    acc = np.zeros((HH, WW), np.float64)
    for nm in train_names:
        mx, my = lin_map(CD[names.index(nm)]); acc += warp(read(nm)[0], mx, my)
    return (acc / len(train_names)).astype(np.float32)

def solve(img, B, p, Tm, iters=60):
    """p = [Θ, Y, 세로, 회전, 크기]. 반환: p, 평균 제곱 잔차, 반복 수"""
    big, gx, gy = img; k_ = len(p)
    V = [B[0], B[1]] + [NUIS[:, j] for j in range(k_ - 2)]           # 각 미지수가 귀퉁이 이동을 바꾸는 방향 (8)
    DX = [(PHI[:, 0, :] @ v).reshape(HH, WW).astype(np.float32) for v in V]    # 픽셀 x 좌표 변화율
    DY = [(PHI[:, 1, :] @ v).reshape(HH, WW).astype(np.float32) for v in V]
    dvec = lambda p: np.r_[p[:2], 1.0] @ B + (NUIS[:, :k_ - 2] @ p[2:] if k_ > 2 else 0)
    def resid(p, ab):
        mx, my = lin_map(dvec(p)); Iw = warp(big, mx, my)
        return Iw - (ab[0] * Tm + ab[1]), mx, my
    ab = np.array([1.0, 0.0]); r, mx, my = resid(p, ab); c = float((r**2).mean()); lam = 1e-4; it = 0
    for it in range(iters):
        gxw = warp(gx, mx, my); gyw = warp(gy, mx, my)
        A = np.stack([(gxw * DX[j] + gyw * DY[j]).ravel() for j in range(k_)] + [-Tm.ravel(), -np.ones(Tm.size, np.float32)], 1).astype(np.float64)
        Hm = A.T @ A; g = -A.T @ r.ravel().astype(np.float64)
        while True:
            st = np.linalg.solve(Hm + lam * np.diag(np.diag(Hm)), g)
            pn, abn = p + st[:k_], ab + st[k_:]
            rn, mxn, myn = resid(pn, abn); cn = float((rn**2).mean())
            if cn < c: p, ab, r, mx, my, c = pn, abn, rn, mxn, myn, cn; lam = max(lam / 10, 1e-9); break
            lam *= 10
            if lam > 1e6: return p, c, it
        if np.abs(st[:k_]).max() < 1e-7: break
    return p, c, it

if __name__ == '__main__':
    import time
    print('선형 근사 최대 좌표 오차 (px): %.5f' % check_linear())
    tr = folds[0][0]; B = np.linalg.lstsq(np.c_[X[tr], np.ones(tr.sum())], CD[tr], rcond=None)[0]
    t0 = time.time(); Tm = mean_template([n for n, t in zip(names, tr) if t][::4]); print('기준 영상 %.1f초' % (time.time() - t0))
    for n in sys.argv[1:] or ['Y+0.10.bmp', 'Y+0.50.bmp', 'T+0.09.bmp', 'Y-1.00.bmp', 'T-0.40.bmp']:
        t0 = time.time(); img = read(n); i = names.index(n); m = q.loc[n, ['dLx', 'dCx', 'dRx']].mean()
        p0 = np.zeros(5); p0[0] = m / 80; p, c, it = solve(img, B, p0, Tm)
        print(f'{n:11s} 정답 Θ{X[i,0]:+.2f} Y{X[i,1]:+.2f} → Θ{p[0]:+.4f} Y{p[1]:+.4f}  (반복 {it}, {time.time()-t0:.1f}초)')

def solve_multi(img, B, m, Tm):
    """출발점 두 곳(가로 이동을 전부 Θ로 / 전부 Y로 설명)에서 풀고 비용이 낮은 쪽을 고른다."""
    best = None
    for p0 in (np.r_[m / 80.0, 0, 0, 0, 0], np.r_[0, m / 14.25, 0, 0, 0]):
        p, c, it = solve(img, B, p0.astype(float), Tm)
        if best is None or c < best[1]: best = (p, c, it)
    return best
