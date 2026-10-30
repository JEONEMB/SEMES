"""G1: 픽셀 직접 가우스-뉴턴. 미지수는 Θ, Y (+ 밝기 a, b) 뿐.
- 학습 사진: 판 귀퉁이 4개의 이동량(v2 호모그래피 추출값)을 [Θ, Y, 1]로 선형 회귀 → 'Θ 1, Y 1mm일 때 판이 어떻게 변형되는지'
- 새 사진: center 사진의 QR 판 영역 픽셀 전부를, (Θ, Y)가 만드는 원근 변환으로 옮겨 새 사진과 비교하고
  밝기 차이 제곱합이 최소가 되도록 Θ, Y, a, b를 가우스-뉴턴(+감쇠)으로 반복해서 푼다."""
import cv2, numpy as np, pandas as pd, os, sys
from pathlib import Path
from common import names, X, folds
HERE = Path(__file__).parent; PROJ = HERE.parent
ROOT = Path(r"C:\Users\SSAFY\Downloads\[기업연계_세메스] 교보재\[기업연계_세메스] 교보재\extracted\SSAFY Test Image Set")
q = pd.read_csv(PROJ/'14_데이터_QR마커위치.csv').set_index('file')
hom = pd.read_csv(PROJ/'analysis_v2/homography_features.csv').set_index('file')
c0 = q.loc['center.bmp']
X0, X1 = int(c0.Lx - 110), int(c0.Rx + 110); Y0, Y1 = int(min(c0.Ly, c0.Ry) - 120), int(max(c0.Ly, c0.Ry) + 120)
MX, MY = 130, 40; BX0, BY0 = X0 - MX, Y0 - MY                              # 큰 자르기 영역의 시작점
CORN = ['좌상', '우상', '우하', '좌하']
SRC = np.float32([[X0, Y0], [X1, Y0], [X1, Y1], [X0, Y1]])
_c = SRC - SRC.mean(0)
NUIS = np.stack([np.tile([0, 1], 4), np.c_[-_c[:, 1], _c[:, 0]].ravel() / 1000, _c.ravel() / 1000], 1)  # 8 x 3
CD = np.array([[hom.loc[n, f'{c}_{a}'] for c in CORN for a in ('dx', 'dy')] for n in names])  # 사진별 귀퉁이 이동 (8)
def read(n):
    p = ROOT/'center.bmp' if n == 'center.bmp' else ROOT/('Theta' if n[0] == 'T' else 'Y')/n
    im = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    big = cv2.GaussianBlur(im[BY0:Y1 + MY, BX0:X1 + MX], (0, 0), 1.0)
    return big, cv2.Sobel(big, cv2.CV_32F, 1, 0, ksize=3) / 8, cv2.Sobel(big, cv2.CV_32F, 0, 1, ksize=3) / 8
cen = read('center.bmp')[0]; T = cen[MY:MY + (Y1 - Y0), MX:MX + (X1 - X0)].copy()
gy_, gx_ = np.mgrid[0:Y1 - Y0, 0:X1 - X0].astype(np.float32)
P0 = np.stack([gx_ + X0, gy_ + Y0], -1).reshape(-1, 1, 2)                 # 판 영역 픽셀의 전체 사진 좌표
def coords(B, p):
    d = np.r_[p[:2], 1.0] @ B + (NUIS @ p[2:] if len(p) > 2 else 0)        # 귀퉁이 이동 8 (+ 방해 요소)
    H = cv2.getPerspectiveTransform(SRC, SRC + d.reshape(4, 2).astype(np.float32))
    m = cv2.perspectiveTransform(P0.astype(np.float64), H).reshape(Y1 - Y0, X1 - X0, 2)
    return (m[..., 0] - BX0).astype(np.float32), (m[..., 1] - BY0).astype(np.float32)
def solve(img, B, p, iters=40, Tm=None):
    TT = T if Tm is None else Tm
    big, gx, gy = img
    def resid(p, ab):
        mx, my = coords(B, p); Iw = cv2.remap(big, mx, my, cv2.INTER_LINEAR)
        return Iw - (ab[0] * TT + ab[1]), mx, my
    ab = np.array([1.0, 0.0]); r, mx, my = resid(p, ab); c = float((r**2).mean()); lam = 1e-4
    for it in range(iters):
        k_ = len(p); e = np.full(k_, 1e-3); J = []
        for k in range(k_):
            dp = np.zeros(k_); dp[k] = e[k]
            ax, ay = coords(B, p + dp); bx, by = coords(B, p - dp)
            J.append(cv2.remap(gx, mx, my, cv2.INTER_LINEAR) * (ax - bx) / (2 * e[k]) + cv2.remap(gy, mx, my, cv2.INTER_LINEAR) * (ay - by) / (2 * e[k]))
        A = np.stack([j.ravel() for j in J] + [-TT.ravel(), -np.ones(TT.size, np.float32)], 1).astype(np.float64)
        H = A.T @ A; g = -A.T @ r.ravel()
        while True:
            st = np.linalg.solve(H + lam * np.diag(np.diag(H)), g)
            pn, abn = p + st[:len(p)], ab + st[len(p):]
            rn, mxn, myn = resid(pn, abn); cn = float((rn**2).mean())
            if cn < c: p, ab, r, mx, my, c = pn, abn, rn, mxn, myn, cn; lam = max(lam / 10, 1e-9); break
            lam *= 10
            if lam > 1e6: return p, c, it
        if np.abs(st[:len(p)]).max() < 1e-7: break
    return p, c, it
def mean_template(train_names):
    acc = np.zeros_like(T); n = 0
    for nm in train_names:
        i = names.index(nm); Bx = np.zeros((3, 8)); Bx[2] = CD[i]
        big = read(nm)[0]; mx, my = coords(Bx, np.zeros(2)); acc += cv2.remap(big, mx, my, cv2.INTER_LINEAR); n += 1
    return acc / n
if __name__ == '__main__':
    test = sys.argv[1:] or ['Y+0.50.bmp', 'T+0.09.bmp', 'Y+0.10.bmp', 'T-1.00.bmp']
    tr = folds[0][0]
    B = np.linalg.lstsq(np.c_[X[tr], np.ones(tr.sum())], CD[tr], rcond=None)[0]
    for n in test:
        img = read(n); i = names.index(n); m = q.loc[n, ['dLx', 'dCx', 'dRx']].mean()
        for init in [(m / 80.0, 0.0), (0.0, m / 14.25)]:
            p, c, it = solve(img, B, np.array(init))
            print(f'{n:12s} 정답 Θ {X[i,0]:+.3f} Y {X[i,1]:+.3f} | 시작 {init[0]:+.3f},{init[1]:+.3f} → Θ {p[0]:+.4f} Y {p[1]:+.4f}  (반복 {it}, 잔차 {c:.2f})')
