"""v6 특징점 이동장 공통: 사진 읽기, 특징점 고르기, 두 가지 위치 재기.
  TM : v1 방식. 정규화 상관 템플릿 매칭 + 포물선 보간 (원본 영상).
  GN : 정수 위치(정규화 상관)에서 출발해 가우스-뉴턴으로 (dx, dy, 밝기 a, b) 정밀화. 흐림 σ1 영상, 3차 스플라인 보간.
원본은 읽기만 한다."""
import os
import cv2, numpy as np, pandas as pd
from scipy import ndimage
from pathlib import Path

HERE = Path(__file__).parent; PROJ = HERE.parent
ROOT = Path(r"C:\Users\SSAFY\Downloads\[기업연계_세메스] 교보재\[기업연계_세메스] 교보재\extracted\SSAFY Test Image Set")
q = pd.read_csv(PROJ/'14_데이터_QR마커위치.csv').set_index('file')
NAMES = list(q.index)                                   # center 포함 281장
AXIS = {n: ('C' if n == 'center.bmp' else ('Y' if n[0] == 'Y' else 'Θ')) for n in NAMES}
VAL = {n: (0.0 if n == 'center.bmp' else float(q.loc[n, 'value'])) for n in NAMES}

def path_of(n): return ROOT/'center.bmp' if n == 'center.bmp' else ROOT/('Theta' if n[0] == 'T' else 'Y')/n
def read_raw(n): return cv2.imdecode(np.fromfile(str(path_of(n)), np.uint8), cv2.IMREAD_GRAYSCALE).astype(np.float32)
def blur(im): return cv2.GaussianBlur(im, (0, 0), 1.0)

def session_of(n):
    """촬영 구간 6개: Y 0.1간격 +/−, Y 0.05간격(홀수) +/−, Θ +/−."""
    a, v = AXIS[n], VAL[n]
    if a == 'C': return 'C'
    if a == 'Θ': return 'Θ+' if v > 0 else 'Θ-'
    odd = abs(round(v / 0.05)) % 2 == 1
    return ('Y2' if odd else 'Y1') + ('+' if v > 0 else '-')

# ---------- 영역 ----------
X0, X1, Y0, Y1 = 1420, 2392, 948, 1192                  # QR 판 (v5와 같음)
SCREWS = [(1492, 1063), (2312, 1073)]
EXCL = {'라벨': (1740, 1195, 2060, 1360), '케이블': (1840, 0, 2300, 590)}

def region_of(x, y):
    for nm, (a, b, c, d) in EXCL.items():
        if a < x < c and b < y < d: return nm
    if X0 <= x <= X1 and Y0 <= y <= Y1:
        for i, (sx, sy) in enumerate(SCREWS):
            if np.hypot(x - sx, y - sy) < 60: return '판나사'
        for k, i in zip('LCR', '좌중우'):
            if abs(x - q.loc['center.bmp', f'{k}x']) < 95 and abs(y - q.loc['center.bmp', f'{k}y']) < 100: return 'QR' + i
        return '판'
    return '판밖'

def pick_features(cen_raw, hp, min_dist=None, max_n=3000, qual=0.02):
    """흐린 center에서 모서리 후보 → 포화, 제외 영역, 가장자리, 약한 무늬를 뺀다."""
    b = blur(cen_raw)
    md = min_dist or (2 * hp + 1)
    pts = cv2.goodFeaturesToTrack(b, maxCorners=max_n, qualityLevel=qual, minDistance=md, blockSize=7).reshape(-1, 2)
    gx = cv2.Sobel(b, cv2.CV_64F, 1, 0, ksize=3) / 8; gy = cv2.Sobel(b, cv2.CV_64F, 0, 1, ksize=3) / 8
    rows = []
    for x, y in pts:
        xi, yi = int(round(x)), int(round(y))
        if not (160 < xi < 3840 - 160 and 60 < yi < 2160 - 60): continue   # 큰 이동 + 큰 템플릿 여유
        sl = (slice(yi - hp, yi + hp + 1), slice(xi - hp, xi + hp + 1))
        sat = (cen_raw[sl] >= 250).mean()
        a, c, bb = (gx[sl]**2).sum(), (gy[sl]**2).sum(), (gx[sl] * gy[sl]).sum()
        lmin = (a + c) / 2 - np.sqrt(((a - c) / 2)**2 + bb**2)
        rows.append(dict(x=xi, y=yi, region=region_of(xi, yi), sat=sat, lmin=lmin, gxx=a, gyy=c, gxy=bb))
    F = pd.DataFrame(rows)
    F = F[(F.sat == 0) & ~F.region.isin(list(EXCL))].reset_index(drop=True)
    return F

# ---------- 위치 재기 ----------
SX, SY = 110, 16          # 정수 탐색 범위 (Θ ±81px, Y ±29px, 세로 ±1px 여유)

def _parab(a, i):
    if 0 < i < len(a) - 1:
        d = a[i - 1] - 2 * a[i] + a[i + 1]
        return i + (0.5 * (a[i - 1] - a[i + 1]) / d if d != 0 else 0)
    return float(i)

def track_tm(img, tmpls, F, hp, init=None):
    """v1 방식. img 원본, tmpls: 특징점별 템플릿 (원본에서 자름). 반환 (N, 2) 이동, (N,) 점수."""
    out = np.zeros((len(F), 2)); sc = np.zeros(len(F))
    for k, (x, y) in enumerate(zip(F.x.values, F.y.values)):
        ox, oy = (0, 0) if init is None else (int(round(init[k, 0])), int(round(init[k, 1])))
        sx, sy = (SX, SY) if init is None else (4, 4)
        x0, y0 = x + ox - hp - sx, y + oy - hp - sy
        s = img[y0:y0 + 2 * hp + 1 + 2 * sy, x0:x0 + 2 * hp + 1 + 2 * sx]
        r = cv2.matchTemplate(s, tmpls[k], cv2.TM_CCOEFF_NORMED); _, mx, _, (bx, by) = cv2.minMaxLoc(r)
        out[k] = (_parab(r[by, :], bx) - sx + ox, _parab(r[:, bx], by) - sy + oy); sc[k] = mx
    return out, sc

HPC = 40                  # 대략 위치용 큰 템플릿 반폭 (QR 반복 무늬에 잘못 붙는 것 방지)

def coarse_init(img, big, F):
    """큰 템플릿(81x81)으로 넓게 찾아 정수 이동을 정한다. big = cut(center 원본, F, HPC)."""
    out = np.zeros((len(F), 2)); sc = np.zeros(len(F)); h = HPC
    for k, (x, y) in enumerate(zip(F.x.values, F.y.values)):
        y0, x0 = max(y - h - SY, 0), max(x - h - SX, 0)
        s = img[y0:y + h + SY + 1, x0:x + h + SX + 1]
        r = cv2.matchTemplate(s, big[k], cv2.TM_CCOEFF_NORMED); _, mx, _, (bx, by) = cv2.minMaxLoc(r)
        out[k] = (bx + x0 + h - x, by + y0 + h - y); sc[k] = mx
    return out, sc

def track_gn(bimg, tmpls_b, F, hp, d0, iters=15):
    """bimg: 흐린 사진, tmpls_b: 흐린 기준에서 자른 템플릿 (N, 2hp+1, 2hp+1). d0: 출발 이동 (TM 결과)."""
    n = 2 * hp + 1; jj, ii = np.meshgrid(np.arange(-hp, hp + 1), np.arange(-hp, hp + 1))
    out = np.zeros((len(F), 2)); M = 3
    for k, (x, y) in enumerate(zip(F.x.values, F.y.values)):
        cx, cy = x + int(round(d0[k, 0])), y + int(round(d0[k, 1]))
        L = hp + M + 3
        win = bimg[cy - L:cy + L + 1, cx - L:cx + L + 1].astype(np.float64)
        gxw = (np.roll(win, -1, 1) - np.roll(win, 1, 1)) / 2; gyw = (np.roll(win, -1, 0) - np.roll(win, 1, 0)) / 2
        cI, cX, cY = (ndimage.spline_filter(w, order=3) for w in (win, gxw, gyw))
        T = tmpls_b[k].astype(np.float64).ravel()
        d = np.array([d0[k, 0] - (cx - x), d0[k, 1] - (cy - y)])   # 창 중심 기준 소수 이동
        a, b = 1.0, 0.0
        for _ in range(iters):
            yy = (ii + L + d[1]).ravel(); xx = (jj + L + d[0]).ravel()
            smp = lambda c: ndimage.map_coordinates(c, [yy, xx], order=3, prefilter=False, mode='nearest')
            I, Ix, Iy = smp(cI), smp(cX), smp(cY)
            r = a * I + b - T
            J = np.c_[a * Ix, a * Iy, I, np.ones_like(I)]
            delta = -np.linalg.solve(J.T @ J + 1e-9 * np.eye(4), J.T @ r)
            d += np.clip(delta[:2], -0.5, 0.5); a += delta[2]; b += delta[3]
            if np.abs(delta[:2]).max() < 1e-5: break
        out[k] = d + (cx - x, cy - y)
    return out

def cut(img, F, hp):
    return np.stack([img[y - hp:y + hp + 1, x - hp:x + hp + 1] for x, y in zip(F.x.values, F.y.values)]).astype(np.float32)
