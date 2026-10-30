import os, csv, cv2, numpy as np, time
from PIL import Image
ROOT = r"C:\Users\SSAFY\Downloads\[기업연계_세메스] 교보재\[기업연계_세메스] 교보재\extracted\SSAFY Test Image Set"
PROJ = r"C:\Users\SSAFY\Desktop\hwlee\projects\2026-10_SEMES"
SC = os.path.dirname(os.path.abspath(__file__))
L = lambda p: np.array(Image.open(p).convert("L"))
C = L(os.path.join(ROOT, "center.bmp"))
HP, SXR, SYR = 24, 120, 12                      # 템플릿 반폭, 탐색범위 x/y
pts = cv2.goodFeaturesToTrack(C, maxCorners=400, qualityLevel=0.01, minDistance=90, blockSize=9).reshape(-1, 2)
pts = [p for p in pts if HP + SXR + 2 < p[0] < C.shape[1] - HP - SXR - 2 and HP + SYR + 2 < p[1] < C.shape[0] - HP - SYR - 2]
# 파인더 패턴(기존 특징점)은 0번으로 고정
pts = [np.array([1846.7, 1010.3])] + pts
pts = np.array(pts); N = len(pts); print("후보 특징점", N)
T = [C[int(round(y)) - HP:int(round(y)) + HP, int(round(x)) - HP:int(round(x)) + HP].astype(np.float32) for x, y in pts]
def sub(a, i):
    if 0 < i < len(a) - 1:
        d = a[i-1] - 2*a[i] + a[i+1]
        return i + (0.5 * (a[i-1] - a[i+1]) / d if d != 0 else 0)
    return float(i)
def track(img):
    out = np.zeros((N, 3), np.float32)
    for k, (x, y) in enumerate(pts):
        xi, yi = int(round(x)), int(round(y))
        s = img[yi - HP - SYR:yi + HP + SYR, xi - HP - SXR:xi + HP + SXR].astype(np.float32)
        r = cv2.matchTemplate(s, T[k], cv2.TM_CCOEFF_NORMED)
        _, mx, _, (bx, by) = cv2.minMaxLoc(r)
        out[k] = (sub(r[by, :], bx) - SXR + (xi - x), sub(r[:, bx], by) - SYR + (yi - y), mx)
    return out
rows = list(csv.DictReader(open(os.path.join(PROJ, "qr_marker_positions.csv"), encoding="utf-8-sig")))
files, X, M = [], [], []
t0 = time.time()
for r in rows:
    p = os.path.join(ROOT, "center.bmp") if r["file"] == "center.bmp" else os.path.join(ROOT, "Theta" if r["axis"] == "Θ" else "Y", r["file"])
    v = float(r["value"]); files.append((r["axis"], v, r["file"], p))
    X.append([v, 0] if r["axis"] == "Θ" else [0, v]); M.append(track(L(p)))
print("tracking %.0fs" % (time.time() - t0))
np.savez(os.path.join(SC, "tracks.npz"), pts=pts, X=np.array(X), M=np.array(M), files=np.array(files, dtype=object))
