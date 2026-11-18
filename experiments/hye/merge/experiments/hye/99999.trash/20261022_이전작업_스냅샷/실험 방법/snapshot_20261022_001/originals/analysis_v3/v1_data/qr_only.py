import os, json, itertools, cv2, numpy as np
from PIL import Image
SC = os.path.dirname(os.path.abspath(__file__))
ROOT = r"C:\Users\SSAFY\Downloads\[기업연계_세메스] 교보재\[기업연계_세메스] 교보재\extracted\SSAFY Test Image Set"
z = np.load(os.path.join(SC, "tracks.npz"), allow_pickle=True); X, files = z["X"], z["files"]
C = np.array(Image.open(os.path.join(ROOT, "center.bmp")).convert("L"))
# QR 3개 영역 마스크 (center 이미지 기준)
ok, P = cv2.QRCodeDetectorAruco().detectMulti(C)
mask = np.zeros_like(C)
for p in P: cv2.fillConvexPoly(mask, p.astype(np.int32), 255)
pts = cv2.goodFeaturesToTrack(C, maxCorners=300, qualityLevel=0.01, minDistance=12, blockSize=5, mask=mask).reshape(-1, 2)
print("QR 내부 특징점", len(pts))
HP, SXR, SYR = 10, 110, 6
T = [C[int(round(y))-HP:int(round(y))+HP, int(round(x))-HP:int(round(x))+HP].astype(np.float32) for x, y in pts]
def sub(a, i):
    if 0 < i < len(a)-1:
        d = a[i-1]-2*a[i]+a[i+1]; return i + (0.5*(a[i-1]-a[i+1])/d if d else 0)
    return float(i)
D = np.zeros((len(files), len(pts))); Sc = np.zeros_like(D)
for f, (_, _, _, path) in enumerate(files):
    g = np.array(Image.open(path).convert("L"))
    for k, (x, y) in enumerate(pts):
        xi, yi = int(round(x)), int(round(y))
        s = g[yi-HP-SYR:yi+HP+SYR, xi-HP-SXR:xi+HP+SXR].astype(np.float32)
        r = cv2.matchTemplate(s, T[k], cv2.TM_CCOEFF_NORMED); _, mx, _, (bx, by) = cv2.minMaxLoc(r)
        D[f, k] = sub(r[by, :], bx) - SXR + (xi - x); Sc[f, k] = mx
rms = lambda a: float(np.sqrt(np.mean(a**2)))
good = []
for k in range(len(pts)):
    ab = np.linalg.lstsq(X, D[:, k], rcond=None)[0]; res = D[:, k] - X @ ab
    if abs(res).max() < 2 and Sc[:, k].min() > 0.8: good.append((k, ab[1]/ab[0]))
print("안정적인 QR 특징점", len(good), " b/a 범위", np.round([min(g[1] for g in good), max(g[1] for g in good)], 4))
np.savez(os.path.join(SC, "qr_tracks.npz"), pts=pts, D=D, good=np.array([g[0] for g in good]))
