import os, time, cv2, numpy as np
from PIL import Image
SC = os.path.dirname(os.path.abspath(__file__))
ROOT = r"C:\Users\SSAFY\Downloads\[기업연계_세메스] 교보재\[기업연계_세메스] 교보재\extracted\SSAFY Test Image Set"
z = np.load(os.path.join(SC, "tracks.npz"), allow_pickle=True); X, files = z["X"], z["files"]
L = lambda p: np.array(Image.open(p).convert("L"))
C = L(os.path.join(ROOT, "center.bmp"))
ok, P = cv2.QRCodeDetectorAruco().detectMulti(C); P = sorted(P, key=lambda p: p[:, 0].mean())
X0, Y0, X1, Y1 = 1450, 940, 2370, 1200
sub = [i for i in range(len(X)) if X[i, 1] == 0][::3]          # Θ 사진 67장
imgs = [L(files[i][3]) for i in sub]
def run(blur, shrink, gf):
    t = C[Y0:Y1, X0:X1].astype(np.float32)
    if blur: t = cv2.GaussianBlur(t, (0, 0), blur)
    m = np.zeros(t.shape, np.uint8)
    for p in P: cv2.fillConvexPoly(m, (p - [X0, Y0] + (np.sign(p - p.mean(0)) * shrink)).astype(np.int32), 255)
    out = []
    for g0 in imgs:
        g = g0[Y0:Y1, X0:X1].astype(np.float32)
        if blur: g = cv2.GaussianBlur(g, (0, 0), blur)
        r = cv2.matchTemplate(g, t[40:220, 150:770], cv2.TM_CCOEFF_NORMED); _, _, _, (bx, by) = cv2.minMaxLoc(r)
        w = np.array([[1, 0, bx - 150], [0, 1, by - 40], [0, 0, 1]], np.float32)
        _, w = cv2.findTransformECC(t, g, w, cv2.MOTION_HOMOGRAPHY, (3, 500, 1e-9), m, gf)
        out.append(w[2, 0])
    v = np.array(out); th = X[sub, 0]; p = np.polyfit(th, v, 1); r = v - np.polyval(p, th)
    return abs(r.std() / p[0])
for blur, shrink, gf in [(0, 0, 5), (0, 0, 1), (1.5, 0, 1), (3, 0, 1), (0, 12, 5), (0, 25, 5), (3, 25, 1)]:
    t0 = time.time(); e = run(blur, shrink, gf)
    print(f"blur {blur:3}  마스크확장 {shrink:2}px  gauss {gf}  →  h31로 본 Θ 잡음 {e:.4f}   ({time.time()-t0:.0f}s)")
