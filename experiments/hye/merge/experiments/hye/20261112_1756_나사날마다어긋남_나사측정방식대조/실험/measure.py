import sys, pathlib, numpy as np, cv2
S = pathlib.Path(sys.argv[1]); Z = np.load(S / "screw/crops.npz"); ref, crops, off = Z["ref"].astype(np.float32), Z["crops"], Z["off"]
H = 70; yy, xx = np.mgrid[-H:H + 1, -H:H + 1]; rr = np.hypot(xx, yy)
def grad(a): a = cv2.GaussianBlur(a, (0, 0), 1.5); return cv2.magnitude(cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3))
crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-8)
def ecc(t, im, mask):
    W = np.eye(2, 3, dtype=np.float32)
    try: _, W = cv2.findTransformECC(t, im, W, cv2.MOTION_TRANSLATION, crit, mask, 5)
    except cv2.error: return np.array([np.nan, np.nan])
    return W[:, 2].astype(float)       # im(x + d) ~ t(x): 현재 영상 안에서 나사가 d 만큼 이동
ANG = np.deg2rad(np.arange(0, 360, 5)); RAD = np.arange(12, 32, .25, dtype=np.float32)
def circle(a, c0=(H, H)):
    a = cv2.GaussianBlur(a, (0, 0), 1.0); c = np.array(c0, float)
    for _ in range(3):
        mx = (c[0] + np.outer(np.cos(ANG), RAD)).astype(np.float32); my = (c[1] + np.outer(np.sin(ANG), RAD)).astype(np.float32)
        P = cv2.remap(a, mx, my, cv2.INTER_CUBIC); g = np.gradient(P, axis=1)           # 바깥으로 갈수록 밝아지는 경계
        k = np.clip(g.argmax(1), 1, len(RAD) - 2); i = np.arange(len(ANG)); den = g[i, k - 1] - 2 * g[i, k] + g[i, k + 1]
        r = RAD[k] + .25 * np.where(abs(den) > 1e-9, .5 * (g[i, k - 1] - g[i, k + 1]) / den, 0)
        px, py = c[0] + r * np.cos(ANG), c[1] + r * np.sin(ANG); keep = np.ones(len(px), bool)
        for _ in range(3):
            A = np.c_[2 * px[keep], 2 * py[keep], np.ones(keep.sum())]; s = np.linalg.lstsq(A, px[keep] ** 2 + py[keep] ** 2, rcond=None)[0]
            R = np.sqrt(s[2] + s[0] ** 2 + s[1] ** 2); d = abs(np.hypot(px - s[0], py - s[1]) - R); keep = d <= max(2.5 * np.median(d[keep]) * 1.4826, .15)
        c = s[:2]
    return c, R, keep.sum()
ring = ((rr >= 13) & (rr <= 32)).astype(np.uint8); full = (rr <= 45).astype(np.uint8)
gref = [grad(r) for r in ref]; cref = [circle(r) for r in ref]; print("센터 사진 원 맞춤: 반지름", [round(float(c[1]), 2) for c in cref], "쓴 점", [int(c[2]) for c in cref])
out = {k: np.zeros((len(crops), 4, 2)) for k in ("기울기영상_전체", "기울기영상_테두리만", "밝기_테두리만", "원테두리")}
for i in range(len(crops)):
    for k in range(4):
        a = crops[i, k].astype(np.float32); g = grad(a)
        out["기울기영상_전체"][i, k] = off[i, k] + ecc(gref[k], g, full); out["기울기영상_테두리만"][i, k] = off[i, k] + ecc(gref[k], g, ring)
        out["밝기_테두리만"][i, k] = off[i, k] + ecc(ref[k], a, ring); c, R, n = circle(a); out["원테두리"][i, k] = off[i, k] + (c - cref[k][0])
    if i % 200 == 0: print(i, flush=True)
np.savez(S / "screw/meas.npz", **{k: v.reshape(len(crops), 8) for k, v in out.items()}, ids=Z["ids"])
print({k: int(np.isnan(v).any(axis=(1, 2)).sum()) for k, v in out.items()})
