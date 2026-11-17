import os, numpy as np, itertools, json
SC = os.path.dirname(os.path.abspath(__file__))
z = np.load(os.path.join(SC, "tracks.npz"), allow_pickle=True)
pts, X, M = z["pts"], z["X"], z["M"]           # M: [img, feat, (dx, dy, score)]
D = M[:, :, 0]; S = M[:, :, 2]
n, N = D.shape
tr = np.arange(n) % 2 == 0; te = ~tr
rms = lambda a: float(np.sqrt(np.mean(a ** 2)))
# 1) 특징점별 선형모델  dx = a*Θ + b*Y
info = []
for k in range(N):
    ab = np.linalg.lstsq(X, D[:, k], rcond=None)[0]; res = D[:, k] - X @ ab
    info.append(dict(k=k, x=float(pts[k][0]), y=float(pts[k][1]), a=float(ab[0]), b=float(ab[1]),
                     ratio=float(ab[1] / ab[0]), sig=rms(res), maxres=float(abs(res).max()), minscore=float(S[:, k].min())))
good = [i["k"] for i in info if i["maxres"] < 2.0 and i["minscore"] > 0.8]
print("좋은 특징점", len(good), "/", N)
def solve(cols, w=None, train=tr, test=te):
    Dd = D[:, cols]
    J = np.array([np.linalg.lstsq(X[train], Dd[train, j], rcond=None)[0] for j in range(len(cols))])   # (m,2)
    W = np.ones(len(cols)) if w is None else w
    A = J * W[:, None]
    est = np.linalg.lstsq(A, (Dd[test] * W).T, rcond=None)[0].T
    e = est - X[test]
    return J, e, est
def score(cols, w=None):
    J, e, _ = solve(cols, w)
    return rms(e[:, 0]), rms(e[:, 1]), float(np.linalg.cond(J))
res = {}
res["기존: 파인더 + 볼트(아래오른쪽)"] = None
# 아래 오른쪽 볼트 근처 특징점 찾기 (3576,1802)
bolt = min(good, key=lambda k: np.hypot(pts[k][0] - 3576, pts[k][1] - 1802))
print("볼트 근처 특징점", bolt, pts[bolt].round(1), "거리", np.hypot(pts[bolt][0]-3576, pts[bolt][1]-1802).round(1))
# 2) 모든 쌍
pairs = []
for i, j in itertools.combinations(good, 2):
    t, y, c = score([i, j]); pairs.append((y, t, c, i, j))
pairs.sort()
print("\n[2개 조합 상위 10]  (교차검증 RMS)")
for y, t, c, i, j in pairs[:10]:
    print(f"  #{i:2d}({pts[i][0]:.0f},{pts[i][1]:.0f}) + #{j:2d}({pts[j][0]:.0f},{pts[j][1]:.0f})  Θ {t:.4f}  Y {y:.3f}mm  cond {c:.1f}")
pf = [p for p in pairs if 0 in (p[3], p[4])]; pf.sort()
print("\n[파인더(#0) + 1개 상위 5]")
for y, t, c, i, j in pf[:5]:
    o = j if i == 0 else i
    print(f"  #0 + #{o:2d}({pts[o][0]:.0f},{pts[o][1]:.0f})  Θ {t:.4f}  Y {y:.3f}mm  cond {c:.1f}")
t, y, c = score([0, bolt]); print(f"\n기존(파인더+볼트#{bolt}) Θ {t:.4f}  Y {y:.3f}mm  cond {c:.1f}")
# 3) 전부 사용 (동일가중 / 노이즈 가중)
sig = np.array([info[k]["sig"] for k in good])
t1, y1, c1 = score(good); t2, y2, c2 = score(good, 1 / sig)
print(f"\n[전체 {len(good)}개 동일가중] Θ {t1:.4f}  Y {y1:.3f}mm  cond {c1:.1f}")
print(f"[전체 {len(good)}개 노이즈가중] Θ {t2:.4f}  Y {y2:.3f}mm")
# 4) greedy 추가
sel = [pairs[0][3], pairs[0][4]]; hist = [(2, pairs[0][1], pairs[0][0])]
rest = [k for k in good if k not in sel]
while rest and len(sel) < 30:
    best = min(rest, key=lambda k: score(sel + [k])[1]); sel.append(best); rest.remove(best)
    t, y, _ = score(sel); hist.append((len(sel), t, y))
print("\n[greedy 추가]"); [print(f"  {m:2d}개  Θ {t:.4f}  Y {y:.3f}mm") for m, t, y in hist if m in (2,3,4,5,6,8,10,15,20,25,30)]
kbest = min(hist, key=lambda h: h[2])[0]
print("greedy 최적 개수", kbest, "Y", min(h[2] for h in hist))
json.dump(dict(info=info, good=good, bolt=int(bolt), best_pair=[int(pairs[0][3]), int(pairs[0][4])],
               greedy=[int(k) for k in sel[:kbest]], hist=hist), open(os.path.join(SC, "combos.json"), "w"))
