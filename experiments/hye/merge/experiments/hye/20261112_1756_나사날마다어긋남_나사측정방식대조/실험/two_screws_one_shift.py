import sys, pathlib, numpy as np
sys.path.insert(0, "apps/distribution"); import reoof
S = pathlib.Path(sys.argv[1]); M = np.load(S / "screw/meas.npz")
tf = reoof.tsv("training_features.tsv"); rows = [r for r in tf if int(r["id"]) != 702 and abs(float(r["T"])) <= 1.0001]
ids = np.array([int(r["id"]) for r in rows]); X = np.array([[float(v) for v in r["features66"].split(";")] for r in rows]); L = np.array([[float(r["Y"]), float(r["T"])] for r in rows]); src = np.array([r["source"] for r in rows])
L[np.where(ids == 420)[0][0], 1] = .32; SS = sorted(set(src)); ok = lambda E: (abs(E[:, 0]) <= .045) & (abs(E[:, 1]) <= .01)
def fit(F, Y):
    mean = F.mean(0); sc = np.maximum(F.std(0), 1e-6); A = np.c_[np.ones(len(F)), (F - mean) / sc]; pen = np.eye(A.shape[1]) * (1e-7 * len(F)); pen[0, 0] = 0; w = np.ones(len(F))
    for _ in range(6):
        sw = np.sqrt(w)[:, None]; B = np.linalg.lstsq((A * sw).T @ (A * sw) + pen, (A * sw).T @ (Y * sw), rcond=None)[0]; P = A @ B
        r = np.maximum(abs(P[:, 0] - Y[:, 0]) / .045, abs(P[:, 1] - Y[:, 1]) / .01); w = np.minimum(1, 1 / np.maximum(r, 1e-12)); w *= len(w) / w.sum()
    return lambda G: np.c_[np.ones(len(G)), (G - mean) / sc] @ B
COMBOS = [("4개 전부", [0, 1, 2, 3]), ("대각선 S1+S4", [0, 3]), ("대각선 S2+S3", [1, 2]), ("위 2개", [0, 1]), ("아래 2개", [2, 3]), ("왼쪽 2개", [0, 2]), ("오른쪽 2개", [1, 3])]
for mname, S8 in [("지금 나사 측정", X[:, 58:]), ("기울기 영상 나사 측정", M["기울기영상_전체"])]:
    print("== " + mname + " : 나사 하나만 가로로 +0.1 / +0.3 px 틀어졌을 때 통과% (틀어진 나사별)")
    for name, pk in COMBOS:
        cols = [c for k in pk for c in (2 * k, 2 * k + 1)]; F = np.c_[X[:, :58], S8[:, cols]]; res = {(j, d): [] for j in range(len(pk)) for d in (.1, .3)}
        for seed in range(5):
            f = np.random.default_rng(seed).permutation(len(L)) % 5; E = {k: np.zeros_like(L) for k in res}
            for k in range(5):
                te = f == k; m = fit(F[~te], L[~te])
                for (j, d) in res: G = F[te].copy(); G[:, 58 + 2 * j] += d; E[(j, d)][te] = m(G) - L[te]
            for k in res: res[k].append(ok(E[k]).mean() * 100)
        a1 = np.mean([np.mean(res[(j, .1)]) for j in range(len(pk))]); a3 = np.mean([np.mean(res[(j, .3)]) for j in range(len(pk))])
        print(name, "|", "  ".join("S%d: %.0f / %.0f" % (pk[j] + 1, np.mean(res[(j, .1)]), np.mean(res[(j, .3)])) for j in range(len(pk))), "| 평균 %.0f / %.0f" % (a1, a3))
