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
COMBOS = [("4개 전부 (지금)", [0, 1, 2, 3]), ("대각선 S1+S4 (왼쪽 위, 오른쪽 아래)", [0, 3]), ("대각선 S2+S3 (오른쪽 위, 왼쪽 아래)", [1, 2]), ("위 2개 S1+S2", [0, 1]), ("아래 2개 S3+S4", [2, 3]),
          ("왼쪽 2개 S1+S3", [0, 2]), ("오른쪽 2개 S2+S4", [1, 3]), ("1개 S1", [0]), ("1개 S4", [3]), ("나사 없음", [])]
for mname, S8 in [("지금 나사 측정", X[:, 58:]), ("기울기 영상 나사 측정", M["기울기영상_전체"])]:
    print("\n== " + mname + "\n조합\t섞은80/20 탈락\tσY\tσT\t50/50 탈락\t세션뺀시험%\t독립잡음0.05px 통과%\t0.1px\t한 나사 x+0.3px\t전체 x+0.1px")
    for name, pk in COMBOS:
        cols = [c for k in pk for c in (2 * k, 2 * k + 1)]; F = np.c_[X[:, :58], S8[:, cols]]; ns = len(cols); fl, sy, st, h, n05, n10, one, com = [], [], [], [], [], [], [], []
        for seed in range(10):
            rng = np.random.default_rng(seed); f = rng.permutation(len(L)) % 5; E = np.zeros_like(L); E5, E10, E1, Ec = [np.zeros_like(L) for _ in range(4)]
            for k in range(5):
                te = f == k; m = fit(F[~te], L[~te]); G = F[te]; E[te] = m(G) - L[te]
                if ns:
                    for Eo, sd in ((E5, .05), (E10, .1)): Gp = G.copy(); Gp[:, 58:] += rng.normal(0, sd, (len(G), ns)); Eo[te] = m(Gp) - L[te]
                    Gp = G.copy(); Gp[:, 58] += .3; E1[te] = m(Gp) - L[te]; Gp = G.copy(); Gp[:, 58::2] += .1; Ec[te] = m(Gp) - L[te]
            fl.append((~ok(E)).sum()); sy.append(E[:, 0].std()); st.append(E[:, 1].std()); n05.append(ok(E5).mean() * 100); n10.append(ok(E10).mean() * 100); one.append(ok(E1).mean() * 100); com.append(ok(Ec).mean() * 100)
            g = rng.permutation(len(L)) % 2; Eh = np.zeros_like(L)
            for k in (0, 1): te = g == k; Eh[te] = fit(F[~te], L[~te])(F[te]) - L[te]
            h.append((~ok(Eh)).sum())
        lo = sum(ok(fit(F[src != s], L[src != s])(F[src == s]) - L[src == s]).sum() for s in SS) / len(L) * 100
        print(f"{name}\t{np.mean(fl):.1f}\t{np.mean(sy):.4f}\t{np.mean(st):.4f}\t{np.mean(h):.1f}\t{lo:.1f}\t" + ("\t".join(f"{np.mean(v):.1f}" for v in (n05, n10, one, com)) if ns else "-\t-\t-\t-"))
