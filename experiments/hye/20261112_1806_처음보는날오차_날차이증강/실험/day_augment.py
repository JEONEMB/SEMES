import sys, pathlib, numpy as np
sys.path.insert(0, "apps/distribution"); import reoof
S = pathlib.Path(sys.argv[1]); M = np.load(S / "screw/meas.npz")
tf = reoof.tsv("training_features.tsv"); rows = [r for r in tf if int(r["id"]) != 702 and abs(float(r["T"])) <= 1.0001]
ids = np.array([int(r["id"]) for r in rows]); X0 = np.array([[float(v) for v in r["features66"].split(";")] for r in rows]); L0 = np.array([[float(r["Y"]), float(r["T"])] for r in rows]); src = np.array([r["source"] for r in rows])
L0[np.where(ids == 420)[0][0], 1] = .32; SS = sorted(set(src)); NM = dict(zip(SS, ["1차", "2차", "3차", "4차"])); WIDE = [s for s in SS if len(set(L0[src == s, 0])) > 5]
ok = lambda E: (abs(E[:, 0]) <= .045) & (abs(E[:, 1]) <= .01)
def fp(Ftr, Ltr, Fte, w0=None):
    mean = Ftr.mean(0); sc = np.maximum(Ftr.std(0), 1e-6); A = np.c_[np.ones(len(Ftr)), (Ftr - mean) / sc]; pen = np.eye(A.shape[1]) * (1e-7 * len(Ftr)); pen[0, 0] = 0; w = np.ones(len(Ftr))
    for _ in range(6):
        sw = np.sqrt(w)[:, None]; B = np.linalg.lstsq((A * sw).T @ (A * sw) + pen, (A * sw).T @ (Ltr * sw), rcond=None)[0]; P = A @ B
        r = np.maximum(abs(P[:, 0] - Ltr[:, 0]) / .045, abs(P[:, 1] - Ltr[:, 1]) / .01); w = np.minimum(1, 1 / np.maximum(r, 1e-12)); w *= len(w) / w.sum()
    return np.c_[np.ones(len(Fte)), (Fte - mean) / sc] @ B
def prep(X):
    m = np.isin(src, WIDE); A = np.c_[np.array([(src[m] == s) for s in WIDE]).T.astype(float), L0[m]]; J = np.linalg.lstsq(A, X[m], rcond=None)[0][len(WIDE):]
    L = L0.copy(); R = {}
    for s in SS:
        a = (X[src == s] - L0[src == s] @ J).mean(0); c0 = np.linalg.lstsq(J.T, -a, rcond=None)[0]; L[src == s] -= c0; R[s] = a + c0 @ J     # 움직임으로 설명 안 되는 차이(센터 기준)
    return L, R
BLK = [slice(0, 54, 2), slice(1, 54, 2), slice(54, 58), slice(58, 66, 2), slice(59, 66, 2)]
def run(X, name):
    L, R = prep(X); rng = np.random.default_rng(0); K = 8
    def loso(aug):
        per = []; tot = 0
        for s in SS:
            te = src == s; tr = ~te; Ftr, Ltr = X[tr], L[tr]
            if aug:
                V = np.array([R[q] for q in SS if q != s]); kind, scale = aug; parts = [Ftr]; labs = [Ltr]
                for _ in range(K):
                    if kind == "실측 방향":      # 학습한 세 날의 차이 벡터와 0(센터) 사이·바깥을 무작위로 섞음
                        z = rng.normal(0, scale, (len(Ftr), len(V))); parts.append(Ftr + z @ V)
                    else:                         # 모르는 방향: 근거 묶음마다 공통 이동을 무작위로 줌
                        d = np.zeros_like(Ftr)
                        for b in BLK: d[:, b] += rng.normal(0, scale, (len(Ftr), 1))
                        parts.append(Ftr + d)
                    labs.append(Ltr)
                Ftr, Ltr = np.vstack(parts), np.vstack(labs)
            E = fp(Ftr, Ltr, X[te]) - L[te]; per.append(ok(E).mean() * 100); tot += ok(E).sum()
        return tot / len(L) * 100, per
    def mixed(aug):
        fl, sy, st = [], [], []
        for seed in range(3):
            f = np.random.default_rng(seed).permutation(len(L)) % 5; E = np.zeros_like(L)
            for k in range(5):
                te = f == k; Ftr, Ltr = X[~te], L[~te]
                if aug:
                    kind, scale = aug; parts = [Ftr]; labs = [Ltr]
                    for _ in range(K):
                        if kind == "실측 방향": V = np.array(list(R.values())); parts.append(Ftr + rng.normal(0, scale, (len(Ftr), len(V))) @ V)
                        else:
                            d = np.zeros_like(Ftr)
                            for b in BLK: d[:, b] += rng.normal(0, scale, (len(Ftr), 1))
                            parts.append(Ftr + d)
                        labs.append(Ltr)
                    Ftr, Ltr = np.vstack(parts), np.vstack(labs)
                E[te] = fp(Ftr, Ltr, X[te]) - L[te]
            fl.append((~ok(E)).sum()); sy.append(E[:, 0].std()); st.append(E[:, 1].std())
        return np.mean(fl), np.mean(sy), np.mean(st)
    print(f"\n== {name}")
    for label, aug in [("증강 없음", None), ("실측 방향 ×0.5", ("실측 방향", .5)), ("실측 방향 ×1", ("실측 방향", 1.)), ("실측 방향 ×2", ("실측 방향", 2.)),
                       ("모르는 방향 0.1px", ("무작위", .1)), ("모르는 방향 0.3px", ("무작위", .3))]:
        t, per = loso(aug); fl, sy, st = mixed(aug)
        print(f"{label}\t처음 보는 날 {t:.1f}% (" + " ".join(f"{NM[s]} {p:.1f}" for s, p in zip(SS, per)) + f")\t같은 날 섞은 평가 탈락 {fl:.1f}장 σY {sy:.4f} σT {st:.4f}")
run(X0, "지금 나사 측정")
run(np.c_[X0[:, :58], M["기울기영상_전체"]], "기울기 영상 나사 측정")
