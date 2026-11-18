import sys, pathlib, numpy as np
sys.path.insert(0, "apps/distribution"); import reoof
S = pathlib.Path(sys.argv[1]); M = np.load(S / "screw/meas.npz")
tf = reoof.tsv("training_features.tsv"); rows = [r for r in tf if int(r["id"]) != 702 and abs(float(r["T"])) <= 1.0001]
ids = np.array([int(r["id"]) for r in rows]); X = np.array([[float(v) for v in r["features66"].split(";")] for r in rows]); L = np.array([[float(r["Y"]), float(r["T"])] for r in rows]); src = np.array([r["source"] for r in rows])
L[np.where(ids == 420)[0][0], 1] = .32; ok = lambda E: (abs(E[:, 0]) <= .045) & (abs(E[:, 1]) <= .01)
keys = [(a, y, t) for a, (y, t) in zip(src, L)]; uk = sorted(set(keys)); gi = np.array([uk.index(k) for k in keys]); gs = np.array([k[0] for k in uk])
def cv(F, f):
    E = np.zeros_like(L)
    for k in sorted(set(f)): te = f == k; E[te] = reoof.predict(reoof.fit(F[~te], L[~te]), F[te]) - L[te]
    return E
for name, S8 in [("지금 방식", X[:, 58:]), ("기울기 영상 맞춤", M["기울기영상_전체"]), ("기울기 영상, 테두리만", M["기울기영상_테두리만"])]:
    F = np.c_[X[:, :58], S8]; out = []
    for mode in ("사진 단위 80/20", "자세 묶음 단위 80/20", "세션별 절반씩 50/50"):
        fl, sy, st, mY, mT, bad = [], [], [], 0, 0, set()
        for seed in range(10):
            rng = np.random.default_rng(seed)
            if mode.startswith("사진"): f = rng.permutation(len(L)) % 5
            elif mode.startswith("자세"): f = (rng.permutation(len(uk)) % 5)[gi]
            else:
                h = np.zeros(len(uk), int)
                for s in set(gs): idx = np.where(gs == s)[0]; h[rng.permutation(idx)[: len(idx) // 2]] = 1
                f = h[gi]
            E = cv(F, f); fl.append(int((~ok(E)).sum())); sy.append(E[:, 0].std()); st.append(E[:, 1].std()); mY = max(mY, abs(E[:, 0]).max()); mT = max(mT, abs(E[:, 1]).max()); bad |= set(ids[~ok(E)].tolist())
        out.append(f"{mode}: 탈락 {fl} 평균 {np.mean(fl):.1f}장, σY {np.mean(sy):.4f} σT {np.mean(st):.4f}, 최대 |eY| {mY:.4f} |eT| {mT:.4f}, 탈락 ID {sorted(bad)}")
    print(name + "\n   " + "\n   ".join(out))
