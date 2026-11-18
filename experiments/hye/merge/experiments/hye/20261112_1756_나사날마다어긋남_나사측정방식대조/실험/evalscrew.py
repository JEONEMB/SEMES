import sys, pathlib, collections, numpy as np
sys.path.insert(0, "apps/distribution"); import reoof
S = pathlib.Path(sys.argv[1]); M = np.load(S / "screw/meas.npz")
tf = reoof.tsv("training_features.tsv"); rows = [r for r in tf if int(r["id"]) != 702 and abs(float(r["T"])) <= 1.0001]
ids = np.array([int(r["id"]) for r in rows]); assert (ids == M["ids"]).all()
X = np.array([[float(v) for v in r["features66"].split(";")] for r in rows]); L = np.array([[float(r["Y"]), float(r["T"])] for r in rows]); src = np.array([r["source"] for r in rows])
L[np.where(ids == 420)[0][0], 1] = .32; SS = sorted(set(src)); NM = dict(zip(SS, ["1차", "2차", "3차", "4차"]))
WIDE = [s for s in SS if len(set(L[src == s, 0])) > 5]; ok = lambda E: (abs(E[:, 0]) <= .045) & (abs(E[:, 1]) <= .01)
g = collections.defaultdict(list)
for i, r in enumerate(rows): g[(r["source"], r["Y"], r["T"])].append(i)
G = [v for v in g.values() if len(v) >= 2]
def rep(F): W = np.vstack([F[v] - F[v].mean(0) for v in G]); return np.sqrt((W ** 2).sum(0) / sum(len(v) - 1 for v in G))
qm = X[:, :54].reshape(-1, 27, 2).mean(1)
def fp(Ftr, Ltr, Fte): return reoof.predict(reoof.fit(Ftr, Ltr), Fte) if Ftr.shape[1] == 66 else None
def evaluate(name, S8):
    F = np.c_[X[:, :58], S8]; sm = S8.reshape(-1, 4, 2).mean(1)
    m = np.isin(src, WIDE); A = np.c_[np.array([(src[m] == s) for s in WIDE]).T.astype(float), L[m]]; J = np.linalg.lstsq(A, F[m], rcond=None)[0][len(WIDE):]
    res = []
    for s in SS:
        a = (F[src == s] - L[src == s] @ J).mean(0); c0 = np.linalg.lstsq(J.T, -a, rcond=None)[0]; r = -a - c0 @ J; res.append(np.sqrt((r[58:] ** 2).mean()))
    # 나사만 따로: 세션별 (나사 평균 - QR 평균) 의 명령 0 에서의 값
    d = sm - qm; Jd = np.linalg.lstsq(A, d[m], rcond=None)[0][len(WIDE):]; par = [(d[src == s] - L[src == s] @ Jd).mean(0) for s in SS]
    lo = [ok(fp(F[src != s], L[src != s], F[src == s]) - L[src == s]).mean() * 100 for s in SS]; tot = sum(ok(fp(F[src != s], L[src != s], F[src == s]) - L[src == s]).sum() for s in SS) / len(L) * 100
    ps, fl = [], []
    for seed in range(5):
        f = np.random.default_rng(seed).permutation(len(L)) % 5; E = np.zeros_like(L)
        for k in range(5): te = f == k; E[te] = fp(F[~te], L[~te], F[te]) - L[te]
        ps.append(ok(E).mean() * 100); fl.append((~ok(E)).sum())
    print(f"\n[{name}]")
    print("  반복 흔들림: 나사평균-QR평균 x %.4f y %.4f | 나사 하나-나사평균 x %.4f" % (*rep(sm - qm), rep((S8.reshape(-1, 4, 2) - sm[:, None]).reshape(len(S8), -1))[0::2].mean()))
    print("  움직임으로 설명 안 되는 나사 차이(px): " + " ".join(f"{NM[s]} {v:.3f}" for s, v in zip(SS, res)))
    print("  명령 0 에서 (나사-QR) x: " + " ".join(f"{NM[s]} {p[0]:+.3f}" for s, p in zip(SS, par)) + f" | 세션 간 퍼짐 {np.std([p[0] for p in par]):.3f}  y: " + " ".join(f"{p[1]:+.3f}" for p in par))
    print(f"  섞은 80/20 통과 {np.mean(ps):.2f}% 탈락 {np.mean(fl):.1f}장 | 세션 뺀 시험 전체 {tot:.1f}% " + " ".join(f"{NM[s]} {v:.1f}" for s, v in zip(SS, lo)))
old = X[:, 58:]; evaluate("지금 방식 (밝기 조각 맞춤)", old)
for k in ("밝기_테두리만", "기울기영상_전체", "기울기영상_테두리만", "원테두리"):
    new = M[k]; diff = new - old
    evaluate(k, new)
    print("  지금 방식과의 차이(px): 전체 RMS %.3f | 세션별 평균 x " % np.sqrt((diff ** 2).mean()) + " ".join(f"{NM[s]} {diff[src == s][:, 0::2].mean():+.3f}" for s in SS) + " | y " + " ".join(f"{diff[src == s][:, 1::2].mean():+.3f}" for s in SS))
