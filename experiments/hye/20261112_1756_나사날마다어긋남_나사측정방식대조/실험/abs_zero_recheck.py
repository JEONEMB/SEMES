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
WIDE = [q for q in SS if len(set(L[src == q, 0])) > 5]; m_ = np.isin(src, WIDE)
A_ = np.c_[np.array([(src[m_] == q) for q in WIDE]).T.astype(float), L[m_]]; J = np.linalg.lstsq(A_, X[m_], rcond=None)[0][len(WIDE):]; L0 = L.copy(); LZ = L.copy()
print("세션별 절대 영점 위치(그날 명령 기준 Y, T):")
for q in SS:
    a = (X[src == q] - L0[src == q] @ J).mean(0); c0 = np.linalg.lstsq(J.T, -a, rcond=None)[0]; LZ[src == q] -= c0; print("  ", q, np.round(c0, 4))
import csv
with open(S / "labels_abs_zero.tsv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, delimiter="\t"); w.writerow(["id", "source", "Y_cmd", "T_cmd", "Y_abs", "T_abs"])
    for i in range(len(L)): w.writerow([ids[i], src[i], f"{L0[i,0]:.2f}", f"{L0[i,1]:.2f}", f"{LZ[i,0]:.5f}", f"{LZ[i,1]:.5f}"])
COMBOS = [("나사 4개", [0, 1, 2, 3]), ("대각선 S1+S4", [0, 3]), ("대각선 S2+S3", [1, 2])]
for lname, LL in (("그날 명령 기준 라벨", L0), ("절대 영점 기준 라벨", LZ)):
    for mname, S8 in [("지금 나사 측정", X[:, 58:]), ("기울기 영상 나사 측정", M["기울기영상_전체"])]:
        for name, pk in COMBOS:
            cols = [c for k in pk for c in (2 * k, 2 * k + 1)]; F = np.c_[X[:, :58], S8[:, cols]]; fl, h, sy, st = [], [], [], []
            for seed in range(10):
                rng = np.random.default_rng(seed); f = rng.permutation(len(LL)) % 5; E = np.zeros_like(LL)
                for k in range(5): te = f == k; E[te] = fit(F[~te], LL[~te])(F[te]) - LL[te]
                fl.append((~ok(E)).sum()); sy.append(E[:, 0].std()); st.append(E[:, 1].std()); g = rng.permutation(len(LL)) % 2; Eh = np.zeros_like(LL)
                for k in (0, 1): te = g == k; Eh[te] = fit(F[~te], LL[~te])(F[te]) - LL[te]
                h.append((~ok(Eh)).sum())
            lo = [ok(fit(F[src != q], LL[src != q])(F[src == q]) - LL[src == q]).mean() * 100 for q in SS]; tot = sum(ok(fit(F[src != q], LL[src != q])(F[src == q]) - LL[src == q]).sum() for q in SS) / len(LL) * 100
            print(f"{lname} | {mname} | {name} | 80/20 탈락 {np.mean(fl):.1f} (0장인 횟수 {sum(v == 0 for v in fl)}/10) | 50/50 탈락 {np.mean(h):.1f} | σY {np.mean(sy):.4f} σT {np.mean(st):.4f} | 처음 보는 날 {tot:.1f}% " + " ".join(f"{v:.0f}" for v in lo))
