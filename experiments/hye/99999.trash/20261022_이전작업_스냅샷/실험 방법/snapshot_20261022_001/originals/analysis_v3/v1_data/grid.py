import os, csv, json, numpy as np
SC = os.path.dirname(os.path.abspath(__file__)); PROJ = r"C:\Users\SSAFY\Desktop\hwlee\projects\2026-10_SEMES"
z = np.load(os.path.join(SC, "tracks.npz"), allow_pickle=True); files = z["files"]; Xall = z["X"]; DT = z["M"][:, :, 0]
Fe = np.load(os.path.join(SC, "ecc_feats.npy")); DQ = np.load(os.path.join(SC, "qr_tracks.npz"))["D"]; gq = list(np.load(os.path.join(SC, "qr_tracks.npz"))["good"])
cb = json.load(open(os.path.join(SC, "combos.json"))); st = json.load(open(os.path.join(SC, "stable.json")))
# 중복 center 제거
keep = []; seen = False
for i, f in enumerate(files):
    if f[2] == "center.bmp":
        if seen: continue
        seen = True
    keep.append(i)
keep = np.array(keep); X = Xall[keep]; DT = DT[keep]; Fe = Fe[keep]; DQ = DQ[keep]; names = [files[i][2] for i in keep]
c14 = {r["file"]: r for r in csv.DictReader(open(os.path.join(PROJ, "14_데이터_QR마커위치.csv"), encoding="utf-8-sig"))}
c15 = {r["file"]: r for r in csv.DictReader(open(os.path.join(PROJ, "15_데이터_파인더_볼트이동량.csv"), encoding="utf-8-sig"))}
col = lambda d, cols, zero=0.0: np.array([[float(d[n][c]) if n in d else zero for c in cols] for n in names])
G14 = col(c14, ["dLx", "dCx", "dRx", "dLy", "dCy", "dRy", "Lw", "Cw", "Rw", "Lh", "Ch", "Rh", "LC_gap", "CR_gap"])
G14[:, 6:] -= G14[[names.index("center.bmp")], 6:]          # 크기/간격은 center 대비 변화량으로
F15 = col(c15, ["finder_dx", "finder_dy", "bolt_dx"])
qmean = G14[:, :3].mean(1, keepdims=True)
FEAT = {
 "Q1 QR 3개 평균 이동량 (1개)": ("QR만", qmean),
 "Q2 파인더 이동량 (1개)": ("QR만", F15[:, :1]),
 "Q3 QR 3개 가로/세로 이동량 (6개)": ("QR만", G14[:, :6]),
 "Q4 QR 3개 이동량+크기+간격 (14개)": ("QR만", G14),
 "Q5 QR 전체 호모그래피 (8개)": ("QR만", Fe[:, :8]),
 "Q6 QR 호모그래피+QR별 affine (26개)": ("QR만", Fe),
 "Q7 QR 내부 특징점 (263개)": ("QR만", DQ[:, gq]),
 "O1 파인더+볼트 (2개)": ("QR 밖 포함", F15[:, [0, 2]]),
 "O2 조합A 파인더+벽볼트 (2개)": ("QR 밖 포함", DT[:, [0, 47]]),
 "O3 조합B2 브래킷 윗판+아래 (2개)": ("QR 밖 포함", DT[:, st["best"]]),
 "O4 안 바뀌는 특징점 전부 (51개)": ("QR 밖 포함", DT[:, st["ok"]]),
 "O5 특징점 전부, 라벨 포함 (74개)": ("QR 밖 포함", DT[:, cb["good"]]),
 "O6 QR 기하 14개 + 조합B2 2개 (16개)": ("QR 밖 포함", np.c_[G14, DT[:, st["best"]]]),
}
idx = np.arange(len(X)); folds = [(idx % 2 == 0, idx % 2 == 1), (idx % 2 == 1, idx % 2 == 0)]
def std(Ftr, Fte):
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-12; return (Ftr - mu) / sd, (Fte - mu) / sd
def a_ratio(F, tr, te):                       # C1: Θ = 이동량 / 계수 (Y=0 가정)
    if F.shape[1] != 1: return None
    p = np.polyfit(X[tr & (X[:, 1] == 0), 0], F[tr & (X[:, 1] == 0), 0], 1)
    return np.c_[(F[te, 0] - p[1]) / p[0], np.zeros(te.sum())]
def a_ols(F, tr, te):                         # C2: 역회귀 [Θ,Y] = W f + b
    A, B = std(F[tr], F[te]); A = np.c_[A, np.ones(len(A))]; B = np.c_[B, np.ones(len(B))]
    if A.shape[1] > A.shape[0] * 0.8: return None
    return B @ np.linalg.lstsq(A, X[tr], rcond=None)[0]
def a_ridge(F, tr, te):                       # C3: 릿지 역회귀 (학습 데이터 안에서 alpha 선택)
    A, B = std(F[tr], F[te]); Xtr = X[tr]; n = len(A); k = np.arange(n)
    def fit(A_, y_, al):
        A1 = np.c_[A_, np.ones(len(A_))]; R = al * np.eye(A1.shape[1]); R[-1, -1] = 0
        return np.linalg.solve(A1.T @ A1 + R, A1.T @ y_)
    best = None
    for al in [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]:
        e = []
        for p in range(2):
            m = k % 2 == p; W = fit(A[~m], Xtr[~m], al); e.append(np.c_[A[m], np.ones(m.sum())] @ W - Xtr[m])
        sc = np.sqrt(np.mean(np.concatenate(e)[:, 1] ** 2) / 14.3 ** -2 * 0 + np.mean((np.concatenate(e) * [80, 14.3]) ** 2))
        if best is None or sc < best[0]: best = (sc, al)
    return np.c_[B, np.ones(len(B))] @ fit(A, Xtr, best[1])
def jac(F, tr):
    A = np.c_[X[tr], np.ones(tr.sum())]; Bc = np.linalg.lstsq(A, F[tr], rcond=None)[0]; R = F[tr] - A @ Bc
    return Bc[:2].T, Bc[2], R
def a_jls(F, tr, te):                         # C4: 야코비안 최소제곱 (f = J[Θ,Y] + b 를 Θ,Y에 대해 풂)
    if F.shape[1] < 2: return None
    J, b, R = jac(F, tr); sd = R.std(0) + 1e-12
    Jn = J / sd[:, None]; y = (F[te] - b) / sd
    return np.linalg.lstsq(Jn, y.T, rcond=None)[0].T
def a_jgls(F, tr, te):                        # C5: 야코비안 + 잡음 공분산 가중 (GLS)
    if F.shape[1] < 2: return None
    J, b, R = jac(F, tr); S = np.cov(R.T); S = S + np.eye(len(S)) * np.trace(S) / len(S) * 1e-2
    Si = np.linalg.inv(S); M = np.linalg.solve(J.T @ Si @ J, J.T @ Si)
    return ((F[te] - b) @ M.T)
ALG = {"C1 이동량÷계수 (Y=0 가정)": a_ratio, "C2 선형 역회귀": a_ols, "C3 릿지 역회귀": a_ridge,
       "C4 야코비안 최소제곱": a_jls, "C5 야코비안 가중(GLS)": a_jgls}
rows = []
for fn, (grp, F) in FEAT.items():
    for an, fa in ALG.items():
        est = np.full(X.shape, np.nan); ok = True
        for tr, te in folds:
            r = fa(F, tr, te)
            if r is None: ok = False; break
            est[te] = r
        if not ok: continue
        e = est - X; eT, eY = np.abs(e[:, 0]), np.abs(e[:, 1])
        isT = X[:, 1] == 0; isY = X[:, 0] == 0
        rows.append(dict(영역=grp, 특징점=fn, 알고리즘=an,
            T_평균=eT.mean(), T_RMS=np.sqrt((eT ** 2).mean()), T_최대=eT.max(),
            Y_평균=eY.mean(), Y_RMS=np.sqrt((eY ** 2).mean()), Y_최대=eY.max(),
            Y_005안=np.mean(eY <= 0.05) * 100, Y사진_T착각=np.sqrt((e[isY & ~isT, 0] ** 2).mean()), T사진_Y착각=np.sqrt((e[isT & ~isY, 1] ** 2).mean())))
out = os.path.join(PROJ, "17_결과표_특징점조합x알고리즘.csv")
with open(out, "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader()
    for r in rows: w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
json.dump(rows, open(os.path.join(SC, "grid.json"), "w"), ensure_ascii=False, default=float)
for r in rows: print(f"{r['특징점'][:30]:32s} {r['알고리즘'][:14]:15s} T평균 {r['T_평균']:.4f} RMS {r['T_RMS']:.4f} | Y평균 {r['Y_평균']:.3f} RMS {r['Y_RMS']:.3f} 최대 {r['Y_최대']:.3f} 안 {r['Y_005안']:5.1f}%")
