"""이상 센터(특징이 모두 0 = center.bmp 와 같은 화면) 기준으로 라벨을 옮기고,
세션 사이에 남는 "움직임으로 설명 안 되는 차이"를 지운 뒤 새 세션 통과율을 본다.

- 자료: 커밋 46152fd 의 training_features.tsv 66개 특징(QR 54 = QR 3개 × ECC homography 3×3 투영점 dx,dy / 바 4 / 나사 8),
  536점 이동은 A0 캐시와 20261112_1504 실험의 4차 캐시. ID 702 제외, T=+2.00 12장 제외, ID 420 라벨 T 0.32 → 702장.
- 라벨 옮기기: 넓게 찍은 세션(1차, 4차)에서 명령 1단위당 특징 이동 J 를 구하고, 세션마다 특징이 0 과 가장 가까워지는 명령 c0 를 구해 라벨에서 뺀다.
  (학습 자료 정리 단계다. 추론은 사진 한 장만 쓴다.)
- 평가: 세션 하나를 통째로 빼고 나머지로 학습(00004 Huber 학습식). 통과 = |eY| ≤ 0.045 그리고 |eT| ≤ 0.01. 정답은 옮긴 라벨.
- 방법
    그대로      : 특징 그대로 학습
    세션내학습  : 학습 세션마다 특징·라벨의 세션 평균을 빼고 학습(세션 사이 차이를 기울기 학습에 쓰지 않음), 추론은 특징 그대로
    차이제거    : 학습 세션들의 "명령 0 에서의 특징"(움직임으로 설명 안 되는 차이) 방향을 특징에서 지우고 학습·추론
    상한        : 시험 세션의 차이 방향까지 지움(정답을 본 것, 도달 가능한 위쪽 한계)
- QR 3×3 투영점 순서는 행 우선으로 가정했다(가운데 = 4번, 모서리 = 0,2,6,8).
분석용 Python 이며 팀 전달 C++ 도구를 대체하지 않는다.

실행: 저장소 루트에서  python "<이 폴더>/실험/ideal_center.py"
"""
import csv, pathlib, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[4]; sys.path.insert(0, str(ROOT / "apps/distribution"))
import reoof
OUT = pathlib.Path(__file__).resolve().parent.parent / "결과"
A0 = ROOT / "experiments/hye/20261102_124309_A0_분할측정고정_계층분할과점측정캐시/결과/cache"
R4 = ROOT / "experiments/hye/20261112_1504_QR요약54개_536점대조/결과/cache_round4"

tf = reoof.tsv("training_features.tsv")
rows = [r for r in tf if int(r["id"]) != 702 and abs(float(r["T"])) <= 1.0001]
ids = np.array([int(r["id"]) for r in rows]); X = np.array([[float(v) for v in r["features66"].split(";")] for r in rows])
L0 = np.array([[float(r["Y"]), float(r["T"])] for r in rows]); src = np.array([r["source"] for r in rows])
L0[np.where(ids == 420)[0][0], 1] = .32
S = sorted(set(src)); NAME = dict(zip(S, ["1차", "2차", "3차", "4차"]))
WIDE = [s for s in S if len(set(L0[src == s, 0])) > 5]

# 536점
shots = list(csv.DictReader(open(A0 / "shots.csv", encoding="utf-8-sig"))); pts = list(csv.DictReader(open(A0 / "points.csv", encoding="utf-8-sig")))
sel = np.array([int(p["selected"]) for p in pts], bool)
disp = np.fromfile(A0 / "disp.f32", np.float32).reshape(len(shots), len(pts), 2)
by_path = {f"data/clean/{s['set']}/{s['rel']}": i for i, s in enumerate(shots)}
ids4 = [int(x) for x in (R4 / "ids.txt").read_text().split()]; disp4 = np.fromfile(R4 / "disp.f32", np.float32).reshape(len(ids4), len(pts), 2)
by4 = {i: k for k, i in enumerate(ids4)}
D = np.array([(disp4[by4[int(r["id"])]] if int(r["id"]) in by4 else disp[by_path[r["path"]]])[sel].reshape(-1) for r in rows], float)
assert np.isfinite(D).all()

def slope(F, L, mask):
    ss = [s for s in WIDE if (mask & (src == s)).any()]
    A = np.c_[np.array([(src[mask] == s) for s in ss]).T.astype(float), L[mask]]
    return np.linalg.lstsq(A, F[mask], rcond=None)[0][len(ss):]

# 라벨 옮기기 (66개 특징 전체 기준)
J = slope(X, L0, np.ones(len(L0), bool)); L = L0.copy(); shift = {}
for s in S:
    m = src == s; a = (X[m] - L0[m] @ J).mean(0); c0 = np.linalg.lstsq(J.T, -a, rcond=None)[0]; res = -a - c0 @ J
    L[m] -= c0; shift[s] = (c0, [np.sqrt((res[i] ** 2).mean()) for i in (slice(0, 54), slice(54, 58), slice(58, 66))])
    # 근거별로 따로 구한 센터(정의의 모호함)
    alt = [np.linalg.lstsq(J[:, i].T, -a[i], rcond=None)[0] for i in (slice(0, 54), slice(58, 66))]
    shift[s] += (alt,)

ok = lambda E: (abs(E[:, 0]) <= .045) & (abs(E[:, 1]) <= .01)
QRm = X[:, :54].reshape(-1, 3, 9, 2); B4, S8 = X[:, 54:58], X[:, 58:]
qmean = QRm.mean(2).reshape(len(X), -1); qcen = QRm[:, :, 4].reshape(len(X), -1); qcor = QRm[:, :, [0, 2, 6, 8]].reshape(len(X), -1)
Q54 = X[:, :54]
def pca(k):
    def f(tr): mu = D[tr].mean(0); Vt = np.linalg.svd(D[tr] - mu, full_matrices=False)[2][:k]; return (D - mu) @ Vt.T
    return f
fixed = lambda F: (lambda tr: F)
SETS = [("QR 3개 평균 이동만 (6)", fixed(qmean)), ("QR 가운데 점 3개 (6)", fixed(qcen)), ("QR 모서리 4점씩 12개 (24)", fixed(qcor)), ("QR 9점씩 27개 (54)", fixed(Q54)),
        ("QR 54 + 바 4", fixed(np.c_[Q54, B4])), ("QR 54 + 나사 8", fixed(np.c_[Q54, S8])), ("QR 54 + 바 4 + 나사 8 (현재 66)", fixed(X)),
        ("QR 평균 6 + 바 4 + 나사 8", fixed(np.c_[qmean, B4, S8])), ("QR 모서리 24 + 바 4 + 나사 8", fixed(np.c_[qcor, B4, S8])),
        ("바 4 + 나사 8 (QR 없음)", fixed(np.c_[B4, S8])),
        ("536점 주성분 20 + 바 4 + 나사 8", lambda tr: np.c_[pca(20)(tr), B4, S8]), ("536점 주성분 54 + 바 4 + 나사 8", lambda tr: np.c_[pca(54)(tr), B4, S8])]

def run(F, te, method):
    tr = ~te
    if method == "그대로": return reoof_fit_predict(F[tr], L[tr], F[te])
    if method == "세션내학습":
        Fc, Lc = F[tr].copy(), L[tr].copy()
        for s in S:
            m = src[tr] == s
            if m.any(): Fc[m] -= Fc[m].mean(0); Lc[m] -= Lc[m].mean(0)
        return reoof_fit_predict(Fc, Lc, F[te])
    Jf = slope(F, L, tr & np.isin(src, WIDE)) if (tr & np.isin(src, WIDE)).any() else slope(F, L, np.isin(src, WIDE))
    use = [s for s in S if (src == s)[tr].any()] + ([S[[i for i, s in enumerate(S) if (src == s)[te].any()][0]]] if method == "상한" else [])
    N = np.array([(F[src == s] - L[src == s] @ Jf).mean(0) for s in use])
    Qb = np.linalg.svd(N, full_matrices=False)[2]; Qb = Qb[: np.linalg.matrix_rank(N, tol=1e-9)]
    G = F - (F @ Qb.T) @ Qb
    return reoof_fit_predict(G[tr], L[tr], G[te])

def reoof_fit_predict(Ftr, Ltr, Fte):
    mean = Ftr.mean(0); sc = np.maximum(Ftr.std(0), 1e-6); A = np.c_[np.ones(len(Ftr)), (Ftr - mean) / sc]
    pen = np.eye(A.shape[1]) * (1e-7 * len(Ftr)); pen[0, 0] = 0; w = np.ones(len(Ftr))
    for _ in range(6):
        sw = np.sqrt(w)[:, None]; Bc = np.linalg.lstsq((A * sw).T @ (A * sw) + pen, (A * sw).T @ (Ltr * sw), rcond=None)[0]; P = A @ Bc
        r = np.maximum(abs(P[:, 0] - Ltr[:, 0]) / .045, abs(P[:, 1] - Ltr[:, 1]) / .01); w = np.minimum(1, 1 / np.maximum(r, 1e-12)); w *= len(w) / w.sum()
    return np.c_[np.ones(len(Fte)), (Fte - mean) / sc] @ Bc

print("세션별 센터 위치(그날 명령) 와 움직임으로 설명 안 되는 차이")
with (OUT / "센터위치.tsv").open("w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, delimiter="\t"); w.writerow(["세션", "센터Y(66전체)", "센터T(66전체)", "남은차이QR_px", "남은차이바_px", "남은차이나사_px", "센터Y(QR만)", "센터T(QR만)", "센터Y(나사만)", "센터T(나사만)"])
    for s in S:
        c0, res, alt = shift[s]; row = [NAME[s], f"{c0[0]:+.4f}", f"{c0[1]:+.4f}"] + [f"{v:.3f}" for v in res] + [f"{alt[0][0]:+.4f}", f"{alt[0][1]:+.4f}", f"{alt[1][0]:+.4f}", f"{alt[1][1]:+.4f}"]
        w.writerow(row); print("  ", row)

METHODS = ["그대로", "세션내학습", "차이제거", "상한"]
with (OUT / "결과.tsv").open("w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, delimiter="\t"); w.writerow(["특징", "방법", "전체통과%", "1차%", "2차%", "3차%", "4차%", "세션평균오차 최대|eY|", "최대|eT|", "세션내 sdY", "sdT"])
    for name, build in SETS:
        for mth in METHODS:
            per = []; tot = 0; mY = mT = 0; W = []
            for s in S:
                te = src == s; F = build(~te); E = run(F, te, mth) - L[te]
                per.append(ok(E).mean() * 100); tot += ok(E).sum(); mY = max(mY, abs(E[:, 0].mean())); mT = max(mT, abs(E[:, 1].mean())); W.append(E - E.mean(0))
            W = np.vstack(W); row = [name, mth, f"{tot/len(L)*100:.1f}"] + [f"{p:.1f}" for p in per] + [f"{mY:.4f}", f"{mT:.4f}", f"{W[:,0].std():.4f}", f"{W[:,1].std():.4f}"]
            w.writerow(row); print("\t".join(row))
