"""봉우리 폭(OOF 오차 σ)을 줄이는 모델 구조 비교.

같은 715장 특징(커밋 46152fd evidence/training_features.tsv), 같은 그룹 5-fold, 같은 Huber 학습식으로 후보를 비교한다.
라벨 정정: ID 420 T 0.33→0.32 (fold 는 T+0.32 묶음과 같게), ID 702 학습·평가 제외. apps/distribution/reoof.py 와 같은 조건이다.
표준화·주성분·계수는 모두 학습 fold 에서만 구한다. 시험 fold 의 라벨은 채점에만 쓴다.
분석용 Python 이며 팀 전달 C++ 도구를 대체하지 않는다.

실행: 저장소 루트에서  python "experiments/hye/20261112_1321_봉우리폭넓음_모델구조비교/실험/compare.py"
"""
import collections, csv, io, pathlib, subprocess
import numpy as np

COMMIT = "46152fd"
EV = "experiments/hye/00004_final_Y_T/evidence"
HERE = pathlib.Path(__file__).resolve().parent.parent / "결과"
QR = list(range(54)); BAR = [54, 55, 56, 57]; PIN = list(range(58, 66))
QX, QY = QR[0::2], QR[1::2]; PX, PY = PIN[0::2], PIN[1::2]


def tsv(name):
    raw = subprocess.run(["git", "show", f"{COMMIT}:{EV}/{name}"], capture_output=True, check=True).stdout
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8")), delimiter="\t"))


def huber_fit(F, Y):
    """00004 model.cpp fit 과 같은 식 (열 수만 일반화)."""
    mean = F.mean(0); sc = np.maximum(F.std(0), 1e-6)
    A = np.c_[np.ones(len(F)), (F - mean) / sc]
    pen = np.eye(A.shape[1]) * (1e-7 * len(F)); pen[0, 0] = 0
    w = np.ones(len(F))
    for _ in range(6):
        sw = np.sqrt(w)[:, None]
        B = np.linalg.lstsq((A * sw).T @ (A * sw) + pen, (A * sw).T @ (Y * sw), rcond=None)[0]
        P = A @ B
        r = np.maximum(abs(P[:, 0] - Y[:, 0]) / .045, abs(P[:, 1] - Y[:, 1]) / .01)
        w = np.minimum(1, 1 / np.maximum(r, 1e-12)); w *= len(w) / w.sum()
    return lambda G: np.c_[np.ones(len(G)), (G - mean) / sc] @ B


def phys(X):
    """물리 성분: 공통 이동, 시차(나사−QR), QR 사이 차(원근·회전), 바 기울기."""
    cx = X[:, QX].mean(1); cy = X[:, QY].mean(1)
    px = X[:, PX].mean(1) - cx; py = X[:, PY].mean(1) - cy
    q13 = X[:, QX[18:27]].mean(1) - X[:, QX[0:9]].mean(1)     # QR3 − QR1 가로 이동 차
    bt = X[:, [55, 57]].mean(1)
    return np.c_[cx, cy, px, py, q13, bt]


def quad(C):
    k = C.shape[1]
    return np.c_[[C[:, i] * C[:, j] for i in range(k) for j in range(i, k)]].T


def build(kind, Xtr, Xte):
    """후보별 입력 행렬. 주성분은 학습 fold 에서만 구한다."""
    if kind == "Huber66 (현재)":
        return Xtr, Xte
    if kind.startswith("66 + 주성분"):
        k = int(kind.split("주성분")[1].split("개")[0])
        mu, s = Xtr.mean(0), Xtr.std(0) + 1e-9
        Vt = np.linalg.svd((Xtr - mu) / s, full_matrices=False)[2][:k]
        pc = lambda X: ((X - mu) / s) @ Vt.T
        return np.c_[Xtr, quad(pc(Xtr))], np.c_[Xte, quad(pc(Xte))]
    if kind == "66 + 물리성분 2차항":
        return np.c_[Xtr, quad(phys(Xtr))], np.c_[Xte, quad(phys(Xte))]
    if kind == "66 + 물리성분 3차항까지":
        def cub(X):
            C = phys(X); k = C.shape[1]
            c3 = np.c_[[C[:, i] * C[:, j] * C[:, l] for i in range(k) for j in range(i, k) for l in range(j, k)]].T
            return np.c_[X, quad(C), c3]
        return cub(Xtr), cub(Xte)
    raise ValueError(kind)


def main():
    tf = [r for r in tsv("training_features.tsv") if r["id"] != "702"]
    fold_of = {int(r["id"]): int(r["fold"]) for r in tsv("group_oof.tsv")}
    ids = np.array([int(r["id"]) for r in tf])
    X = np.array([[float(v) for v in r["features66"].split(";")] for r in tf])
    L = np.array([[float(r["Y"]), float(r["T"])] for r in tf])
    fold = np.array([fold_of[i] for i in ids])
    k420 = np.where(ids == 420)[0][0]; L[k420, 1] = 0.32; fold[k420] = fold_of[418]
    src = np.array([r["source"] for r in tf])
    labs = np.array(sorted(set(map(tuple, np.round(L, 2)))))
    keyg = collections.defaultdict(list)
    for i in range(len(ids)): keyg[(src[i], round(L[i, 0], 2), round(L[i, 1], 2))].append(i)
    grp = [v for v in keyg.values() if len(v) >= 2]

    pin_shift = np.zeros(66); pin_shift[PX] = 0.5           # README 교란: 네 나사 x 에 모두 +0.5px
    kinds = ["Huber66 (현재)", "66 + 주성분8개 2차항", "66 + 주성분12개 2차항", "66 + 물리성분 2차항", "66 + 물리성분 3차항까지"]
    rows = []; Ci0 = None   # 이웃 판정은 모든 후보를 현재 모델(Huber66)의 Σ 하나로 같은 기준에서 센다
    for kind in kinds:
        P = np.zeros_like(L); Pp = np.zeros_like(L)
        for f in sorted(set(fold)):
            te = fold == f
            Ftr, Fte = build(kind, X[~te], X[te]); ncol = Ftr.shape[1]
            model = huber_fit(Ftr, L[~te])
            P[te] = model(Fte)
            _, Fte_p = build(kind, X[~te], X[te] + pin_shift)
            Pp[te] = model(Fte_p)
        E = P - L
        C = np.cov(E.T)
        if Ci0 is None: Ci0 = np.linalg.inv(C)
        Ci = Ci0
        mis = sum(not np.allclose(labs[np.einsum('ij,jk,ik->i', labs - p, Ci, labs - p).argmin()], y) for p, y in zip(P, L))
        W = np.vstack([E[v] - E[v].mean(0) for v in grp]); dof = sum(len(v) - 1 for v in grp)
        rep = np.sqrt((W ** 2).sum(0) / dof)
        ok = lambda P: int(((abs(P[:, 0] - L[:, 0]) <= .045) & (abs(P[:, 1] - L[:, 1]) <= .01)).sum())
        rows.append(dict(kind=kind, sY=E[:, 0].std(ddof=1), sT=E[:, 1].std(ddof=1), rho=C[0, 1] / np.sqrt(C[0, 0] * C[1, 1]),
                         repY=rep[0], repT=rep[1], mis=mis, pass_=ok(P), pass_pin=ok(Pp), n=len(L),
                         ncol=ncol))
        print(rows[-1])
    out = HERE / "결과.tsv"
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["모델", "입력수", "σY", "σT", "ρ", "반복σY", "반복σT", "다른라벨더가까움", "동시통과", "나사x+0.5px통과", "n"])
        for r in rows:
            w.writerow([r["kind"], r["ncol"], f"{r['sY']:.5f}", f"{r['sT']:.5f}", f"{r['rho']:.2f}", f"{r['repY']:.5f}", f"{r['repT']:.5f}",
                        r["mis"], r["pass_"], r["pass_pin"], r["n"]])
    print("저장:", out)


if __name__ == "__main__":
    main()
