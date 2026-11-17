"""QR 요약 54개(00004)와 고정 536점 직접 측정을 같은 사진·같은 분할에서 비교한다.

- 사진: 1~4차 중 536점이 모두 유효한 사진(T=+2.00 12장 제외), ID 702 제외.
- 4차 536점은 이 실험의 `실험/measure630`(control tracker 를 그대로 링크한 C++ 도구)로 재서 `결과/cache_round4` 에 두었다.
  같은 도구로 1~3차 12장을 다시 재어 A0 캐시와 최대 차 0.0 px 임을 확인했다.
- QR 54·바 4·나사 8: 커밋 46152fd `experiments/hye/00004_final_Y_T/evidence/training_features.tsv`
- 536점 이동(dx, dy): `experiments/hye/20261102_124309_A0_.../결과/cache/disp.f32` (630점 중 selected=1 인 536점)
- 분할: 00004 의 그룹 5-fold(같은 자세 반복은 한 fold). ID 420 은 라벨 T 0.32, fold 는 T+0.32 묶음과 같게 한다.
- 학습식: 00004 model.cpp 와 같은 표준화 + ridge + Huber 6회. 주성분·규제 세기는 학습 fold 안에서만 정한다.
- 모든 후보에 바 4·나사 8 을 똑같이 붙인다. 바꾸는 것은 QR 표현뿐이다.
분석용 Python 이며 팀 전달 C++ 도구를 대체하지 않는다.

실행: 저장소 루트에서  python "experiments/hye/20261112_1504_QR요약54개_536점대조/실험/compare536.py"
"""
import collections, csv, io, pathlib, subprocess
import numpy as np

COMMIT = "46152fd"; EV = "experiments/hye/00004_final_Y_T/evidence"
ROOT = pathlib.Path(__file__).resolve().parents[4]
A0 = ROOT / "experiments/hye/20261102_124309_A0_분할측정고정_계층분할과점측정캐시/결과/cache"
OUT = pathlib.Path(__file__).resolve().parent.parent / "결과"


def tsv(name):
    raw = subprocess.run(["git", "show", f"{COMMIT}:{EV}/{name}"], capture_output=True, check=True, cwd=ROOT).stdout
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8")), delimiter="\t"))


def huber(F, Y, lam=1e-7):
    mean = F.mean(0); sc = np.maximum(F.std(0), 1e-6)
    A = np.c_[np.ones(len(F)), (F - mean) / sc]
    pen = np.eye(A.shape[1]) * (lam * len(F)); pen[0, 0] = 0
    w = np.ones(len(F))
    for _ in range(6):
        sw = np.sqrt(w)[:, None]
        B = np.linalg.lstsq((A * sw).T @ (A * sw) + pen, (A * sw).T @ (Y * sw), rcond=None)[0]
        P = A @ B
        r = np.maximum(abs(P[:, 0] - Y[:, 0]) / .045, abs(P[:, 1] - Y[:, 1]) / .01)
        w = np.minimum(1, 1 / np.maximum(r, 1e-12)); w *= len(w) / w.sum()
    return lambda G: np.c_[np.ones(len(G)), (G - mean) / sc] @ B


def main():
    shots = list(csv.DictReader(open(A0 / "shots.csv", encoding="utf-8-sig")))
    pts = list(csv.DictReader(open(A0 / "points.csv", encoding="utf-8-sig")))
    sel = np.array([int(p["selected"]) for p in pts], bool); region = np.array([int(p["region"]) for p in pts])
    disp = np.fromfile(A0 / "disp.f32", np.float32).reshape(len(shots), len(pts), 2).astype(float)
    by_path = {f"data/clean/{s['set']}/{s['rel']}": i for i, s in enumerate(shots)}
    R4 = OUT / "cache_round4"
    ids4 = [int(x) for x in (R4 / "ids.txt").read_text().split()]
    disp4 = np.fromfile(R4 / "disp.f32", np.float32).reshape(len(ids4), len(pts), 2).astype(float)
    vs4 = [int(r["valid_selected"]) for r in csv.DictReader(open(R4 / "shots.csv", encoding="utf-8"))]
    by_id4 = {i: k for k, i in enumerate(ids4)}

    tf = tsv("training_features.tsv"); fold_of = {int(r["id"]): int(r["fold"]) for r in tsv("group_oof.tsv")}
    ids, X66, L, fold, D, src = [], [], [], [], [], []
    for r in tf:
        rid = int(r["id"])
        if rid == 702: continue
        if rid in by_id4:
            if vs4[by_id4[rid]] != 536: continue
            d = disp4[by_id4[rid]]
        else:
            i = by_path.get(r["path"])
            if i is None or int(shots[i]["valid_selected"]) != 536: continue
            d = disp[i]
        ids.append(rid); X66.append([float(v) for v in r["features66"].split(";")])
        L.append([float(r["Y"]), float(r["T"])]); fold.append(fold_of[rid]); D.append(d[sel].reshape(-1)); src.append(r["source"])
    ids = np.array(ids); X66 = np.array(X66); L = np.array(L); fold = np.array(fold); D = np.array(D); src = np.array(src)
    k = np.where(ids == 420)[0][0]; L[k, 1] = 0.32; fold[k] = fold_of[418]
    QR, REST = X66[:, :54], X66[:, 54:]
    print(f"사진 {len(ids)}장 (세션별 {dict(collections.Counter(src))}), 536점 특징 {D.shape[1]}개")

    # 536점의 QR별 평균·아핀 요약 (QR 3개 × 6 = 18)
    xy = np.array([[float(p["x"]), float(p["y"])] for p in pts])[sel]; reg = region[sel]
    def affine(Dm):
        out = []
        for q in sorted(set(reg)):
            m = reg == q; A = np.c_[np.ones(m.sum()), xy[m] - xy[m].mean(0)]
            d = Dm.reshape(len(Dm), -1, 2)[:, m]
            out.append(np.einsum('pk,npc->nkc', np.linalg.pinv(A).T, d).reshape(len(Dm), -1))
        return np.hstack(out)
    AFF = affine(D)

    def pca(k):
        def build(tr, te):
            mu = D[tr].mean(0); Vt = np.linalg.svd(D[tr] - mu, full_matrices=False)[2][:k]
            return np.c_[(D[tr] - mu) @ Vt.T, REST[tr]], np.c_[(D[te] - mu) @ Vt.T, REST[te]]
        return build
    def plain(F): return lambda tr, te: (np.c_[F[tr], REST[tr]], np.c_[F[te], REST[te]])
    def both(k):
        def build(tr, te):
            mu = D[tr].mean(0); Vt = np.linalg.svd(D[tr] - mu, full_matrices=False)[2][:k]
            return np.c_[QR[tr], (D[tr] - mu) @ Vt.T, REST[tr]], np.c_[QR[te], (D[te] - mu) @ Vt.T, REST[te]]
        return build

    cands = [("QR 요약 54개 (현재)", plain(QR), 1e-7), ("536점 QR별 아핀 요약 18개", plain(AFF), 1e-7),
             ("536점 주성분 20개", pca(20), 1e-7), ("536점 주성분 54개", pca(54), 1e-7), ("536점 주성분 100개", pca(100), 1e-7),
             ("536점 전부 1072개, 규제 선택", plain(D), None),
             ("QR 54 + 536점 주성분 20개", both(20), 1e-7), ("QR 54 + 536점 주성분 54개", both(54), 1e-7)]
    key = collections.defaultdict(list)
    for i in range(len(ids)): key[(src[i], L[i, 0], L[i, 1])].append(i)
    grp = [v for v in key.values() if len(v) >= 2]
    labs = np.array(sorted(set(map(tuple, np.round(L, 2)))))
    rows = []; Ci0 = None
    for name, build, lam in cands:
        P = np.zeros_like(L); lam_used = []
        for f in sorted(set(fold)):
            te = fold == f; tr = ~te
            Ftr, Fte = build(tr, te)
            l = lam
            if l is None:   # 학습 fold 안에서 묶음 단위 3-fold 로 규제 세기 선택
                gid = np.array([hash((src[i], L[i, 0], L[i, 1])) % 3 for i in np.where(tr)[0]]); best = None
                for c in (1e-4, 1e-3, 1e-2, 1e-1, 1):
                    e = np.zeros((tr.sum(), 2))
                    for g in range(3): e[gid == g] = huber(Ftr[gid != g], L[tr][gid != g], c)(Ftr[gid == g]) - L[tr][gid == g]
                    s = np.hypot(e[:, 0].std() / .05, e[:, 1].std() / .01)
                    if best is None or s < best[0]: best = (s, c)
                l = best[1]; lam_used.append(l)
            P[te] = huber(Ftr, L[tr], l)(Fte)
        E = P - L; C = np.cov(E.T)
        if Ci0 is None: Ci0 = np.linalg.inv(C)
        mis = sum(not np.allclose(labs[np.einsum('ij,jk,ik->i', labs - p, Ci0, labs - p).argmin()], y) for p, y in zip(P, L))
        W = np.vstack([E[v] - E[v].mean(0) for v in grp]); rep = np.sqrt((W ** 2).sum(0) / sum(len(v) - 1 for v in grp))
        ok = int(((abs(E[:, 0]) <= .045) & (abs(E[:, 1]) <= .01)).sum())
        rows.append([name, Ftr.shape[1], f"{E[:,0].std(ddof=1):.5f}", f"{E[:,1].std(ddof=1):.5f}", f"{abs(E[:,0]).max():.4f}", f"{abs(E[:,1]).max():.4f}",
                     f"{rep[0]:.5f}", f"{rep[1]:.5f}", mis, f"{ok}/{len(L)}", ",".join(map(str, lam_used))])
        print(rows[-1])
    OUT.mkdir(exist_ok=True)
    with (OUT / "결과.tsv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["QR 표현 (+바4+나사8)", "입력수", "σY", "σT", "최대|eY|", "최대|eT|", "반복σY", "반복σT", "다른라벨더가까움", "동시통과", "선택된규제"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
