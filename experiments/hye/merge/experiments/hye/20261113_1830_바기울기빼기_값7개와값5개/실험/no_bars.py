"""바 기울기 2개를 뺀 값 5개(QR 1 + 나사 4)와 지금 값 7개의 정확도를 같은 조건으로 비교한다.

QR 값은 지금 모듈과 같은 NCC 포물선(실험 20261113_1700 / 모듈 측정 값). 학습식은 같은 Huber(combo.py fit).
설정은 20261113_1510 robustness.py 와 같다.
  A = ssafy 방식 : 11-12 개발 사진(rig_1st, rig_grid), 라벨 L2
  B = ssafy_2 방식: 11-12 + 11-13 역순 개발 사진, 라벨 avg (rig_1st 는 L2)
  공통 시험 728장, 처음 보는 날(A 만: 11-13 역순 1679장), 폴드 검정(개발 2910장, 자세 묶음 5조각 10회), 값 교란(보고서 크기)
정답은 data/labels/rig_20261113_avg. 통과 |eY| ≤ 0.045 mm 그리고 |eT| ≤ 0.01 도.
입력: <ncc 학습 자료 폴더>/features7.tsv (QR 값이 NCC 포물선인 3638장, .batch_build/ncc/s2 에서 만든 것과 같다)
출력: 결과/결과.tsv, 결과/요약.txt
실행: 저장소 루트에서  python "<이 폴더>/no_bars.py" <features7.tsv>
"""
import csv, json, pathlib, sys
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent; ROOT = HERE.parents[3]; OUT = HERE.parent / "결과"
S2 = ROOT / "experiments/hye/20261113_1345_역순격자로라벨보정확인_정방향보정값옮겨적용/결과/ssafy2"
L2F = ROOT / "experiments/hye/20261113_0950_우리장비사진거부_장비기준자료와라벨보정/결과/labels_L2.tsv"
PTS = ROOT / "experiments/hye/00000_final_Y_T_ssafy/assets/points.json"
NAMES = ["qr_dx_mean", "bar_tilt_upper", "bar_tilt_lower", "screw1_x", "screw2_x", "screw3_x", "screw4_x"]
TY, TT, SEED = .045, .010, 20261113


def tsv(p):
    return list(csv.DictReader(pathlib.Path(p).open(encoding="utf-8"), delimiter="\t"))


def fit(F, Y):
    mean = F.mean(0); sc = np.maximum(F.std(0), 1e-6); A = np.c_[np.ones(len(F)), (F - mean) / sc]
    pen = np.eye(A.shape[1]) * (1e-7 * len(F)); pen[0, 0] = 0; w = np.ones(len(F))
    for _ in range(6):
        Aw = A * w[:, None]; B = np.linalg.lstsq(Aw.T @ A + pen, Aw.T @ Y, rcond=None)[0]; P = A @ B
        r = np.maximum(abs(P[:, 0] - Y[:, 0]) / TY, abs(P[:, 1] - Y[:, 1]) / TT); w = np.minimum(1, 1 / np.maximum(r, 1e-12)); w *= len(w) / w.sum()
    return lambda G: np.c_[np.ones(len(G)), (G - mean) / sc] @ B


def ok(E):
    return (abs(E[:, 0]) <= TY) & (abs(E[:, 1]) <= TT)


def main():
    OUT.mkdir(exist_ok=True); log = []; rows = []
    def say(*a):
        s = " ".join(str(x) for x in a); print(s, flush=True); log.append(s)
    F = {r["id"]: r for r in tsv(sys.argv[1])}; L = tsv(S2 / "labels.tsv"); SP = {r["id"]: r for r in tsv(S2 / "splits.tsv")}
    l2 = {r["id"]: (float(r["Y_abs"]), float(r["T_abs"])) for r in tsv(L2F)}
    ids = [r["id"] for r in L]; src = np.array([r["source"] for r in L])
    X = np.array([[float(F[i][n]) for n in NAMES] for i in ids])
    Yavg = np.array([[float(r["Y_abs"]), float(r["T_abs"])] for r in L]); YL2 = np.array([l2.get(i, (np.nan, np.nan)) for i in ids])
    final = np.array([SP[i]["final"] == "1" for i in ids]); dev = ~final
    cv = [np.array([int(SP[i][f"cv{k}"]) for i in ids]) for k in range(10)]
    rev = src == "rig_grid_rev"
    R = {"A ssafy": (dev & ~rev, YL2), "B ssafy_2": (dev, Yavg)}
    pts = json.loads(PTS.read_text(encoding="utf-8"))["points"]
    nq = np.array([sum(1 for p in pts if int(p[2]) == k and int(p[3])) for k in range(3)])
    tilt_sd = X[dev][:, 1:3].std(0); prng = np.random.default_rng(SEED)
    CFG = {"값 7개 (지금)": [0, 1, 2, 3, 4, 5, 6], "값 5개 (바 빼기)": [0, 3, 4, 5, 6], "값 6개 (위 바만)": [0, 1, 3, 4, 5, 6], "값 6개 (아래 바만)": [0, 2, 3, 4, 5, 6]}
    say("구성 | 방식 | 공통 시험 728 통과 σY σT | 처음 보는 날 1679 통과 σY σT | 폴드 탈락(2910, 10회) | 교란 최소(QR 하나 / 나사 하나)")
    preds = {}
    for cname, cols in CFG.items():
        for rname, (pool, Y) in R.items():
            Xc = X[:, cols]; m = fit(Xc[pool], Y[pool])
            Et = m(Xc[final]) - Yavg[final]; preds[(cname, rname)] = m(Xc)
            fails = []
            for k in range(10):
                E = np.full((len(ids), 2), np.nan)
                for j in range(5):
                    te = dev & (cv[k] == j); tr = pool & (cv[k] != j); E[te] = fit(Xc[tr], Y[tr])(Xc[te]) - Yavg[te]
                fails.append(int((~ok(E[dev])).sum()))
            if rname.startswith("A"):
                Er = m(Xc[rev]) - Yavg[rev]; rv = f"{ok(Er).mean()*100:6.2f}% {Er[:,0].std()*1e3:5.1f} {Er[:,1].std()*1e3:4.2f}"; rv_pass = round(ok(Er).mean() * 100, 2)
            else:
                rv = "   해당 없음        "; rv_pass = ""
            # 교란: QR 하나 +0.3 px(그 QR 몫 1/3), 나사 하나 +0.3 px — 시험 사진에만
            qr1 = []; sc1 = []
            for k in range(3):
                Z = X[final].copy(); Z[:, 0] += .3 / 3; qr1.append(ok(m(Z[:, cols]) - Yavg[final]).mean() * 100)
            for k in range(4):
                Z = X[final].copy(); Z[:, 3 + k] += .3; sc1.append(ok(m(Z[:, cols]) - Yavg[final]).mean() * 100)
            row = dict(구성=cname, 방식=rname, 값수=len(cols), 공통시험_통과=round(ok(Et).mean() * 100, 2), 공통시험_σY_um=round(Et[:, 0].std() * 1e3, 2), 공통시험_σT_mdeg=round(Et[:, 1].std() * 1e3, 2),
                       처음보는날_통과=rv_pass, 폴드_탈락평균=round(np.mean(fails), 1), QR하나_교란=round(np.mean(qr1), 1), 나사하나_교란=round(np.mean(sc1), 1))
            rows.append(row)
            say(f"  {cname:14s} {rname:9s} | {ok(Et).mean()*100:6.2f}% {Et[:,0].std()*1e3:5.1f} {Et[:,1].std()*1e3:4.2f} | {rv} | {np.mean(fails):5.1f} | {np.mean(qr1):5.1f}% / {np.mean(sc1):5.1f}%")
    say("\n같은 사진에서 출력 차이 (값 5개 − 값 7개), 처음 보는 날 1679장 (A) / 전체 3638장 (B): 흔들림 Y µm, T mdeg (최대)")
    for rname in R:
        d = preds[("값 5개 (바 빼기)", rname)] - preds[("값 7개 (지금)", rname)]
        d = d[rev] if rname.startswith("A") else d
        say(f"  {rname}: Y {d[:,0].std()*1e3:.2f} ({abs(d[:,0]).max()*1e3:.1f}), T {d[:,1].std()*1e3:.3f} ({abs(d[:,1]).max()*1e3:.2f})")
    with (OUT / "결과.tsv").open("w", encoding="utf-8", newline="") as fo:
        w = csv.DictWriter(fo, list(rows[0].keys()), delimiter="\t"); w.writeheader(); w.writerows(rows)
    (OUT / "요약.txt").write_bytes(("\n".join(log) + "\n").encode("utf-8"))


if __name__ == "__main__":
    main()
