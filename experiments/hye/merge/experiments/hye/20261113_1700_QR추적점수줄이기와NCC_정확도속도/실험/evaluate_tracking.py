"""QR 값(값 7개 중 1개)을 만드는 방법을 바꿨을 때 정확도를 비교한다. 속도는 trackx.exe time 결과(결과/시간.txt)를 함께 적는다.

QR 값 후보
  정밀 맞춤 N점: 지금 방식(LK → 점마다 17×17 스플라인 정밀 맞춤)에서 536점 중 N점만 평균. 점은 QR 마다 고르게(id 순서로 같은 간격) 고른다.
                 N = 536, 268, 134, 67, 33, 15, 9, 3. 67점, 15점은 무작위로 20번 뽑은 결과도 본다.
  LK N점      : 정밀 맞춤 없이 LK 광학 흐름까지만
  NCC 정수    : QR 3개 각각 통째로 NCC 정수 이동의 평균(가로)
  NCC 포물선  : 위에 상관 봉우리 3점 포물선 맞춤으로 소수 이동을 더한 것
입력: 실험/trackx 로 뽑은 점별 이동(rig.bin, rev.bin: 사진마다 NCC 정수·포물선 3×2, LK 630×2, 정밀 맞춤 630×2, 유효 630).
      역순 사진도 QR 추적은 원래 기준 자료(rig_1st/center.bmp)로 바로 잰다(세로 14 px 는 탐색 범위 22 px 안). 나머지 값 6개는 측정 기준을 옮긴 값.
학습·평가 (20261113_1510 robustness.py 와 같은 설정, A = ssafy 방식)
  학습: 11-12 개발 사진(rig_1st, rig_grid), 라벨 L2. 값 7개 Huber 직선식(QR 값만 후보로 바꿈)
  공통 시험 728장, 처음 보는 날(11-13 역순 1679장), 폴드 검정(자세 묶음 5조각 10회, 개발 2910장). 정답은 data/labels/rig_20261113_avg
출력: 결과/결과.tsv, 결과/요약.txt
실행: 저장소 루트에서  python "<이 폴더>/evaluate_tracking.py" <rig.bin> <rev.bin>
"""
import csv, json, pathlib, sys
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent; ROOT = HERE.parents[3]; OUT = HERE.parent / "결과"
S2 = ROOT / "experiments/hye/20261113_1345_역순격자로라벨보정확인_정방향보정값옮겨적용/결과/ssafy2"
L2F = ROOT / "experiments/hye/20261113_0950_우리장비사진거부_장비기준자료와라벨보정/결과/labels_L2.tsv"
F7 = ROOT / "experiments/hye/20261113_0950_우리장비사진거부_장비기준자료와라벨보정/결과/features7.tsv"
PTS = ROOT / "experiments/hye/00000_final_Y_T_ssafy/assets/points.json"
NAMES = ["qr_dx_mean", "bar_tilt_upper", "bar_tilt_lower", "screw1_x", "screw2_x", "screw3_x", "screw4_x"]
TY, TT, SEED = .045, .010, 20261113
W = 12 + 1260 + 1260 + 630


def tsv(p):
    return list(csv.DictReader(p.open(encoding="utf-8"), delimiter="\t"))


def fit(F, Y):
    mean = F.mean(0); sc = np.maximum(F.std(0), 1e-6); A = np.c_[np.ones(len(F)), (F - mean) / sc]
    pen = np.eye(A.shape[1]) * (1e-7 * len(F)); pen[0, 0] = 0; w = np.ones(len(F))
    for _ in range(6):
        Aw = A * w[:, None]; B = np.linalg.lstsq(Aw.T @ A + pen, Aw.T @ Y, rcond=None)[0]; P = A @ B
        r = np.maximum(abs(P[:, 0] - Y[:, 0]) / TY, abs(P[:, 1] - Y[:, 1]) / TT); w = np.minimum(1, 1 / np.maximum(r, 1e-12)); w *= len(w) / w.sum()
    return lambda G: np.c_[np.ones(len(G)), (G - mean) / sc] @ B


def ok(E):
    return (abs(E[:, 0]) <= TY) & (abs(E[:, 1]) <= TT)


def load_bin(path):
    a = np.fromfile(path, np.float32); ids = [l.strip() for l in open(str(path) + ".ids", encoding="utf-8") if l.strip()]
    return dict(zip(ids, a.reshape(len(ids), W)))


def main():
    OUT.mkdir(exist_ok=True); log = []; rows = []
    def say(*a):
        s = " ".join(str(x) for x in a); print(s, flush=True); log.append(s)
    T = {**load_bin(sys.argv[1]), **load_bin(sys.argv[2])}
    pts = json.loads(PTS.read_text(encoding="utf-8"))["points"]
    sel = np.array([int(p[3]) for p in pts], bool); reg = np.array([int(p[2]) for p in pts])
    sel_ids = [np.where(sel & (reg == k))[0] for k in range(3)]

    F = {r["id"]: r for r in tsv(S2 / "features7.tsv")}; L = tsv(S2 / "labels.tsv"); SP = {r["id"]: r for r in tsv(S2 / "splits.tsv")}
    l2 = {r["id"]: (float(r["Y_abs"]), float(r["T_abs"])) for r in tsv(L2F)}
    ids = [r["id"] for r in L]; src = np.array([r["source"] for r in L])
    X6 = np.array([[float(F[i][n]) for n in NAMES] for i in ids])
    Yavg = np.array([[float(r["Y_abs"]), float(r["T_abs"])] for r in L]); YL2 = np.array([l2.get(i, (np.nan, np.nan)) for i in ids])
    final = np.array([SP[i]["final"] == "1" for i in ids]); dev = ~final
    cv = [np.array([int(SP[i][f"cv{k}"]) for i in ids]) for k in range(10)]
    day1 = src != "rig_grid_rev"; rev = src == "rig_grid_rev"; poolA = dev & day1
    D = np.array([T[i] for i in ids]); ncc_int = D[:, 0:6:2]; ncc_sub = D[:, 6:12:2]
    LK = D[:, 12:12 + 1260:2]; GN = D[:, 12 + 1260:12 + 2520:2]; VAL = D[:, 12 + 2520:] > .5

    # 검산: 정밀 맞춤 536점 평균 = 값 7개의 QR 값
    f7 = {r["id"]: r for r in tsv(F7)}
    q536 = GN[:, sel].mean(1)
    d_day1 = max(abs(q536[n] - float(f7[i]["qr_dx_mean"])) for n, i in enumerate(ids) if day1[n])
    d_rev = abs(q536[rev] - X6[rev, 0])
    say(f"검산: 11-12 사진의 536점 평균과 기존 값 최대 차이 {d_day1:.2e} px / 역순: 직접 잰 값 − 측정 기준을 옮긴 값 평균 {np.mean(q536[rev]-X6[rev,0]):+.4f} px, 흔들림 {np.std(q536[rev]-X6[rev,0]):.4f} px")
    say(f"유효하지 않은 선택점이 있는 사진: {int((~VAL[:, sel]).any(1).sum())}장")

    def even(N):
        m = np.zeros(630, bool)
        for k in range(3):
            want = max(1, int(round(N * len(sel_ids[k]) / 536)))
            m[sel_ids[k][[j * len(sel_ids[k]) // want for j in range(want)]]] = True
        return m
    rng = np.random.default_rng(SEED)
    def rand(N):
        m = np.zeros(630, bool)
        for k in range(3):
            want = max(1, int(round(N * len(sel_ids[k]) / 536))); m[rng.choice(sel_ids[k], want, replace=False)] = True
        return m

    def score(q, name, kind, n_points):
        X = X6.copy(); X[:, 0] = q
        m = fit(X[poolA], YL2[poolA])
        Et = m(X[final]) - Yavg[final]; Er = m(X[rev]) - Yavg[rev]
        fails = []
        for k in range(10):
            E = np.full((len(ids), 2), np.nan)
            for j in range(5):
                te = dev & (cv[k] == j); tr = poolA & (cv[k] != j); E[te] = fit(X[tr], YL2[tr])(X[te]) - Yavg[te]
            fails.append(int((~ok(E[dev])).sum()))
        noise = np.sqrt(np.mean((q - q536) ** 2))
        row = dict(방식=name, 종류=kind, 점수=n_points, QR값_536점과_rms차_px=round(noise, 4),
                   공통시험_통과=round(ok(Et).mean() * 100, 2), 공통시험_σY_um=round(Et[:, 0].std() * 1e3, 2), 공통시험_σT_mdeg=round(Et[:, 1].std() * 1e3, 2),
                   처음보는날_통과=round(ok(Er).mean() * 100, 2), 처음보는날_σY_um=round(Er[:, 0].std() * 1e3, 2), 처음보는날_σT_mdeg=round(Er[:, 1].std() * 1e3, 2),
                   폴드_탈락평균=round(np.mean(fails), 1))
        rows.append(row); return row

    say("\n방식 | 점 | QR값 536점과 rms 차 px | 공통 시험 728 통과 σY σT | 처음 보는 날 1679 통과 σY σT | 폴드 탈락(2910장, 10회 평균)")
    def show(r):
        say(f"  {r['방식']:16s} {r['점수']:>4} | {r['QR값_536점과_rms차_px']:.4f} | {r['공통시험_통과']:6.2f}% {r['공통시험_σY_um']:5.1f} {r['공통시험_σT_mdeg']:4.2f} | "
            f"{r['처음보는날_통과']:6.2f}% {r['처음보는날_σY_um']:5.1f} {r['처음보는날_σT_mdeg']:4.2f} | {r['폴드_탈락평균']:5.1f}")
    for N in (536, 268, 134, 67, 33, 15, 9, 3):
        m = even(N); show(score(GN[:, m].mean(1), f"정밀 맞춤 {N}", "고른 간격", int(m.sum())))
    for N in (67, 15):
        res = [score(GN[:, rand(N)].mean(1), f"정밀 맞춤 {N} 무작위", "무작위", N) for _ in range(20)]
        for key in ("공통시험_통과", "처음보는날_통과", "폴드_탈락평균", "QR값_536점과_rms차_px"):
            v = [r[key] for r in res]; say(f"  정밀 맞춤 {N} 무작위 20회 {key}: 평균 {np.mean(v):.3f} 최저 {min(v):.3f} 최고 {max(v):.3f}")
        del rows[-20:]
        rows.append(dict(방식=f"정밀 맞춤 {N} 무작위 20회", 종류="무작위", 점수=N, QR값_536점과_rms차_px=round(np.mean([r['QR값_536점과_rms차_px'] for r in res]), 4),
                         공통시험_통과=round(np.mean([r['공통시험_통과'] for r in res]), 2), 공통시험_σY_um="", 공통시험_σT_mdeg="",
                         처음보는날_통과=round(np.mean([r['처음보는날_통과'] for r in res]), 2), 처음보는날_σY_um="", 처음보는날_σT_mdeg="",
                         폴드_탈락평균=round(np.mean([r['폴드_탈락평균'] for r in res]), 1)))
    for N in (536, 67):
        m = even(N); show(score(LK[:, m].mean(1), f"LK {N}", "LK 까지만", int(m.sum())))
    show(score(ncc_int.mean(1), "NCC 정수", "NCC", 0))
    show(score(ncc_sub.mean(1), "NCC 포물선", "NCC", 0))

    with (OUT / "결과.tsv").open("w", encoding="utf-8", newline="") as fo:
        w = csv.DictWriter(fo, list(rows[0].keys()), delimiter="\t"); w.writeheader(); w.writerows(rows)
    (OUT / "요약.txt").write_bytes(("\n".join(log) + "\n").encode("utf-8"))


if __name__ == "__main__":
    main()
