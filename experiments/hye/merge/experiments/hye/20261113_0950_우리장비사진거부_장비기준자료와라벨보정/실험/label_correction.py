"""우리 장비 사진의 절대 영점을 다시 잡고, 명령 라벨과 실제 자세의 차이를 재서 라벨을 보정한다.

분석용 Python 이다. 값 7개 측정과 학습·평가는 C++ 도구(experiments/hye/00000_final_Y_T_ssafy, final00_tool)가 한다.

입력: 결과/촬영목록.tsv (prepare_rig_inputs.py), 결과/features7.tsv (final00_tool measure, 장비 기준 자료)

1. 영상으로 잰 자세 (vision pose)
   rig_1st 의 Y축 사진 80장, T축 사진 200장에서 값 5개(QR 536점 가로 평균, 나사 4개 가로 이동)의 명령 1단위당 이동 J(5×2)를 구한다.
   사진마다 값 5개 = J·(Y, T) 를 최소제곱으로 풀어 (visY, visT) 를 낸다. 값이 모두 0 이면 (0, 0) 이므로 영점은 센터 사진(rig_1st/center.bmp)이다.
   바 기울기 2개는 자세와 무관하게 변하는 성분이 있어 자세 계산에는 쓰지 않는다(모델 입력 7개에는 그대로 들어간다).
   차이 d = (visY, visT) − 명령 이 "라벨과 실제 이동의 차이"다.

2. 라벨 세 가지
   L0 명령 그대로
   L1 절대 영점(SEMES 와 같은 방식): 세트(rig_1st, rig_grid)마다 d 의 중앙값 하나를 명령에 더한다
   L2 절대 영점 + 라벨 보정: 명령 + 같은 조건에서 찍은 다른 사진들의 d 로 구한 보정값. 자기 사진의 d 는 쓰지 않는다
      rig_grid   d = 줄(같은 T 명령 41장) 효과 + 칸(같은 Y 명령, Y 스캔 방향별) 효과 의 덧셈 모형. 그 사진을 빼고 맞춘 예측값(leave-one-out)
      rig_1st    촬영 순서로 앞뒤 5장씩(자기 제외 10장)의 d 중앙값. T축 앞 절반, Y축, T축 뒤 절반을 따로 한다

3. 보정이 진짜인지 확인: rig_1st T축 사진의 d 곡선과 rig_grid 줄 효과를 같은 T 명령에서 비교한다(서로 다른 사진, 다른 시각).

출력(결과/): 라벨보정.tsv, labels_L0.tsv, labels_L1.tsv, labels_L2.tsv, splits.tsv, features7_shots.tsv, 보정요약.txt, 그림_라벨차이.png

실행: 저장소 루트에서  python "<이 폴더>/label_correction.py"
"""
import csv, pathlib
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent / "결과"
NAMES = ["qr_dx_mean", "bar_tilt_upper", "bar_tilt_lower", "screw1_x", "screw2_x", "screw3_x", "screw4_x"]
POSE = [0, 3, 4, 5, 6]          # 자세 계산에 쓰는 값: QR 가로 평균, 나사 4개 가로
SEED = 20261112                 # SEMES 고정 검증(20261112_1835 combo.py)과 같은 씨앗
HALF = 5                        # rig_1st 이동 중앙값의 한쪽 장수


def load():
    shots = list(csv.DictReader((OUT / "촬영목록.tsv").open(encoding="utf-8"), delimiter="\t"))
    feat = {int(r["id"]): r for r in csv.DictReader((OUT / "features7.tsv").open(encoding="utf-8"), delimiter="\t")}
    failed = [r for r in shots if feat[int(r["id"])]["measured"] != "1"]
    shots = [r for r in shots if feat[int(r["id"])]["measured"] == "1"]
    shots.sort(key=lambda r: (int(r["seconds"]), int(r["id"])))
    d = dict(id=np.array([int(r["id"]) for r in shots]), set=np.array([r["set"] for r in shots]), kind=np.array([r["kind"] for r in shots]),
             seq=np.array([int(r["seq"]) for r in shots]), sec=np.array([int(r["seconds"]) for r in shots]), file=[r["file"] for r in shots],
             cmd=np.array([[float(r["Y_cmd"]), float(r["T_cmd"])] for r in shots]))
    d["X"] = np.array([[float(feat[i][n]) for n in NAMES] for i in d["id"]])
    return d, failed, feat


def slope(x, F):
    return np.linalg.lstsq(np.c_[np.ones(len(x)), x], F, rcond=None)[0][1]


def loo_additive(y, groups):
    """y ≈ 여러 범주 효과의 합. 최소제곱으로 맞추고, 사진마다 그 사진을 뺀 예측값을 돌려준다."""
    cols = []
    for g in groups:
        for v in sorted(set(g)): cols.append((g == v).astype(float))
    A = np.array(cols).T
    Q, _ = np.linalg.qr(A)                      # 열이 겹쳐도(순위 부족) 사영은 같다
    r = np.linalg.matrix_rank(A)
    U = np.linalg.svd(A, full_matrices=False)[0][:, :r]
    fit = U @ (U.T @ y); h = (U ** 2).sum(1)
    return y - (y - fit) / (1 - h), fit


def loo_running_median(y, half):
    out = np.empty_like(y)
    for i in range(len(y)):
        j = [k for k in range(max(0, i - half), min(len(y), i + half + 1)) if k != i]
        out[i] = np.median(y[j])
    return out


def make_splits(src, cmd):
    keys = [(a, round(y, 2), round(t, 2)) for a, (y, t) in zip(src, cmd)]; uk = sorted(set(keys)); gi = np.array([uk.index(k) for k in keys]); gs = np.array([k[0] for k in uk])
    rng = np.random.default_rng(SEED); final_g = np.zeros(len(uk), bool)
    for s in sorted(set(gs)):
        idx = np.where(gs == s)[0]; final_g[rng.permutation(idx)[: round(len(idx) * .2)]] = True
    final = final_g[gi]; dg = np.where(~final_g)[0]; cv, half = [], []
    for seed in range(10):
        r = np.random.default_rng(1000 + seed); f = np.full(len(uk), -1); f[r.permutation(dg)] = np.arange(len(dg)) % 5; cv.append(f[gi])
        h = np.full(len(uk), -1)
        for s in sorted(set(gs)):
            idx = np.array([g for g in dg if gs[g] == s]); p = r.permutation(idx); h[p[: len(p) // 2]] = 0; h[p[len(p) // 2:]] = 1
        half.append(h[gi])
    return final, cv, half


def main():
    d, failed, feat = load()
    X, cmd, st, kind, sec = d["X"], d["cmd"], d["set"], d["kind"], d["sec"]
    shot = kind == "shot"; one = shot & (st == "rig_1st"); grid = shot & (st == "rig_grid")
    ys, ts = one & (cmd[:, 1] == 0), one & (cmd[:, 0] == 0)
    log = []
    def say(*a):
        s = " ".join(str(x) for x in a); print(s); log.append(s)
    say(f"측정 성공 {len(d['id'])}장, 실패 {len(failed)}장:", [(r["file"], feat[int(r["id"])]["error"]) for r in failed])

    # 1. 영상으로 잰 자세
    J = np.c_[slope(cmd[ys, 0], X[ys][:, POSE]), slope(cmd[ts, 1], X[ts][:, POSE])]
    say("명령 1단위당 이동 (px): 값 순서 QR, 나사1~4")
    say("  Y 1 mm :", np.round(J[:, 0], 3)); say("  T 1 도 :", np.round(J[:, 1], 3))
    vis = np.linalg.lstsq(J, X[:, POSE].T, rcond=None)[0].T
    res = X[:, POSE] - vis @ J.T
    say("자세 2개로 설명 안 되는 값 (px rms): QR, 나사1~4 =", np.round(np.sqrt((res[shot] ** 2).mean(0)), 3))
    dif = vis - cmd

    # 2. 라벨
    L0 = cmd.copy(); L1 = cmd.copy(); L2 = cmd.copy(); const = {}
    for s in ("rig_1st", "rig_grid"):
        m = shot & (st == s); const[s] = np.median(dif[m], 0); L1[m] += const[s]
    # rig_grid: 줄 + 스캔 방향별 칸
    order = np.zeros(len(cmd), int); direction = np.zeros(len(cmd), int)
    for t in sorted(set(cmd[grid, 1])):
        m = np.where(grid & (cmd[:, 1] == t))[0]; m = m[np.argsort(sec[m], kind="stable")]
        order[m] = np.arange(len(m)); direction[m] = 1 if cmd[m[-1], 0] > cmd[m[0], 0] else -1
    row = np.round(cmd[grid, 1] * 100).astype(int); col = np.round(cmd[grid, 0] * 100).astype(int) * 10 + direction[grid]
    fit_grid = np.zeros((grid.sum(), 2))
    for a in range(2):
        loo, fit_grid[:, a] = loo_additive(dif[grid, a], [row, col]); L2[grid, a] += loo
    # rig_1st: 촬영 순서 이동 중앙값. T축 앞 절반(T<0) → Y축 → T축 뒤 절반(T>0) 순서로 찍혔다
    parts = [ts & (cmd[:, 1] < 0), ys, ts & (cmd[:, 1] > 0)]
    assert sum(p.sum() for p in parts) == one.sum()
    for p in parts:
        k = np.where(p)[0]; k = k[np.argsort(sec[k], kind="stable")]
        for a in range(2): L2[k, a] += loo_running_median(dif[k, a], HALF)

    say("\n라벨과 실제 이동의 차이 d = 영상 자세 − 명령  (T 는 mdeg = 0.001 도, Y 는 µm)")
    for nm, m in (("rig_1st T축 200장", ts), ("rig_1st Y축 80장", ys), ("rig_grid 1679장", grid)):
        say(f"  {nm:18s} T: 중앙 {np.median(dif[m,1])*1e3:+6.1f}, 표준편차 {dif[m,1].std()*1e3:5.1f}, 범위 {dif[m,1].min()*1e3:+6.1f} ~ {dif[m,1].max()*1e3:+6.1f}"
            f"   Y: 중앙 {np.median(dif[m,0])*1e3:+6.1f}, 표준편차 {dif[m,0].std()*1e3:5.1f}, 범위 {dif[m,0].min()*1e3:+6.1f} ~ {dif[m,0].max()*1e3:+6.1f}")
    say("  세트 상수(L1 에서 명령에 더하는 값): " + ", ".join(f"{s} Y {c[0]*1e3:+.1f} µm, T {c[1]*1e3:+.1f} mdeg" for s, c in const.items()))
    for nm, L in (("L0 명령", L0), ("L1 절대 영점", L1), ("L2 절대 영점+보정", L2)):
        e = vis - L
        say(f"  {nm:16s} 남는 차이 표준편차: T {e[shot,1].std()*1e3:5.2f} mdeg, Y {e[shot,0].std()*1e3:5.1f} µm,  |T 차이| > 10 mdeg {int((abs(e[shot,1]) > .010).sum())}장, |Y 차이| > 45 µm {int((abs(e[shot,0]) > .045).sum())}장 / {shot.sum()}")

    # 직선(선형) 성분이 있는지: 시간, 명령에 대한 기울기
    say("\n직선 성분 (d 를 한 변수에 직선으로 맞춘 기울기와 설명 비율 R²)")
    for nm, m in (("rig_grid", grid), ("rig_1st", one)):
        for vn, v, unit in (("촬영 시각", sec[m] / 60.0, "분"), ("T 명령", cmd[m, 1], "도"), ("Y 명령", cmd[m, 0], "mm")):
            if np.ptp(v) == 0: continue
            for a, an, sc, u in ((1, "T", 1e3, "mdeg"), (0, "Y", 1e3, "µm")):
                c = np.polyfit(v, dif[m, a], 1); r2 = 1 - ((dif[m, a] - np.polyval(c, v)) ** 2).sum() / ((dif[m, a] - dif[m, a].mean()) ** 2).sum()
                say(f"  {nm:8s} d{an} 대 {vn}: {c[0]*sc:+7.2f} {u}/{unit}, R² {r2:.2f}")

    # 3. 보정이 진짜인지: 1st T축 곡선과 grid 줄 효과
    kt = np.where(ts)[0]; curve = {round(cmd[k, 1], 2): v for k, v in zip(kt, loo_running_median(dif[kt[np.argsort(sec[kt], kind='stable')], 1], HALF))}
    kt = kt[np.argsort(sec[kt], kind="stable")]; curve = dict(zip(np.round(cmd[kt, 1], 2), loo_running_median(dif[kt, 1], HALF)))
    rows = sorted(set(np.round(cmd[grid, 1], 2))); rmean = np.array([dif[grid & (np.round(cmd[:, 1], 2) == t), 1].mean() for t in rows])
    both = [i for i, t in enumerate(rows) if t in curve]; c1 = np.array([curve[rows[i]] for i in both]); g1 = rmean[both]
    say(f"\n같은 T 명령에서 rig_1st T축 곡선과 rig_grid 줄 평균 비교 ({len(both)}개 T)")
    say(f"  상관 {np.corrcoef(c1, g1)[0,1]:.2f}.  줄 평균의 흩어짐 {g1.std()*1e3:.1f} mdeg → 1st 곡선을 뺀 뒤 {(g1 - c1 - (g1 - c1).mean()).std()*1e3:.1f} mdeg (평균 차 {(g1 - c1).mean()*1e3:+.1f} mdeg)")
    neg = [i for i, t in enumerate([rows[b] for b in both]) if t <= 0.5]
    say(f"  T ≤ +0.5 구간만: 상관 {np.corrcoef(c1[neg], g1[neg])[0,1]:.2f}, 뺀 뒤 {(g1[neg] - c1[neg] - (g1[neg] - c1[neg]).mean()).std()*1e3:.1f} mdeg")
    inrow = np.array([dif[grid & (np.round(cmd[:, 1], 2) == t), :].std(0) for t in rows])
    say(f"  rig_grid 한 줄 안의 흩어짐(중앙값): T {np.median(inrow[:,1])*1e3:.1f} mdeg, Y {np.median(inrow[:,0])*1e3:.1f} µm  ← 영상 자세의 잡음 상한")
    rc = kind != "shot"
    say("\n다시 찍은 센터(명령 0, 0)의 영상 자세: 세트 순번 T(mdeg) Y(µm)")
    for k in np.where(rc)[0]: say(f"  {st[k]:8s} {d['seq'][k]:5d}  {vis[k,1]*1e3:+6.1f}  {vis[k,0]*1e3:+6.1f}")

    # 저장
    with (OUT / "라벨보정.tsv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t"); w.writerow(["id", "set", "kind", "seq", "seconds", "file", "Y_cmd", "T_cmd", "visY", "visT", "Y_L1", "T_L1", "Y_L2", "T_L2", "scan_dir", "row_order"])
        for k in range(len(cmd)):
            w.writerow([d["id"][k], st[k], kind[k], d["seq"][k], sec[k], d["file"][k], f"{cmd[k,0]:.2f}", f"{cmd[k,1]:.2f}", f"{vis[k,0]:.5f}", f"{vis[k,1]:.5f}",
                        f"{L1[k,0]:.5f}", f"{L1[k,1]:.5f}", f"{L2[k,0]:.5f}", f"{L2[k,1]:.5f}", direction[k], order[k]])
    for nm, L in (("L0", L0), ("L1", L1), ("L2", L2)):
        with (OUT / f"labels_{nm}.tsv").open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t"); w.writerow(["id", "source", "Y_cmd", "T_cmd", "Y_abs", "T_abs"])
            for k in np.where(shot)[0]: w.writerow([d["id"][k], st[k], f"{cmd[k,0]:.2f}", f"{cmd[k,1]:.2f}", f"{L[k,0]:.5f}", f"{L[k,1]:.5f}"])
    final, cv, half = make_splits(st[shot], cmd[shot])
    with (OUT / "splits.tsv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t"); w.writerow(["id", "final"] + [f"cv{k}" for k in range(10)] + [f"half{k}" for k in range(10)])
        for n, k in enumerate(np.where(shot)[0]): w.writerow([d["id"][k], int(final[n])] + [cv[j][n] for j in range(10)] + [half[j][n] for j in range(10)])
    lines = (OUT / "features7.tsv").read_text(encoding="utf-8").splitlines(); keep = set(str(i) for i in d["id"][shot])
    (OUT / "features7_shots.tsv").write_bytes(("\n".join([lines[0]] + [l for l in lines[1:] if l.split("\t")[0] in keep]) + "\n").encode("utf-8"))
    say(f"\n평가용 사진 {shot.sum()}장 (rig_1st {one.sum()}, rig_grid {grid.sum()}), 마지막 확인 {int(final.sum())}장")
    (OUT / "보정요약.txt").write_bytes(("\n".join(log) + "\n").encode("utf-8"))

    # 그림
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Malgun Gothic", "sans-serif"]; plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(2, 2, figsize=(15, 9))
    a = ax[0, 0]; a.plot(cmd[kt, 1], dif[kt, 1] * 1e3, ".-", ms=3, lw=.6, color="#1d4ed8", label="rig_1st T축 사진 (0.01도 간격)")
    a.plot(rows, rmean * 1e3, "o", color="#d03b30", label="rig_grid 줄 평균 (41장씩)"); a.axhline(0, color="#888", lw=.8)
    a.set_xlabel("T 명령 (도)"); a.set_ylabel("영상 T − 명령 T (mdeg)"); a.set_title("T 라벨과 실제 각도의 차이: 두 세트에서 같은 곡선이 반복된다"); a.legend(); a.grid(alpha=.3)
    a = ax[0, 1]; a.plot(sec[grid] / 60, dif[grid, 1] * 1e3, ".", ms=2, color="#d03b30", label="rig_grid"); a.plot(sec[one] / 60, dif[one, 1] * 1e3, ".", ms=2, color="#1d4ed8", label="rig_1st")
    a.plot(sec[rc] / 60, vis[rc, 1] * 1e3, "k*", ms=8, label="다시 찍은 센터"); a.set_xlabel("촬영 시각 (분)"); a.set_ylabel("mdeg"); a.set_title("같은 차이를 시간 순으로"); a.legend(); a.grid(alpha=.3)
    a = ax[1, 0]; a.plot(sec[grid] / 60, dif[grid, 0] * 1e3, ".", ms=2, color="#d03b30", label="rig_grid"); a.plot(sec[one] / 60, dif[one, 0] * 1e3, ".", ms=2, color="#1d4ed8", label="rig_1st")
    a.plot(sec[rc] / 60, vis[rc, 0] * 1e3, "k*", ms=8, label="다시 찍은 센터"); a.set_xlabel("촬영 시각 (분)"); a.set_ylabel("영상 Y − 명령 Y (µm)"); a.set_title("Y 라벨과 실제 이동의 차이"); a.legend(); a.grid(alpha=.3)
    a = ax[1, 1]
    for nm, L, c in (("L0 명령", L0, "#8a94a6"), ("L1 절대 영점", L1, "#e07b00"), ("L2 절대 영점+보정", L2, "#0f7b55")):
        a.hist((vis[shot, 1] - L[shot, 1]) * 1e3, bins=np.arange(-50, 30.5, 1), histtype="step", lw=1.6, color=c, label=nm)
    a.axvspan(-10, 10, color="#0f7b55", alpha=.07); a.set_xlabel("영상 T − 라벨 T (mdeg), 연두색 띠는 허용 ±10"); a.set_ylabel("사진 수"); a.set_title("라벨별 남는 T 차이"); a.legend(); a.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(OUT / "그림_라벨차이.png", dpi=110)


if __name__ == "__main__":
    main()
