"""역순 격자(rig_grid_rev)로 라벨 보정(L2)이 맞았는지 확인한다.

분석용 Python 이다. 값 7개 측정과 모델 예측은 C++ 도구(experiments/hye/00000_final_Y_T_ssafy, final00_tool measure / predict)가 한다.

질문
  정방향 격자(rig_grid, 11-12)에서 잰 "명령과 실제 자세의 차이" d 가 스테이지 위치 때문인가, 촬영 순서(시간·직전 이동) 때문인가.
  위치 때문이라면 같은 자세를 반대 순서로 찍어도 d 가 같아야 하고, 정방향에서 만든 보정값을 역순 사진에 옮겨 써도 맞아야 한다.

방법
  1. 영상 자세: 앞 실험(20261113_0950)과 똑같이 rig_1st 축 사진으로 구한 J(값 5개 × Y,T)로 사진마다 (visY, visT) 를 푼다.
     정방향의 영점은 장비 모듈의 센터(rig_1st/center.bmp). 역순은 장비 모듈 자료로는 1698장 모두 바 측정에서 거부되므로
     역순 센터로 다시 만든 기준 자료(make_rev_assets.py)로 재고, 영점은 역순 센터다(README: 11-12 센터 자리를 되찾은 곳).
     정방향도 자기 세트(grid) 센터로 다시 맞춘 값을 함께 본다.
  2. 같은 자세(같은 파일 이름) 끼리 정방향 d 와 역순 d 를 비교한다.
  3. 정방향에서 만든 보정값을 역순 명령에 더해 라벨을 만들고, 역순 영상 자세와의 남는 차이를 본다.
       C0 명령 그대로 / C1 정방향 세트 상수 / C2 정방향 줄+칸 덧셈 모형(스캔 방향 구분) / C3 정방향 같은 자세의 d 그대로(표)
     참고로 역순 사진 자신으로 만든 L2(leave-one-out, 앞 실험과 같은 방식)도 본다. 이것은 옮겨 쓴 것이 아니라 상한이다.
  4. 장비 모델(1959장 L2 학습, 역순 사진은 처음 봄)의 예측을 영상 자세, 옮긴 라벨, 명령과 비교한다.

입력: 결과/촬영목록_rev.tsv, 결과/features7_rev.tsv, 결과/predictions_rev.tsv, 앞 실험의 결과 폴더
출력(결과/): 역순비교.tsv, 평가요약.txt, 그림_역순비교.png
실행: 저장소 루트에서  python "<이 폴더>/evaluate_rev.py"
"""
import csv, importlib.util, pathlib
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent / "결과"
PREV = HERE.parents[1] / "20261113_0950_우리장비사진거부_장비기준자료와라벨보정"
TOL_Y, TOL_T = 0.045, 0.010

spec = importlib.util.spec_from_file_location("lc", PREV / "실험" / "label_correction.py"); lc = importlib.util.module_from_spec(spec); spec.loader.exec_module(lc)


def tsv(p):
    return list(csv.DictReader(p.open(encoding="utf-8"), delimiter="\t"))


def main():
    log = []
    def say(*a):
        s = " ".join(str(x) for x in a); print(s); log.append(s)

    # ---- 정방향(앞 실험) 다시 계산: J 와 d
    f, failed_f, _ = lc.load()
    Xf, cf, stf, kf, secf = f["X"], f["cmd"], f["set"], f["kind"], f["sec"]
    one = (kf == "shot") & (stf == "rig_1st")
    ys, ts = one & (cf[:, 1] == 0), one & (cf[:, 0] == 0)
    J = np.c_[lc.slope(cf[ys, 0], Xf[ys][:, lc.POSE]), lc.slope(cf[ts, 1], Xf[ts][:, lc.POSE])]
    visf = np.linalg.lstsq(J, Xf[:, lc.POSE].T, rcond=None)[0].T
    prev = {int(r["id"]): r for r in tsv(PREV / "결과" / "라벨보정.tsv")}
    chk = max(abs(visf[i, 1] - float(prev[int(k)]["visT"])) for i, k in enumerate(f["id"]))
    say(f"앞 실험 영상 자세 재현: 최대 차이 T {chk*1e3:.4f} mdeg")
    gridf = (kf == "shot") & (stf == "rig_grid")
    fid = {pathlib.PurePosixPath(fn).name: i for i, fn in enumerate(f["file"]) if gridf[i]}
    cen_f = np.where((stf == "rig_grid") & (kf == "center"))[0][0]

    # ---- 역순
    shots = tsv(OUT / "촬영목록_rev.tsv"); feat = {int(r["id"]): r for r in tsv(OUT / "features7_rev.tsv")}
    bad = [(r["file"], feat[int(r["id"])]["error"]) for r in shots if feat[int(r["id"])]["measured"] != "1"]
    shots = [r for r in shots if feat[int(r["id"])]["measured"] == "1"]
    say(f"역순 측정 성공 {len(shots)}장, 실패 {len(bad)}장: {bad}")
    rid = np.array([int(r["id"]) for r in shots]); kr = np.array([r["kind"] for r in shots]); secr = np.array([int(r["seconds"]) for r in shots])
    cr = np.array([[float(r["Y_cmd"]), float(r["T_cmd"])] for r in shots]); fr = [r["file"] for r in shots]
    Xr = np.array([[float(feat[i][n]) for n in lc.NAMES] for i in rid])
    # 측정 기준 바꾸기: 역순 값은 역순 센터 기준으로 쟀다. 장비 모듈 측정 기준(rig_1st/center.bmp) 값 = 역순 기준 값 + 기준 차이.
    # 기준 차이는 센터 계열 18장을 두 기준으로 모두 재서 평균한 상수다(rev_center_offset.py). 라벨 영점이 아니다.
    off = np.loadtxt(OUT / "측정기준차이_값5개.txt")
    Xr[:, lc.POSE] += off
    zoff = np.linalg.lstsq(J, off, rcond=None)[0]
    say(f"측정 기준 차이(값 5개) {np.round(off, 4)} → 자세로 Y {zoff[0]*1e3:+.1f} µm, T {zoff[1]*1e3:+.2f} mdeg")
    visr = np.linalg.lstsq(J, Xr[:, lc.POSE].T, rcond=None)[0].T
    res = Xr[:, lc.POSE] - visr @ J.T
    sh = kr == "shot"; rc = kr != "shot"
    say("자세 2개로 설명 안 되는 값 (px rms) QR, 나사1~4: 역순", np.round(np.sqrt((res[sh] ** 2).mean(0)), 3),
        " 정방향", np.round(np.sqrt(((Xf[gridf][:, lc.POSE] - visf[gridf] @ J.T) ** 2).mean(0)), 3))
    say("역순 센터 명령(0, 0) 사진 18장의 영상 자세 (절대 영점 기준, 분, Y µm, T mdeg):")
    for k in np.where(rc)[0]: say(f"  {secr[k]/60:5.1f}  {visr[k,0]*1e3:+6.1f}  {visr[k,1]*1e3:+6.1f}")

    # 같은 자세끼리 짝
    pair = np.array([fid.get(pathlib.PurePosixPath(fn).name, -1) for fn in fr])
    S = np.where(sh & (pair >= 0))[0]; P = pair[S]
    assert np.allclose(cr[S], cf[P]), "명령 불일치"
    say(f"짝지은 자세 {len(S)}개 (정방향 grid {gridf.sum()}장)")
    dR = visr[S] - cr[S]; dF = visf[P] - cf[P]
    # 세트 영점 = 그 세트 사진 전체로 구한 "명령 → 절대 영점" 상수(보고서의 절대 영점 정의, L1 과 같은 방식)
    kR = np.median(dR, 0)
    say(f"세트 영점(사진 전체 회귀 상수, 영상 − 명령의 중앙값): 정방향 grid Y {np.median(dF[:,0])*1e3:+.1f} µm T {np.median(dF[:,1])*1e3:+.1f} mdeg,"
        f" 역순 Y {kR[0]*1e3:+.1f} µm T {kR[1]*1e3:+.1f} mdeg")

    # 스캔 방향·줄 안 순서
    def scan(cmd, sec, idx):
        direction = np.zeros(len(idx), int); order = np.zeros(len(idx), int)
        for t in sorted(set(cmd[idx, 1])):
            m = np.where(cmd[idx, 1] == t)[0]; m = m[np.argsort(sec[idx][m], kind="stable")]
            order[m] = np.arange(len(m)); direction[m] = 1 if cmd[idx[m[-1]], 0] > cmd[idx[m[0]], 0] else -1
        return direction, order
    dirR, ordR = scan(cr, secr, S); dirF, ordF = scan(cf, secf, P)
    say(f"스캔 방향이 정방향과 반대인 자세 {(dirR != dirF).sum()} / {len(S)}")

    # ---- 2. 같은 자세의 d 비교
    say("\n같은 자세의 d = 영상 − 명령 (T mdeg, Y µm)")
    for nm, a, sc in (("T", 1, 1e3), ("Y", 0, 1e3)):
        say(f"  {nm}: 정방향 중앙 {np.median(dF[:,a])*sc:+6.1f} 표준편차 {dF[:,a].std()*sc:5.1f} | 역순 중앙 {np.median(dR[:,a])*sc:+6.1f} 표준편차 {dR[:,a].std()*sc:5.1f}"
            f" | 상관 {np.corrcoef(dF[:,a], dR[:,a])[0,1]:.2f} | 차이(역순−정방향) 중앙 {np.median(dR[:,a]-dF[:,a])*sc:+6.1f} 표준편차 {(dR[:,a]-dF[:,a]).std()*sc:5.1f}")
    rows = sorted(set(np.round(cr[S, 1], 2)))
    rmR = np.array([dR[np.round(cr[S, 1], 2) == t, 1].mean() for t in rows]); rmF = np.array([dF[np.round(cr[S, 1], 2) == t, 1].mean() for t in rows])
    say(f"  T 줄 평균 41개: 상관 {np.corrcoef(rmR, rmF)[0,1]:.2f}, 정방향 흩어짐 {rmF.std()*1e3:.1f}, 역순 흩어짐 {rmR.std()*1e3:.1f},"
        f" 차이의 흩어짐(상수 뺀 뒤) {(rmR-rmF-(rmR-rmF).mean()).std()*1e3:.1f} mdeg")
    cols = sorted(set(np.round(cr[S, 0], 2)))
    cmR = np.array([dR[np.round(cr[S, 0], 2) == y, 0].mean() for y in cols]); cmF = np.array([dF[np.round(cr[S, 0], 2) == y, 0].mean() for y in cols])
    say(f"  Y 칸 평균 41개: 상관 {np.corrcoef(cmR, cmF)[0,1]:.2f}, 정방향 흩어짐 {cmF.std()*1e3:.1f}, 역순 흩어짐 {cmR.std()*1e3:.1f},"
        f" 차이의 흩어짐(상수 뺀 뒤) {(cmR-cmF-(cmR-cmF).mean()).std()*1e3:.1f} µm")
    # 시간 성분: 차이를 두 세트의 촬영 시각으로
    for nm, a in (("T", 1), ("Y", 0)):
        diff = dR[:, a] - dF[:, a]
        for vn, v in (("역순 시각(분)", secr[S] / 60), ("정방향 grid 시각(분)", (secf[P] - secf[gridf].min()) / 60)):
            c = np.polyfit(v, diff, 1); r2 = 1 - ((diff - np.polyval(c, v)) ** 2).sum() / ((diff - diff.mean()) ** 2).sum()
            say(f"  차이 {nm} 대 {vn}: {c[0]*1e3:+.2f} /분, R² {r2:.2f}")
    # 줄 첫 사진(큰 T 이동 직후) 효과
    for nm, a in (("T", 1), ("Y", 0)):
        say(f"  줄 안 순서 0번(T 이동 직후)과 나머지의 d 평균 차: 정방향 {(dF[ordF==0,a].mean()-dF[ordF>0,a].mean())*1e3:+.1f}, 역순 {(dR[ordR==0,a].mean()-dR[ordR>0,a].mean())*1e3:+.1f} ({nm})")

    # ---- 3. 정방향 보정값을 옮겨 쓴다
    gF = np.where(gridf)[0]
    dirG, _ = scan(cf, secf, gF)
    rowG = np.round(cf[gF, 1] * 100).astype(int); colG = np.round(cf[gF, 0] * 100).astype(int) * 10 + dirG
    dG = visf[gF] - cf[gF]
    def additive_fit(y, groups):
        cols_ = []; keys = []
        for gi, g in enumerate(groups):
            for v in sorted(set(g)): cols_.append((g == v).astype(float)); keys.append((gi, v))
        A = np.array(cols_).T; b = np.linalg.lstsq(A, y, rcond=None)[0]
        return dict(zip(keys, b))
    corr = {}
    corr["C0 명령"] = np.zeros((len(S), 2))
    corr["C1 정방향 상수"] = np.tile(np.median(dG, 0), (len(S), 1))
    for nm, use_dir in (("C2 정방향 줄+칸(방향 구분)", True), ("C2b 정방향 줄+칸(방향 무시)", False)):
        c2 = np.zeros((len(S), 2)); colk = colG if use_dir else np.round(cf[gF, 0] * 100).astype(int) * 10
        rowR = np.round(cr[S, 1] * 100).astype(int); colR = np.round(cr[S, 0] * 100).astype(int) * 10 + (dirR if use_dir else 0)
        for a in range(2):
            b = additive_fit(dG[:, a], [rowG, colk])
            c2[:, a] = [b[(0, r)] + b[(1, c)] for r, c in zip(rowR, colR)]
        corr[nm] = c2
    corr["C3 정방향 같은 자세 d"] = dF.copy()
    # 상한: 역순 자기 L2 (leave-one-out, 앞 실험과 같은 덧셈 모형)
    own = np.zeros((len(S), 2))
    rowR = np.round(cr[S, 1] * 100).astype(int); colR = np.round(cr[S, 0] * 100).astype(int) * 10 + dirR
    for a in range(2): own[:, a] = lc.loo_additive(dR[:, a], [rowR, colR])[0]
    corr["참고 역순 자기 L2(LOO)"] = own

    def score(e):
        okT = np.abs(e[:, 1]) <= TOL_T; okY = np.abs(e[:, 0]) <= TOL_Y
        return (f"중앙 T {np.median(e[:,1])*1e3:+6.1f} Y {np.median(e[:,0])*1e3:+6.1f} | 표준편차 T {e[:,1].std()*1e3:5.2f} Y {e[:,0].std()*1e3:5.1f}"
                f" | |T|>10 {int((~okT).sum()):4d}  |Y|>45 {int((~okY).sum()):3d}  둘 다 통과 {int((okT&okY).sum()):4d}/{len(e)}")
    say("\n보정 라벨(명령 + 보정값)과 역순 영상 자세의 남는 차이 e = 영상 − 라벨 (T mdeg, Y µm)")
    say(" [가) 절대 영점 기준 그대로: 정방향 보정값을 상수까지 그대로 옮김]")
    labels = {}
    for nm, c in corr.items():
        labels[nm] = cr[S] + c; say(f"  {nm:24s} {score(visr[S] - labels[nm])}")
    say(" [나) 세트 영점은 각 세트 사진 전체 회귀로 구하고(역순 상수 = 역순 사진 1679장), 정방향에서는 상수를 뺀 모양만 옮김]")
    cG = np.median(dG, 0)
    for nm, c in corr.items():
        if nm.startswith("참고"): lab_ = cr[S] + c                      # 자기 L2 는 이미 자기 상수를 담고 있다
        else: lab_ = cr[S] + kR + (c - cG if not nm.startswith("C0") else 0)
        labels[nm + " (세트 영점)"] = lab_
        say(f"  {nm:24s} {score(visr[S] - lab_)}")

    # ---- 3b. 다시 찍은 센터(약 2분마다 실제로 잰 시간 흐름)를 d 에서 빼고 같은 자세끼리 비교한다.
    #  (한 세트 안에서는 T 줄 순서 = 촬영 순서라 T 위치 곡선과 시간 흐름을 자료만으로 가를 수 없다. 그래서 잰 시간 흐름만 쓴다)
    # 다시 찍은 센터로 시간 효과를 쟀다면: 센터 사진의 영상 자세를 시각으로 이어 정방향·역순 d 에서 빼고 같은 자세끼리 비교
    def center_track(vis, sec, kind, st_mask):
        k = np.where(st_mask & (kind != "shot"))[0]; o = np.argsort(sec[k]); return sec[k][o].astype(float), vis[k][o]
    cfT, cfV = center_track(visf, secf, kf, stf == "rig_grid"); crT, crV = center_track(visr, secr, kr, np.ones(len(kr), bool))
    say("  다시 찍은 센터(약 2분마다)를 시각으로 이어 d 에서 뺐을 때 같은 자세 비교:")
    for nm, a in (("T", 1), ("Y", 0)):
        aF = dF[:, a] - np.interp(secf[P].astype(float), cfT, cfV[:, a]); aR = dR[:, a] - np.interp(secr[S].astype(float), crT, crV[:, a])
        tol = TOL_T if a == 1 else TOL_Y
        e0 = aR - np.median(aR); e1 = (aR - aF) - np.median(aR - aF)
        say(f"    {nm}: 같은 자세 상관 {np.corrcoef(aF, aR)[0,1]:.2f} (빼기 전 {np.corrcoef(dF[:,a], dR[:,a])[0,1]:.2f}).  역순 남는 차이(상수 뺌): 표 없이 표준편차 {e0.std()*1e3:5.1f} 허용 밖 {int((abs(e0)>tol).sum())}"
            f" → 정방향 같은 자세 값(시간 흐름 뺀 것)을 옮겨 빼면 {e1.std()*1e3:5.1f} 허용 밖 {int((abs(e1)>tol).sum())}")
    # ---- 4. 모델 예측 (역순 센터 기준 자료로 낸 값이라 측정 기준 차이를 자세로 푼 zoff 를 더한다. 바 기울기 2개의 기준 차이는 옮기지 못해 근사)
    if not (OUT / "predictions_rev.tsv").exists():
        say("\n(모델 예측 파일이 아직 없음)"); (OUT / "평가요약.txt").write_bytes(("\n".join(log) + "\n").encode("utf-8")); return
    pred = {int(r["id"]): r for r in tsv(OUT / "predictions_rev.tsv")}
    for r in pred.values():
        if r["valid"] == "1": r["Y"], r["T"] = str(float(r["Y"]) + zoff[0]), str(float(r["T"]) + zoff[1])
    okp = np.array([i in pred and pred[i]["valid"] == "1" for i in rid[S]])
    say(f"\n장비 모델 예측 (공개 API, 1959장 L2 학습, 역순 사진은 처음 봄): 측정 성립 {okp.sum()}/{len(S)}")
    py = np.array([[float(pred[i]["Y"]), float(pred[i]["T"])] if (i in pred and pred[i]["valid"] == "1") else [np.nan, np.nan] for i in rid[S]])
    m = okp
    say(f"  학습 범위 밖 표시 {sum(pred[i]['outside_training_range'] == '1' for i in rid[S] if i in pred)}장")
    say(" 비교 대상별 오차 e = 예측 − 기준 (절대 영점 기준)")
    say(f"  {'영상 자세(같은 사진, 값 5개)':24s} {score(py[m] - visr[S][m])}")
    for nm in ("C0 명령", "C0 명령 (세트 영점)", "C2 정방향 줄+칸(방향 구분) (세트 영점)", "참고 역순 자기 L2(LOO)"):
        say(f"  {nm:24s} {score(py[m] - labels[nm][m])}")
    # 바 기울기 변화 (카메라 숙임 영향 확인)
    say(f"\n바 기울기 (px/1000px) 같은 자세 평균 차(역순−정방향): 위 {np.mean(Xr[S,1]-Xf[P,1]):+.3f}, 아래 {np.mean(Xr[S,2]-Xf[P,2]):+.3f}"
        f"  (정방향 사진 간 표준편차 위 {Xf[P,1].std():.3f}, 아래 {Xf[P,2].std():.3f})")

    # 저장
    with (OUT / "역순비교.tsv").open("w", encoding="utf-8", newline="") as fo:
        w = csv.writer(fo, delimiter="\t")
        w.writerow(["id", "file", "Y_cmd", "T_cmd", "seconds", "scan_dir", "row_order", "visY_rev", "visT_rev", "visY_fwd", "visT_fwd", "Y_C2", "T_C2", "Y_C3", "T_C3", "predY", "predT"])
        for n, k in enumerate(S):
            w.writerow([rid[k], fr[k], f"{cr[k,0]:.2f}", f"{cr[k,1]:.2f}", secr[k], dirR[n], ordR[n], f"{visr[k,0]:.5f}", f"{visr[k,1]:.5f}", f"{visf[P[n],0]:.5f}", f"{visf[P[n],1]:.5f}",
                        f"{labels['C2 정방향 줄+칸(방향 구분)'][n,0]:.5f}", f"{labels['C2 정방향 줄+칸(방향 구분)'][n,1]:.5f}", f"{labels['C3 정방향 같은 자세 d'][n,0]:.5f}", f"{labels['C3 정방향 같은 자세 d'][n,1]:.5f}",
                        f"{py[n,0]:.5f}", f"{py[n,1]:.5f}"])
    (OUT / "평가요약.txt").write_bytes(("\n".join(log) + "\n").encode("utf-8"))

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Malgun Gothic", "sans-serif"]; plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(2, 2, figsize=(15, 9))
    a = ax[0, 0]; a.plot(rows, rmF * 1e3, "o-", color="#d03b30", ms=4, label="정방향 rig_grid (11-12)"); a.plot(rows, rmR * 1e3, "s-", color="#1d4ed8", ms=4, label="역순 rig_grid_rev (11-13)")
    a.set_xlabel("T 명령 (도)"); a.set_ylabel("줄 평균 영상 T − 명령 T (mdeg)"); a.set_title("T 차이: 같은 자세, 반대 순서"); a.legend(); a.grid(alpha=.3)
    a = ax[0, 1]; a.plot(cols, cmF * 1e3, "o-", color="#d03b30", ms=4, label="정방향"); a.plot(cols, cmR * 1e3, "s-", color="#1d4ed8", ms=4, label="역순")
    a.set_xlabel("Y 명령 (mm)"); a.set_ylabel("칸 평균 영상 Y − 명령 Y (µm)"); a.set_title("Y 차이: 같은 자세, 반대 순서"); a.legend(); a.grid(alpha=.3)
    a = ax[1, 0]; a.plot(secr[S] / 60, (dR[:, 1] - dF[:, 1]) * 1e3, ".", ms=2, color="#555"); a.set_xlabel("역순 촬영 시각 (분)"); a.set_ylabel("역순 d − 정방향 d, T (mdeg)")
    a.set_title("같은 자세의 T 차이를 역순 시각 순으로"); a.grid(alpha=.3)
    a = ax[1, 1]
    for nm, c in (("C0 명령", "#8a94a6"), ("C1 정방향 상수", "#e07b00"), ("C2 정방향 줄+칸(방향 구분)", "#0f7b55"), ("참고 역순 자기 L2(LOO)", "#1d4ed8")):
        a.hist((visr[S, 1] - labels[nm][:, 1]) * 1e3, bins=np.arange(-50, 40.5, 1), histtype="step", lw=1.6, color=c, label=nm)
    a.axvspan(-10, 10, color="#0f7b55", alpha=.07); a.set_xlabel("역순 영상 T − 라벨 T (mdeg), 띠는 ±10"); a.set_ylabel("사진 수"); a.set_title("라벨별 남는 T 차이 (역순 사진)"); a.legend(); a.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(OUT / "그림_역순비교.png", dpi=110)


if __name__ == "__main__":
    main()
