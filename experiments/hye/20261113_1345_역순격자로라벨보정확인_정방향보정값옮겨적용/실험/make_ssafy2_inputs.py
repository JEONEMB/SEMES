"""00000_final_Y_T_ssafy_2 학습 입력을 만든다: 정방향·역순 격자의 같은 자세 평균으로 다시 붙인 라벨. 자료 준비만 한다(학습·평가는 C++ final00_tool).

라벨 (절대 영점 = 값이 0 인 상태. 세트 영점은 세트 사진 전체 회귀, 보고서 정의)
  d = 영상 자세 − 명령, 세트 상수 k = d 의 중앙값(세트 사진 전체), d_c = d − k
  rig_grid (11-12 정방향) 1679장 : 명령 + k_정방향 + (d_c정방향 + d_c역순)/2
  rig_grid_rev (11-13 역순) 1679장: 명령 + k_역순   + (d_c정방향 + d_c역순)/2
  rig_1st 280장                   : 역순 짝이 없어 앞 실험 L2 그대로
  반대 순서로 찍은 두 사진을 평균하므로 시간에 따른 직선 흐름이 상쇄된다. 남는 라벨 잡음(라벨 − 영상 자세)은 σT 약 4.3 mdeg 다.
값 7개
  rig_1st, rig_grid: 앞 실험 features7.tsv (장비 측정 기준 rig_1st/center.bmp)
  rig_grid_rev: 역순 센터 기준으로 잰 C++ 값 + 측정 기준 차이(값 5개: rev_center_offset.py, 바 2개: bar_offset.py, 각 18장 평균)
분할 (보고서의 고정 검증과 같은 방식, lc.make_splits)
  자세 묶음 = (자료 계열, Y, T). 정방향과 역순의 같은 자세는 한 묶음이라 학습과 평가로 갈리지 않는다.
  계열(rig_1st, grid)마다 묶음 20% 를 마지막 확인으로 먼저 떼고, 나머지에서 섞은 5조각 10회(주), 계열별 절반 10회(보조).
  세트 통째로 빼기는 도구가 라벨의 source(rig_1st, rig_grid, rig_grid_rev)로 한다. rig_grid_rev 를 빼면 "11-12 로 학습 → 11-13 역순 평가" 다.
출력: 결과/ssafy2/ features7.tsv, labels.tsv, splits.tsv, 라벨표.tsv, 요약.txt
실행: 저장소 루트에서  python "<이 폴더>/make_ssafy2_inputs.py"
"""
import csv, importlib.util, pathlib
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
RES = HERE.parent / "결과"; OUT = RES / "ssafy2"
PREV = HERE.parents[1] / "20261113_0950_우리장비사진거부_장비기준자료와라벨보정"
spec = importlib.util.spec_from_file_location("lc", PREV / "실험" / "label_correction.py"); lc = importlib.util.module_from_spec(spec); spec.loader.exec_module(lc)


def tsv(p):
    return list(csv.DictReader(p.open(encoding="utf-8"), delimiter="\t"))


def main():
    OUT.mkdir(exist_ok=True); log = []
    def say(*a):
        s = " ".join(str(x) for x in a); print(s); log.append(s)
    f, _, feat_f = lc.load()
    Xf, cf, stf, kf = f["X"], f["cmd"], f["set"], f["kind"]
    one = (kf == "shot") & (stf == "rig_1st"); ys, ts = one & (cf[:, 1] == 0), one & (cf[:, 0] == 0)
    J = np.c_[lc.slope(cf[ys, 0], Xf[ys][:, lc.POSE]), lc.slope(cf[ts, 1], Xf[ts][:, lc.POSE])]
    visf = np.linalg.lstsq(J, Xf[:, lc.POSE].T, rcond=None)[0].T
    gridf = (kf == "shot") & (stf == "rig_grid")
    L2 = {int(r["id"]): (float(r["Y_abs"]), float(r["T_abs"])) for r in tsv(PREV / "결과/labels_L2.tsv")}

    name = lambda fn: pathlib.PurePosixPath(fn).name
    gi = {name(fn): i for i, fn in enumerate(f["file"]) if gridf[i]}
    shots = [r for r in tsv(RES / "촬영목록_rev.tsv") if r["kind"] == "shot"]
    nopair = [r["file"] for r in shots if name(r["file"]) not in gi]
    shots = [r for r in shots if name(r["file"]) in gi]
    say(f"역순 사진 중 정방향 짝이 없어 뺀 것 {len(nopair)}장: {nopair} (정방향 쪽이 측정 실패)")
    feat_r = {r["id"]: r for r in tsv(RES / "features7_rev.tsv")}
    off5 = np.loadtxt(RES / "측정기준차이_값5개.txt"); offb = np.loadtxt(RES / "측정기준차이_바.txt")
    off7 = np.array([off5[0], offb[0], offb[1], off5[1], off5[2], off5[3], off5[4]])
    say("역순 측정 기준 차이(값 7개, 장비 기준 = 역순 기준 + 이 값):", np.round(off7, 4))
    Xr = np.array([[float(feat_r[r["id"]][n]) for n in lc.NAMES] for r in shots]) + off7
    cr = np.array([[float(r["Y_cmd"]), float(r["T_cmd"])] for r in shots]); rid = [int(r["id"]) for r in shots]
    visr = np.linalg.lstsq(J, Xr[:, lc.POSE].T, rcond=None)[0].T

    P = np.array([gi[name(r["file"])] for r in shots])
    dF = visf[P] - cf[P]; dR = visr - cr; kF = np.median(dF, 0); kR = np.median(dR, 0)
    avg = ((dF - kF) + (dR - kR)) / 2
    labF = cf[P] + kF + avg; labR = cr + kR + avg
    say(f"세트 상수 (영상 − 명령 중앙값): 정방향 Y {kF[0]*1e3:+.1f} µm T {kF[1]*1e3:+.1f} mdeg, 역순 Y {kR[0]*1e3:+.1f} µm T {kR[1]*1e3:+.1f} mdeg")
    for nm, vis, lab in (("정방향", visf[P], labF), ("역순", visr, labR)):
        e = vis - lab; ok = (abs(e[:, 0]) <= .045) & (abs(e[:, 1]) <= .01)
        say(f"{nm} 라벨 − 영상 자세: σY {e[:,0].std()*1e3:.1f} µm σT {e[:,1].std()*1e3:.2f} mdeg, 허용 안 {ok.sum()}/{len(e)}")

    rows = []   # id, source, family, Y_cmd, T_cmd, Y_abs, T_abs, X
    for k in np.where(one)[0]:
        i = int(f["id"][k]); rows.append((i, "rig_1st", "rig_1st", cf[k], L2[i], Xf[k]))
    for n, k in enumerate(P):
        rows.append((int(f["id"][k]), "rig_grid", "grid", cf[k], labF[n], Xf[k]))
    for n in range(len(rid)):
        rows.append((rid[n], "rig_grid_rev", "grid", cr[n], labR[n], Xr[n]))
    with (OUT / "features7.tsv").open("w", encoding="utf-8", newline="") as fo:
        fo.write("\t".join(["id"] + lc.NAMES + ["valid536", "qr_correlation", "line_correlation", "measured", "error"]) + "\n")
        for i, s, _, _, _, x in rows:
            m = feat_f[i] if s != "rig_grid_rev" else feat_r[str(i)]      # 측정 기록(유효점, 상관)은 원래 측정 그대로
            fo.write("\t".join([str(i)] + [repr(float(v)) for v in x] + [m["valid536"], m["qr_correlation"], m["line_correlation"], m["measured"], ""]) + "\n")
    with (OUT / "labels.tsv").open("w", encoding="utf-8", newline="") as fo:
        fo.write("id\tsource\tY_cmd\tT_cmd\tY_abs\tT_abs\n")
        for i, s, _, c, l, _ in rows: fo.write(f"{i}\t{s}\t{c[0]:.2f}\t{c[1]:.2f}\t{l[0]:.5f}\t{l[1]:.5f}\n")
    fam = np.array([r[2] for r in rows]); cmd = np.array([r[3] for r in rows])
    final, cv, half = lc.make_splits(fam, cmd)
    with (OUT / "splits.tsv").open("w", encoding="utf-8", newline="") as fo:
        w = csv.writer(fo, delimiter="\t"); w.writerow(["id", "final"] + [f"cv{k}" for k in range(10)] + [f"half{k}" for k in range(10)])
        for n, r in enumerate(rows): w.writerow([r[0], int(final[n])] + [cv[j][n] for j in range(10)] + [half[j][n] for j in range(10)])
    # 같은 자세가 학습과 평가로 갈리지 않는지 확인
    key = [(r[2], round(r[3][0], 2), round(r[3][1], 2)) for r in rows]
    for j in range(10):
        g = {}
        for kk, v in zip(key, cv[j]): g.setdefault(kk, set()).add(int(v))
        assert all(len(s) == 1 for s in g.values()), "같은 자세 묶음이 갈렸다"
    with (OUT / "라벨표.tsv").open("w", encoding="utf-8", newline="") as fo:
        w = csv.writer(fo, delimiter="\t"); w.writerow(["id_fwd", "id_rev", "Y_cmd", "T_cmd", "visY_fwd", "visT_fwd", "visY_rev", "visT_rev", "Y_lab_fwd", "T_lab_fwd", "Y_lab_rev", "T_lab_rev"])
        for n, k in enumerate(P):
            w.writerow([int(f["id"][k]), rid[n], f"{cr[n,0]:.2f}", f"{cr[n,1]:.2f}"] + [f"{v:.5f}" for v in (visf[k, 0], visf[k, 1], visr[n, 0], visr[n, 1], labF[n, 0], labF[n, 1], labR[n, 0], labR[n, 1])])
    say(f"학습 자료 {len(rows)}장 (rig_1st {one.sum()}, rig_grid {len(P)}, rig_grid_rev {len(rid)}), 마지막 확인 {int(final.sum())}장")
    say(f"라벨 범위 Y {min(r[4][0] for r in rows):+.4f} ~ {max(r[4][0] for r in rows):+.4f}, T {min(r[4][1] for r in rows):+.4f} ~ {max(r[4][1] for r in rows):+.4f}")
    (OUT / "요약.txt").write_bytes(("\n".join(log) + "\n").encode("utf-8"))


if __name__ == "__main__":
    main()
