"""ssafy 방식과 ssafy_2 방식의 강건성을 같은 시험 사진, 같은 정답으로 비교한다.

학습 방식 (값 7개, 같은 Huber 학습식: 20261112_1835 combo.py 의 fit 과 같다)
  A = ssafy 방식  : 11-12 사진(rig_1st, rig_grid)만, 라벨 L2
  B = ssafy_2 방식: 11-12 + 11-13 역순(rig_grid_rev), 라벨 = 격자는 정방향·역순 같은 자세 평균, rig_1st 는 L2
공통 시험 사진: ssafy_2 분할의 마지막 확인 728장(자세 묶음 20%, 세 세트 모두, 학습에 절대 안 씀)
공통 정답: data/labels/rig_20261113_avg/labels.tsv (격자 avg, rig_1st L2)
  A 는 학습 라벨(L2)과 시험 정답(avg)이 다르다. 실제로 쓸 때도 정답은 모르므로 "더 정직한 정답"으로 둘 다 채점한다.
  정답 자체의 잡음(라벨 − 영상 자세)이 σT 4.3 mdeg 라 통과율 상한이 100% 가 아니다.
통과: |eY| ≤ 0.045 mm 그리고 |eT| ≤ 0.01 도.

시험
  1 데이터 줄이기: 학습 사진 수 n 을 자세 묶음 단위로 무작위로 뽑아(20회) 학습 → 공통 시험
  2 폴드 검정: 개발 자료를 자세 묶음 5조각으로 나눠 10회. 시험 조각은 세 세트 모두에서 채점
  3 범위 밖: (a) |Y| ≤ 1, |T| ≤ 0.5 안쪽만 학습 → 공통 시험 중 바깥 / (b) Y < 0 만 학습 → Y > 0 / (c) T < 0 만 학습 → T > 0
  4 다른 날: A 를 11-12 로 학습 → 11-13 역순 전체(1679장). B 는 두 날을 다 배워 해당 없음
  5 값 교란(보고서와 같은 크기): QR 하나 +0.3 px, QR 마다 잡음 0.1 px, 나사 하나 +0.3 px, 나사마다 잡음 0.1 px, 바 기울기 하나 +10% 표준편차
  6 계수 안정성: 학습 자료를 복원 추출 30회 → 공통 시험 예측의 흔들림(사진별 표준편차의 중앙값)
출력: 결과/결과.tsv, 결과/요약.txt
실행: 저장소 루트에서  python "<이 폴더>/robustness.py"
"""
import csv, json, pathlib
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[3]; OUT = HERE.parent / "결과"
S2 = ROOT / "experiments/hye/20261113_1345_역순격자로라벨보정확인_정방향보정값옮겨적용/결과/ssafy2"
L2F = ROOT / "experiments/hye/20261113_0950_우리장비사진거부_장비기준자료와라벨보정/결과/labels_L2.tsv"
PTS = ROOT / "experiments/hye/00000_final_Y_T_ssafy/assets/points.json"
NAMES = ["qr_dx_mean", "bar_tilt_upper", "bar_tilt_lower", "screw1_x", "screw2_x", "screw3_x", "screw4_x"]
SEED = 20261113; TY, TT = .045, .010


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


def main():
    OUT.mkdir(exist_ok=True); log = []; rows = []
    def say(*a):
        s = " ".join(str(x) for x in a); print(s, flush=True); log.append(s)
    def rec(test, recipe, cond, E, extra=""):
        rows.append(dict(시험=test, 방식=recipe, 조건=cond, 장수=len(E), 통과율=round(ok(E).mean() * 100, 2), 탈락=int((~ok(E)).sum()),
                         σY_um=round(E[:, 0].std() * 1e3, 2), σT_mdeg=round(E[:, 1].std() * 1e3, 2), 최대Y_um=round(abs(E[:, 0]).max() * 1e3, 1), 최대T_mdeg=round(abs(E[:, 1]).max() * 1e3, 2), 비고=extra))
        return f"{ok(E).mean()*100:6.2f}% (탈락 {int((~ok(E)).sum())}/{len(E)}) σY {E[:,0].std()*1e3:5.1f} σT {E[:,1].std()*1e3:5.2f}"

    F = {r["id"]: r for r in tsv(S2 / "features7.tsv")}; L = tsv(S2 / "labels.tsv"); SP = {r["id"]: r for r in tsv(S2 / "splits.tsv")}
    l2 = {r["id"]: (float(r["Y_abs"]), float(r["T_abs"])) for r in tsv(L2F)}
    ids = [r["id"] for r in L]; src = np.array([r["source"] for r in L])
    X = np.array([[float(F[i][n]) for n in NAMES] for i in ids])
    Yavg = np.array([[float(r["Y_abs"]), float(r["T_abs"])] for r in L])
    YL2 = np.array([l2.get(i, (np.nan, np.nan)) for i in ids])        # 역순은 L2 가 없다(A 학습에 안 씀)
    cmd = np.array([[float(r["Y_cmd"]), float(r["T_cmd"])] for r in L])
    final = np.array([SP[i]["final"] == "1" for i in ids]); dev = ~final
    cv = [np.array([int(SP[i][f"cv{k}"]) for i in ids]) for k in range(10)]
    day1 = src != "rig_grid_rev"
    fam = np.where(src == "rig_1st", "1st", "grid")
    gkey = np.array([f"{a}|{y:.2f}|{t:.2f}" for a, (y, t) in zip(fam, cmd)])
    R = {"A ssafy": dict(pool=dev & day1, Y=YL2), "B ssafy_2": dict(pool=dev, Y=Yavg)}
    say(f"사진 {len(ids)} (rig_1st {int((src=='rig_1st').sum())}, rig_grid {int((src=='rig_grid').sum())}, rig_grid_rev {int((src=='rig_grid_rev').sum())}), 공통 시험 {final.sum()}장")
    for nm, r in R.items(): say(f"  {nm}: 학습 후보 {r['pool'].sum()}장")
    Te = final

    # 0 전체 학습 → 공통 시험 (기준)
    say("\n[0] 학습 후보 전체로 학습 → 공통 시험 728장")
    M = {}
    for nm, r in R.items():
        M[nm] = fit(X[r["pool"]], r["Y"][r["pool"]]); E = M[nm](X[Te]) - Yavg[Te]
        say(f"  {nm:10s} 전체 {rec('0 기준', nm, '전체', E)}")
        for s in ("rig_1st", "rig_grid", "rig_grid_rev"):
            m = src[Te] == s; say(f"  {'':10s} {s:13s} {rec('0 기준', nm, s, E[m])}")

    # 1 데이터 줄이기
    say("\n[1] 데이터 줄이기: 자세 묶음 단위로 n 장 뽑아 학습(20회) → 공통 시험. 통과율 평균 [최저 ~ 최고], σT 평균")
    rng = np.random.default_rng(SEED)
    for nm, r in R.items():
        pool = np.where(r["pool"])[0]; groups = {}
        for i in pool: groups.setdefault(gkey[i], []).append(i)
        gl = list(groups.values())
        for n in (15, 30, 60, 120, 250, 500, 1000, 2000, len(pool)):
            if n > len(pool): continue
            P_, S_ = [], []
            for rep in range(20 if n < len(pool) else 1):
                take = []
                for g in rng.permutation(len(gl)):
                    take += gl[g]
                    if len(take) >= n: break
                E = fit(X[take], r["Y"][take])(X[Te]) - Yavg[Te]; P_.append(ok(E).mean() * 100); S_.append(E[:, 1].std() * 1e3)
            rows.append(dict(시험="1 데이터 줄이기", 방식=nm, 조건=f"n={n}", 장수=int(Te.sum()), 통과율=round(np.mean(P_), 2), 탈락="", σY_um="", σT_mdeg=round(np.mean(S_), 2), 최대Y_um="", 최대T_mdeg="", 비고=f"최저 {min(P_):.1f} 최고 {max(P_):.1f}"))
            say(f"  {nm:10s} n={n:5d}: {np.mean(P_):6.2f}% [{min(P_):6.2f} ~ {max(P_):6.2f}]  σT {np.mean(S_):5.2f}")

    # 2 폴드 검정
    say("\n[2] 폴드 검정: 개발 자료 자세 묶음 5조각 × 10회. 시험 조각은 세 세트 모두 채점(정답 avg)")
    for nm, r in R.items():
        Eall = {s: [] for s in ("rig_1st", "rig_grid", "rig_grid_rev")}; fails = []
        for k in range(10):
            E = np.full((len(ids), 2), np.nan)
            for j in range(5):
                te = dev & (cv[k] == j); tr = r["pool"] & (cv[k] != j)
                E[te] = fit(X[tr], r["Y"][tr])(X[te]) - Yavg[te]
            fails.append(int((~ok(E[dev])).sum()))
            for s in Eall: Eall[s].append(E[dev & (src == s)])
        say(f"  {nm:10s} 탈락(10회) {fails} 평균 {np.mean(fails):.1f} / {dev.sum()}")
        for s, v in Eall.items(): say(f"  {'':10s} {s:13s} {rec('2 폴드', nm, s, np.vstack(v), '10회 합침')}")

    # 3 범위 밖
    say("\n[3] 범위 밖 예측 (정답 avg)")
    regions = {"안쪽(|Y|≤1,|T|≤0.5) → 바깥": (lambda c: (abs(c[:, 0]) <= 1) & (abs(c[:, 1]) <= .5), lambda c: ~((abs(c[:, 0]) <= 1) & (abs(c[:, 1]) <= .5))),
               "Y<0 → Y>0": (lambda c: c[:, 0] < 0, lambda c: c[:, 0] > 0),
               "T<0 → T>0": (lambda c: c[:, 1] < 0, lambda c: c[:, 1] > 0)}
    for cond, (inside, outside) in regions.items():
        for nm, r in R.items():
            tr = r["pool"] & inside(cmd); te = Te & outside(cmd)
            E = fit(X[tr], r["Y"][tr])(X[te]) - Yavg[te]; say(f"  {cond:24s} {nm:10s} 학습 {tr.sum():4d}장 → {rec('3 범위 밖', nm, cond, E)}")

    # 4 다른 날
    say("\n[4] 다른 날: A 를 11-12 개발 자료로 학습 → 11-13 역순 전체 (정답 avg)")
    te = src == "rig_grid_rev"; E = M["A ssafy"](X[te]) - Yavg[te]
    say(f"  A ssafy → 역순 1679장 {rec('4 다른 날', 'A ssafy', '11-12 → 11-13 역순 전체', E)}")
    say("  B ssafy_2 는 두 날을 모두 배워 처음 보는 날이 없다(평가 전용 rig_eval 필요).")

    # 5 값 교란
    say("\n[5] 값 교란 (학습은 깨끗한 값, 공통 시험 값에만 교란). 통과율")
    pts = json.loads(PTS.read_text(encoding="utf-8"))["points"]
    nq = np.array([sum(1 for p in pts if int(p[2]) == k and int(p[3])) for k in range(3)]); wq = nq / nq.sum()
    tilt_sd = X[dev][:, 1:3].std(0); prng = np.random.default_rng(SEED + 1)
    def perturbed():
        out = []
        for k in range(3): Z = X[Te].copy(); Z[:, 0] += .3 * wq[k]; out.append(("QR 하나 +0.3 px", Z))
        Z = X[Te].copy(); Z[:, 0] += (prng.normal(0, .1, (Te.sum(), 3)) * wq).sum(1); out.append(("QR 마다 잡음 0.1 px", Z))
        for k in range(4): Z = X[Te].copy(); Z[:, 3 + k] += .3; out.append(("나사 하나 +0.3 px", Z))
        Z = X[Te].copy(); Z[:, 3:7] += prng.normal(0, .1, (Te.sum(), 4)); out.append(("나사 마다 잡음 0.1 px", Z))
        for b in range(2): Z = X[Te].copy(); Z[:, 1 + b] += .1 * tilt_sd[b]; out.append(("바 기울기 하나 +10% 표준편차", Z))
        return out
    PERT = perturbed()
    for nm in R:
        res = {}
        for pn, Z in PERT: res.setdefault(pn, []).append(ok(M[nm](Z) - Yavg[Te]).mean() * 100)
        base = ok(M[nm](X[Te]) - Yavg[Te]).mean() * 100
        for pn, v in res.items():
            rows.append(dict(시험="5 값 교란", 방식=nm, 조건=pn, 장수=int(Te.sum()), 통과율=round(np.mean(v), 2), 탈락="", σY_um="", σT_mdeg="", 최대Y_um="", 최대T_mdeg="", 비고=f"교란 없음 {base:.2f}"))
        say(f"  {nm:10s} 교란 없음 {base:.1f}% | " + " | ".join(f"{pn} {np.mean(v):.1f}%" for pn, v in res.items()) + f" | 최소 {min(np.mean(v) for v in res.values()):.1f}%")

    # 6 계수 안정성
    say("\n[6] 계수 안정성: 학습 자료 복원 추출 30회 → 공통 시험 예측의 사진별 표준편차 (중앙값 / 최대)")
    for nm, r in R.items():
        pool = np.where(r["pool"])[0]; P_ = []
        for rep in range(30):
            b = rng.choice(pool, len(pool)); P_.append(fit(X[b], r["Y"][b])(X[Te]))
        sd = np.std(np.array(P_), 0)
        rows.append(dict(시험="6 계수 안정성", 방식=nm, 조건="복원 추출 30회", 장수=int(Te.sum()), 통과율="", 탈락="", σY_um=round(np.median(sd[:, 0]) * 1e3, 3), σT_mdeg=round(np.median(sd[:, 1]) * 1e3, 3),
                         최대Y_um=round(sd[:, 0].max() * 1e3, 3), 최대T_mdeg=round(sd[:, 1].max() * 1e3, 3), 비고="사진별 예측 표준편차"))
        say(f"  {nm:10s} Y {np.median(sd[:,0])*1e3:.2f} µm (최대 {sd[:,0].max()*1e3:.2f}), T {np.median(sd[:,1])*1e3:.3f} mdeg (최대 {sd[:,1].max()*1e3:.3f})")

    with (OUT / "결과.tsv").open("w", encoding="utf-8", newline="") as fo:
        w = csv.DictWriter(fo, list(rows[0].keys()), delimiter="\t"); w.writeheader(); w.writerows(rows)
    (OUT / "요약.txt").write_bytes(("\n".join(log) + "\n").encode("utf-8"))


if __name__ == "__main__":
    main()
