"""우리 장비(rig_1st, rig_grid) 사진의 Y·T 라벨별 3D 봉우리 화면용 데이터 생성.

입력: experiments/hye/20261113_0950_우리장비사진거부_장비기준자료와라벨보정/결과/ 의
      라벨보정.tsv (명령, 영상 자세, 라벨 L1·L2), features7.tsv (값 7개), 촬영목록.tsv
출력: 같은 폴더의 index.html (template.html 에 데이터와 plotly.js 를 넣어 생성). apps/web 의 "데이터 분포" 탭이 띄운다

라벨 세 가지(명령 그대로, 절대 영점, 영점+보정)마다 값 7개 Huber 선형식을 5-fold OOF 로 다시 학습한다.
학습식은 experiments/hye/00000_final_Y_T/src/huber_fit.cpp 와 같다(apps/distribution/reoof.py 의 fit 을 값 7개에 맞춘 것).
이 스크립트는 화면용 분석 도구다. 팀 전달 추론·학습 도구(C++)를 대체하지 않는다.

실행: 저장소 루트에서  python apps/distribution/rig/make_page.py
"""
import csv, json, pathlib
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EXP = ROOT / "experiments/hye/20261113_0950_우리장비사진거부_장비기준자료와라벨보정/결과"
NAMES = ["qr_dx_mean", "bar_tilt_upper", "bar_tilt_lower", "screw1_x", "screw2_x", "screw3_x", "screw4_x"]
MODES = [("L0", "명령 그대로"), ("L1", "절대 영점"), ("L2", "영점+보정")]
SEED = 20261113


def fit(X, Y):
    mean = X.mean(0); scale = np.maximum(X.std(0), 1e-6)
    A = np.c_[np.ones(len(X)), (X - mean) / scale]
    pen = np.eye(A.shape[1]) * (1e-7 * len(X)); pen[0, 0] = 0
    w = np.ones(len(X))
    for _ in range(6):
        sw = np.sqrt(w)[:, None]
        B = np.linalg.lstsq((A * sw).T @ (A * sw) + pen, (A * sw).T @ (Y * sw), rcond=None)[0]
        P = A @ B
        r = np.maximum(np.abs(P[:, 0] - Y[:, 0]) / .045, np.abs(P[:, 1] - Y[:, 1]) / .01)
        w = np.minimum(1, 1 / np.maximum(r, 1e-12)); w *= len(w) / w.sum()
    return lambda Z: np.c_[np.ones(len(Z)), (Z - mean) / scale] @ B


def main():
    rows = list(csv.DictReader((EXP / "라벨보정.tsv").open(encoding="utf-8"), delimiter="\t"))
    feat = {r["id"]: r for r in csv.DictReader((EXP / "features7.tsv").open(encoding="utf-8"), delimiter="\t")}
    failed = [r["file"] for r in csv.DictReader((EXP / "촬영목록.tsv").open(encoding="utf-8"), delimiter="\t") if feat[r["id"]]["measured"] != "1"]
    X = np.array([[float(feat[r["id"]][n]) for n in NAMES] for r in rows])
    cmd = np.array([[float(r["Y_cmd"]), float(r["T_cmd"])] for r in rows])
    vis = np.array([[float(r["visY"]), float(r["visT"])] for r in rows])
    lab = {"L0": cmd, "L1": np.array([[float(r["Y_L1"]), float(r["T_L1"])] for r in rows]), "L2": np.array([[float(r["Y_L2"]), float(r["T_L2"])] for r in rows])}
    shot = np.array([r["kind"] == "shot" for r in rows]); st = np.array([r["set"] for r in rows])
    ks = np.where(shot)[0]; kc = np.where(~shot)[0]
    fold = np.empty(len(ks), int); fold[np.random.default_rng(SEED).permutation(len(ks))] = np.arange(len(ks)) % 5

    # 봉우리: 명령 좌표. 원점은 센터 사진의 자리(사진 22장)
    cnt = {}
    for k in ks: cnt[(round(cmd[k, 0], 2), round(cmd[k, 1], 2))] = cnt.get((round(cmd[k, 0], 2), round(cmd[k, 1], 2)), 0) + 1
    cnt[(0.0, 0.0)] = cnt.get((0.0, 0.0), 0) + len(kc)
    labs = sorted(cnt); LA = np.array(labs)

    pts = [dict(id=int(r["id"]), s=r["set"], f=r["file"].split("/", 1)[1], Y=round(float(r["Y_cmd"]), 2), T=round(float(r["T_cmd"]), 2), m=[]) for r in rows]
    for k in kc: pts[k]["center"] = True
    for n, k in enumerate(ks): pts[k]["fold"] = int(fold[n])
    modes = []
    for key, name in MODES:
        L = lab[key]
        P = np.full((len(rows), 2), np.nan)
        for f in range(5):
            te = ks[fold == f]; tr = ks[fold != f]
            P[te] = fit(X[tr], L[tr])(X[te])
        P[kc] = fit(X[ks], L[ks])(X[kc])        # 센터 사진은 전체 사진으로 학습한 모델로 예측한다
        E = P[ks] - L[ks]
        mY, mT = E.mean(0); c = np.cov(E.T); sYY, sTT, sYT = c[0, 0], c[1, 1], c[0, 1]
        inv = np.linalg.inv(c)
        q = lambda dy, dt: inv[0, 0] * dy * dy + 2 * inv[0, 1] * dy * dt + inv[1, 1] * dt * dt
        cover = float((q(E[:, 0] - mY, E[:, 1] - mT) <= 5.991464547).mean())
        ok = (np.abs(E[:, 0]) <= .045) & (np.abs(E[:, 1]) <= .01)
        # 점의 자리: 명령 + (예측 − 라벨). 센터 사진은 라벨이 (0, 0) 이라 예측 그대로다
        pos = cmd + (P - L); pos[kc] = P[kc]
        own = np.sqrt(q(pos[:, 0] - cmd[:, 0], pos[:, 1] - cmd[:, 1]))
        d2 = q(pos[:, 0][:, None] - LA[:, 0][None, :], pos[:, 1][:, None] - LA[:, 1][None, :])
        near = d2.argmin(1)
        for k in range(len(rows)):
            ly, lt = (0.0, 0.0) if not shot[k] else L[k]
            pts[k]["m"].append([round(float(pos[k, 0]), 5), round(float(pos[k, 1]), 5), round(float(own[k]), 3), labs[near[k]][0], labs[near[k]][1],
                                round(float(np.sqrt(d2[k, near[k]])), 3), round(float(ly), 5), round(float(lt), 5)])
        sets = {s: dict(n=int((st[ks] == s).sum()), sdY=float(E[st[ks] == s, 0].std(ddof=1)), sdT=float(E[st[ks] == s, 1].std(ddof=1)), fail=int((~ok[st[ks] == s]).sum())) for s in ("rig_1st", "rig_grid")}
        modes.append(dict(id=key, name=name, n=len(ks), fail=int((~ok).sum()), sets=sets,
                          cov=dict(mY=float(mY), mT=float(mT), sYY=float(sYY), sTT=float(sTT), sYT=float(sYT), n=len(ks), cover95=cover)))
        mis = int(sum(1 for k in ks if (labs[near[k]][0], labs[near[k]][1]) != (pts[k]["Y"], pts[k]["T"])))
        print(f"{key} {name}: 탈락 {int((~ok).sum())}/{len(ks)}  sdY {sYY**.5:.5f} sdT {sTT**.5:.5f} rho {sYT/(sYY*sTT)**.5:+.2f} cover95 {cover:.3f}  다른 라벨이 더 가까움 {mis}"
              f"  센터 22장 예측 T {P[kc,1].min()*1e3:+.1f}~{P[kc,1].max()*1e3:+.1f} mdeg, Y {P[kc,0].min()*1e3:+.1f}~{P[kc,0].max()*1e3:+.1f} µm")

    dif = (vis - cmd)[ks]
    const = {s: [float(v) for v in np.median((lab["L1"] - cmd)[shot & (st == s)], 0)] for s in ("rig_1st", "rig_grid")}
    data = dict(modes=modes, labels=[dict(Y=a, T=b, n=cnt[(a, b)], est=(a, b) == (0.0, 0.0)) for a, b in labs], pts=pts,
                summary=dict(failed=failed, const=const,
                             dif=dict(medY=float(np.median(dif[:, 0])), minY=float(dif[:, 0].min()), maxY=float(dif[:, 0].max()),
                                      medT=float(np.median(dif[:, 1])), minT=float(dif[:, 1].min()), maxT=float(dif[:, 1].max()))))
    html = (HERE / "template.html").read_text(encoding="utf-8").replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    # 오프라인·file:// 에서도 열리도록 plotly.js 를 HTML 안에 넣는다 (없으면 CDN 태그 유지)
    vendor = HERE.parent / "vendor" / "plotly-2.35.3.min.js"
    tag = '<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.35.3/plotly.min.js"></script>'
    if vendor.exists():
        html = html.replace(tag, "<script>" + vendor.read_text(encoding="utf-8").replace("</script", "<\\/script") + "</script>")
    (HERE / "index.html").write_text(html, encoding="utf-8")
    print(f"사진 {len(ks)}장 + 센터 {len(kc)}장, 라벨 {len(labs)}개 → {HERE / 'index.html'}")


if __name__ == "__main__":
    main()
