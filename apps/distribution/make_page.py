"""Y·T 라벨별 3D 봉우리 시각화용 데이터 생성.

입력: data/group_oof_corrected.tsv (reoof.py 가 만든 정정 OOF: ID 420 라벨 T 0.32, ID 702 학습·평가 제외, T=+2.00 12장 학습 제외)
      + 커밋 46152fd 의 experiments/hye/00004_final_Y_T/evidence/training_features.tsv (파일명·차수, git show 로 읽음)
출력: 같은 폴더의 index.html (template.html 에 데이터와 plotly.js 를 넣어 생성). apps/web 의 "데이터 분포" 탭이 띄운다

실행: 저장소 루트에서  python apps/distribution/make_page.py
"""
import collections, csv, io, json, math, pathlib, subprocess

COMMIT = "46152fd"
EV = "experiments/hye/00004_final_Y_T/evidence"
EXCLUDE = {702}  # 4차 Y-0.8 T-0.96.bmp (README: 명령/촬영상태 불일치 의심, 이번 분석 제외)
RND = {"20261014": 1, "20261020": 2, "20261029": 3, "20261109_round4": 4}
HERE = pathlib.Path(__file__).resolve().parent

# 원점 (0,0) 센터 사진 4장의 예측값. 715장 학습 목록에는 원점 라벨이 없다.
# 값 출처: 2026-11-12 다른 작업 세션이 00004 배포 모델로 추정해 사용자에게 보고한 값(이 스크립트가 다시 계산하지 않음).
# 학습에 쓰지 않은 사진이라 OOF 와 같은 성격으로 본다. 원점 반복 촬영 분산은 없으므로 원점 봉우리는 공통 공분산을 적용한 추정 봉우리다.
# 1차 자료에서 원점을 역추정한 6개(예전 630점 방식)는 기준이 달라 넣지 않는다.
CENTERS = [
    dict(id=9001, r=1, f="clean_20261014/center.bmp", pY=0.02376, pT=-0.00053),
    dict(id=9002, r=2, f="clean_20261020/기준 위치/center1.bmp", pY=0.01631, pT=0.00491),
    dict(id=9003, r=2, f="clean_20261020/기준 위치/center2.bmp", pY=0.01151, pT=0.00496),
    dict(id=9004, r=2, f="clean_20261020/기준 위치/center3.bmp", pY=0.02114, pT=0.00523),
]


def tsv(name):
    raw = subprocess.run(["git", "show", f"{COMMIT}:{EV}/{name}"], capture_output=True, check=True).stdout
    return {r["id"]: r for r in csv.DictReader(io.StringIO(raw.decode("utf-8")), delimiter="\t")}


def cov(ev):
    m = len(ev)
    a = sum(x for x, _ in ev) / m
    b = sum(y for _, y in ev) / m
    return (a, b,
            sum((x - a) ** 2 for x, _ in ev) / (m - 1),
            sum((y - b) ** 2 for _, y in ev) / (m - 1),
            sum((x - a) * (y - b) for x, y in ev) / (m - 1))


def main():
    tf = tsv("training_features.tsv")
    corr = HERE / "data" / "group_oof_corrected.tsv"
    oof = {r["id"]: r for r in csv.DictReader(corr.open(encoding="utf-8"), delimiter="	")}
    assert not set(oof) - set(tf) and not (set(tf) - set(oof)) - {str(i) for i in EXCLUDE}
    rows = []
    for i, r in oof.items():
        t = tf[i]
        rows.append(dict(id=int(i), r=RND[t["source"]], f=t["path"].split("/")[-1],
                         Y=round(float(r["Y"]), 2), T=round(float(r["T"]), 2),
                         pY=float(r["pred_Y"]), pT=float(r["pred_T"]), fold=int(r["fold"]),
                         note=r["label_note"] or None))
    rows.sort(key=lambda r: r["id"])
    # 공분산·차수별 폭·95% 비율은 범위 안(|T| ≤ 1) 사진만으로 구한다. 범위 밖 12장은 학습에서 뺀 외삽 예측이라 표시만 한다.
    for r in rows: r["out"] = abs(r["T"]) > 1.0001
    inr = [r for r in rows if not r["out"]]
    e = [(r["pY"] - r["Y"], r["pT"] - r["T"]) for r in inr]
    mY, mT, sYY, sTT, sYT = cov(e)
    det = sYY * sTT - sYT ** 2
    iYY, iTT, iYT = sTT / det, sYY / det, -sYT / det

    def m2(dy, dt):
        return iYY * dy * dy + 2 * iYT * dy * dt + iTT * dt * dt

    rounds = {}
    for k in (1, 2, 3, 4):
        c = cov([e[j] for j, r in enumerate(inr) if r["r"] == k])
        rounds[k] = dict(n=sum(1 for r in inr if r["r"] == k), sdY=c[2] ** .5, sdT=c[3] ** .5,
                         rho=c[4] / (c[2] * c[3]) ** .5)
    centers = [dict(c, Y=0.0, T=0.0, fold=None, center=True) for c in CENTERS]
    cnt = collections.Counter((r["Y"], r["T"]) for r in rows + centers)
    labs = sorted(cnt)
    for r in rows + centers:
        dy, dt = r["pY"] - r["Y"], r["pT"] - r["T"]
        best = min(labs, key=lambda L: m2(r["pY"] - L[0], r["pT"] - L[1]))
        r["dOwn"] = math.sqrt(m2(dy, dt))
        r["nY"], r["nT"] = best
        r["dNear"] = math.sqrt(m2(r["pY"] - best[0], r["pT"] - best[1]))
    cover = sum(1 for dy, dt in e if m2(dy - mY, dt - mT) <= 5.991464547) / len(e)
    # 이름의 이동 방향은 experiments/hye/20261022_1750_trial원점_절편외삽역산/결과/결과.md 의 first/last 명령 기준이다.
    # 1차 trial 6개의 역추정 원점: trial 마다 OOF 예측 = a + b·명령 을 맞춰 명령 0 에서의 예측(a). 사진이 아니라 계산값이다.
    import numpy as np
    one = [r for r in rows if r["r"] == 1]
    def trial(name, sel, ax):
        c = [r for r in one if sel(r)]
        x = np.array([r["Y"] if ax == 0 else r["T"] for r in c])
        aY = np.polyfit(x, [r["pY"] for r in c], 1)[1]; aT = np.polyfit(x, [r["pT"] for r in c], 1)[1]
        return dict(name=name, n=len(c), pY=float(aY), pT=float(aT))
    k20 = lambda r: round(r["Y"] * 20)
    origins = [trial("S1 Y +0.1→+2.0 (+ 방향 이동)", lambda r: r["T"] == 0 and r["Y"] > 0 and k20(r) % 2 == 0, 0),
               trial("S2 Y −0.1→−2.0 (− 방향 이동)", lambda r: r["T"] == 0 and r["Y"] < 0 and k20(r) % 2 == 0, 0),
               trial("S3 T +0.01→+1.00 (+ 방향 이동)", lambda r: r["Y"] == 0 and r["T"] > 0, 1),
               trial("S4 T −0.01→−1.00 (− 방향 이동)", lambda r: r["Y"] == 0 and r["T"] < 0, 1),
               trial("S5 Y +0.05→+1.95 (+ 방향 이동)", lambda r: r["T"] == 0 and r["Y"] > 0 and k20(r) % 2 == 1, 0),
               trial("S6 Y −1.95→−0.05 (+ 방향 이동)", lambda r: r["T"] == 0 and r["Y"] < 0 and k20(r) % 2 == 1, 0)]
    data = dict(commit=COMMIT, origins=origins, excluded=sorted(EXCLUDE),
                cov=dict(mY=mY, mT=mT, sYY=sYY, sTT=sTT, sYT=sYT, n=len(inr), nAll=len(rows), cover95=cover),
                rounds=rounds,
                labels=[dict(Y=a, T=b, n=cnt[(a, b)], est=(a, b) == (0.0, 0.0)) for a, b in labs],
                pts=rows + centers)
    html = (HERE / "template.html").read_text(encoding="utf-8").replace(
        "/*__DATA__*/null", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    # 오프라인·file:// 에서도 열리도록 plotly.js 를 HTML 안에 넣는다 (없으면 CDN 태그 유지)
    vendor = HERE / "vendor" / "plotly-2.35.3.min.js"
    tag = '<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.35.3/plotly.min.js"></script>'
    if vendor.exists():
        js = vendor.read_text(encoding="utf-8").replace("</script", "<\\/script")
        html = html.replace(tag, "<script>" + js + "</script>")
    (HERE / "index.html").write_text(html, encoding="utf-8")
    mis = sum(1 for r in inr if (r["nY"], r["nT"]) != (r["Y"], r["T"]))
    for r in rows:
        if r["out"]: print("out", r["id"], (r["Y"], r["T"]), "pred %.4f %.4f" % (r["pY"], r["pT"]), "near", (r["nY"], r["nT"]))
    for c in centers:
        print("center", c["id"], c["f"], "d0 %.2f" % c["dOwn"], "near", (c["nY"], c["nT"]), "%.2f" % c["dNear"])
    print(f"images {len(rows)} (in-range {len(inr)}) labels {len(labs)} sdY {sYY**.5:.5f} sdT {sTT**.5:.5f} "
          f"rho {sYT/(sYY*sTT)**.5:.3f} cover95 {cover:.3f} nearer-other {mis}")


if __name__ == "__main__":
    main()
