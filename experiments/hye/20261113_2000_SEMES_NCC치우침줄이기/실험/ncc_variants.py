"""SEMES 사진에서 NCC 포물선의 치우침(소수 위치에 따른 S자, ±0.03 px)을 줄이는 방법을 비교한다.

과적합을 피하려고 후보를 미리 정해 두고(아래 VARIANTS), 결과를 보며 수치를 바꾸지 않는다.
  P3   : 지금 방식. NCC 봉우리 + 좌우 3점 포물선
  G3   : 3점 가우스 맞춤(상관 값의 로그에 포물선)
  Q5   : 봉우리 ±2 화소 5점에 2차식 최소제곱
  B1   : 센터 조각과 사진을 가우시안 σ1 로 흐린 뒤 P3
  B2   : 가우시안 σ2 로 흐린 뒤 P3
  (소수 위치별 치우침 표를 빼는 방법은 536점 결과로 표를 학습해야 해 과적합 위험이 커서 넣지 않았다.)
고르는 기준: 개발 560장 주 검증(섞은 80/20 10회)과 1차 축 사진의 사진 한 장 흔들림(라벨이 필요 없는 지표). 마지막 확인 142장은 고른 뒤 한 번만 본다.
나머지 6개 값(바 2, 나사 4)은 C++ 모듈 측정값 그대로 쓰고 QR 값만 바꾼다. 학습식은 C++ 과 같은 Huber(combo.py fit).
입력: <C++ 측정 features7.tsv(NCC)> <C++ 측정 features7.tsv(536점)> <photos.tsv> <labels.tsv> <splits.tsv>
출력: 결과/qr_variants.tsv(사진별 QR 값), 결과/결과.tsv, 결과/요약.txt
실행: 저장소 루트에서  python "<이 폴더>/ncc_variants.py" ...
"""
import csv, pathlib, sys
import cv2, numpy as np

HERE = pathlib.Path(__file__).resolve().parent; OUT = HERE.parent / "결과"; ROOT = HERE.parents[3]
BOX = [(1552, 969, 184, 198), (1809, 971, 184, 198), (2067, 973, 185, 198)]   # SEMES QR 상자 (site.cpp 원본 상수)
SX, SY = 110, 22
NAMES = ["qr_dx_mean", "bar_tilt_upper", "bar_tilt_lower", "screw1_x", "screw2_x", "screw3_x", "screw4_x"]
VARIANTS = ["P3", "G3", "Q5", "B1", "B2"]
TY, TT = .045, .010


def read(p):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_GRAYSCALE)


def sub(c, p, kind):
    y, x = p
    if kind == "Q5":
        if x < 2 or x > c.shape[1] - 3: return 0.0
        v = c[y, x - 2:x + 3].astype(np.float64); a, b, _ = np.polyfit(np.arange(-2, 3), v, 2)
        return float(np.clip(-b / (2 * a), -1, 1)) if a < 0 else 0.0
    l, m, r = float(c[y, x - 1]), float(c[y, x]), float(c[y, x + 1])
    if kind == "G3":
        if min(l, m, r) <= 0: kind = "P3"
        else:
            l, m, r = np.log(l), np.log(m), np.log(r)
    den = l - 2 * m + r
    return .5 * (l - r) / den if den < 0 else 0.0


def qr_values(img, center):
    out = {}
    for v in VARIANTS:
        s = {"B1": 1.0, "B2": 2.0}.get(v)
        im = cv2.GaussianBlur(img, (0, 0), s) if s else img; ce = cv2.GaussianBlur(center, (0, 0), s) if s else center
        dx = []
        for x, y, w, h in BOX:
            c = cv2.matchTemplate(im[y - SY:y + h + SY, x - SX:x + w + SX], ce[y:y + h, x:x + w], cv2.TM_CCOEFF_NORMED)
            _, _, _, loc = cv2.minMaxLoc(c); px, py = loc
            dx.append(px - SX + sub(c, (py, px), "P3" if v in ("B1", "B2") else v))
        out[v] = float(np.mean(dx))
    return out


def tsv(p):
    return list(csv.DictReader(open(p, encoding="utf-8"), delimiter="\t"))


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
    OUT.mkdir(exist_ok=True); log = []
    def say(*a):
        s = " ".join(str(x) for x in a); print(s, flush=True); log.append(s)
    fN, f5, photos, labels, splits = sys.argv[1:6]
    FN = {r["id"]: r for r in tsv(fN)}; F5 = {r["id"]: r for r in tsv(f5)}
    P = {l.split("\t")[0]: l.strip().split("\t")[1] for l in open(photos, encoding="utf-8") if l.strip()}
    L = {r["id"]: r for r in tsv(labels)}; SP = {r["id"]: r for r in tsv(splits)}
    ids = list(P)
    cache = OUT / "qr_variants.tsv"
    if cache.exists():
        Q = {r["id"]: {v: float(r[v]) for v in VARIANTS} for r in tsv(cache)}
    else:
        center = read(ROOT / "experiments/hye/00000_final_Y_T_semes/assets/center.bmp"); Q = {}
        for n, i in enumerate(ids):
            Q[i] = qr_values(read(P[i]), center)
            if n % 100 == 0: print("측정", n, flush=True)
        with cache.open("w", encoding="utf-8", newline="") as fo:
            w = csv.writer(fo, delimiter="\t"); w.writerow(["id"] + VARIANTS)
            for i in ids: w.writerow([i] + [f"{Q[i][v]:.6f}" for v in VARIANTS])
    p3cpp = np.array([float(FN[i]["qr_dx_mean"]) for i in ids]); p3py = np.array([Q[i]["P3"] for i in ids])
    say(f"검산: Python P3 − C++ NCC 포물선 최대 차이 {abs(p3py - p3cpp).max():.2e} px")
    g536 = np.array([float(F5[i]["qr_dx_mean"]) for i in ids])
    X6 = np.array([[float(FN[i][n]) for n in NAMES] for i in ids]); Y = np.array([[float(L[i]["Y_abs"]), float(L[i]["T_abs"])] for i in ids])
    final = np.array([SP[i]["final"] == "1" for i in ids]); dev = ~final; cv = [np.array([int(SP[i][f"cv{k}"]) for i in ids]) for k in range(10)]
    # 1차 축 사진 (라벨이 필요 없는 흔들림 지표)
    axes = {}
    for nm, cond, key in (("T축", lambda r: r["source"] == "20261014" and float(r["Y_cmd"]) == 0 and float(r["T_cmd"]) != 0, "T_cmd"),
                          ("Y축", lambda r: r["source"] == "20261014" and float(r["T_cmd"]) == 0 and float(r["Y_cmd"]) != 0, "Y_cmd")):
        axes[nm] = sorted([n for n, i in enumerate(ids) if cond(L[i])], key=lambda n: float(L[ids[n]][key]))
    def noise(q):
        r = []
        for nm, idx in axes.items():
            d = q[idx] - g536[idx]; e = d[1:-1] - (d[:-2] + d[2:]) / 2; r.append(e.std() / np.sqrt(1.5))
        return r
    def bias(q):
        d = q - g536; f = g536 - np.round(g536); k = np.clip(((f + .5) * 10).astype(int), 0, 9)
        prof = np.array([d[k == j].mean() for j in range(10)]); return prof.max() - prof.min()
    rows = []
    say("\n후보 | 536점과 rms 차 | 소수 위치 치우침 폭 | (후보 − 536점) 흔들림 T축 / Y축 | 개발 주 검증 탈락(560장, 10회) rmsY rmsT")
    cand = {"536점(기준)": g536, **{v: np.array([Q[i][v] for i in ids]) for v in VARIANTS}}
    for name, q in cand.items():
        X = X6.copy(); X[:, 0] = q; fails = []; ey = []; et = []
        for k in range(10):
            E = np.full((len(ids), 2), np.nan)
            for j in range(5):
                te = dev & (cv[k] == j); tr = dev & (cv[k] != j); E[te] = fit(X[tr], Y[tr])(X[te]) - Y[te]
            fails.append(int((~ok(E[dev])).sum())); ey.append(np.sqrt(np.mean(E[dev, 0] ** 2))); et.append(np.sqrt(np.mean(E[dev, 1] ** 2)))
        nz = noise(q)
        row = dict(후보=name, rms차_px=round(float(np.sqrt(np.mean((q - g536) ** 2))), 4), 치우침폭_px=round(float(bias(q)), 4), 흔들림_T축=round(nz[0], 4), 흔들림_Y축=round(nz[1], 4),
                   주검증_탈락=round(float(np.mean(fails)), 2), rmsY=round(float(np.mean(ey)), 5), rmsT=round(float(np.mean(et)), 5))
        rows.append(row)
        say(f"  {name:10s} | {row['rms차_px']:.4f} | {row['치우침폭_px']:.4f} | {row['흔들림_T축']:.4f} / {row['흔들림_Y축']:.4f} | {row['주검증_탈락']:.2f}  {row['rmsY']:.5f} {row['rmsT']:.5f}")
    with (OUT / "결과.tsv").open("w", encoding="utf-8", newline="") as fo:
        w = csv.DictWriter(fo, list(rows[0].keys()), delimiter="\t"); w.writeheader(); w.writerows(rows)
    (OUT / "요약.txt").write_bytes(("\n".join(log) + "\n").encode("utf-8"))


if __name__ == "__main__":
    main()
