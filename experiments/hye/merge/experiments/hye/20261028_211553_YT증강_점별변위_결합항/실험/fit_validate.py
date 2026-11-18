"""점별 반응 + 결합항 적합과 검증 (실험용 Python, 팀 전달 코드 아님).

입력: inputs/measurements.npz  (341장 x 고정630점 x (dx, dy) px, 흐린 경계 직접 맞춤 측정)
      inputs/manifest.csv, inputs/points.json (control 선택 536점 표시)
출력: results/*.csv, results/fit.npz, results/summary.json
"""
import csv, json, re, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
INP, OUT = HERE / "inputs", HERE / "results"
OUT.mkdir(exist_ok=True)

m = np.load(INP / "measurements.npz")
D, VALID = m["D"], m["valid"]                       # (341,630,2), (341,630)
N, P = VALID.shape
rows = list(csv.DictReader(open(INP / "manifest.csv", encoding="utf-8-sig")))
SEL = np.array([r[3] for r in json.load(open(INP / "points.json"))["points"]], bool)
TRIAL = {"Y1+": "S1", "Y1-": "S2", "T+": "S3", "T-": "S4", "Y2+": "S5", "Y2-": "S6"}

info = []
for i, r in enumerate(rows):
    date = "20261020" if r["relative"].startswith("20261020") else "20261014"
    k = re.search(r"\((\d)\)", r["file"])
    info.append(dict(i=i, date=date, Y=float(r["Y"]), T=float(r["T"]), scope=r["scope"],
                     trial=TRIAL.get(r["group"], "C14" if date == "20261014" else ("C20" if r["scope"] == "center" else "MIX")),
                     rep=int(k.group(1)) if k else (int(re.search(r"(\d)", r["file"]).group(1)) if r["scope"] == "center" and date == "20261020" else 1),
                     pose=f'Y{float(r["Y"]):+.2f} T{float(r["T"]):+.2f}', file=r["relative"]))
idx = lambda f: [x["i"] for x in info if f(x)]
I14 = idx(lambda x: x["date"] == "20261014")
IC20 = idx(lambda x: x["trial"] == "C20")
IMIX = idx(lambda x: x["trial"] == "MIX")
TRIALS = ["S1", "S2", "S3", "S4", "S5", "S6"]


FAMILIES = {                                         # 반응식 후보 (Y 차수, T 차수, T 부호별 분리)
    "Y1_T1": (1, 1, False), "Y1_T1s": (1, 1, True), "Y2_T2s": (2, 2, True),
    "Y3_T2s": (3, 2, True), "Y3_T3": (3, 3, False), "Y3_T3s": (3, 3, True)}


def basis(Y, T, fam):
    dy, dt, sg = FAMILIES[fam]
    Y, T = np.asarray(Y, float), np.asarray(T, float)
    cols = [Y ** d for d in range(1, dy + 1)]
    if sg: cols += [np.maximum(T, 0) ** d for d in range(1, dt + 1)] + [np.minimum(T, 0) ** d for d in range(1, dt + 1)]
    else: cols += [T ** d for d in range(1, dt + 1)]
    return np.stack(cols, -1)


def fit_single(ids, fam):
    """1014 단일축: 점마다 trial 영점 + Y 다항식 + T 다항식. center 사진의 영점은 0으로 고정."""
    tr = [info[i]["trial"] for i in ids]
    trials = [t for t in TRIALS if t in tr]
    X = np.concatenate([np.array([[t == s for s in trials] for t in tr], float),
                        basis([info[i]["Y"] for i in ids], [info[i]["T"] for i in ids], fam)], 1)
    off = np.zeros((len(TRIALS), P, 2)); coef = np.zeros((X.shape[1] - len(trials), P, 2))
    for p in range(P):
        ok = VALID[ids, p]
        sol, *_ = np.linalg.lstsq(X[ok], D[ids][ok, p, :], rcond=None)
        for j, s in enumerate(trials): off[TRIALS.index(s), p] = sol[j]
        coef[:, p] = sol[len(trials):]
    return dict(fam=fam, off=off, coef=coef, trials=trials)


def response(fit, Y, T):
    """(Y, T) -> (630, 2) 반응. 결합항 제외."""
    b = basis([Y], [T], fit["fam"])[0]
    return np.tensordot(b, fit["coef"], 1)


def jac(fit, Y, T, h=1e-3):
    return np.stack([(response(fit, Y + h, T) - response(fit, Y - h, T)) / (2 * h),
                     (response(fit, Y, T + h) - response(fit, Y, T - h)) / (2 * h)], -1)  # (630,2,2)


def score(res, ok, J=None, mask=None):
    """잔차 RMS(px), 판 평균 이동(dx, dy), 그리고 잔차를 Y/T로 환산한 값."""
    mk = ok & (SEL if mask is None else mask)
    r = res[mk]
    out = dict(rms=float(np.sqrt(np.mean(r ** 2))), mdx=float(r[:, 0].mean()), mdy=float(r[:, 1].mean()))
    if J is not None:
        A = J[mk].reshape(-1, 2); sol, *_ = np.linalg.lstsq(A, r.reshape(-1), rcond=None)
        out.update(eqY=float(sol[0]), eqT=float(sol[1]))
    return out


summary = {"points": int(P), "selected536": int(SEL.sum()), "photos": int(N)}

# ---------- 1. 반복 잡음 바닥 (20261020 같은 자세 3장) ----------
poses = sorted({info[i]["pose"] for i in IMIX})
dev, dev_rep = [], {1: [], 2: [], 3: []}
POOL_DEV, POOL_SHOT = [], []                             # 생성용 반복 잡음 표본 (점별, 같은 자세 평균 대비)
for ps in poses + ["C20"]:
    ids = IC20 if ps == "C20" else [i for i in IMIX if info[i]["pose"] == ps]
    ok = VALID[ids].all(0) & SEL
    mu = D[ids].mean(0)
    for i in ids:
        d = (D[i] - mu)[ok] * np.sqrt(len(ids) / (len(ids) - 1))
        dev.append(d); dev_rep[info[i]["rep"]].append(d.mean(0))
        full = np.where(ok[:, None], (D[i] - mu) * np.sqrt(len(ids) / (len(ids) - 1)), 0.0)
        POOL_DEV.append(full); POOL_SHOT.append(info[i]["rep"])
dev_all = np.concatenate(dev)
summary["noise_floor"] = dict(
    rms=float(np.sqrt(np.mean(dev_all ** 2))), rms_x=float(np.sqrt(np.mean(dev_all[:, 0] ** 2))),
    rms_y=float(np.sqrt(np.mean(dev_all[:, 1] ** 2))),
    shot_mean_dx={k: float(np.mean([v[0] for v in dev_rep[k]])) for k in dev_rep},
    shot_mean_dy={k: float(np.mean([v[1] for v in dev_rep[k]])) for k in dev_rep})

# ---------- 2. 1014 점별 반응식 고르기: trial 안에서 명령값 순서로 4겹 교차검증 ----------
# 모든 trial이 학습에 남으므로 trial 영점은 학습 사진에서만 추정하고, 시험 사진은 그대로 예측한다.
rank = {}
for s in TRIALS:
    ids = sorted((i for i in I14 if info[i]["trial"] == s), key=lambda i: abs(info[i]["Y"]) + abs(info[i]["T"]))
    for r, i in enumerate(ids): rank[i] = r % 4
shape_rows = []
for fam in FAMILIES:
    for f in range(4):
        test = [i for i in I14 if rank.get(i, -1) == f]
        fit = fit_single([i for i in I14 if rank.get(i, -1) != f], fam)
        for i in test:
            j = TRIALS.index(info[i]["trial"])
            sc = score(D[i] - fit["off"][j] - response(fit, info[i]["Y"], info[i]["T"]), VALID[i], jac(fit, info[i]["Y"], info[i]["T"]))
            shape_rows.append(dict(family=fam, fold=f, trial=info[i]["trial"], file=info[i]["file"], Y=info[i]["Y"], T=info[i]["T"], **sc))
with open(OUT / "shape_cv_1014.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, shape_rows[0].keys()); w.writeheader(); w.writerows(shape_rows)
agg = {}
for fam in FAMILIES:
    L = [x for x in shape_rows if x["family"] == fam]
    agg[fam] = dict(rms=float(np.sqrt(np.mean([x["rms"] ** 2 for x in L]))),
                    rms_Y=float(np.sqrt(np.mean([x["rms"] ** 2 for x in L if x["trial"] in ("S1", "S2", "S5", "S6")]))),
                    rms_T=float(np.sqrt(np.mean([x["rms"] ** 2 for x in L if x["trial"] in ("S3", "S4")]))),
                    mdx_maxabs=float(max(abs(x["mdx"]) for x in L)))
summary["shape_cv_1014"] = agg
FAM = min(agg, key=lambda k: agg[k]["rms"])
summary["family_selected"] = FAM
FIT = fit_single(I14, FAM)

# ---------- 3. 20261020 혼합 57장: 세션 영점과 결합항 (자세 하나 빼기) ----------
O20 = D[IC20].mean(0)
ONE = lambda Y, T: 1.0
COUP = {"none": [], "YT": [lambda Y, T: Y * T], "YT+YT2+Y2T": [lambda Y, T: Y * T, lambda Y, T: Y * T * T, lambda Y, T: Y * Y * T]}
MIXMODELS = {f"{o}|{c}": ([ONE] if o == "세션영점" else []) + t for o in ("센터영점", "세션영점") for c, t in COUP.items()}


def mix_fit(ids_fit, terms):
    c = np.zeros((len(terms), P, 2))
    if terms:
        X = np.array([[t(info[i]["Y"], info[i]["T"]) for t in terms] for i in ids_fit])
        B = np.stack([D[i] - O20 - response(FIT, info[i]["Y"], info[i]["T"]) for i in ids_fit])
        for p in range(P):
            ok = VALID[ids_fit, p]
            c[:, p] = np.linalg.lstsq(X[ok], B[ok, p], rcond=None)[0]
    return c


def mix_pred(i, terms, c):
    g = np.tensordot(np.array([t(info[i]["Y"], info[i]["T"]) for t in terms]), c, 1) if terms else 0
    return O20 + response(FIT, info[i]["Y"], info[i]["T"]) + g


mix_rows, coupling = [], {}
for name, terms in MIXMODELS.items():
    for pool_tag in ("T<=1", "T2포함"):
        pool = [i for i in IMIX if pool_tag == "T2포함" or info[i]["T"] <= 1.0]
        for ps in poses:
            test = [i for i in IMIX if info[i]["pose"] == ps]
            c = mix_fit([i for i in pool if info[i]["pose"] != ps], terms)
            for i in test:
                sc = score(D[i] - mix_pred(i, terms, c), VALID[i], jac(FIT, info[i]["Y"], info[i]["T"]))
                mix_rows.append(dict(model=name, fit_pool=pool_tag, file=info[i]["file"], Y=info[i]["Y"], T=info[i]["T"], rep=info[i]["rep"], **sc))
        L = [x for x in mix_rows if x["model"] == name and x["fit_pool"] == pool_tag]
        for sub, keep in (("T<=1", lambda x: x["T"] <= 1.0), ("T=2", lambda x: x["T"] > 1.0)):
            S = [x for x in L if keep(x)]
            coupling[f"{name}|학습 {pool_tag}|평가 {sub}"] = dict(
                n=len(S), rms=float(np.sqrt(np.mean([x["rms"] ** 2 for x in S]))),
                mdx_mean=float(np.mean([x["mdx"] for x in S])), mdx_maxabs=float(max(abs(x["mdx"]) for x in S)),
                eqY_maxabs=float(max(abs(x["eqY"]) for x in S)), eqT_maxabs=float(max(abs(x["eqT"]) for x in S)))
with open(OUT / "mixed_lopo.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, mix_rows[0].keys()); w.writeheader(); w.writerows(mix_rows)
summary["mixed_lopo"] = coupling
POOL = [i for i in IMIX if info[i]["T"] <= 1.0]
SES_C = mix_fit(POOL, [ONE])[0]                           # 세션 영점 (결합항 없이), 생성에 쓰는 11번째 영점
summary["mix_session_origin"] = dict(mdx=float((O20 + SES_C)[SEL, 0].mean()), mdy=float((O20 + SES_C)[SEL, 1].mean()))
COUP_ALL = mix_fit(POOL, [ONE] + COUP["YT"])             # 진단용: 세션 영점 + YT

# ---------- 4. 실측 영점 10개 ----------
origins = {"C14": np.zeros((P, 2))}
for j, s in enumerate(TRIALS): origins[s] = FIT["off"][j]
for i in IC20: origins[f"C20_{info[i]['rep']}"] = D[i]
origins["MIX20"] = O20 + SES_C
summary["origins"] = {k: dict(mdx=float(v[SEL, 0].mean()), mdy=float(v[SEL, 1].mean()),
                              shape_rms=float(np.sqrt(np.mean((v[SEL] - v[SEL].mean(0)) ** 2)))) for k, v in origins.items()}

photo_rows = []
for i in range(N):
    x = info[i]
    if x["trial"] in TRIALS: src = next(r for r in shape_rows if r["family"] == FAM and r["file"] == x["file"]); kind = "1014 단일축 4겹 CV"
    elif x["trial"] == "MIX": src = next(r for r in mix_rows if r["model"] == "세션영점|none" and r["fit_pool"] == "T<=1" and r["file"] == x["file"]); kind = "1020 혼합 자세 빼기" + (" (T=2 외삽)" if x["T"] > 1 else "")
    else: src = None; kind = "센터 (영점 자체)"
    photo_rows.append(dict(file=x["file"], date=x["date"], trial=x["trial"], Y=x["Y"], T=x["T"], rep=x["rep"], eval=kind,
                           actual_mdx=float(D[i][SEL & VALID[i], 0].mean()), actual_mdy=float(D[i][SEL & VALID[i], 1].mean()),
                           resid_rms=src["rms"] if src else "", resid_mdx=src["mdx"] if src else "", resid_mdy=src["mdy"] if src else ""))
with open(OUT / "photos_validation.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, photo_rows[0].keys()); w.writeheader(); w.writerows(photo_rows)

json.dump(summary, open(OUT / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
np.savez_compressed(OUT / "fit.npz", family=FAM, yt_coupling=COUP_ALL[1], coef=FIT["coef"], off=FIT["off"], O20=O20, sel=SEL,
                    points=m["points"], origin_names=np.array(list(origins)), origins=np.stack(list(origins.values())),
                    pool_dev=np.stack(POOL_DEV), pool_shot=np.array(POOL_SHOT), noise_floor_rms=summary["noise_floor"]["rms"])
print("saved", OUT)
