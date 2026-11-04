"""증강을 채택 모듈(joint, control)의 입력 공간에서 다시 만들고 평가한다 (실험용 Python, 팀 전달 코드 아님).

입력: inputs/module_features/ (tool/dump_features.exe 가 실측 341장에서 뽑은 값)
  joint_obs.csv     joint Observer 102관측
  control_disp.csv  control Tracker 630점 변위 - center 변위 (+ 유효 표시)
  predictions.csv   C++ 모듈의 실제 추정값 (Python 이식 검사용)

1. Python 이식 검사: 같은 입력에서 C++ 추정과 같은지
2. 증강 자체 평가: 실측 사진마다, 그 사진을 빼고 맞춘 생성기로 같은 자세·영점의 증강 쌍둥이를 만들어
   실측 추정값과 쌍둥이 추정값(잡음 없는 것 1개 + 잡음 표본 전부)을 비교한다.
3. Y-T 평면 오차 지도: 격자 16,281점 x 영점 15개 x n회를 두 모듈로 채점한다.
반복 잡음은 두 가지로 따로 넣는다 (2026-10-29):
  1020반복  1020 같은 자세 3장의 편차 (촬영 순번 유지, T <= 1 자세 15개 + 센터 = 48개)
  1014연속  1014 단일축 280장의 점별 반응식 잔차 (자유도 보정). 1014는 같은 자세 반복이 없어서
            잔차로 추정하며, 반응식이 못 맞춘 부분이 있으면 그만큼 크게 잡힌다.
결과: results/module_eval_2noise/ (잡음 1종 결과 results/module_eval/ 는 그대로 둔다)
"""
import csv, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
MF, OUT = HERE / "inputs" / "module_features", HERE / "results" / "module_eval_2noise"
OUT.mkdir(parents=True, exist_ok=True)
HYE = HERE.parent
sys.path.insert(0, str(HERE))
from generate import basis, FAMILIES  # noqa: E402

TOL, N_AUG, SEED = 0.09, 3, 20261028
TRIALS = ["S1", "S2", "S3", "S4", "S5", "S6"]
TRIAL = {"Y1+": "S1", "Y1-": "S2", "T+": "S3", "T-": "S4", "Y2+": "S5", "Y2-": "S6"}

# ---------- 사진 정보 (fit_validate.py 와 같은 해석) ----------
import re
rows = list(csv.DictReader(open(HERE / "inputs" / "manifest.csv", encoding="utf-8-sig")))
info = []
for i, r in enumerate(rows):
    date = "20261020" if r["relative"].startswith("20261020") else "20261014"
    k = re.search(r"\((\d)\)", r["file"])
    trial = TRIAL.get(r["group"], "C14" if date == "20261014" else ("C20" if r["scope"] == "center" else "MIX"))
    rep = int(k.group(1)) if k else (int(re.search(r"(\d)", r["file"]).group(1)) if trial == "C20" else 1)
    info.append(dict(i=i, date=date, Y=float(r["Y"]), T=float(r["T"]), trial=trial, rep=rep, file=r["relative"], pose=(float(r["Y"]), float(r["T"]))))
N = len(info)
I14 = [x["i"] for x in info if x["date"] == "20261014"]
IC20 = [x["i"] for x in info if x["trial"] == "C20"]
IMIX = [x["i"] for x in info if x["trial"] == "MIX"]
POSES = sorted({info[i]["pose"] for i in IMIX})

# ---------- 측정값 ----------
def read_rows(name):
    with open(MF / name, encoding="utf-8") as f:
        out = {}
        for line in f:
            if line.startswith("index"): continue
            p = line.rstrip("\n").split(","); out[int(p[0])] = p[1:]
        return out
jo = read_rows("joint_obs.csv"); cd = read_rows("control_disp.csv"); pr = read_rows("predictions.csv")
XJ = np.array([[float(v) for v in jo[i]] for i in range(N)])                     # (341, 102)
CDr = np.array([[float(v) for v in cd[i]] for i in range(N)])
XC, CVAL = CDr[:, :1260], CDr[:, 1260:].astype(bool)                            # (341, 1260) 점 순서 (p, c)
CPP = dict(jy=np.array([float(pr[i][0]) for i in range(N)]), jt=np.array([float(pr[i][1]) for i in range(N)]),
           cax=np.array([pr[i][2] for i in range(N)]), cval=np.array([float(pr[i][4]) for i in range(N)]))

# ---------- 모듈 추정 (Python 이식) ----------
JM = json.load(open(HYE / "00000_final_Y_T_joint" / "assets" / "both_model.json", encoding="utf-8"))["model"]
J_O, J_P, J_I, J_S = np.array(JM["origin"]).reshape(-1), np.array(JM["projector"]), np.array(JM["initial"]), np.array(JM["scale"])
def joint_predict(X):
    X = np.atleast_2d(X); h = X[:, 0:54:2].sum(1, keepdims=True) / 2700.0
    q = (np.concatenate([X, X * h], 1) - J_O) @ J_P @ J_I
    return q[:, 0] * J_S[0], q[:, 1] * J_S[1]

CM = json.load(open(HYE / "00000_final_Y_T control" / "assets" / "model.json", encoding="utf-8"))
PTS = json.load(open(HERE / "inputs" / "points.json"))["points"]
SEL = np.array([p[3] for p in PTS], bool)
_u = np.array([(p[0] - 1906) / 486 for p in PTS]); _v = np.array([(p[1] - 1070) / 122 for p in PTS])
_B = np.stack([np.ones(630), _u, _v, _u * _u, _u * _v, _v * _v], 1)[SEL]
P12 = np.linalg.pinv(_B)                                                           # (6, 536)
def control_features(XCflat):
    D = np.atleast_2d(XCflat).reshape(-1, 630, 2)[:, SEL]                          # (n, 536, 2)
    return np.einsum("jp,npc->njc", P12, D).reshape(len(D), 12)                    # 행 우선 (6 x 2) → 12
W = np.array(CM["W"])
BR = [(name, b[0], np.array(b[1]), np.array(b[2]), np.array(b[3])) for name, b in CM["branches"].items()]
def control_predict(F):
    F = np.atleast_2d(F); best_cost = np.full(len(F), np.inf); best_val = np.zeros(len(F)); best_ax = np.empty(len(F), object)
    for name, ax, o, s, c in BR:
        z = (F - o) @ W.T; s_, c_ = W @ s, W @ c
        val = z @ s_ / (s_ @ s_)
        for _ in range(8):
            j = s_[None] + 2 * val[:, None] * c_[None]; r = z - val[:, None] * s_[None] - (val ** 2)[:, None] * c_[None]
            val = val + (j * r).sum(1) / (j * j).sum(1)
        res = z - val[:, None] * s_[None] - (val ** 2)[:, None] * c_[None]; cost = (res * res).sum(1)
        cost[val * (1 if name.endswith("+") else -1) < 0] = np.inf
        upd = cost < best_cost; best_cost[upd] = cost[upd]; best_val[upd] = val[upd]; best_ax[upd] = ax
    y = np.where(best_ax == "Y", best_val, 0.0); t = np.where(best_ax == "T", best_val, 0.0)
    return y, t, best_ax

pj = joint_predict(XJ); pc = control_predict(control_features(XC))
summary = {"port_check": dict(
    joint_max_abs_diff=float(max(np.abs(pj[0] - CPP["jy"]).max(), np.abs(pj[1] - CPP["jt"]).max())),
    control_axis_agree=int((pc[2] == CPP["cax"]).sum()), control_max_abs_diff=float(np.abs(np.where(pc[2] == "Y", pc[0], pc[1]) - CPP["cval"]).max()))}
print("port check", summary["port_check"], flush=True)

# ---------- 생성기 (임의 측정 공간, 성분 K개) ----------
def fit_single(X, ids, fam):
    tr = [info[i]["trial"] for i in ids]; trials = [t for t in TRIALS if t in tr]
    A = np.concatenate([np.array([[t == s for s in trials] for t in tr], float), basis([info[i]["Y"] for i in ids], [info[i]["T"] for i in ids], fam)], 1)
    sol = np.linalg.lstsq(A, X[ids], rcond=None)[0]
    off = np.zeros((len(TRIALS), X.shape[1]))
    for j, s in enumerate(trials): off[TRIALS.index(s)] = sol[j]
    return dict(fam=fam, off=off, coef=sol[len(trials):])
resp = lambda fit, Y, T: basis(np.atleast_1d(Y), np.atleast_1d(T), fit["fam"]) @ fit["coef"]

def resid_pool(X, ids, fit):
    """1014연속 잡음 표본: 맞춤에 쓴 사진들의 잔차 필드, 자유도 보정 sqrt(n / (n - p))"""
    # 1014 센터 사진(C14)은 fit_single 처럼 영점 0으로 맞춘다
    off = lambda i: fit["off"][TRIALS.index(info[i]["trial"])] if info[i]["trial"] in TRIALS else 0
    r = np.array([X[i] - off(i) - resp(fit, info[i]["Y"], info[i]["T"])[0] for i in ids])
    p = len({info[i]["trial"] for i in ids} & set(TRIALS)) + fit["coef"].shape[0]
    return r * np.sqrt(len(ids) / (len(ids) - p))

rank = {}
for s in TRIALS:
    ids = sorted((i for i in I14 if info[i]["trial"] == s), key=lambda i: abs(info[i]["Y"]) + abs(info[i]["T"]))
    for r, i in enumerate(ids): rank[i] = r % 4

def build(X):
    """그 공간의 생성기 전체: 반응식 선택, 영점 15개, 백래시, 잡음 표본"""
    cv = {}
    for fam in FAMILIES:
        err = []
        for f in range(4):
            fit = fit_single(X, [i for i in I14 if rank.get(i, -1) != f], fam)
            for i in (i for i in I14 if rank.get(i, -1) == f):
                err.append(X[i] - fit["off"][TRIALS.index(info[i]["trial"])] - resp(fit, info[i]["Y"], info[i]["T"])[0])
        cv[fam] = float(np.sqrt(np.mean(np.square(err))))
    fam = min(cv, key=cv.get); fit = fit_single(X, I14, fam)
    O20 = X[IC20].mean(0)
    pool = [i for i in IMIX if info[i]["T"] <= 1]
    ses = np.mean([X[i] - O20 - resp(fit, info[i]["Y"], info[i]["T"])[0] for i in pool], 0)
    O = {"C14": np.zeros(X.shape[1]), **{s: fit["off"][j] for j, s in enumerate(TRIALS)}}
    for i in IC20: O[f"C20_{info[i]['rep']}"] = X[i]
    O["MIX20"] = O20 + ses
    bplus = np.mean([O["S1"], O["S3"], O["S5"], O["MIX20"] - O20], 0); bminus = np.mean([O["S2"], O["S4"], O["S6"]], 0)
    M = np.stack(list(O.values())); mu = M.mean(0); _, S, Vt = np.linalg.svd(M - mu, full_matrices=False); sg = S / np.sqrt(len(M) - 1)
    for k in range(2):
        for s in (1, -1): O[f"PC{k + 1}{'+' if s > 0 else '-'}2s"] = mu + s * 2 * sg[k] * Vt[k]
    dev, shot, dpose = [], [], []
    # T = 2 자세는 생성 범위 밖이고 control LK 추적이 수십 점씩 튀어서 (반복 잡음이 아님) 뺀다. 잡음 1종 결과(results/module_eval/)는 넣었었다
    for ps in [q for q in POSES if q[1] <= 1] + ["C20"]:
        ids = IC20 if ps == "C20" else [i for i in IMIX if info[i]["pose"] == ps]
        m = X[ids].mean(0)
        for i in ids: dev.append((X[i] - m) * np.sqrt(len(ids) / (len(ids) - 1))); shot.append(info[i]["rep"]); dpose.append(ps)
    return dict(fam=fam, cv=cv, fit=fit, O=O, O20=O20, ses=ses, bplus=bplus, bminus=bminus, dev=np.array(dev), shot=np.array(shot), dpose=dpose,
                dev14=resid_pool(X, I14, fit))

GJ, GC = build(XJ), build(XC)
CENTER_KIND = {"C14", "C20_1", "C20_2", "C20_3"}
summary["generator"] = {nm: dict(family=g["fam"], shape_cv_rms=g["cv"]) for nm, g in (("joint", GJ), ("control", GC))}

# ---------- 2. 증강 자체 평가: 실측 사진 vs 증강 쌍둥이 ----------
def twins(X, G, i):
    """사진 i를 빼고 맞춘 생성기로 만든 쌍둥이: (잡음 없는 1개, 잡음 표본들)"""
    x = info[i]
    if x["trial"] in TRIALS:  # 1014연속 잡음도 그 겹을 뺀 맞춤의 잔차만 쓴다
        train = [k for k in I14 if rank.get(k, -1) != rank[i]]
        fit = fit_single(X, train, G["fam"])
        base = fit["off"][TRIALS.index(x["trial"])] + resp(fit, x["Y"], x["T"])[0]
        return base, {"1020반복": base[None] + G["dev"], "1014연속": base[None] + resid_pool(X, train, fit)}
    else:  # 1020 혼합: 1014 전체 반응 + 1020 센터 + 다른 자세로 맞춘 세션 영점, 잡음 표본도 그 자세 것은 뺀다
        pool = [k for k in IMIX if info[k]["T"] <= 1 and info[k]["pose"] != x["pose"]]
        ses = np.mean([X[k] - G["O20"] - resp(G["fit"], info[k]["Y"], info[k]["T"])[0] for k in pool], 0)
        base = G["O20"] + ses + resp(G["fit"], x["Y"], x["T"])[0]
        keep = np.array([p != x["pose"] for p in G["dpose"]])
    return base, {"1020반복": base[None] + G["dev"][keep], "1014연속": base[None] + G["dev14"]}

NOISES = ["1020반복", "1014연속"]
# 잡음 크기 비교: control 공간은 px 변위라 선택 536점 RMS로 본다
def sel_rms(D):
    D = D.reshape(len(D), 630, 2)[:, SEL]
    return dict(n=len(D), rms=float(np.sqrt(np.mean(D ** 2))), x=float(np.sqrt(np.mean(D[..., 0] ** 2))), y=float(np.sqrt(np.mean(D[..., 1] ** 2))),
                plate_dx_sd=float(D[..., 0].mean(1).std()))
summary["noise_control_px"] = {"1020반복": sel_rms(GC["dev"]), "1014연속": sel_rms(GC["dev14"])}
print("noise", summary["noise_control_px"], flush=True)

trow = []
pct = lambda real, s: float((np.sum(s < real) + 0.5 * np.sum(s == real)) / len(s))
for i in range(N):
    x = info[i]
    if x["trial"] in ("C14", "C20"): continue
    bj, SJ = twins(XJ, GJ, i); bc, SC = twins(XC, GC, i)
    ry, rt = pj[0][i], pj[1][i]; ty, tt = joint_predict(bj)
    cy, ct, cax = pc[0][i], pc[1][i], pc[2][i]; qy, qt, qax = control_predict(control_features(bc))
    noisy = {}
    for nz in NOISES:
        sy, st = joint_predict(SJ[nz]); wy, wt, wax = control_predict(control_features(SC[nz]))
        noisy.update({f"joint_twin_noisy_pass@{nz}": float(np.mean((np.abs(sy - x["Y"]) <= TOL) & (np.abs(st - x["T"]) <= TOL))),
                      f"joint_pct_y@{nz}": pct(ry, sy), f"joint_pct_t@{nz}": pct(rt, st), f"joint_twin_sd_y@{nz}": float(sy.std()), f"joint_twin_sd_t@{nz}": float(st.std()),
                      f"control_twin_axis_agree@{nz}": float(np.mean(wax == cax)),
                      f"control_twin_noisy_pass@{nz}": float(np.mean((np.abs(wy - x["Y"]) <= TOL) & (np.abs(wt - x["T"]) <= TOL))),
                      f"control_pct@{nz}": pct(cy + ct, wy + wt)})
    trow.append(dict(
        file=x["file"], set="1014 단일축" if x["date"] == "20261014" else ("1020 혼합 T=2 (외삽)" if x["T"] > 1 else "1020 혼합"),
        Y=x["Y"], T=x["T"],
        joint_real_y=ry, joint_real_t=rt, joint_twin_y=float(ty[0]), joint_twin_t=float(tt[0]),
        joint_diff_y=float(ry - ty[0]), joint_diff_t=float(rt - tt[0]),
        joint_real_pass=bool(abs(ry - x["Y"]) <= TOL and abs(rt - x["T"]) <= TOL), joint_twin_pass=bool(abs(ty[0] - x["Y"]) <= TOL and abs(tt[0] - x["T"]) <= TOL),
        control_real_axis=cax, control_twin_axis=qax[0],
        control_real_y=cy, control_real_t=ct, control_twin_y=float(qy[0]), control_twin_t=float(qt[0]),
        control_diff=float((cy - qy[0]) + (ct - qt[0])) if cax == qax[0] else float("nan"),
        control_real_pass=bool(abs(cy - x["Y"]) <= TOL and abs(ct - x["T"]) <= TOL), control_twin_pass=bool(abs(qy[0] - x["Y"]) <= TOL and abs(qt[0] - x["T"]) <= TOL),
        **noisy))
with open(OUT / "twin_compare.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, trow[0].keys()); w.writeheader(); w.writerows(trow)

agg = {}
for st in ("1014 단일축", "1020 혼합", "1020 혼합 T=2 (외삽)"):
    L = [r for r in trow if r["set"] == st]
    a = np.array; dy = a([r["joint_diff_y"] for r in L]); dt = a([r["joint_diff_t"] for r in L])
    inside = lambda k: float(np.mean([0.025 <= r[k] <= 0.975 for r in L]))
    cdf = a([r["control_diff"] for r in L]); cdf = cdf[np.isfinite(cdf)]
    agg[st] = dict(
        n=len(L),
        joint=dict(diff_y_rms=float(np.sqrt(np.mean(dy ** 2))), diff_y_max=float(np.abs(dy).max()), diff_t_rms=float(np.sqrt(np.mean(dt ** 2))), diff_t_max=float(np.abs(dt).max()),
                   real_pass=int(sum(r["joint_real_pass"] for r in L)), twin_pass=int(sum(r["joint_twin_pass"] for r in L)),
                   pass_agree=int(sum(r["joint_real_pass"] == r["joint_twin_pass"] for r in L)),
                   noisy={nz: dict(twin_noisy_pass_mean=float(np.mean([r[f"joint_twin_noisy_pass@{nz}"] for r in L])),
                                   real_inside_twin95_y=inside(f"joint_pct_y@{nz}"), real_inside_twin95_t=inside(f"joint_pct_t@{nz}"),
                                   twin_noise_sd_y=float(np.mean([r[f"joint_twin_sd_y@{nz}"] for r in L])), twin_noise_sd_t=float(np.mean([r[f"joint_twin_sd_t@{nz}"] for r in L])))
                          for nz in NOISES}),
        control=dict(axis_agree=int(sum(r["control_real_axis"] == r["control_twin_axis"] for r in L)),
                     diff_rms=float(np.sqrt(np.mean(cdf ** 2))) if len(cdf) else None, diff_max=float(np.abs(cdf).max()) if len(cdf) else None,
                     real_pass=int(sum(r["control_real_pass"] for r in L)), twin_pass=int(sum(r["control_twin_pass"] for r in L)),
                     pass_agree=int(sum(r["control_real_pass"] == r["control_twin_pass"] for r in L)),
                     noisy={nz: dict(twin_noisy_pass_mean=float(np.mean([r[f"control_twin_noisy_pass@{nz}"] for r in L])),
                                     real_inside_twin95=inside(f"control_pct@{nz}"),
                                     axis_agree_mean=float(np.mean([r[f"control_twin_axis_agree@{nz}"] for r in L]))) for nz in NOISES}))
summary["twin"] = agg
print(json.dumps(agg, ensure_ascii=False, indent=1), flush=True)

# ---------- 3. Y-T 평면 오차 지도 ----------
Ys, Ts = np.round(np.linspace(-2, 2, 81), 2), np.round(np.linspace(-1, 1, 201), 2)
YY, TT = np.meshgrid(Ys, Ts, indexing="ij"); Yf, Tf = YY.ravel(), TT.ravel()
names = list(GJ["O"])
sgn = np.sign(Tf); sgn[sgn == 0] = np.sign(Yf[sgn == 0])
RES = {}
for nz in NOISES:
    # 1020반복은 잡음 1종 때와 같은 난수 순서라 그 결과를 그대로 재현한다
    rng = np.random.default_rng(SEED if nz == "1020반복" else SEED + 1)
    res = {k: np.zeros((len(names), N_AUG, len(Yf)), np.float32) for k in ("jy", "jt", "cy", "ct")}
    res["cax"] = np.zeros((len(names), N_AUG, len(Yf)), np.int8)
    for o, nm in enumerate(names):
        # 잡음 표본은 사진 단위로 뽑아 joint와 control이 같은 실측 편차 사진을 쓰게 한다 (두 생성기의 표본 순서가 같다)
        if nz == "1020반복": picks = [rng.choice(np.flatnonzero(GJ["shot"] == k % 3 + 1), len(Yf)) for k in range(N_AUG)]
        else: picks = [rng.choice(len(GJ["dev14"]), len(Yf)) for k in range(N_AUG)]
        for G, X, key in ((GJ, XJ, "j"), (GC, XC, "c")):
            base = G["O"][nm][None] + basis(Yf, Tf, G["fam"]) @ G["fit"]["coef"]
            if nm in CENTER_KIND: base = base + np.where(sgn[:, None] > 0, G["bplus"][None], np.where(sgn[:, None] < 0, G["bminus"][None], 0))
            pool = G["dev"] if nz == "1020반복" else G["dev14"]
            for k in range(N_AUG):
                S = base + pool[picks[k]]
                if key == "j":
                    y, t = joint_predict(S); res["jy"][o, k], res["jt"][o, k] = y, t
                else:
                    y, t, ax = control_predict(control_features(S)); res["cy"][o, k], res["ct"][o, k] = y, t; res["cax"][o, k] = (ax == "T").astype(np.int8)
        print("scored", nz, nm, flush=True)
    np.savez_compressed(OUT / f"grid_predictions_{nz}.npz", names=np.array(names), Y=Ys, T=Ts, n=N_AUG, seed=SEED, **res)
    RES[nz] = res

def maps(y, t):
    ey, et = np.abs(y - Yf), np.abs(t - Tf); ok = (ey <= TOL) & (et <= TOL)
    return dict(ey=ey.mean((0, 1)), et=et.mean((0, 1)), eymax=ey.max((0, 1)), etmax=et.max((0, 1)), pass_=ok.mean((0, 1)), pass_by_origin=ok.mean(1))
MAPS = {nz: {"joint": maps(RES[nz]["jy"], RES[nz]["jt"]), "control": maps(RES[nz]["cy"], RES[nz]["ct"])} for nz in NOISES}
for nz in NOISES:
  for nm, m in MAPS[nz].items():
    ok_all = m["pass_"]; inner = (np.abs(Yf) <= 1) & (np.abs(Tf) <= 1)
    summary[f"grid_{nm}@{nz}"] = dict(pass_rate=float(ok_all.mean()), pass_rate_absY_le1=float(ok_all[inner].mean()),
                                 points_all_pass=int((ok_all == 1).sum()), points_never_pass=int((ok_all == 0).sum()),
                                 ey_mean=float(m["ey"].mean()), et_mean=float(m["et"].mean()), ey_max=float(m["eymax"].max()), et_max=float(m["etmax"].max()),
                                 pass_by_origin={n: float(v) for n, v in zip(names, m["pass_by_origin"].mean(1))})
    # 단일축 직선만 (control의 원래 사용 조건: T=0 위 Y, Y=0 위 T)
    line = (Tf == 0) | (Yf == 0)
    summary[f"grid_{nm}@{nz}"]["pass_rate_on_axes"] = float(ok_all[line].mean())
json.dump(summary, open(OUT / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

# 대시보드용
r4 = lambda a: np.round(np.asarray(a, float), 4).tolist()
dash = dict(names=names, n=N_AUG, noises=NOISES, noise=summary["noise_control_px"], twin=agg, port=summary["port_check"], gen=summary["generator"],
            grid={nz: {nm: dict(ey=r4(m["ey"]), et=r4(m["et"]), eymax=r4(m["eymax"]), etmax=r4(m["etmax"]), pass_=r4(m["pass_"]),
                                by_origin=["".join(str(int(round(v * N_AUG))) for v in row) for row in m["pass_by_origin"]], summary=summary[f"grid_{nm}@{nz}"])
                       for nm, m in MAPS[nz].items()} for nz in NOISES},
            twins=[{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in trow])
(HERE.parents[2] / "apps" / "augment" / "eval.js").write_text("window.AUG_EVAL = " + json.dumps(dash, ensure_ascii=False, separators=(",", ":")).replace("NaN", "null") + ";\n", encoding="utf-8")
print(json.dumps({k: v for k, v in summary.items() if k.startswith("grid_")}, ensure_ascii=False, indent=1))
