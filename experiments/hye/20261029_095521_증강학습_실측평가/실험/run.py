"""증강 자료로만 학습한 joint, control을 실측 사진으로 평가한다 (실험용 Python, 팀 전달 코드 아님).

계획은 PROTOCOL.md에 실행 전 고정했다.
입력: ../20261028_211553_YT증강_점별변위_결합항/inputs (채택 C++ 모듈이 실측 341장에서 뽑은 특징, 다시 측정하지 않음)
결과: results/ (predictions.csv, summary.json, check.json)
"""
import csv, json, re, hashlib
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
HYE = HERE.parent
SRC = HYE / "20261028_211553_YT증강_점별변위_결합항" / "inputs"
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)

TOL, SEED = 0.09, 20261029
N_JOINT_AUG = 20000
CTRL_STEP = {"Y": 0.01, "T": 0.002}
CTRL_REPEAT = 5
TRIALS = ["S1", "S2", "S3", "S4", "S5", "S6"]
TRIAL = {"Y1+": "S1", "Y1-": "S2", "T+": "S3", "T-": "S4", "Y2+": "S5", "Y2-": "S6"}
SESSION = {v: k for k, v in TRIAL.items()}  # S1 -> "Y1+"
FAMILIES = {"Y1_T1": (1, 1, False), "Y1_T1s": (1, 1, True), "Y2_T2s": (2, 2, True),
            "Y3_T2s": (3, 2, True), "Y3_T3": (3, 3, False), "Y3_T3s": (3, 3, True)}


def basis(Y, T, fam):  # 앞선 실험 generate.py 와 같은 식 (그 파일은 import 하면 결과를 다시 써서 복사했다)
    dy, dt, sg = FAMILIES[fam]
    Y, T = np.asarray(Y, float), np.asarray(T, float)
    cols = [Y ** d for d in range(1, dy + 1)]
    if sg: cols += [np.maximum(T, 0) ** d for d in range(1, dt + 1)] + [np.minimum(T, 0) ** d for d in range(1, dt + 1)]
    else: cols += [T ** d for d in range(1, dt + 1)]
    return np.stack(cols, -1)


# ---------- 사진 정보 (앞선 실험 module_eval.py 와 같은 해석) ----------
rows = list(csv.DictReader(open(SRC / "manifest.csv", encoding="utf-8-sig")))
info = []
for i, r in enumerate(rows):
    date = "20261020" if r["relative"].startswith("20261020") else "20261014"
    k = re.search(r"\((\d)\)", r["file"])
    trial = TRIAL.get(r["group"], "C14" if date == "20261014" else ("C20" if r["scope"] == "center" else "MIX"))
    info.append(dict(i=i, date=date, Y=float(r["Y"]), T=float(r["T"]), trial=trial, file=r["relative"]))
N = len(info)
I14 = [x["i"] for x in info if x["date"] == "20261014"]
I14S = [i for i in I14 if info[i]["trial"] in TRIALS]
I20 = [x["i"] for x in info if x["date"] == "20261020"]
LY, LT = np.array([x["Y"] for x in info]), np.array([x["T"] for x in info])
rank = {}
for s in TRIALS:
    ids = sorted((i for i in I14 if info[i]["trial"] == s), key=lambda i: abs(info[i]["Y"]) + abs(info[i]["T"]))
    for r_, i in enumerate(ids): rank[i] = r_ % 4


def read_rows(name):
    out = {}
    for line in open(SRC / "module_features" / name, encoding="utf-8"):
        if line.startswith("index"): continue
        p = line.rstrip("\n").split(","); out[int(p[0])] = p[1:]
    return out
jo, cd = read_rows("joint_obs.csv"), read_rows("control_disp.csv")
XJ = np.array([[float(v) for v in jo[i]] for i in range(N)])
XC = np.array([[float(v) for v in cd[i]] for i in range(N)])[:, :1260]

# ---------- 특징 ----------
PTS = json.load(open(SRC / "points.json"))["points"]
SEL = np.array([p[3] for p in PTS], bool)
_u = np.array([(p[0] - 1906) / 486 for p in PTS]); _v = np.array([(p[1] - 1070) / 122 for p in PTS])
P12 = np.linalg.pinv(np.stack([np.ones(630), _u, _v, _u * _u, _u * _v, _v * _v], 1)[SEL])
def p12(X):
    D = np.atleast_2d(X).reshape(-1, 630, 2)[:, SEL]
    return np.einsum("jp,npc->njc", P12, D).reshape(len(D), 12)
def fmap(X):  # 채택 joint feature_map 과 같은 식
    X = np.atleast_2d(X); h = X[:, 0:54:2].sum(1, keepdims=True) / 2700.0
    return np.concatenate([X, X * h], 1)


# ---------- 학습기 ----------
def shrink_corr(R, dof, shrink):
    cov = R.T @ R / dof; sd = np.sqrt(np.maximum(np.diag(cov), 1e-14))
    return cov, sd, (1 - shrink) * cov / np.outer(sd, sd) + shrink * np.eye(len(sd))

def train_joint(X, Y, T):
    Z = fmap(X); A = np.stack([np.ones(len(Z)), Y / 2, T / 2], 1)
    sol = np.linalg.lstsq(A, Z, rcond=None)[0]; origin, coef = sol[0], sol[1:]
    _, sd, corr = shrink_corr(Z - A @ sol, len(Z) - 3, 0.01)
    Sinv = np.linalg.inv(corr * np.outer(sd, sd))
    proj = Sinv @ coef.T; gram = coef @ proj
    return dict(origin=origin, proj=proj, initial=np.linalg.inv(gram))

def predict_joint(m, X):
    q = (fmap(X) - m["origin"]) @ m["proj"] @ m["initial"] * 2
    return q[:, 0], q[:, 1]

NAMES = ["Y1+", "Y1-", "Y2+", "Y2-", "T+", "T-"]
def train_control(F, session, value):
    """P12_vs_J204 src/models.cpp train_yort 를 옮김"""
    fitted = {}; F = np.asarray(F); session = np.asarray(session); value = np.asarray(value, float)
    for axis in ("Y", "T"):
        sess = [n for n in NAMES if n[0] == axis]; k = np.array([s[0] == axis for s in session])
        A = np.zeros((k.sum(), len(sess) + 2)); A[:, 0] = value[k]; A[:, 1] = value[k] ** 2
        for r_, s in enumerate(session[k]): A[r_, 2 + sess.index(s)] = 1
        c = np.linalg.lstsq(A, F[k], rcond=None)[0]
        for j, s in enumerate(sess): fitted[s] = (axis, c[j + 2], c[0], c[1])
    R = np.array([F[r_] - fitted[s][1] - value[r_] * fitted[s][2] - value[r_] ** 2 * fitted[s][3] for r_, s in enumerate(session)])
    _, sd, corr = shrink_corr(R, max(len(R) - 16, 1), 0.05)
    lam, vec = np.linalg.eigh(corr)
    W = (vec / np.sqrt(lam)).T / sd[None]
    return dict(W=W, branches=[(n, *fitted[n]) for n in NAMES])

def predict_control(m, F):
    F = np.atleast_2d(F); W = m["W"]; best_cost = np.full(len(F), np.inf); best_val = np.zeros(len(F)); best_ax = np.full(len(F), "", object)
    for name, ax, o, s, c in m["branches"]:
        z = (F - o) @ W.T; s_, c_ = W @ s, W @ c; val = z @ s_ / (s_ @ s_)
        for _ in range(8):
            j = s_[None] + 2 * val[:, None] * c_[None]; r_ = z - val[:, None] * s_[None] - (val ** 2)[:, None] * c_[None]
            val = val + (j * r_).sum(1) / (j * j).sum(1)
        res = z - val[:, None] * s_[None] - (val ** 2)[:, None] * c_[None]; cost = (res * res).sum(1)
        cost[val * (1 if name.endswith("+") else -1) < 0] = np.inf
        upd = cost < best_cost; best_cost[upd] = cost[upd]; best_val[upd] = val[upd]; best_ax[upd] = ax
    return np.where(best_ax == "Y", best_val, 0.0), np.where(best_ax == "T", best_val, 0.0), best_ax

# 채택 동결 모델 (참고, 재대입)
JM = json.load(open(HYE / "20261028_1258_final_Y_T_joint" / "assets" / "both_model.json", encoding="utf-8"))["model"]
FROZEN_J = dict(origin=np.array(JM["origin"]).reshape(-1), proj=np.array(JM["projector"]), initial=np.array(JM["initial"]))
assert np.allclose(JM["scale"], [2, 2])
CM = json.load(open(HYE / "20261028_1327_final_Y_T control" / "assets" / "model.json", encoding="utf-8"))
FROZEN_C = dict(W=np.array(CM["W"]), branches=[(n, b[0], np.array(b[1]), np.array(b[2]), np.array(b[3])) for n, b in CM["branches"].items()])
FROZEN_C["branches"].sort(key=lambda b: NAMES.index(b[0]))

FC = p12(XC)
sess_of = lambda ids: [SESSION[info[i]["trial"]] for i in ids]
val_of = lambda ids: [info[i]["Y"] if SESSION[info[i]["trial"]][0] == "Y" else info[i]["T"] for i in ids]

# ---------- 0. 학습기 확인 ----------
check = {}
retr = train_control(FC[I14S], sess_of(I14S), val_of(I14S))
fy, ft, fa = predict_control(FROZEN_C, FC); ry, rt, ra = predict_control(retr, FC)
check["control_retrain_vs_frozen"] = dict(
    W_max_abs_diff=float(np.abs(np.abs(retr["W"]) - np.abs(FROZEN_C["W"])).max()),
    branch_max_abs_diff=float(max(np.abs(a[k] - b[k]).max() for a, b in zip(retr["branches"], FROZEN_C["branches"]) for k in (2, 3, 4))),
    axis_agree=int((fa == ra).sum()), estimate_max_abs_diff=float(np.abs((fy + ft) - (ry + rt)).max()), photos=N)
jr = train_joint(XJ, LY, LT); a_ = predict_joint(FROZEN_J, XJ); b_ = predict_joint(jr, XJ)
check["joint_retrain_vs_frozen_341"] = dict(note="원래 학습 코드가 없어 같은 구조로 다시 만든 학습기. 계수는 같지 않다.",
    y_max_abs_diff=float(np.abs(a_[0] - b_[0]).max()), t_max_abs_diff=float(np.abs(a_[1] - b_[1]).max()),
    frozen_pass=int(((np.abs(a_[0] - LY) <= TOL) & (np.abs(a_[1] - LT) <= TOL)).sum()),
    retrained_pass=int(((np.abs(b_[0] - LY) <= TOL) & (np.abs(b_[1] - LT) <= TOL)).sum()))
print(json.dumps(check, ensure_ascii=False, indent=1), flush=True)


# ---------- 생성기 (1014 사진만) ----------
def fit_single(X, ids, fam):
    tr = [info[i]["trial"] for i in ids]; trials = [t for t in TRIALS if t in tr]
    A = np.concatenate([np.array([[t == s for s in trials] for t in tr], float), basis([info[i]["Y"] for i in ids], [info[i]["T"] for i in ids], fam)], 1)
    sol = np.linalg.lstsq(A, X[ids], rcond=None)[0]
    off = np.zeros((len(TRIALS), X.shape[1]))
    for j, s in enumerate(trials): off[TRIALS.index(s)] = sol[j]
    return dict(fam=fam, off=off, coef=sol[len(trials):], ntr=len(trials))
resp = lambda fit, Y, T: basis(np.atleast_1d(Y), np.atleast_1d(T), fit["fam"]) @ fit["coef"]
offset = lambda fit, i: fit["off"][TRIALS.index(info[i]["trial"])] if info[i]["trial"] in TRIALS else 0

def build(X, ids):
    """ids (1014 사진) 만으로 맞춘 생성기"""
    folds = sorted({rank[i] for i in ids if i in rank}); cv = {}
    for fam in FAMILIES:
        err = []
        for f in folds:
            fit = fit_single(X, [i for i in ids if rank.get(i, -1) != f], fam)
            err += [X[i] - offset(fit, i) - resp(fit, info[i]["Y"], info[i]["T"])[0] for i in ids if rank.get(i, -1) == f]
        cv[fam] = float(np.sqrt(np.mean(np.square(err))))
    fam = min(cv, key=cv.get); fit = fit_single(X, ids, fam)
    O = {"C14": np.zeros(X.shape[1]), **{s: fit["off"][j] for j, s in enumerate(TRIALS)}}
    M = np.stack(list(O.values())); mu = M.mean(0); _, S, Vt = np.linalg.svd(M - mu, full_matrices=False); sg = S / np.sqrt(len(M) - 1)
    for k in range(2):
        for s in (1, -1): O[f"PC{k + 1}{'+' if s > 0 else '-'}2s"] = mu + s * 2 * sg[k] * Vt[k]
    r_ = np.array([X[i] - offset(fit, i) - resp(fit, info[i]["Y"], info[i]["T"])[0] for i in ids])
    p = fit["ntr"] + fit["coef"].shape[0]
    return dict(fam=fam, cv=cv, fit=fit, O=O, bplus=np.mean([O["S1"], O["S3"], O["S5"]], 0), bminus=np.mean([O["S2"], O["S4"], O["S6"]], 0),
                noise=r_ * np.sqrt(len(ids) / (len(ids) - p)))

def joint_samples(G, rng):
    Y = rng.uniform(-2, 2, N_JOINT_AUG); T = rng.uniform(-1, 1, N_JOINT_AUG)
    names = list(G["O"]); pick = rng.integers(len(names), size=N_JOINT_AUG)
    base = np.stack([G["O"][names[k]] for k in pick]) + basis(Y, T, G["fam"]) @ G["fit"]["coef"]
    sgn = np.where(T != 0, np.sign(T), np.sign(Y)); center = np.array([names[k] == "C14" for k in pick])
    base[center] += np.where(sgn[center, None] > 0, G["bplus"][None], G["bminus"][None])
    return base + G["noise"][rng.integers(len(G["noise"]), size=N_JOINT_AUG)], Y, T

def control_samples(G, ids, rng):
    X, sess, val = [], [], []
    for s in TRIALS:
        real = [info[i]["Y"] if SESSION[s][0] == "Y" else info[i]["T"] for i in ids if info[i]["trial"] == s]
        if not real: continue
        ax = SESSION[s][0]; step = CTRL_STEP[ax]; lo, hi = min(real), max(real)
        vals = np.round(np.arange(lo, hi + step / 2, step), 6)
        Y = vals if ax == "Y" else np.zeros_like(vals); T = vals if ax == "T" else np.zeros_like(vals)
        base = G["O"][s][None] + basis(Y, T, G["fam"]) @ G["fit"]["coef"]
        for _ in range(CTRL_REPEAT):
            X.append(base + G["noise"][rng.integers(len(G["noise"]), size=len(vals))]); sess += [SESSION[s]] * len(vals); val += list(vals)
    return np.concatenate(X), sess, val


# ---------- 실행 ----------
rng = np.random.default_rng(SEED)
pred = {}   # (모델, 사진) -> (y, t, 축)
gens = {}
def run_split(name, train_ids, test_ids):
    tr_s = [i for i in train_ids if info[i]["trial"] in TRIALS]
    GJ, GC = build(XJ, train_ids), build(XC, train_ids)
    gens[name] = dict(joint=dict(family=GJ["fam"], shape_cv=GJ["cv"]), control=dict(family=GC["fam"], shape_cv=GC["cv"]))
    xa, ya, ta = joint_samples(GJ, rng)
    xc, sc, vc = control_samples(GC, tr_s, rng)
    gens[name]["samples"] = dict(joint=len(xa), control=len(xc))
    models = {"joint_실측": ("j", train_joint(XJ[train_ids], LY[train_ids], LT[train_ids])),
              "joint_증강": ("j", train_joint(xa, ya, ta)),
              "control_실측": ("c", train_control(FC[tr_s], sess_of(tr_s), val_of(tr_s))),
              "control_증강": ("c", train_control(p12(xc), sc, vc))}
    for mname, (kind, m) in models.items():
        if kind == "j": y, t = predict_joint(m, XJ[test_ids]); ax = np.full(len(test_ids), "", object)
        else: y, t, ax = predict_control(m, FC[test_ids])
        for k, i in enumerate(test_ids): pred[(mname, i)] = (float(y[k]), float(t[k]), str(ax[k]), name)
    print("done", name, gens[name], flush=True)

for f in range(4):
    test = [i for i in I14S if rank[i] == f]
    run_split(f"A{f}", [i for i in I14 if i not in test], test)
run_split("B", I14, I20)
fy, ft, fa = predict_control(FROZEN_C, FC); jy, jt = predict_joint(FROZEN_J, XJ)
for i in I14S + I20:
    pred[("joint_동결", i)] = (float(jy[i]), float(jt[i]), "", "재대입")
    pred[("control_동결", i)] = (float(fy[i]), float(ft[i]), str(fa[i]), "재대입")

# ---------- 채점 ----------
def group(i):
    x = info[i]
    if x["date"] == "20261014": return "A 1014 단일축"
    if x["trial"] == "C20": return "B 1020 센터"
    return "B 1020 혼합 T≤1" if x["T"] <= 1 else "B 1020 혼합 T=2 (범위 밖)"
MODELS = ["joint_실측", "joint_증강", "joint_동결", "control_실측", "control_증강", "control_동결"]
with open(OUT / "predictions.csv", "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.writer(fh); w.writerow(["file", "group", "label_y", "label_t", "model", "split", "est_y", "est_t", "axis", "error_y", "error_t", "pass"])
    for mname in MODELS:
        for i in I14S + I20:
            y, t, ax, sp = pred[(mname, i)]; ey, et = abs(y - LY[i]), abs(t - LT[i])
            w.writerow([info[i]["file"], group(i), LY[i], LT[i], mname, sp, y, t, ax, ey, et, int(ey <= TOL and et <= TOL)])

summary = {"check": check, "generators": gens, "results": {}}
GROUPS = ["A 1014 단일축", "B 1020 센터", "B 1020 혼합 T≤1", "B 1020 혼합 T=2 (범위 밖)"]
for mname in MODELS:
    summary["results"][mname] = {}
    for g in GROUPS:
        ids = [i for i in I14S + I20 if group(i) == g]
        e = np.array([[abs(pred[(mname, i)][0] - LY[i]), abs(pred[(mname, i)][1] - LT[i])] for i in ids])
        d = dict(n=len(ids), pass_=int(((e[:, 0] <= TOL) & (e[:, 1] <= TOL)).sum()), y_mean=float(e[:, 0].mean()), y_max=float(e[:, 0].max()),
                 t_mean=float(e[:, 1].mean()), t_max=float(e[:, 1].max()))
        if g.startswith("A"):
            for ax, lab in (("Y", LT == 0), ("T", LY == 0)):
                k = [i for i in ids if (info[i]["trial"] in TRIALS and SESSION[info[i]["trial"]][0] == ax)]
                d[f"{ax}만_{'y' if ax == 'Y' else 't'}_mean"] = float(np.mean([abs(pred[(mname, i)][0 if ax == 'Y' else 1] - (LY if ax == 'Y' else LT)[i]) for i in k]))
                d[f"{ax}만_pass"] = int(sum(abs(pred[(mname, i)][0] - LY[i]) <= TOL and abs(pred[(mname, i)][1] - LT[i]) <= TOL for i in k))
            if mname.startswith("control"):
                d["axis_correct"] = int(sum(pred[(mname, i)][2] == SESSION[info[i]["trial"]][0] for i in ids))
        summary["results"][mname][g] = d
summary["inputs_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [SRC / "manifest.csv", SRC / "points.json", SRC / "module_features" / "joint_obs.csv", SRC / "module_features" / "control_disp.csv"]}
json.dump(summary, open(OUT / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
for mname in MODELS:
    print(mname, {g: (f"{v['pass_']}/{v['n']}", round(v["y_mean"], 4), round(v["y_max"], 4), round(v["t_mean"], 4), round(v["t_max"], 4)) for g, v in summary["results"][mname].items()})
