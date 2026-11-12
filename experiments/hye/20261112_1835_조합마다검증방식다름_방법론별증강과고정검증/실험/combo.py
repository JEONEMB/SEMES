"""방법론(QR 표현 × 나사 측정 × 바 구성)마다 날 차이 증강 자료를 만들고, 모든 조합을 한 가지 검증 방식으로 채점한다.

고정 조건
- 사진: 범위 안 702장(ID 702 제외, T=+2.00 12장 제외). 커밋 46152fd `training_features.tsv`.
- 라벨: 절대 영점. `experiments/hye/20261112_1731_.../결과/절대영점_라벨_702장.tsv` 를 모든 조합에 똑같이 쓴다.
- 학습식: 00004 Huber(표준화, ridge 1e-7·n, Huber 6회). 통과 = |eY| ≤ 0.045 이고 |eT| ≤ 0.01.
- 분할: 자세 묶음(세션, 명령 Y, 명령 T 가 같은 사진) 단위. 맨 처음 20% 를 마지막 확인용으로 떼고(세션별 층화), 나머지 80%(개발)에서만 고른다.
  주 검증 = 개발 자료를 섞은 80/20(5조각 번갈아) 10회. 보조 = 세션별 절반 50/50 10회. 참고 = 세션 통째로 빼기.
  분할은 `결과/splits.json` 에 저장하고 모든 조합이 같은 분할을 쓴다.

방법론 48가지
- QR: ECC54(9곳), ECC24(모서리 4곳), ECC6(QR별 평균), P536_2(536점 평균), P536_12(2차식 P12), P536_20(3차식)
- 나사(4개): old(밝기 조각 맞춤, 00004), grad(기울기 영상 맞춤, 20261112_1756 실험)
- 바: both, up(위만), lo(아래만), none
증강: 없음 / 날 차이 증강(학습 자료의 세션별 "움직임으로 설명 안 되는 차이" 벡터에 N(0, 2²) 배율을 곱해 더한 복사본 8개, 정답 그대로).
  검증 안에서는 학습 조각만으로 벡터를 다시 구해 증강한다(시험 사진 정보가 새지 않게).

강건성: 주 검증 앞 3회의 모델에, 시험 사진의 원시 측정값을 흔들어 넣었을 때 통과율.
  QR 하나 가로 +0.3 px / QR마다 독립 잡음 0.1 px / 나사 하나 가로 +0.3 px / 나사마다 독립 잡음 0.1 px / 바 하나 +0.3 px.
  숫자 교란이며 사진을 바꿔 다시 잰 것이 아니다.

출력
- 결과/splits.json, 결과/결과.tsv(96행), 결과/마지막확인.tsv
- data/augmented/<이 실험 이름>/결과/증강자료/<방법론>/ : augmented.npz, day_vectors.tsv (702장 전체로 만든 학습용 자료)
분석용 Python 이며 팀 전달 C++ 도구를 대체하지 않는다.

실행: 저장소 루트에서  python "<이 폴더>/실험/combo.py"
"""
import csv, json, pathlib, sys, itertools
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "apps/distribution"))
import reoof

EXP = pathlib.Path(__file__).resolve().parent.parent
OUT = EXP / "결과"
AUG_OUT = ROOT / "data/augmented" / EXP.name / "결과" / "증강자료"
A0 = ROOT / "experiments/hye/20261102_124309_A0_분할측정고정_계층분할과점측정캐시/결과/cache"
R4 = ROOT / "experiments/hye/20261112_1504_QR요약54개_536점대조/결과/cache_round4"
LAB = ROOT / "experiments/hye/20261112_1731_세션마다영점다름_이상센터라벨과세션차이제거/결과/절대영점_라벨_702장.tsv"
MEAS = ROOT / "experiments/hye/20261112_1756_나사날마다어긋남_나사측정방식대조/결과/meas.npz"
SCALE, COPIES, SEED = 2.0, 8, 20261112
QR_OPTS = ["ECC54", "ECC24", "ECC6", "P536_2", "P536_12", "P536_20"]
SCREW_OPTS = ["old", "grad"]
BAR_OPTS = ["both", "up", "lo", "none"]
NAMES = {"20261014": "1차", "20261020": "2차", "20261029": "3차", "20261109_round4": "4차"}
ok = lambda E: (abs(E[:, 0]) <= .045) & (abs(E[:, 1]) <= .01)


def load():
    tf = reoof.tsv("training_features.tsv")
    rows = [r for r in tf if int(r["id"]) != 702 and abs(float(r["T"])) <= 1.0001]
    ids = np.array([int(r["id"]) for r in rows]); src = np.array([r["source"] for r in rows])
    X = np.array([[float(v) for v in r["features66"].split(";")] for r in rows])
    lab = {int(r["id"]): r for r in csv.DictReader(open(LAB, encoding="utf-8"), delimiter="\t")}
    L = np.array([[float(lab[i]["Y_abs"]), float(lab[i]["T_abs"])] for i in ids])
    cmd = np.array([[float(lab[i]["Y_cmd"]), float(lab[i]["T_cmd"])] for i in ids])
    shots = list(csv.DictReader(open(A0 / "shots.csv", encoding="utf-8-sig"))); pts = list(csv.DictReader(open(A0 / "points.csv", encoding="utf-8-sig")))
    sel = np.array([int(p["selected"]) for p in pts], bool)
    disp = np.fromfile(A0 / "disp.f32", np.float32).reshape(len(shots), len(pts), 2)
    by_path = {f"data/clean/{s['set']}/{s['rel']}": i for i, s in enumerate(shots)}
    ids4 = [int(x) for x in (R4 / "ids.txt").read_text().split()]
    disp4 = np.fromfile(R4 / "disp.f32", np.float32).reshape(len(ids4), len(pts), 2); by4 = {i: k for k, i in enumerate(ids4)}
    D = np.array([(disp4[by4[int(r["id"])]] if int(r["id"]) in by4 else disp[by_path[r["path"]]])[sel] for r in rows], float)
    xy = np.array([[float(p["x"]), float(p["y"])] for p in pts])[sel]; reg = np.array([int(p["region"]) for p in pts])[sel]
    M = np.load(MEAS); assert (M["ids"] == ids).all()
    raw = dict(qr=X[:, :54].reshape(-1, 3, 9, 2).copy(), d536=D, bars=X[:, 54:58].copy(), s_old=X[:, 58:].reshape(-1, 4, 2).copy(), s_grad=M["기울기영상_전체"].reshape(-1, 4, 2).copy())
    u = (xy - xy.mean(0)) / xy.std(0).max()
    def basis(order): return np.array([u[:, 0] ** i * u[:, 1] ** (d - i) for d in range(order + 1) for i in range(d, -1, -1)]).T
    geo = dict(reg=reg, regs=sorted(set(reg)), pinv2=np.linalg.pinv(basis(2)), pinv3=np.linalg.pinv(basis(3)))
    return ids, src, L, cmd, raw, geo


def build(raw, geo, q, s, b):
    n = len(raw["bars"])
    if q == "ECC54": Q = raw["qr"].reshape(n, 54)
    elif q == "ECC24": Q = raw["qr"][:, :, [0, 2, 6, 8]].reshape(n, 24)
    elif q == "ECC6": Q = raw["qr"].mean(2).reshape(n, 6)
    elif q == "P536_2": Q = raw["d536"].mean(1)
    elif q == "P536_12": Q = np.einsum('kp,npc->nkc', geo["pinv2"], raw["d536"]).reshape(n, -1)
    elif q == "P536_20": Q = np.einsum('kp,npc->nkc', geo["pinv3"], raw["d536"]).reshape(n, -1)
    S = raw["s_old" if s == "old" else "s_grad"].reshape(n, 8)
    B = {"both": raw["bars"], "up": raw["bars"][:, :2], "lo": raw["bars"][:, 2:], "none": raw["bars"][:, :0]}[b]
    return np.c_[Q, B, S]


def perturbations(raw, geo, b, rng):
    """(이름, 교란된 raw) 목록. 같은 이름은 나중에 평균한다."""
    def cp(): return {k: v.copy() for k, v in raw.items()}
    out = []
    for qi, rg in enumerate(geo["regs"]):
        r = cp(); r["qr"][:, qi, :, 0] += .3; r["d536"][:, geo["reg"] == rg, 0] += .3; out.append(("QR 하나 +0.3", r))
    r = cp(); z = rng.normal(0, .1, (len(raw["bars"]), 3, 2))
    for qi, rg in enumerate(geo["regs"]): r["qr"][:, qi] += z[:, qi][:, None, :]; r["d536"][:, geo["reg"] == rg] += z[:, qi][:, None, :]
    out.append(("QR 잡음 0.1", r))
    for k in range(4):
        r = cp(); r["s_old"][:, k, 0] += .3; r["s_grad"][:, k, 0] += .3; out.append(("나사 하나 +0.3", r))
    r = cp(); z = rng.normal(0, .1, (len(raw["bars"]), 4, 2)); r["s_old"] += z; r["s_grad"] += z; out.append(("나사 잡음 0.1", r))
    for bi in {"both": [0, 1], "up": [0], "lo": [1], "none": []}[b]:
        r = cp(); r["bars"][:, 2 * bi] += .3; out.append(("바 하나 +0.3", r))
    return out


def fit(F, Y):
    mean = F.mean(0); sc = np.maximum(F.std(0), 1e-6); A = np.c_[np.ones(len(F)), (F - mean) / sc]
    pen = np.eye(A.shape[1]) * (1e-7 * len(F)); pen[0, 0] = 0; w = np.ones(len(F))
    for _ in range(6):
        Aw = A * w[:, None]; B = np.linalg.lstsq(Aw.T @ A + pen, Aw.T @ Y, rcond=None)[0]; P = A @ B
        r = np.maximum(abs(P[:, 0] - Y[:, 0]) / .045, abs(P[:, 1] - Y[:, 1]) / .01); w = np.minimum(1, 1 / np.maximum(r, 1e-12)); w *= len(w) / w.sum()
    return lambda G: np.c_[np.ones(len(G)), (G - mean) / sc] @ B


def day_vectors(F, L, src, cmd):
    """학습 자료만으로 세션별 '움직임으로 설명 안 되는 차이' 벡터를 구한다."""
    S = sorted(set(src)); wide = [s for s in S if len(set(np.round(cmd[src == s, 0], 2))) > 5] or S
    m = np.isin(src, wide); A = np.c_[np.array([(src[m] == s) for s in wide]).T.astype(float), L[m]]
    J = np.linalg.lstsq(A, F[m], rcond=None)[0][len(wide):]
    V = []
    for s in S:
        a = (F[src == s] - L[src == s] @ J).mean(0); c = np.linalg.lstsq(J.T, a, rcond=None)[0]; V.append(a - c @ J)
    return S, np.array(V)


def augment(F, L, src, cmd, rng):
    S, V = day_vectors(F, L, src, cmd)
    Z = rng.normal(0, SCALE, (COPIES, len(F), len(S)))
    return np.vstack([F] + [F + Z[k] @ V for k in range(COPIES)]), np.vstack([L] * (COPIES + 1)), Z, S, V


def make_splits(ids, src, cmd):
    keys = [(a, round(y, 2), round(t, 2)) for a, (y, t) in zip(src, cmd)]; uk = sorted(set(keys)); gi = np.array([uk.index(k) for k in keys]); gs = np.array([k[0] for k in uk])
    rng = np.random.default_rng(SEED); final_g = np.zeros(len(uk), bool)
    for s in sorted(set(gs)):
        idx = np.where(gs == s)[0]; final_g[rng.permutation(idx)[: round(len(idx) * .2)]] = True
    final = final_g[gi]; dev = ~final
    dg = np.where(~final_g)[0]; cv, half = [], []
    for seed in range(10):
        r = np.random.default_rng(1000 + seed); f = np.full(len(uk), -1); f[r.permutation(dg)] = np.arange(len(dg)) % 5; cv.append(f[gi])
        h = np.full(len(uk), -1)
        for s in sorted(set(gs)):
            idx = np.array([g for g in dg if gs[g] == s]); p = r.permutation(idx); h[p[: len(p) // 2]] = 0; h[p[len(p) // 2:]] = 1
        half.append(h[gi])
    return dev, final, cv, half


def main():
    ids, src, L, cmd, raw, geo = load()
    dev, final, cv, half = make_splits(ids, src, cmd)
    OUT.mkdir(exist_ok=True)
    (OUT / "splits.json").write_text(json.dumps(dict(dev=ids[dev].tolist(), final=ids[final].tolist(),
        cv=[{str(int(i)): int(f) for i, f in zip(ids[dev], c[dev])} for c in cv], half=[{str(int(i)): int(f) for i, f in zip(ids[dev], h[dev])} for h in half]), ensure_ascii=False), encoding="utf-8")
    print(f"사진 {len(ids)}장: 개발 {dev.sum()}, 마지막 확인 {final.sum()}", {NAMES[s]: (int((dev & (src == s)).sum()), int((final & (src == s)).sum())) for s in sorted(set(src))}, flush=True)
    S_all = sorted(set(src)); rows = []; store = {}
    for q, s, b in itertools.product(QR_OPTS, SCREW_OPTS, BAR_OPTS):
        name = f"QR-{q}_나사-{s}_바-{b}"; F = build(raw, geo, q, s, b)
        P = [(nm, build(r, geo, q, s, b)) for nm, r in perturbations(raw, geo, b, np.random.default_rng(SEED))]
        # 학습용 증강 자료(702장 전체)
        d = AUG_OUT / name
        if not d.exists():
            Fa, La, Z, Ss, V = augment(F, L, src, cmd, np.random.default_rng(SEED)); d.mkdir(parents=True)
            np.savez_compressed(d / "augmented.npz", features=Fa, labels_abs=La, id=np.tile(ids, COPIES + 1), copy=np.repeat(np.arange(COPIES + 1), len(ids)),
                                source=np.tile(src, COPIES + 1), z=np.vstack([np.zeros((len(ids), len(Ss)))] + list(Z)), day_vectors=V, sessions=np.array(Ss))
            with (d / "day_vectors.tsv").open("w", encoding="utf-8", newline="") as f:
                w = csv.writer(f, delimiter="\t"); w.writerow(["session", "name", "rms_px", "vector"])
                for ss, v in zip(Ss, V): w.writerow([ss, NAMES[ss], f"{np.sqrt((v ** 2).mean()):.4f}", ";".join(repr(float(x)) for x in v)])
        for aug in (False, True):
            def train(tr, seed):
                Ft, Lt = F[tr], L[tr]
                if aug: Ft, Lt = augment(Ft, Lt, src[tr], cmd[tr], np.random.default_rng(SEED + seed))[:2]
                return fit(Ft, Lt)
            fails, sy, st, my, mt = [], [], [], 0, 0; rob = {}
            for k, f in enumerate(cv):
                E = np.zeros_like(L); EP = [np.zeros_like(L) for _ in P]
                for j in range(5):
                    te = f == j; tr = dev & ~te; m = train(tr, 10 * k + j); E[te] = m(F[te]) - L[te]
                    if k < 3:
                        for e, (_, Fp) in zip(EP, P): e[te] = m(Fp[te]) - L[te]
                Ed = E[dev]; fails.append(int((~ok(Ed)).sum())); sy.append(Ed[:, 0].std()); st.append(Ed[:, 1].std()); my = max(my, abs(Ed[:, 0]).max()); mt = max(mt, abs(Ed[:, 1]).max())
                if k < 3:
                    for e, (nm, _) in zip(EP, P): rob.setdefault(nm, []).append(ok(e[dev]).mean() * 100)
            hf = []
            for k, h in enumerate(half):
                E = np.zeros_like(L)
                for j in (0, 1): te = h == j; m = train(dev & ~te, 500 + 2 * k + j); E[te] = m(F[te]) - L[te]
                hf.append(int((~ok(E[dev])).sum()))
            lo = []
            for si, ss in enumerate(S_all):
                te = dev & (src == ss); m = train(dev & ~te, 900 + si); lo.append(ok(m(F[te]) - L[te]).mean() * 100)
            lo_all = sum(v * (dev & (src == ss)).sum() for v, ss in zip(lo, S_all)) / dev.sum()
            R = {nm: float(np.mean(v)) for nm, v in rob.items()}
            row = dict(방법론=name, QR=q, 나사=s, 바=b, 증강="날 차이" if aug else "없음", 값수=F.shape[1], 탈락평균=np.mean(fails), 탈락최악=max(fails), 탈락0회수=sum(v == 0 for v in fails),
                       sY=np.mean(sy), sT=np.mean(st), 최대eY=my, 최대eT=mt, 절반탈락평균=np.mean(hf), 절반탈락최악=max(hf), 처음보는날=lo_all, **{f"처음_{NAMES[ss]}": v for ss, v in zip(S_all, lo)},
                       **{f"강건_{nm}": R.get(nm, float("nan")) for nm in ("QR 하나 +0.3", "QR 잡음 0.1", "나사 하나 +0.3", "나사 잡음 0.1", "바 하나 +0.3")}, 강건최소=min(R.values()))
            rows.append(row); store[(name, aug)] = (q, s, b)
            print(f"{name}\t{row['증강']}\t값 {row['값수']}\t탈락 {row['탈락평균']:.1f} (최악 {row['탈락최악']})\t절반 {row['절반탈락평균']:.1f}\t강건최소 {row['강건최소']:.1f}\t처음 {row['처음보는날']:.1f}", flush=True)
    cols = list(rows[0])
    with (OUT / "결과.tsv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t"); w.writerow(cols)
        for r in rows: w.writerow([f"{r[c]:.5f}" if isinstance(r[c], float) else r[c] for c in cols])
    # 고르기: 주 검증 탈락이 최소+0.5장 이내인 것 중 강건최소가 큰 순, 다음 절반 탈락, 다음 값 수
    best = min(r["탈락평균"] for r in rows); cand = sorted([r for r in rows if r["탈락평균"] <= best + .5], key=lambda r: (-r["강건최소"], r["절반탈락평균"], r["값수"]))
    print("\n주 검증 최소 탈락", best, "| 후보", len(cand))
    with (OUT / "마지막확인.tsv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t"); w.writerow(["순위", "방법론", "증강", "값수", "개발_탈락평균", "개발_절반탈락", "개발_강건최소", "마지막확인_탈락", "마지막확인_장수", "마지막확인_최대eY", "마지막확인_최대eT", "마지막확인_강건최소"])
        base = next(r for r in rows if r["방법론"] == "QR-ECC54_나사-old_바-both" and r["증강"] == "없음")
        for rank, r in enumerate(cand[:5] + [base], 1):
            q, s, b = store[(r["방법론"], r["증강"] != "없음")]; F = build(raw, geo, q, s, b); Ft, Lt = F[dev], L[dev]
            if r["증강"] != "없음": Ft, Lt = augment(Ft, Lt, src[dev], cmd[dev], np.random.default_rng(SEED))[:2]
            m = fit(Ft, Lt); E = m(F[final]) - L[final]
            rb = {}
            for nm, rr in perturbations(raw, geo, b, np.random.default_rng(SEED + 1)): rb.setdefault(nm, []).append(ok(m(build(rr, geo, q, s, b)[final]) - L[final]).mean() * 100)
            out = [rank if rank <= 5 else "base", r["방법론"], r["증강"], r["값수"], f"{r['탈락평균']:.1f}", f"{r['절반탈락평균']:.1f}", f"{r['강건최소']:.1f}", int((~ok(E)).sum()), int(final.sum()),
                   f"{abs(E[:,0]).max():.4f}", f"{abs(E[:,1]).max():.4f}", f"{min(np.mean(v) for v in rb.values()):.1f}"]
            w.writerow(out); print("\t".join(map(str, out)))


if __name__ == "__main__":
    main()
