"""보고서(00000_보고서.md) 빈칸 계산. 기준 구성(값 7개)에서 한 가지씩만 바꾼다.

20261112_1835 실험의 combo.py 에서 load / make_splits / fit / perturbations 를 그대로 불러 쓴다(같은 702장, 절대 영점 라벨, 같은 분할, 같은 학습식, 같은 교란).
바꾸는 것: 쓰는 QR(3개 중), 쓰는 나사(4개 중), 쓰는 바 기울기(위·아래), 계산식. 기준 구성의 회차별·조각별·세션별 오차와 값별 반복 흔들림도 낸다.
분석용 Python 이며 팀 전달 C++ 도구를 대체하지 않는다.

실행: 저장소 루트에서  python "<이 폴더>/실험/one_at_a_time.py" > "<이 폴더>/결과/출력.txt"   → 결과/결과.tsv
"""
import sys, pathlib, itertools, csv, io
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = pathlib.Path.cwd()
sys.path.insert(0, str(ROOT / "experiments/hye/20261112_1835_조합마다검증방식다름_방법론별증강과고정검증/실험"))
import combo
from combo import fit, ok, SEED, NAMES

ids, src, L, cmd, raw, geo = combo.load()
dev, final, cv, half = combo.make_splits(ids, src, cmd)
S_all = sorted(set(src)); n = len(ids)
regs = geo["regs"]; reg = geo["reg"]

# QR 영역의 좌우 순서 확인
pts = list(csv.DictReader(open(combo.A0 / "points.csv", encoding="utf-8-sig")))
sel = np.array([int(p["selected"]) for p in pts], bool)
px = np.array([float(p["x"]) for p in pts])[sel]
print("QR 영역별 평균 x, 점 수:", {int(r): (round(float(px[reg == r].mean())), int((reg == r).sum())) for r in regs})
order = np.argsort([px[reg == r].mean() for r in regs]); QN = {int(order[0]): "왼쪽", int(order[1]): "가운데", int(order[2]): "오른쪽"}
print("QR 이름:", QN)


def build(r, qr, screws, bars):
    """qr: ('P', [영역 인덱스]) 536점 가로 평균 1값 / ('Pper', [..]) 영역별 평균 / ('E', [..]) ECC QR별 가로 평균 / None
    screws: 나사 인덱스 목록(grad 가로). bars: 'tilt' | 'up' | 'lo' | 'none'"""
    cols = []
    if qr:
        kind, idx = qr
        if kind == "P": cols.append(r["d536"][:, np.isin(reg, [regs[i] for i in idx]), 0].mean(1)[:, None])
        elif kind == "Pper": cols.append(np.c_[tuple(r["d536"][:, reg == regs[i], 0].mean(1) for i in idx)])
        elif kind == "E": cols.append(r["qr"][..., 0][:, idx].mean(2))
        elif kind == "Emean": cols.append(r["qr"][..., 0][:, idx].mean((1, 2))[:, None])
    cols.append({"tilt": r["bars"][:, [1, 3]], "up": r["bars"][:, [1]], "lo": r["bars"][:, [3]], "none": r["bars"][:, :0]}[bars])
    cols.append(r["s_grad"][:, screws, 0])
    return np.hstack(cols)


PERT = combo.perturbations(raw, geo, "none", np.random.default_rng(SEED))
# 바 기울기 교란(기준 구성은 바 세로 이동을 쓰지 않아 combo 의 바 교란이 해당 없음): 기울기 값에 그 값 표준편차의 10% 를 더한다
tilt_sd = raw["bars"][:, [1, 3]].std(0)


def evaluate(name, spec, fitter=fit, detail=False):
    F = build(raw, *spec); P = [(nm, build(r, *spec)) for nm, r in PERT]
    fails, sy, st, my, mt = [], [], [], 0, 0; rob = {}; Eall = []
    for k, f in enumerate(cv):
        E = np.zeros_like(L); EP = [np.zeros_like(L) for _ in P]
        for j in range(5):
            te = f == j; m = fitter(F[dev & ~te], L[dev & ~te]); E[te] = m(F[te]) - L[te]
            if k < 3:
                for e, (_, Fp) in zip(EP, P): e[te] = m(Fp[te]) - L[te]
        Ed = E[dev]; fails.append(int((~ok(Ed)).sum())); sy.append(Ed[:, 0].std()); st.append(Ed[:, 1].std())
        my = max(my, abs(Ed[:, 0]).max()); mt = max(mt, abs(Ed[:, 1]).max()); Eall.append(E.copy())
        if k < 3:
            for e, (nm, _) in zip(EP, P): rob.setdefault(nm, []).append(ok(e[dev]).mean() * 100)
    hf = []
    for k, h in enumerate(half):
        E = np.zeros_like(L)
        for j in (0, 1): te = h == j; m = fitter(F[dev & ~te], L[dev & ~te]); E[te] = m(F[te]) - L[te]
        hf.append(int((~ok(E[dev])).sum()))
    lo = []; Elo = np.zeros_like(L)
    for ss in S_all:
        te = dev & (src == ss); m = fitter(F[dev & ~te], L[dev & ~te]); Elo[te] = m(F[te]) - L[te]; lo.append(ok(Elo[te]).mean() * 100)
    lo_all = ok(Elo[dev]).mean() * 100
    m = fitter(F[dev], L[dev]); Ef = m(F[final]) - L[final]
    R = {nm: float(np.mean(v)) for nm, v in rob.items()}
    row = dict(구성=name, 값수=F.shape[1], 탈락평균=np.mean(fails), 탈락최악=max(fails), 탈락0회=sum(v == 0 for v in fails), sY=np.mean(sy), sT=np.mean(st), 최대eY=my, 최대eT=mt,
               절반탈락=np.mean(hf), 처음보는날=lo_all, **{f"처음_{NAMES[s]}": v for s, v in zip(S_all, lo)}, **{f"강건_{k}": v for k, v in R.items()}, 강건최소=min(R.values()),
               마지막탈락=int((~ok(Ef)).sum()), 마지막최대eY=abs(Ef[:, 0]).max(), 마지막최대eT=abs(Ef[:, 1]).max())
    print(f"{name}\t값 {F.shape[1]}\t탈락 {row['탈락평균']:.1f} (최악 {row['탈락최악']}, 0장 {row['탈락0회']}회)\t절반 {row['절반탈락']:.1f}\tσY {row['sY']:.4f} σT {row['sT']:.4f}\t최대 {my:.4f}/{mt:.4f}\t"
          f"강건 QR {R['QR 하나 +0.3']:.0f}/{R['QR 잡음 0.1']:.0f} 나사 {R['나사 하나 +0.3']:.0f}/{R['나사 잡음 0.1']:.0f} 최소 {row['강건최소']:.0f}\t처음 {lo_all:.0f} ({' '.join(f'{v:.0f}' for v in lo)})\t마지막 {row['마지막탈락']}/{final.sum()}", flush=True)
    if detail: return row, F, Eall, Elo, Ef, fails, hf
    return row


ALLQ, ALLS = [0, 1, 2], [0, 1, 2, 3]
rows = []
print("\n== 기준 구성 ==")
base, F0, Eall, Elo, Ef, fails, hf = evaluate("기준: QR 536점 가로 평균 1 + 나사 4 가로 + 바 기울기 2", (("P", ALLQ), ALLS, "tilt"), detail=True); rows.append(base)
print("주 검증 10회 탈락:", fails, " 50/50 10회 탈락:", hf)

print("\n-- 주 검증: 회차별·조각별 (기준 구성) --")
for k, f in enumerate(cv):
    E = Eall[k]; print(f"회차 {k + 1}: " + " | ".join(f"조각{j + 1} n={int((dev & (f == j)).sum())} 탈락 {int((~ok(E[dev & (f == j)])).sum())} σY {E[dev & (f == j), 0].std():.4f} σT {E[dev & (f == j), 1].std():.4f}" for j in range(5)))
Ecat = np.stack(Eall)  # 10 x n x 2
print("\n-- 주 검증: 세션별 (10회 합쳐서, 개발 560장) --")
for s in S_all:
    m = dev & (src == s); E = Ecat[:, m].reshape(-1, 2)
    print(f"{NAMES[s]} n={int(m.sum())}: 평균 eY {E[:, 0].mean():+.4f} eT {E[:, 1].mean():+.5f} | σY {E[:, 0].std():.4f} σT {E[:, 1].std():.5f} | MAE {abs(E[:, 0]).mean():.4f} / {abs(E[:, 1]).mean():.5f} | 최대 {abs(E[:, 0]).max():.4f} / {abs(E[:, 1]).max():.5f} | 탈락(10회 평균) {(~ok(E)).sum() / 10:.1f}")
E = Ecat[:, dev].reshape(-1, 2)
print(f"전체 n=560: 평균 eY {E[:, 0].mean():+.4f} eT {E[:, 1].mean():+.5f} | σY {E[:, 0].std():.4f} σT {E[:, 1].std():.5f} | MAE {abs(E[:, 0]).mean():.4f} / {abs(E[:, 1]).mean():.5f} | p95 {np.percentile(abs(E[:, 0]), 95):.4f} / {np.percentile(abs(E[:, 1]), 95):.5f} | 최대 {abs(E[:, 0]).max():.4f} / {abs(E[:, 1]).max():.5f} | 상관 {np.corrcoef(E.T)[0, 1]:+.2f}")
print("\n-- 처음 보는 날(세션 통째로 빼기): 세션별 (기준 구성) --")
for s in S_all:
    m = dev & (src == s); E = Elo[m]
    print(f"{NAMES[s]} n={int(m.sum())}: 통과 {ok(E).mean() * 100:.1f}% | 평균 eY {E[:, 0].mean():+.4f} eT {E[:, 1].mean():+.5f} | σY {E[:, 0].std():.4f} σT {E[:, 1].std():.5f} | 최대 {abs(E[:, 0]).max():.4f} / {abs(E[:, 1]).max():.5f}")
print("\n-- 마지막 확인 142장: 세션별 (기준 구성, 개발 560장 전부로 학습) --")
for s in S_all:
    m = src[final] == s; E = Ef[m]
    print(f"{NAMES[s]} n={int(m.sum())}: 탈락 {int((~ok(E)).sum())} | σY {E[:, 0].std():.4f} σT {E[:, 1].std():.5f} | 최대 {abs(E[:, 0]).max():.4f} / {abs(E[:, 1]).max():.5f}")
print(f"전체 n={final.sum()}: σY {Ef[:, 0].std():.4f} σT {Ef[:, 1].std():.5f} | MAE {abs(Ef[:, 0]).mean():.4f} / {abs(Ef[:, 1]).mean():.5f} | 최대 {abs(Ef[:, 0]).max():.4f} / {abs(Ef[:, 1]).max():.5f}")

# 바 기울기 교란
print("\n-- 바 기울기 교란(기준 구성, 주 검증 앞 3회): 기울기 한쪽 값에 +10% 표준편차 --")
for bi, bn in ((0, "위"), (1, "아래")):
    res = []
    for k in range(3):
        f = cv[k]; E = np.zeros_like(L)
        for j in range(5):
            te = f == j; m = fit(F0[dev & ~te], L[dev & ~te]); Fp = F0[te].copy(); Fp[:, 1 + bi] += 0.1 * tilt_sd[bi]; E[te] = m(Fp) - L[te]
        res.append(ok(E[dev]).mean() * 100)
    print(f"{bn} 바 기울기 +{0.1 * tilt_sd[bi]:.4f} (표준편차 {tilt_sd[bi]:.4f} 의 10%): 통과 {np.mean(res):.1f}%")

# 값별 잡음 명세: 같은 자세 반복(세션, 명령 Y, 명령 T 같음)의 흔들림과 한 칸 민감도
print("\n-- 값별 잡음 명세 (기준 구성 7개 값) --")
keys = [(a, round(y, 2), round(t, 2)) for a, (y, t) in zip(src, cmd)]; uk = sorted(set(keys)); gi = np.array([uk.index(k) for k in keys])
cnt = np.bincount(gi); rep = cnt[gi] >= 2
print(f"반복 묶음 {int((cnt >= 2).sum())}개, 사진 {int(rep.sum())}장")
names7 = ["QR 가로 평균", "위 바 기울기", "아래 바 기울기", "나사 S1 가로", "나사 S2 가로", "나사 S3 가로", "나사 S4 가로"]
res = F0[rep] - np.array([F0[gi == g].mean(0) for g in gi[rep]])
dof = rep.sum() - (cnt >= 2).sum(); sd = np.sqrt((res ** 2).sum(0) / dof)
A = np.c_[np.array([(src == s) for s in S_all]).T.astype(float), L]; J = np.linalg.lstsq(A, F0, rcond=None)[0][len(S_all):]
for i, nm in enumerate(names7):
    print(f"{nm}: 반복 흔들림 {sd[i]:.4f} | Y 1mm 당 {J[0, i]:+.3f}, Y 한 칸(0.05) {J[0, i] * .05:+.4f} | T 1 당 {J[1, i]:+.3f}, T 한 칸(0.01) {J[1, i] * .01:+.4f}")
par = F0[:, 3:] - F0[:, [0]]; resp = par[rep] - np.array([par[gi == g].mean(0) for g in gi[rep]]); sdp = np.sqrt((resp ** 2).sum(0) / dof)
Jp = np.linalg.lstsq(A, par, rcond=None)[0][len(S_all):]
print("나사 − QR 가로 시차(나사별): 반복 흔들림", np.round(sdp, 4), "| Y 한 칸", np.round(Jp[0] * .05, 4), "| T 한 칸", np.round(Jp[1] * .01, 4))
pm = par.mean(1); r2 = pm[rep] - np.array([pm[gi == g].mean() for g in gi[rep]]); jm = np.linalg.lstsq(A, pm, rcond=None)[0][len(S_all):]
print(f"나사 4개 평균 − QR 가로 시차: 반복 흔들림 {np.sqrt((r2 ** 2).sum() / dof):.4f} | Y 한 칸 {jm[0] * .05:+.4f} | T 한 칸 {jm[1] * .01:+.4f}")
# 예측의 반복 흔들림 (전체 학습 모델)
m = fit(F0, L); Pa = m(F0); rp = Pa[rep] - np.array([Pa[gi == g].mean(0) for g in gi[rep]]); print("예측의 반복 흔들림 (전부 학습): σY %.4f σT %.5f" % tuple(np.sqrt((rp ** 2).sum(0) / dof)))

print("\n== QR: 3개 중 어느 것을 쓸지 (나사 4 가로 + 바 기울기 2 고정) ==")
for idx in ([0], [1], [2], [0, 1], [0, 2], [1, 2]):
    rows.append(evaluate("QR 536점 가로 평균: " + "+".join(QN[i] for i in idx), (("P", idx), ALLS, "tilt")))
rows.append(evaluate("QR 536점 QR별 가로 평균 3값", (("Pper", ALLQ), ALLS, "tilt")))
rows.append(evaluate("QR ECC 3개 전체 가로 평균 1값", (("Emean", ALLQ), ALLS, "tilt")))
for idx in ([0], [1], [2]):
    rows.append(evaluate("QR ECC 가로 평균: " + QN[idx[0]], (("E", idx), ALLS, "tilt")))
rows.append(evaluate("QR ECC QR별 가로 평균 3값", (("E", ALLQ), ALLS, "tilt")))
rows.append(evaluate("QR 없음", (None, ALLS, "tilt")))

print("\n== 나사: 4개 중 어느 것을 쓸지 (QR 536점 가로 평균 + 바 기울기 2 고정) ==  S1 왼쪽 위, S2 오른쪽 위, S3 왼쪽 아래, S4 오른쪽 아래")
SN = {(0,): "S1", (1,): "S2", (2,): "S3", (3,): "S4", (0, 3): "대각선 S1+S4", (1, 2): "대각선 S2+S3", (0, 1): "위 2개", (2, 3): "아래 2개", (0, 2): "왼쪽 2개", (1, 3): "오른쪽 2개",
      (0, 1, 2): "S4 빼고 3개", (0, 1, 3): "S3 빼고 3개", (0, 2, 3): "S2 빼고 3개", (1, 2, 3): "S1 빼고 3개", (): "나사 없음"}
for idx, nm in SN.items():
    rows.append(evaluate("나사 " + nm, (("P", ALLQ), list(idx), "tilt")))

print("\n== 바: 위·아래 중 어느 것을 쓸지 (QR 536점 가로 평균 + 나사 4 가로 고정) ==")
for b, nm in (("up", "위 바 기울기만"), ("lo", "아래 바 기울기만"), ("none", "바 없음")):
    rows.append(evaluate("바 " + nm, (("P", ALLQ), ALLS, b)))

print("\n== Y/T 계산식 (기준 구성 7개 값) ==")
def ols(F, Y):
    A = np.c_[np.ones(len(F)), F]; B = np.linalg.lstsq(A, Y, rcond=None)[0]; return lambda G: np.c_[np.ones(len(G)), G] @ B
def ridge(lam):
    def f(F, Y):
        mean = F.mean(0); sc = np.maximum(F.std(0), 1e-6); A = np.c_[np.ones(len(F)), (F - mean) / sc]; pen = np.eye(A.shape[1]) * lam * len(F); pen[0, 0] = 0
        B = np.linalg.solve(A.T @ A + pen, A.T @ Y); return lambda G: np.c_[np.ones(len(G)), (G - mean) / sc] @ B
    return f
def poly(deg, base):
    def expand(F, mean, sc):
        Z = (F - mean) / sc; cols = [Z]
        if deg >= 2: cols.append(np.array([Z[:, i] * Z[:, j] for i in range(Z.shape[1]) for j in range(i, Z.shape[1])]).T)
        if deg >= 3: cols.append(Z ** 3)
        return np.hstack(cols)
    def f(F, Y):
        mean = F.mean(0); sc = np.maximum(F.std(0), 1e-6); m = base(expand(F, mean, sc), Y); return lambda G: m(expand(G, mean, sc))
    return f
spec0 = (("P", ALLQ), ALLS, "tilt")
rows.append(evaluate("식: 보통 최소제곱 (1차)", spec0, ols))
rows.append(evaluate("식: Ridge 1e-3·n (1차)", spec0, ridge(1e-3)))
rows.append(evaluate("식: Huber 1차 (기준과 같음)", spec0, fit))
rows.append(evaluate("식: Huber + 2차항 전부 (값 35)", spec0, poly(2, fit)))
rows.append(evaluate("식: Huber + 2차항 + 각 값 3제곱 (값 42)", spec0, poly(3, fit)))

out = pathlib.Path(__file__).resolve().parent.parent / "결과" / "결과.tsv"; ks = []
for r in rows:
    for k in r:
        if k not in ks: ks.append(k)
with out.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, ks, delimiter="\t"); w.writeheader()
    for r in rows: w.writerow({k: (f"{v:.5f}" if isinstance(v, float) else v) for k, v in r.items()})
print("\n저장:", out.relative_to(ROOT).as_posix())
