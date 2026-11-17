"""바 세로 이동을 여러 방식(창 크기, 밝기/기울기, 가로 구간)으로 다시 재고 날 차이가 줄어드는지 본다."""
import sys, pathlib, numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter1d
S = pathlib.Path(sys.argv[1]); EXP = pathlib.Path("experiments/hye/20261112_1835_조합마다검증방식다름_방법론별증강과고정검증/실험")
sys.path.insert(0, str(EXP)); import combo
from combo import fit, ok, SEED, NAMES

Z = np.load(S / "bar/profiles.npz"); ref, prof, reg = Z["ref"].astype(float), Z["prof"].astype(float), Z["reg"]
n = len(prof); XC = np.array([[x0 + 32 * j + 16 for j in range(32)] for x0, _, _ in reg])       # 칸 중심 x

def shifts(R, C, half, iters=12):
    """R: (4,32,200) 기준, C: (n,4,32,200) 현재. 창 = 행 100±half. 현재 경계가 기준보다 아래로 s 만큼. gain/offset 맞춤 GN."""
    y = np.arange(100 - half, 101 + half, dtype=float); T = R[:, :, 100 - half:101 + half]                 # (4,32,w)
    Cf = C.reshape(-1, 200); dC = np.gradient(Cf, axis=1); N = len(Cf); s = np.zeros(N); rows = np.repeat(np.arange(N), len(y)).astype(float)
    Tt = np.broadcast_to(T, (n,) + T.shape).reshape(N, -1)
    for _ in range(iters):
        yy = (y[None, :] + s[:, None]).ravel()
        v = map_coordinates(Cf, [rows, yy], order=3, mode="nearest").reshape(N, -1); dv = map_coordinates(dC, [rows, yy], order=3, mode="nearest").reshape(N, -1)
        vm = v.mean(1, keepdims=True); tm = Tt.mean(1, keepdims=True); g = ((Tt - tm) * (v - vm)).sum(1, keepdims=True) / np.maximum(((v - vm) ** 2).sum(1, keepdims=True), 1e-12)
        r = (Tt - tm) - g * (v - vm); J = g * (dv - dv.mean(1, keepdims=True)); d = (r * J).sum(1) / np.maximum((J ** 2).sum(1), 1e-12); s += np.clip(d, -1, 1)
    return s.reshape(n, 4, 32)

def grad(A): return np.gradient(gaussian_filter1d(A, 1.5, axis=-1), axis=-1)
def bars4(sh, cols):
    out = []
    for regs in ((0, 1), (2, 3)):
        x = np.concatenate([XC[k][cols] for k in regs]) / 1000.; v = np.concatenate([sh[:, k][:, cols] for k in regs], axis=1)
        xm = x.mean(); sxx = ((x - xm) ** 2).sum(); out += [v.mean(1), ((v - v.mean(1, keepdims=True)) * (x - xm)).sum(1) / sxx]
    return np.array(out).T                                                                               # 위 평균, 위 기울기, 아래 평균, 아래 기울기

ORIG = slice(4, 28); ALL = slice(0, 32)
gr, gp = grad(ref), grad(prof)
METHODS = {}
for name, (R, C, half, cols) in {
    "밝기 ±10 (지금 방식과 같은 크기)": (ref, prof, 10, ORIG), "밝기 ±30": (ref, prof, 30, ORIG), "밝기 ±60": (ref, prof, 60, ORIG), "밝기 ±60, 가로 구간 넓힘": (ref, prof, 60, ALL),
    "기울기 ±10": (gr, gp, 10, ORIG), "기울기 ±30": (gr, gp, 30, ORIG), "기울기 ±60": (gr, gp, 60, ORIG), "기울기 ±60, 가로 구간 넓힘": (gr, gp, 60, ALL)}.items():
    METHODS[name] = bars4(shifts(R, C, half), cols); print("측정 완료:", name, flush=True)

ids, src, L, cmd, raw, geo = combo.load(); assert (ids == Z["ids"]).all()
dev, final, cv, half_s = combo.make_splits(ids, src, cmd); SS = sorted(set(src)); old = raw["bars"]
METHODS = {"00004 저장값 (지금)": old, **METHODS}
np.savez(S / "bar/bars_meas.npz", ids=ids, **{f"m{i}": v for i, v in enumerate(METHODS.values())}, names=np.array(list(METHODS)))

A = np.c_[np.ones(n), L, L[:, 0] * L[:, 1], L ** 2]; Dm = np.array([(src == s) for s in SS]).T.astype(float)
wide = [SS[0], SS[3]]; mw = np.isin(src, wide); Aw = np.c_[np.array([(src[mw] == s) for s in wide]).T.astype(float), L[mw]]
import collections
g = collections.defaultdict(list)
for i in range(n): g[(src[i], round(cmd[i, 0], 2), round(cmd[i, 1], 2))].append(i)
G = [v for v in g.values() if len(v) >= 2]
def rep(v): W = np.concatenate([v[k] - v[k].mean() for k in G]); return np.sqrt((W ** 2).sum() / sum(len(k) - 1 for k in G))
print("\n방식\t값\t저장값과의 상관\t반복 흔들림(px)\t날이 설명하는 비율\t명령 0 에서의 값: 1차 2차 3차 4차")
for name, Bv in METHODS.items():
    for j, lab in ((0, "위 바 세로"), (2, "아래 바 세로"), (1, "위 바 기울기"), (3, "아래 바 기울기")):
        v = Bv[:, j]; r1 = v - A @ np.linalg.lstsq(A, v, rcond=None)[0]; AD = np.c_[A[:, 1:], Dm]; r2 = v - AD @ np.linalg.lstsq(AD, v, rcond=None)[0]
        Jv = np.linalg.lstsq(Aw, v[mw], rcond=None)[0][2:]; z = [(v[src == s] - L[src == s] @ Jv).mean() for s in SS]
        print(f"{name}\t{lab}\t{np.corrcoef(v, old[:, j])[0, 1]:.4f}\t{rep(v):.4f}\t{(r1.var() - r2.var()) / v.var() * 100:.1f}%\t" + " ".join(f"{x:+.3f}" for x in z))

# 고정 검증: 새 기준(값 7) 과, 거기에 바 세로 이동 2개를 더한 구성
def evaluate(F):
    fails, hf = [], []
    for k, f in enumerate(cv):
        E = np.zeros_like(L)
        for j in range(5): te = f == j; E[te] = fit(F[dev & ~te], L[dev & ~te])(F[te]) - L[te]
        fails.append(int((~ok(E[dev])).sum()))
    for h in half_s:
        E = np.zeros_like(L)
        for j in (0, 1): te = h == j; E[te] = fit(F[dev & ~te], L[dev & ~te])(F[te]) - L[te]
        hf.append(int((~ok(E[dev])).sum()))
    lo = [ok(fit(F[dev & (src != s)], L[dev & (src != s)])(F[dev & (src == s)]) - L[dev & (src == s)]).mean() * 100 for s in SS]
    tot = sum(v * (dev & (src == s)).sum() for v, s in zip(lo, SS)) / dev.sum()
    Ef = fit(F[dev], L[dev])(F[final]) - L[final]
    return f"탈락 {np.mean(fails):.1f} (최악 {max(fails)})\t50/50 {np.mean(hf):.1f}\t처음 보는 날 {tot:.0f}% ({' '.join('%.0f' % v for v in lo)})\t마지막 {int((~ok(Ef)).sum())}/{int(final.sum())}"
Q1 = raw["d536"].mean(1)[:, :1]; SX = raw["s_grad"][:, :, 0]
print("\n고정 검증 (QR 536점 가로 평균 + 나사 grad 가로 + 바)")
print("바 없음\t\t" + evaluate(np.c_[Q1, SX]))
for name, Bv in METHODS.items():
    print(f"{name}\t기울기만\t" + evaluate(np.c_[Q1, Bv[:, [1, 3]], SX]))
    print(f"{name}\t기울기 + 세로 이동\t" + evaluate(np.c_[Q1, Bv, SX]))
