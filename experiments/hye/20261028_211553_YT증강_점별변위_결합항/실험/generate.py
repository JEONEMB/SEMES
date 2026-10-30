"""영점 x 격자 x n회 증강 생성 (실험용 Python, 팀 전달 코드 아님).

fit_validate.py가 만든 results/fit.npz를 읽어서
  1) 추가 영점 후보 4개를 만들고 (실측 영점 11개의 주성분 ±2σ)
  2) results/generator.npz (생성에 필요한 전부)와 대시보드용 apps/augment/data.js를 쓴다.
  3) --write 영점이름 을 주면 그 영점의 81x201xn 샘플을 점별 변위로 results/samples_<영점>.npz에 쓴다.

샘플 = 영점 필드 + 단일축 반응(Y, T±) [+ YT 결합항, 기본 끔] + 반복 촬영 잡음(같은 shot 번호의 실측 편차에서 하나 뽑기)
"""
import argparse, csv, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
APP = HERE.parents[2] / "apps" / "augment"
FAMILIES = {"Y1_T1": (1, 1, False), "Y1_T1s": (1, 1, True), "Y2_T2s": (2, 2, True),
            "Y3_T2s": (3, 2, True), "Y3_T3": (3, 3, False), "Y3_T3s": (3, 3, True)}
LABELS = {"C14": "20261014 센터 (고정 기준)", "S1": "S1 Y 0.1 + 접근", "S2": "S2 Y 0.1 − 접근", "S3": "S3 T + 접근",
          "S4": "S4 T − 접근", "S5": "S5 Y 0.05 홀수 +", "S6": "S6 Y 0.05 홀수 −", "C20_1": "20261020 센터 1",
          "C20_2": "20261020 센터 2", "C20_3": "20261020 센터 3", "MIX20": "20261020 혼합 촬영 세션"}


def basis(Y, T, fam):
    dy, dt, sg = FAMILIES[fam]
    Y, T = np.asarray(Y, float), np.asarray(T, float)
    cols = [Y ** d for d in range(1, dy + 1)]
    if sg: cols += [np.maximum(T, 0) ** d for d in range(1, dt + 1)] + [np.minimum(T, 0) ** d for d in range(1, dt + 1)]
    else: cols += [T ** d for d in range(1, dt + 1)]
    return np.stack(cols, -1)


f = np.load(RES / "fit.npz")
FAM, COEF, SEL = str(f["family"]), f["coef"], f["sel"]
names, O = [str(x) for x in f["origin_names"]], f["origins"]
kinds = ["실측 trial 영점" if n.startswith("S") else ("실측 혼합 세션 영점" if n == "MIX20" else "실측 센터 사진") for n in names]

# 접근 방향 백래시: 센터에서 + 방향으로 들어간 촬영과 − 방향으로 들어간 촬영의 영점 - 그 날 센터
ix = {n: i for i, n in enumerate(names)}
BPLUS = np.mean([O[ix["S1"]], O[ix["S3"]], O[ix["S5"]], O[ix["MIX20"]] - O[[ix["C20_1"], ix["C20_2"], ix["C20_3"]]].mean(0)], 0)
BMINUS = np.mean([O[ix["S2"]], O[ix["S4"]], O[ix["S6"]]], 0)

# 추가 영점: 실측 11개 영점 필드(선택 536점)의 주성분 2개를 따라 ±2σ
X = O[:, SEL].reshape(len(O), -1)
mu = X.mean(0); U, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
sig = S / np.sqrt(len(O) - 1)
extra = {}
for k in range(2):
    for sg in (+1, -1):
        full = np.zeros_like(O[0]); full[SEL] = (mu + sg * 2 * sig[k] * Vt[k]).reshape(-1, 2)
        extra[f"PC{k + 1}{'+' if sg > 0 else '-'}2s"] = full
for n, v in extra.items(): names.append(n); kinds.append("추가 후보 (주성분 ±2σ)"); LABELS[n] = f"추가 후보 {n[:3]} {n[3]}2σ"
O = np.concatenate([O, np.stack(list(extra.values()))])
pc_share = (S ** 2 / (S ** 2).sum())[:3]

np.savez_compressed(RES / "generator.npz", family=FAM, coef=COEF, sel=SEL, points=f["points"], origin_names=np.array(names),
                    origin_kinds=np.array(kinds), origins=O, pool_dev=f["pool_dev"], pool_shot=f["pool_shot"], yt_coupling=f["yt_coupling"],
                    backlash_plus=BPLUS, backlash_minus=BMINUS)
with open(RES / "origins.csv", "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.writer(fh); w.writerow(["name", "label", "kind", "mean_dx_px", "mean_dy_px", "shape_rms_px"])
    for n, k, v in zip(names, kinds, O):
        w.writerow([n, LABELS[n], k, f"{v[SEL, 0].mean():.4f}", f"{v[SEL, 1].mean():.4f}", f"{np.sqrt(((v[SEL] - v[SEL].mean(0)) ** 2).mean()):.4f}"])


def samples(o, n, seed, coupling=False, backlash=True):
    """한 영점의 81 x 201 x n 샘플, (81, 201, n, 630, 2) float32.
    센터 사진 영점에는 접근 방향 백래시를 더한다 (T가 있으면 T 부호, 없으면 Y 부호). trial·세션 영점은 이미 접근 방향을 담고 있다."""
    add_b = backlash and kinds[o] == "실측 센터 사진"
    rng = np.random.default_rng(seed)
    Ys, Ts = np.round(np.linspace(-2, 2, 81), 2), np.round(np.linspace(-1, 1, 201), 2)
    out = np.empty((81, 201, n) + O[o].shape, np.float32)
    shots = f["pool_shot"]
    for a, Y in enumerate(Ys):
        for b, T in enumerate(Ts):
            sgn = np.sign(T) or np.sign(Y)
            bl = (BPLUS if sgn > 0 else BMINUS if sgn < 0 else 0) if add_b else 0
            base = O[o] + bl + np.tensordot(basis([Y], [T], FAM)[0], COEF, 1) + (Y * T * f["yt_coupling"] if coupling else 0)
            for k in range(n):
                cand = np.flatnonzero(shots == k % 3 + 1)
                out[a, b, k] = base + f["pool_dev"][rng.choice(cand)]
    return out


def write_app():
    r = lambda a, d=4: np.round(np.asarray(a, float), d).tolist()
    summ = json.load(open(RES / "summary.json", encoding="utf-8"))
    photos = list(csv.DictReader(open(RES / "photos_validation.csv", encoding="utf-8-sig")))
    data = dict(
        created=str(np.datetime64("now")), experiment=HERE.name, family=FAM, family_spec=FAMILIES[FAM],
        points=r(f["points"][SEL], 1), coef=r(COEF[:, SEL]), yt=r(f["yt_coupling"][SEL]),
        origins=[dict(name=n, label=LABELS[n], kind=k, field=r(v[SEL])) for n, k, v in zip(names, kinds, O)],
        pool=r(f["pool_dev"][:, SEL]), bplus=r(BPLUS[SEL]), bminus=r(BMINUS[SEL]), pool_shot=f["pool_shot"].tolist(), pc_share=r(pc_share, 3),
        noise_floor=summ["noise_floor"], shape_cv=summ["shape_cv_1014"], mixed=summ["mixed_lopo"],
        photos=[dict(file=p["file"], date=p["date"], trial=p["trial"], Y=float(p["Y"]), T=float(p["T"]), rep=int(p["rep"]), eval=p["eval"],
                     mdx=round(float(p["actual_mdx"]), 4), mdy=round(float(p["actual_mdy"]), 4),
                     rms=round(float(p["resid_rms"]), 4) if p["resid_rms"] else None,
                     rdx=round(float(p["resid_mdx"]), 4) if p["resid_mdx"] else None) for p in photos])
    APP.mkdir(parents=True, exist_ok=True)
    (APP / "data.js").write_text("window.AUG = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n", encoding="utf-8")
    print("wrote", APP / "data.js", round((APP / "data.js").stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", nargs="*", default=[], help="샘플을 저장할 영점 이름 (all = 전부)")
    ap.add_argument("--n", type=int, default=3); ap.add_argument("--seed", type=int, default=20261028)
    ap.add_argument("--coupling", action="store_true", help="YT 결합항을 켠다 (검증에서 유의하지 않음)")
    ap.add_argument("--no-backlash", action="store_true", help="센터 사진 영점에 접근 방향 백래시를 더하지 않는다")
    a = ap.parse_args()
    write_app()
    print("origins:", len(names), "PC share:", np.round(pc_share, 3).tolist(), "backlash +/- mean dx:", round(float(BPLUS[SEL, 0].mean()), 3), round(float(BMINUS[SEL, 0].mean()), 3))
    for nm in (names if a.write == ["all"] else a.write):
        o = names.index(nm)
        arr = samples(o, a.n, a.seed + o, a.coupling, not a.no_backlash)
        np.savez_compressed(RES / f"samples_{nm}.npz", D=arr, Y=np.round(np.linspace(-2, 2, 81), 2), T=np.round(np.linspace(-1, 1, 201), 2),
                            origin=nm, n=a.n, seed=a.seed + o, coupling=a.coupling, backlash=not a.no_backlash, family=FAM)
        print("wrote samples", nm, arr.shape)
