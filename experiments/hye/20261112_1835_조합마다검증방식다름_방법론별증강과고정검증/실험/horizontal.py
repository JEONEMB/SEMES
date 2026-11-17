"""추가 조합: 세로 방향 값(바 세로 이동, QR·나사의 세로 이동)을 빼고 가로 이동과 바 기울기만 쓴다.

combo.py 와 같은 사진, 같은 절대 영점 라벨, 같은 분할(seed 고정이라 같은 분할이 다시 만들어진다), 같은 학습식, 같은 교란을 쓴다.
비교용으로 combo.py 의 세 구성(base, ECC6+grad+바 없음, P536_2+grad+바 없음)을 함께 돌려 숫자가 같은지 확인한다.

구성 표기: QR <표현>[x] / 나사 <측정>[x] / 바 <both|tilt|none>.  [x] 는 가로 값만.
바 교란은 바 세로 이동에 넣는다. 바 세로 이동을 안 쓰는 구성에는 해당이 없다.

실행: 저장소 루트에서  python "<이 폴더>/실험/horizontal.py"   → 결과/가로만_결과.tsv
"""
import csv, pathlib, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import combo
from combo import fit, augment, ok, SEED, NAMES

OUT = combo.OUT


def build(raw, geo, spec):
    q, qx, s, sx, b = spec; n = len(raw["bars"])
    if q == "ECC54": Q = raw["qr"].reshape(n, 27, 2)
    elif q == "ECC6": Q = raw["qr"].mean(2)
    elif q == "P536_2": Q = raw["d536"].mean(1)[:, None, :]
    Q = Q[:, :, 0] if qx else Q.reshape(n, -1)
    S = raw["s_old" if s == "old" else "s_grad"]; S = S[:, :, 0] if sx else S.reshape(n, 8)
    B = {"both": raw["bars"], "tilt": raw["bars"][:, [1, 3]], "none": raw["bars"][:, :0]}[b]
    return np.c_[Q, B, S]


SPECS = [("base: QR ECC54 / 나사 old / 바 both", ("ECC54", False, "old", False, "both")),
         ("QR ECC6 / 나사 grad / 바 none", ("ECC6", False, "grad", False, "none")),
         ("QR P536_2 / 나사 grad / 바 none", ("P536_2", False, "grad", False, "none")),
         ("QR ECC6 / 나사 grad / 바 tilt", ("ECC6", False, "grad", False, "tilt")),
         ("QR P536_2 / 나사 grad / 바 tilt", ("P536_2", False, "grad", False, "tilt")),
         ("QR ECC54 / 나사 grad / 바 tilt", ("ECC54", False, "grad", False, "tilt")),
         ("QR ECC6 x / 나사 grad x / 바 tilt", ("ECC6", True, "grad", True, "tilt")),
         ("QR P536_2 x / 나사 grad x / 바 tilt", ("P536_2", True, "grad", True, "tilt")),
         ("QR ECC54 x / 나사 grad x / 바 tilt", ("ECC54", True, "grad", True, "tilt")),
         ("QR ECC6 x / 나사 grad x / 바 none", ("ECC6", True, "grad", True, "none")),
         ("QR P536_2 x / 나사 grad x / 바 none", ("P536_2", True, "grad", True, "none")),
         ("QR ECC54 x / 나사 grad x / 바 none", ("ECC54", True, "grad", True, "none")),
         ("QR ECC6 x / 나사 old x / 바 tilt", ("ECC6", True, "old", True, "tilt")),
         ("QR ECC6 x / 나사 grad (가로·세로) / 바 tilt", ("ECC6", True, "grad", False, "tilt"))]


def main():
    ids, src, L, cmd, raw, geo = combo.load()
    dev, final, cv, half = combo.make_splits(ids, src, cmd); S_all = sorted(set(src)); rows = []
    for name, spec in SPECS:
        bkey = "both" if spec[4] == "both" else "none"
        F = build(raw, geo, spec); P = [(nm, build(r, geo, spec)) for nm, r in combo.perturbations(raw, geo, bkey, np.random.default_rng(SEED))]
        for aug in (False, True):
            def train(tr, seed):
                Ft, Lt = F[tr], L[tr]
                if aug: Ft, Lt = augment(Ft, Lt, src[tr], cmd[tr], np.random.default_rng(SEED + seed))[:2]
                return fit(Ft, Lt)
            fails, sy, st, my, mt = [], [], [], 0, 0; rob = {}
            for k, f in enumerate(cv):
                E = np.zeros_like(L); EP = [np.zeros_like(L) for _ in P]
                for j in range(5):
                    te = f == j; m = train(dev & ~te, 10 * k + j); E[te] = m(F[te]) - L[te]
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
            m = train(dev, 0); Ef = m(F[final]) - L[final]
            R = {nm: float(np.mean(v)) for nm, v in rob.items()}
            row = dict(구성=name, 증강="날 차이" if aug else "없음", 값수=F.shape[1], 탈락평균=float(np.mean(fails)), 탈락최악=max(fails), 탈락0회수=sum(v == 0 for v in fails), sY=float(np.mean(sy)), sT=float(np.mean(st)),
                       최대eY=float(my), 최대eT=float(mt), 절반탈락평균=float(np.mean(hf)), 처음보는날=float(lo_all), **{f"처음_{NAMES[ss]}": float(v) for ss, v in zip(S_all, lo)},
                       **{f"강건_{nm}": R.get(nm, float("nan")) for nm in ("QR 하나 +0.3", "QR 잡음 0.1", "나사 하나 +0.3", "나사 잡음 0.1", "바 하나 +0.3")}, 강건최소=min(R.values()),
                       마지막확인탈락=int((~ok(Ef)).sum()), 마지막확인장수=int(final.sum()))
            rows.append(row)
            print(f"{name}\t{row['증강']}\t값 {row['값수']}\t탈락 {row['탈락평균']:.1f} (최악 {row['탈락최악']}, 0장 {row['탈락0회수']}회)\t절반 {row['절반탈락평균']:.1f}\tσY {row['sY']:.4f} σT {row['sT']:.4f}\t"
                  f"강건 QR {row['강건_QR 하나 +0.3']:.0f}/{row['강건_QR 잡음 0.1']:.0f} 나사 {row['강건_나사 하나 +0.3']:.0f}/{row['강건_나사 잡음 0.1']:.0f} 최소 {row['강건최소']:.0f}\t"
                  f"처음 {row['처음보는날']:.0f} ({' '.join('%.0f' % v for v in lo)})\t마지막 {row['마지막확인탈락']}/{row['마지막확인장수']}", flush=True)
    cols = list(rows[0])
    with (OUT / "가로만_결과.tsv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t"); w.writerow(cols)
        for r in rows: w.writerow([f"{r[c]:.5f}" if isinstance(r[c], float) else r[c] for c in cols])


if __name__ == "__main__":
    main()
