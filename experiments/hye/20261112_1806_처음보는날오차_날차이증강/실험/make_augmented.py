"""가장 잘 된 증강(실제로 잰 날 차이 방향, 배율 2)으로 학습용 증강 자료를 만든다.

- 입력: 커밋 46152fd `training_features.tsv` 66개 특징(QR 54, 바 4, 나사 8, 지금 나사 측정).
  범위 안 702장(ID 702 제외, T=+2.00 12장 제외, ID 420 라벨 T 0.32).
- 라벨: 절대 영점 기준. 세션마다 특징이 0(center.bmp 와 같은 화면)에 가장 가까워지는 명령을 구해 라벨에서 뺀다.
- 날 차이 벡터: 세션마다 "명령 0 에서의 특징"에서 Y, T 움직임으로 설명되는 부분을 뺀 나머지 66개 값. 세션 4개, 벡터 4개.
- 증강: 사진마다 복사본 8개. 복사본 = 원본 특징 + Σ z_k × (세션 k 의 날 차이 벡터), z_k ~ N(0, 2²). 정답은 그대로.
  `day_augment.py` 의 "실측 방향 ×2" 와 같은 생성식이다. 여기서는 배포 학습용이라 네 세션의 벡터를 모두 쓴다.
- 출력(결과/증강자료/): augmented_training_features.tsv, day_vectors.tsv, augmented.npz, 확인.txt
특징 수준 증강이며 영상(BMP)은 만들지 않는다. 분석용 Python 이며 팀 전달 C++ 도구를 대체하지 않는다.

실행: 저장소 루트에서  python "experiments/hye/20261112_1806_처음보는날오차_날차이증강/실험/make_augmented.py"
"""
import csv, pathlib, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "apps/distribution"))
import reoof

OUT = pathlib.Path(__file__).resolve().parent.parent / "결과" / "증강자료"
SCALE, COPIES, SEED = 2.0, 8, 20261112
NAMES = {"20261014": "1차", "20261020": "2차", "20261029": "3차", "20261109_round4": "4차"}


def main():
    if OUT.exists():
        sys.exit(f"이미 있다: {OUT} (덮어쓰지 않는다)")
    tf = reoof.tsv("training_features.tsv")
    rows = [r for r in tf if int(r["id"]) != 702 and abs(float(r["T"])) <= 1.0001]
    ids = np.array([int(r["id"]) for r in rows]); src = np.array([r["source"] for r in rows])
    X = np.array([[float(v) for v in r["features66"].split(";")] for r in rows])
    L0 = np.array([[float(r["Y"]), float(r["T"])] for r in rows]); L0[np.where(ids == 420)[0][0], 1] = .32
    S = sorted(set(src)); wide = [s for s in S if len(set(L0[src == s, 0])) > 5]

    m = np.isin(src, wide)
    A = np.c_[np.array([(src[m] == s) for s in wide]).T.astype(float), L0[m]]
    J = np.linalg.lstsq(A, X[m], rcond=None)[0][len(wide):]            # 명령 1 단위당 특징 이동 (2×66)
    L = L0.copy(); V = []; zero = {}
    for s in S:
        a = (X[src == s] - L0[src == s] @ J).mean(0)                    # 그날 명령 0 에서의 특징
        c0 = np.linalg.lstsq(J.T, -a, rcond=None)[0]                    # 절대 영점이 되는 그날 명령
        L[src == s] -= c0; zero[s] = c0; V.append(a + c0 @ J)           # 움직임으로 설명 안 되는 차이
    V = np.array(V)

    rng = np.random.default_rng(SEED)
    Z = rng.normal(0, SCALE, (COPIES, len(X), len(S)))
    feats = [X] + [X + Z[k] @ V for k in range(COPIES)]
    F = np.vstack(feats); Y = np.vstack([L] * (COPIES + 1))
    copy = np.repeat(np.arange(COPIES + 1), len(X)); oid = np.tile(ids, COPIES + 1); osrc = np.tile(src, COPIES + 1)
    Zall = np.vstack([np.zeros((len(X), len(S)))] + [Z[k] for k in range(COPIES)])

    OUT.mkdir(parents=True)
    with (OUT / "augmented_training_features.tsv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["row", "id", "copy", "source", "Y_abs", "T_abs", "Y_cmd", "T_cmd"] + [f"z_{NAMES[s]}" for s in S] + ["features66"])
        Lc = np.vstack([L0] * (COPIES + 1))
        for i in range(len(F)):
            w.writerow([i, oid[i], copy[i], osrc[i], f"{Y[i,0]:.6f}", f"{Y[i,1]:.6f}", f"{Lc[i,0]:.2f}", f"{Lc[i,1]:.2f}"]
                       + [f"{z:.6f}" for z in Zall[i]] + [";".join(repr(float(v)) for v in F[i])])
    with (OUT / "day_vectors.tsv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["session", "name", "abs_zero_Y_cmd", "abs_zero_T_cmd", "rms_qr_px", "rms_bar_px", "rms_screw_px", "vector66"])
        for s, v in zip(S, V):
            w.writerow([s, NAMES[s], f"{zero[s][0]:.6f}", f"{zero[s][1]:.6f}"] + [f"{np.sqrt((v[b] ** 2).mean()):.4f}" for b in (slice(0, 54), slice(54, 58), slice(58, 66))]
                       + [";".join(repr(float(x)) for x in v)])
    np.savez_compressed(OUT / "augmented.npz", features=F, labels_abs=Y, id=oid, copy=copy, source=osrc, z=Zall, day_vectors=V, sessions=np.array(S), slope=J)

    # 확인: 원본만 학습한 모델과 증강 자료로 학습한 모델을 원본 702장에 적용
    ok = lambda E: (abs(E[:, 0]) <= .045) & (abs(E[:, 1]) <= .01)
    lines = [f"행 {len(F)} (원본 {len(X)} + 복사본 {COPIES} × {len(X)}), 특징 {F.shape[1]}, 배율 {SCALE}, seed {SEED}"]
    for name, (Ft, Yt) in {"원본만 학습": (X, L), "증강 자료로 학습": (F, Y)}.items():
        M = reoof.fit(Ft, Yt); E = reoof.predict(M, X) - L
        # 날 차이 방향으로 민 원본에 대한 반응: 벡터 하나를 그대로 더했을 때 예측이 얼마나 변하나
        d = np.array([abs(reoof.predict(M, X + v) - reoof.predict(M, X)).mean(0) for v in V])
        lines.append(f"{name}: 원본 702장 다시 넣기 통과 {int(ok(E).sum())}/702, σY {E[:,0].std():.5f} σT {E[:,1].std():.5f} | "
                     f"날 차이 벡터를 더했을 때 예측 변화 평균 Y {d[:,0].mean():.5f} T {d[:,1].mean():.5f}")
    lines.append("복사본 특징 − 원본 특징의 RMS(px): QR %.3f 바 %.3f 나사 %.3f" % tuple(np.sqrt(((F[len(X):] - np.vstack([X] * COPIES))[:, b] ** 2).mean()) for b in (slice(0, 54), slice(54, 58), slice(58, 66))))
    (OUT / "확인.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines)); print("저장:", OUT)


if __name__ == "__main__":
    main()
