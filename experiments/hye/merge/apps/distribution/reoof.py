"""시각화용 OOF 재계산: 라벨 정정과 아웃라이어 제외를 반영한다.

00004 Huber66 학습식(experiments/hye/00004_final_Y_T/src/model.cpp LinearModel::fit)을 그대로 옮겼다.
  train 표준화(모표준편차, 최소 1e-6) → 절편 제외 ridge 1e-7·n → 두 축 공유 Huber 가중 6회 풀이
  가중: r = max(|eY|/0.045, |eT|/0.01), w = min(1, 1/max(r,1e-12)), 평균 1로 정규화. 마지막 풀이의 계수를 쓴다.
특징값(features66)과 그룹 fold 배정은 커밋 46152fd 의 evidence 를 그대로 쓴다. 영상에서 특징을 다시 재지 않는다.
이 스크립트는 분포 페이지용 분석 도구다. 팀 전달 추론·학습 도구(C++)를 대체하지 않는다.

정정(2026-11-12 사용자 지시):
  - ID 420 `Y+0.05 T+0.32/Y+0.05 T+0.33 (3).bmp`: 라벨 T 0.33 → 0.32.
    근거: 원본 폴더 T+0.32 안에 있고, 그 폴더에 `T+0.32 (3).txt`만 있고 같은 이름 BMP가 없다.
    동봉 TXT T_Actual -1.350(=T+0.32 묶음, T+0.33 은 -1.340), 촬영 시각 20:35:22 가 T+0.32 (2) 와 (3).txt 사이다.
    파일과 원본 라벨 자료는 고치지 않는다. 그룹 fold 는 T+0.32 묶음(ID 418, 419)의 fold 를 따른다.
  - ID 702 `Y-0.8 T-0.96.bmp`(4차): 아웃라이어로 학습과 평가에서 모두 뺀다(OOF 예측 T +0.717, README 의 최대 T 사례).

  - T=+2.00 12장(2차, 정해진 범위 T ±1 밖): 학습에서만 뺀다. 예측은 그 사진 fold 의 모델(이 12장 없이 학습)로 내어 표시는 유지한다.
    근거: experiments/hye/20261112_1504_QR요약54개_536점대조/결과/결과.md 결과 3
    (섞은 평가는 같고, 세션을 통째로 뺀 시험에서 1차 10.4→56.8%, 4차 8.4→47.6%).

실행: 저장소 루트에서  python apps/distribution/reoof.py   (먼저 원래 결과 재현을 검사한다)
"""
import csv, io, pathlib, subprocess, sys
import numpy as np

COMMIT = "46152fd"
EV = "experiments/hye/00004_final_Y_T/evidence"
HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "data" / "group_oof_corrected.tsv"
RELABEL = {420: dict(T=0.32, fold_like=418)}
EXCLUDE = {702}
T_RANGE = 1.0   # |라벨 T| 가 이보다 크면 학습에서 뺀다(예측은 낸다)


def tsv(name):
    raw = subprocess.run(["git", "show", f"{COMMIT}:{EV}/{name}"], capture_output=True, check=True).stdout
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8")), delimiter="\t"))


def fit(X, Y):
    mean = X.mean(0); scale = np.maximum(X.std(0), 1e-6)
    A = np.c_[np.ones(len(X)), (X - mean) / scale]
    pen = np.eye(67) * (1e-7 * len(X)); pen[0, 0] = 0
    w = np.ones(len(X))
    for _ in range(6):
        sw = np.sqrt(w)[:, None]
        Aw, Yw = A * sw, Y * sw
        B = np.linalg.lstsq(Aw.T @ Aw + pen, Aw.T @ Yw, rcond=None)[0]
        P = A @ B
        r = np.maximum(np.abs(P[:, 0] - Y[:, 0]) / .045, np.abs(P[:, 1] - Y[:, 1]) / .01)
        w = np.minimum(1, 1 / np.maximum(r, 1e-12)); w *= len(w) / w.sum()
    return mean, scale, B


def predict(model, X):
    mean, scale, B = model
    return np.c_[np.ones(len(X)), (X - mean) / scale] @ B


def run_oof(ids, X, Y, fold, train_ok=None):
    P = np.full_like(Y, np.nan)
    if train_ok is None: train_ok = np.ones(len(Y), bool)
    for f in sorted(set(fold)):
        te = fold == f
        P[te] = predict(fit(X[~te & train_ok], Y[~te & train_ok]), X[te])
    return P


def main():
    tf = {int(r["id"]): r for r in tsv("training_features.tsv")}
    oof = {int(r["id"]): r for r in tsv("group_oof.tsv")}
    ids = np.array(sorted(tf))
    X = np.array([[float(v) for v in tf[i]["features66"].split(";")] for i in ids])
    Y = np.array([[float(tf[i]["Y"]), float(tf[i]["T"])] for i in ids])
    fold = np.array([int(oof[i]["fold"]) for i in ids])

    # 1) 원래 조건 재현 검사
    P0 = run_oof(ids, X, Y, fold)
    ref = np.array([[float(oof[i]["pred_Y"]), float(oof[i]["pred_T"])] for i in ids])
    diff = np.abs(P0 - ref).max()
    print(f"재현 검사: 원래 group OOF 와 최대 차 {diff:.2e}")
    if diff > 1e-6:
        sys.exit("재현 실패: 학습식이 원본과 다르다")

    # 2) 정정 반영
    keep = ~np.isin(ids, list(EXCLUDE))
    Y2, fold2 = Y.copy(), fold.copy()
    for i, c in RELABEL.items():
        k = np.where(ids == i)[0][0]
        Y2[k, 1] = c["T"]
        fold2[k] = fold[np.where(ids == c["fold_like"])[0][0]]
    inr = np.abs(Y2[keep][:, 1]) <= T_RANGE + 1e-9
    P_all = run_oof(ids[keep], X[keep], Y2[keep], fold2[keep])            # 비교용: 범위 밖도 학습에 넣음
    P = run_oof(ids[keep], X[keep], Y2[keep], fold2[keep], train_ok=inr)
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["id", "Y", "T", "pred_Y", "pred_T", "fold", "label_note"])
        for k, i in enumerate(ids[keep]):
            note = "relabel T 0.33->0.32" if i in RELABEL else ("train-excluded out-of-range" if not inr[k] else "")
            w.writerow([i, f"{Y2[keep][k,0]:.2f}", f"{Y2[keep][k,1]:.2f}", repr(float(P[k, 0])), repr(float(P[k, 1])), fold2[keep][k], note])
    E0 = (ref - Y)[keep]; Ea = P_all - Y2[keep]; E = P - Y2[keep]
    ok = lambda e: int(((np.abs(e[:, 0]) <= .045) & (np.abs(e[:, 1]) <= .01)).sum())
    print(f"정정 전(702 표시만 제외)        σY {E0[:,0].std(ddof=1):.5f} σT {E0[:,1].std(ddof=1):.5f}")
    print(f"정정 후, 범위 밖도 학습 (전체)  σY {Ea[:,0].std(ddof=1):.5f} σT {Ea[:,1].std(ddof=1):.5f}")
    print(f"  범위 안 {inr.sum()}장만           σY {Ea[inr,0].std(ddof=1):.5f} σT {Ea[inr,1].std(ddof=1):.5f} 통과 {ok(Ea[inr])}")
    print(f"범위 밖 {(~inr).sum()}장 학습 제외, 범위 안  σY {E[inr,0].std(ddof=1):.5f} σT {E[inr,1].std(ddof=1):.5f} 통과 {ok(E[inr])}")
    print("범위 밖 사진의 오차(eY, eT):"); print(np.round(E[~inr], 4))
    k420 = list(ids[keep]).index(420)
    print("ID 420 예측", P[k420], "라벨", Y2[keep][k420])
    print("저장:", OUT)


if __name__ == "__main__":
    main()
