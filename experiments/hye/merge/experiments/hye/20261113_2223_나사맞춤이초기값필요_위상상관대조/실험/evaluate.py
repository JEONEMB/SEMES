"""값 7개 구성에서 나사 4개의 가로 이동만 위상 상관 후보로 바꿔 고정 검증으로 채점한다.

기준 구성: QR 536점 가로 평균 1 + 바 기울기 2 + 나사 가로 4 (기울기 영상 ECC). 증강 없음.
학습식, 라벨(절대 영점), 분할(splits), 교란은 20261112_1835 실험의 combo.py / horizontal.py 를 그대로 불러 쓴다.
horizontal.main 은 증강 있음/없음 두 줄을 내므로 증강 없음 줄만 남긴다.

사진 한 장 흔들림: 1차 T축 사진(T 를 같은 간격으로 바꿈)을 T 순서로 놓고 2차 차분의 표준편차 / √6.
 - "값 자체": 스테이지 흔들림 + 측정 잡음
 - "(후보 − ECC)": 두 측정의 잡음 차이만 남는다

출력: 결과/정확도.tsv, 결과/흔들림.tsv
실행: 저장소 루트에서  python "<이 폴더>/실험/evaluate.py"
"""
import csv, glob, io, contextlib, pathlib, sys, tempfile
import numpy as np

EXP = pathlib.Path(__file__).resolve().parent.parent; OUT = EXP / "결과"
ROOT = EXP.parents[2]
sys.path.insert(0, str(next(ROOT.glob("experiments/hye/20261112_1835_*")) / "실험"))
import combo, horizontal

P = dict(np.load(OUT / "pc_meas.npz")); P.update(np.load(OUT / "pcu_meas.npz"))
VARIANTS = [("기울기 영상 ECC (지금)", None), ("위상 상관 OpenCV 밝기", "pc_raw"), ("위상 상관 OpenCV 기울기", "pc_grad"),
            ("위상 상관 OpenCV 밝기, 초기값 없음", "pc_raw_ni"), ("위상 상관 OpenCV 기울기, 초기값 없음", "pc_grad_ni"),
            ("정밀 위상 상관 밝기", "pcu_raw"), ("정밀 위상 상관 기울기", "pcu_grad"),
            ("정밀 위상 상관 밝기, 초기값 없음", "pcu_raw_ni"), ("정밀 위상 상관 기울기, 초기값 없음", "pcu_grad_ni")]
SPEC = ("P536_2", True, "grad", True, "tilt")
load0 = combo.load
ids0, src0, L0, cmd0, raw0, geo0 = load0(); assert (P["ids"] == ids0).all()

rows = []
for name, key in VARIANTS:
    def load(key=key):
        raw = dict(raw0)
        if key: raw["s_grad"] = P[key].reshape(-1, 4, 2).copy()
        return ids0, src0, L0, cmd0, raw, geo0
    combo.load = load; horizontal.SPECS = [(name, SPEC)]
    with tempfile.TemporaryDirectory() as td:
        horizontal.OUT = pathlib.Path(td)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf): horizontal.main()
        r = [x for x in csv.DictReader(open(pathlib.Path(td) / "가로만_결과.tsv", encoding="utf-8"), delimiter="\t") if x["증강"] == "없음"][0]
    rows.append(r); print(buf.getvalue().splitlines()[0], flush=True)
combo.load = load0
with (OUT / "정확도.tsv").open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t"); w.writeheader(); w.writerows(rows)

# 사진 한 장 흔들림 (1차 T축)
import reoof
tf = {int(r["id"]): r for r in reoof.tsv("training_features.tsv")}
axis = np.array([tf[i]["source"] == "20261014" and "/Theta/" in tf[i]["path"] for i in ids0])
order = np.where(axis)[0][np.argsort(L0[axis, 1], kind="stable")]
ecc = raw0["s_grad"][:, :, 0]
def jit(v): d = np.diff(v[order], 2, axis=0); return d.std(0).mean() / np.sqrt(6)
with (OUT / "흔들림.tsv").open("w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, delimiter="\t"); w.writerow(["후보", "T축 장수", "값 자체 px", "(후보 − ECC) px"])
    for name, key in VARIANTS:
        v = ecc if key is None else P[key].reshape(-1, 4, 2)[:, :, 0]
        w.writerow([name, len(order), f"{jit(v):.4f}", f"{jit(v - ecc):.4f}"]); print(name, len(order), f"{jit(v):.4f}", f"{jit(v - ecc):.4f}")
