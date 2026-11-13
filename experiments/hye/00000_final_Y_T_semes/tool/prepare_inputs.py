"""v2tool 입력 파일을 만든다(자료 준비만 한다. 측정과 학습, 평가는 C++ 도구가 한다).

출력(인자로 준 폴더, ASCII 경로):
  photos.tsv  id<TAB>사진 절대 경로   (702장: ID 702 제외, T=+2.00 12장 제외)
  labels.tsv  절대 영점 라벨 사본
  splits.tsv  20261112_1835 실험의 splits.json 을 표로 옮긴 것(id, final, cv0..cv9, half0..half9)
  python_features7.tsv  Python 분석이 쓴 값 7개(대조용)

실행: 저장소 루트에서  python "<이 폴더>/실험/prepare_inputs.py" <출력 폴더>
"""
import csv, json, pathlib, shutil, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "apps/distribution")); sys.path.insert(0, str(ROOT / "experiments/hye/20261112_1835_조합마다검증방식다름_방법론별증강과고정검증/실험"))
import reoof, combo

OUT = pathlib.Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
E4 = ROOT / "experiments/hye/20261112_1504_QR요약54개_536점대조/결과"
tf = reoof.tsv("training_features.tsv")
p4 = dict(zip([int(x) for x in (E4 / "round4_ids.txt").read_text().split()], [l.strip() for l in (E4 / "round4_list.txt").read_text(encoding="utf-8").splitlines() if l.strip()]))
rows = [r for r in tf if int(r["id"]) != 702 and abs(float(r["T"])) <= 1.0001]
with (OUT / "photos.tsv").open("w", encoding="utf-8", newline="") as f:
    for r in rows:
        rid = int(r["id"]); p = pathlib.Path(p4[rid]) if rid in p4 else ROOT / r["path"]
        assert p.exists(), p
        f.write(f"{rid}\t{p.as_posix()}\n")
shutil.copyfile(combo.LAB, OUT / "labels.tsv")
sp = json.loads((combo.OUT / "splits.json").read_text(encoding="utf-8")); final = set(sp["final"])
with (OUT / "splits.tsv").open("w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, delimiter="\t"); w.writerow(["id", "final"] + [f"cv{k}" for k in range(10)] + [f"half{k}" for k in range(10)])
    for r in rows:
        i = int(r["id"]); s = str(i)
        w.writerow([i, int(i in final)] + [sp["cv"][k].get(s, -1) for k in range(10)] + [sp["half"][k].get(s, -1) for k in range(10)])
ids, src, L, cmd, raw, geo = combo.load()
F = np.c_[raw["d536"].mean(1)[:, :1], raw["bars"][:, [1, 3]], raw["s_grad"][:, :, 0], raw["s_grad"][:, :, 1]]
with (OUT / "python_features7.tsv").open("w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, delimiter="\t"); w.writerow(["id", "qr_dx_mean", "bar_tilt_upper", "bar_tilt_lower"] + [f"screw{k}_x" for k in range(1, 5)] + [f"screw{k}_y" for k in range(1, 5)])
    for i, v in zip(ids, F): w.writerow([int(i)] + [repr(float(x)) for x in v])
print("photos", len(rows), "->", OUT)
