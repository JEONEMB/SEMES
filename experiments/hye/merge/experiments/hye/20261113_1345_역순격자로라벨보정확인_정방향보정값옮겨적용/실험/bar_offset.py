"""역순 사진의 바 기울기 2개를 장비 모듈 측정 기준(rig_1st/center.bmp)으로 옮기는 상수를 잰다.

왜: 역순 사진은 바가 세로로 8~14 px 밀려 C++ 모듈(바 탐색 ±11.5 px)로는 장비 기준으로 잴 수 없다. 그래서 역순 센터 기준으로 쟀다.
    두 날 자료를 한 모델로 학습하려면 바 기울기도 같은 측정 기준이어야 한다.
방법: C++ 와 같은 정의를 Python 으로, 탐색만 ±22 px 로 넓혀 잰다.
  칸(64 px)마다 64열 평균 밝기 프로파일 → 9탭 가우시안 → 기준 프로파일 가운데 21행과 현재 프로파일을 3차 스플라인으로 옮겨 가며
  이득·오프셋을 맞춘 제곱오차가 가장 작은 이동을 찾는다. 바마다 24칸 이동을 칸 가로 위치(1000 px 단위)에 직선으로 맞춘 기울기.
검산: 정방향 grid 센터를 같은 방법으로 재서 C++ 값과 비교한다.
기준 차이: 역순 센터 계열 18장을 장비 기준(이 스크립트)과 역순 기준(C++)으로 모두 재서 차이를 평균한다. 라벨 영점이 아니다.
출력: 결과/측정기준차이_바.txt
실행: 저장소 루트에서  python "<이 폴더>/bar_offset.py"
"""
import csv, json, pathlib
import cv2, numpy as np
from scipy.interpolate import CubicSpline

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[3]
A = ROOT / "experiments/hye/00000_final_Y_T_ssafy/assets"
PREV = HERE.parents[1] / "20261113_0950_우리장비사진거부_장비기준자료와라벨보정"
EXT = 22
K = np.exp(-.5 * np.arange(-4, 5) ** 2); K /= K.sum()


def read(p):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_GRAYSCALE)


def profile(img, x0, x1, y0, n):
    p = img[y0:y0 + n, x0:x1].astype(float).mean(1)
    return np.convolve(np.pad(p, 4, mode="symmetric"), K, "valid")


blocks = json.loads((A / "line_protocol.json").read_text(encoding="utf-8"))["blocks"]
ref = read(A / "center.bmp")
REFP = [profile(ref, *b["read_box"][0::2], b["read_box"][1], 49) for b in blocks]
BX = np.array([(b["read_box"][0] + b["read_box"][2]) / 2 for b in blocks])


def tilts(img):
    shifts = []
    for b, rp in zip(blocks, REFP):
        x0, y0, x1, _ = b["read_box"]
        cur = profile(img, x0, x1, y0 - EXT, 49 + 2 * EXT)   # 현재 프로파일 인덱스 = 기준 인덱스 + EXT
        cs = CubicSpline(np.arange(len(cur)), cur)
        t = rp[14:35]; best = None
        for s in np.arange(-EXT, EXT + 1e-9, .005):
            v = cs(np.arange(21) + 14 + s + EXT); A_ = np.c_[v, np.ones(21)]
            c = np.linalg.lstsq(A_, t, rcond=None)[0]; cost = ((A_ @ c - t) ** 2).mean()
            if best is None or cost < best[0]: best = (cost, s)
        shifts.append(best[1])
    sh = np.array(shifts); out = []
    for bar in range(2):
        x = BX[24 * bar:24 * bar + 24] / 1000.; y = sh[24 * bar:24 * bar + 24]
        out.append(float(np.polyfit(x, y, 1)[0]))
    return np.array(out), sh


def main():
    feat = {r["id"]: r for r in csv.DictReader((PREV / "결과/features7.tsv").open(encoding="utf-8"), delimiter="\t")}
    lines = []
    for fid, p in (("200000", ROOT / "data/clean/rig_20261112/rig_grid/center.bmp"), ("291681", ROOT / "data/clean/rig_20261112/rig_grid/_check/center_end.bmp")):
        v, _ = tilts(read(p))
        lines.append(f"검산 {fid}: Python {v[0]:+.4f} {v[1]:+.4f} | C++ {float(feat[fid]['bar_tilt_upper']):+.4f} {float(feat[fid]['bar_tilt_lower']):+.4f}")
    shots = list(csv.DictReader((HERE.parent / "결과/촬영목록_rev.tsv").open(encoding="utf-8"), delimiter="\t"))
    rel = {r["id"]: r for r in csv.DictReader((HERE.parent / "결과/features7_rev.tsv").open(encoding="utf-8"), delimiter="\t")}
    offs = []
    for r in shots:
        if r["kind"] == "shot": continue
        v, sh = tilts(read(ROOT / "data/clean/rig_20261113_rev" / r["file"]))
        b = np.array([float(rel[r["id"]]["bar_tilt_upper"]), float(rel[r["id"]]["bar_tilt_lower"])]); offs.append(v - b)
        lines.append(f"  {r['id']}: 장비 기준 {v[0]:+.4f} {v[1]:+.4f}  역순 기준 {b[0]:+.4f} {b[1]:+.4f}  차이 {v[0]-b[0]:+.4f} {v[1]-b[1]:+.4f}  (칸 이동 범위 {sh.min():+.1f}~{sh.max():+.1f})")
    offs = np.array(offs); m, sd = offs.mean(0), offs.std(0, ddof=1)
    lines.append(f"기준 차이 평균 ({len(offs)}장): 위 {m[0]:+.4f} 아래 {m[1]:+.4f}  표준편차 {sd[0]:.4f} {sd[1]:.4f}")
    text = "\n".join(lines) + "\n"; print(text)
    np.savetxt(HERE.parent / "결과" / "측정기준차이_바.txt", m[None], fmt="%.6f", header="bar_tilt_upper bar_tilt_lower  (장비 기준 = 역순 기준 + 이 값)")
    (HERE.parent / "결과" / "측정기준차이_바_기록.txt").write_bytes(text.encode("utf-8"))


if __name__ == "__main__":
    main()
