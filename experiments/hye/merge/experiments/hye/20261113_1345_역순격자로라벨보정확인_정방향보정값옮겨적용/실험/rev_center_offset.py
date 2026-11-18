"""역순 사진의 측정 기준 영상(역순 센터)과 장비 모듈의 측정 기준 영상(rig_1st/center.bmp) 사이의 값 차이를 잰다.

왜: 역순 사진은 바가 세로로 8~14 px 밀려 장비 모듈 자료로는 C++ 측정이 바 단계에서 거부된다. 그래서 역순 센터 기준 자료로 쟀다.
    절대 영점으로 옮기려면 "역순 센터의 자세 값 5개(QR 536점 가로 평균, 나사 4개 가로)"가 장비 센터 기준으로 얼마인지 필요하다.
    자세 계산에는 바를 쓰지 않으므로 이 5개만 있으면 된다.
방법(C++ 모듈과 같은 정의를 Python 으로 근사)
  QR: QR 상자마다 ECC homography(장비 센터 조각 → 사진 조각)로 선택된 536점을 옮긴 가로 이동의 평균
  나사: 나사 중심 141×141 조각의 기울기 크기(가우시안 1.5, Sobel 3)를 흐림 5 로 다듬고, 반지름 45 원 마스크로 ECC 평행이동
검산: 정방향 grid 센터(C++ 값이 있음)에 같은 계산을 해서 C++ 값과 비교한다.
출력: 결과/측정기준차이_값5개.txt, 결과/측정기준차이.txt
실행: 저장소 루트에서  python "<이 폴더>/rev_center_offset.py"
"""
import json, pathlib, csv
import cv2, numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[3]
A = ROOT / "experiments/hye/00000_final_Y_T_ssafy/assets"
PREV = HERE.parents[1] / "20261113_0950_우리장비사진거부_장비기준자료와라벨보정"


def read(p):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_GRAYSCALE)


site = json.loads((A / "site.json").read_text()); pts = json.loads((A / "points.json").read_text())["points"]
ref = read(A / "center.bmp")


def grad(p8):
    b = cv2.GaussianBlur(p8.astype(np.float32), (0, 0), 1.5)
    return cv2.magnitude(cv2.Sobel(b, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=3))


disk = np.zeros((141, 141), np.uint8); yy, xx = np.mgrid[-70:71, -70:71]; disk[np.hypot(xx, yy) <= 44] = 255


def five(img):
    qr = []
    for k, (x, y, w, h) in enumerate(site["qr"]):
        pad = 60
        res = cv2.matchTemplate(img[y - pad:y + h + pad, x - pad:x + w + pad], ref[y:y + h, x:x + w], cv2.TM_CCOEFF_NORMED)
        _, _, _, loc = cv2.minMaxLoc(res); dx, dy = loc[0] - pad, loc[1] - pad; m = 12
        tpl = ref[y - m:y + h + m, x - m:x + w + m]; cur = img[y + dy - m:y + dy + h + m, x + dx - m:x + dx + w + m]
        _, warp = cv2.findTransformECC(tpl, cur, np.eye(3, dtype=np.float32), cv2.MOTION_HOMOGRAPHY, (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 500, 1e-9), None, 5)
        T0 = np.array([[1, 0, -(x - m)], [0, 1, -(y - m)], [0, 0, 1]], float); T1 = np.array([[1, 0, x + dx - m], [0, 1, y + dy - m], [0, 0, 1]], float)
        H = T1 @ warp.astype(float) @ T0
        for px, py, kk, sel in pts:
            if int(kk) == k and int(sel):
                v = H @ np.array([px, py, 1.0]); qr.append(v[0] / v[2] - px)
    out = [float(np.mean(qr))]
    for sx, sy in site["screws"]:
        t = cv2.GaussianBlur(grad(ref[sy - 70:sy + 71, sx - 70:sx + 71]), (5, 5), 0)
        best = None
        for oy in range(-20, 21):            # 정수 시작점: 세로로 밀린 사진을 위해 넓게 찾는다
            c = grad(img[sy + oy - 70:sy + oy + 71, sx - 70:sx + 71]); s = float(np.corrcoef(t.ravel(), c.ravel())[0, 1])
            if best is None or s > best[0]: best = (s, oy)
        oy = best[1]; c = grad(img[sy + oy - 70:sy + oy + 71, sx - 70:sx + 71])
        _, warp = cv2.findTransformECC(t, c, np.eye(2, 3, dtype=np.float32), cv2.MOTION_TRANSLATION, (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 200, 1e-8), disk, 1)
        out.append(float(warp[0, 2]))
    return np.array(out)


def main():
    # 1) 검산: 정방향 grid 센터를 Python 으로 재서 C++ 값과 비교
    feat = {r["id"]: r for r in csv.DictReader((PREV / "결과/features7.tsv").open(encoding="utf-8"), delimiter="	")}
    names = ["qr_dx_mean", "screw1_x", "screw2_x", "screw3_x", "screw4_x"]
    lines = []
    v = five(read(ROOT / "data/clean/rig_20261112/rig_grid/center.bmp"))
    lines.append("검산 정방향 grid 센터: Python " + " ".join(f"{x:+.3f}" for x in v) + "  | C++ " + " ".join(f"{float(feat['200000'][n]):+.3f}" for n in names))
    # 2) 측정 기준 바꾸기: 역순 센터 계열 18장(센터 1 + 다시 찍은 센터 17)을 장비 센터 기준(Python)과 역순 센터 기준(C++)으로 모두 재서
    #    두 기준의 차이(사진과 무관한 상수)를 평균한다. 라벨 영점이 아니다. 라벨 영점은 evaluate_rev.py 에서 사진 전체 회귀로 구한다.
    shots = list(csv.DictReader((HERE.parent / "결과/촬영목록_rev.tsv").open(encoding="utf-8"), delimiter="	"))
    rel = {r["id"]: r for r in csv.DictReader((HERE.parent / "결과/features7_rev.tsv").open(encoding="utf-8"), delimiter="	")}
    REV = ROOT / "data/clean/rig_20261113_rev"
    offs = []
    for r in shots:
        if r["kind"] == "shot": continue
        a = five(read(REV / r["file"])); b = np.array([float(rel[r["id"]][n]) for n in names]); offs.append(a - b)
        lines.append(f"  {r['id']} {r['file']}: 장비 기준 " + " ".join(f"{x:+.3f}" for x in a) + "  차이 " + " ".join(f"{x:+.3f}" for x in a - b))
    offs = np.array(offs); m, sd = offs.mean(0), offs.std(0, ddof=1)
    lines.append(f"기준 차이 평균 ({len(offs)}장): " + " ".join(f"{x:+.4f}" for x in m) + "   표준편차 " + " ".join(f"{x:.4f}" for x in sd))
    text = "\n".join(lines) + "\n"
    print(text)
    np.savetxt(HERE.parent / "결과" / "측정기준차이_값5개.txt", m[None], fmt="%.6f", header=" ".join(names) + "  (장비 센터 기준 값 = 역순 센터 기준 값 + 이 값)")
    (HERE.parent / "결과" / "측정기준차이.txt").write_bytes(text.encode("utf-8"))


if __name__ == "__main__":
    main()
