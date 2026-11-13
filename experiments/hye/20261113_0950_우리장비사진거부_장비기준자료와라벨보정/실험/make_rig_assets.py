"""우리 장비(rig) 센터 사진에 맞는 기준 자료를 만든다. 자료 준비만 한다(측정·학습·평가는 C++ 도구가 한다).

SEMES 자료(experiments/hye/00000_final_Y_T/assets)의 정의를 그대로 옮긴다. 새로 고르는 것은 없다.
  center.bmp          rig_1st/center.bmp 사본. 이 사진과 화면이 같은 자세가 절대 영점 (0, 0) 이다
  site.json           QR 상자 3개, 536점 추적 영역, 나사 4개 중심, QR 정수 이동 탐색 반폭
  points.json         SEMES 의 QR 구조 모서리 630점(그중 536점 선택)을 QR 별 homography 로 장비 센터에 옮긴 좌표
  geometry.json       같은 630점(관측기의 점검용 좌표)
  line_protocol.json  위·아래 바의 측정 칸 48개(가로 범위는 SEMES 와 같고, 세로 위치만 장비 센터의 가장자리에 맞춤)

옮기는 방법
  QR 상자: SEMES 센터의 QR 상자를 장비 센터에서 NCC 로 찾은 정수 이동만큼 옮긴다(크기는 그대로).
  630점:   QR 마다 SEMES 센터 조각 → 장비 센터 조각의 homography 를 ECC 로 구해 점을 옮긴다(같은 QR 무늬의 같은 모서리).
  나사:    장비 센터에서 나사 4개의 기울기 크기 무게중심(정수 반올림).
  바:      SEMES 와 같은 가로 범위의 64 px 칸마다, 장비 센터에서 가장 뚜렷한 가로 가장자리 행을 기준 행으로 잡는다.

실행: 저장소 루트에서  python "<이 폴더>/make_rig_assets.py"
"""
import hashlib, json, pathlib, shutil
import cv2, numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SEMES = ROOT / "experiments/hye/00000_final_Y_T/assets"
RIG = ROOT / "data/clean/rig_20261112"
OUT = HERE / "assets_rig"
QR = [(1552, 969, 184, 198), (1809, 971, 184, 198), (2067, 973, 185, 198)]
FULL_CROP = (1340, 800, 1130, 530)
SCREW_GUESS = [(951, 1390), (2805, 1426), (997, 1706), (2744, 1743)]   # 눈으로 본 대략 위치
SEARCH_X, SEARCH_Y = 125, 22   # SEMES 는 110. 장비는 Y+2, T+1 에서 QR 이 114 px 움직인다. QR 간격 257 px 의 절반(128)보다 작게 둔다
# 바의 기준 가장자리(장비 센터에서 본 대략 행, 밝기 변화 부호). 가로 범위는 SEMES 와 같다
BARS = [("upper_left", "upper", 400, 964, 1), ("upper_right", "upper", 2700, 993, 1),
        ("lower_left", "lower", 400, 1935, -1), ("lower_right", "lower", 2600, 1961, -1)]


def read(path):
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_GRAYSCALE)


def main():
    OUT.mkdir(exist_ok=True)
    shutil.copyfile(RIG / "rig_1st/center.bmp", OUT / "center.bmp")
    s, c = read(SEMES / "center.bmp"), read(OUT / "center.bmp")
    assert s.shape == c.shape == (2160, 3840)

    # QR 상자와 QR 별 homography
    boxes, H = [], []
    for x, y, w, h in QR:
        pad = 150
        res = cv2.matchTemplate(c[y - pad:y + h + pad, x - pad:x + w + pad], s[y:y + h, x:x + w], cv2.TM_CCOEFF_NORMED)
        _, ncc, _, loc = cv2.minMaxLoc(res)
        dx, dy = loc[0] - pad, loc[1] - pad
        boxes.append((x + dx, y + dy, w, h))
        m = 12   # 상자보다 12 px 넓게 맞춘다
        tpl = s[y - m:y + h + m, x - m:x + w + m]
        cur = c[y + dy - m:y + dy + h + m, x + dx - m:x + dx + w + m]
        warp = np.eye(3, dtype=np.float32)
        cc, warp = cv2.findTransformECC(tpl, cur, warp, cv2.MOTION_HOMOGRAPHY, (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 500, 1e-9), None, 5)
        # ECC: tpl(p) ≈ cur(warp·p). 전체 좌표로 바꾼다
        T0 = np.array([[1, 0, -(x - m)], [0, 1, -(y - m)], [0, 0, 1]], float)
        T1 = np.array([[1, 0, x + dx - m], [0, 1, y + dy - m], [0, 0, 1]], float)
        H.append(T1 @ warp.astype(float) @ T0)
        print(f"QR {len(boxes)}: 정수 이동 ({dx:+d}, {dy:+d}) NCC {ncc:.3f}  ECC 상관 {cc:.4f}")
        assert ncc > .5 and cc > .9

    def move(k, x, y):
        v = H[k] @ np.array([x, y, 1.0])
        return float(v[0] / v[2]), float(v[1] / v[2])

    pts = json.loads((SEMES / "points.json").read_text(encoding="utf-8"))["points"]
    new = []
    for x, y, k, sel in pts:
        nx, ny = move(int(k), x, y)
        bx, by, bw, bh = boxes[int(k)]
        assert bx <= int(nx) < bx + bw and by <= int(ny) < by + bh, (x, y, nx, ny)
        new.append([nx, ny, int(k), int(sel)])
    (OUT / "points.json").write_bytes(json.dumps({"points": new, "source": "SEMES points.json moved by per-QR ECC homography to the rig center"}, indent=1).encode())
    geo = json.loads((SEMES / "geometry.json").read_text(encoding="utf-8"))
    for n in geo["corners"]:
        n["cx"], n["cy"] = move(int(n["qr"]) - 1, n["cx"], n["cy"])
    (OUT / "geometry.json").write_bytes(json.dumps(geo).encode())

    # 나사: 기울기 크기 무게중심
    screws = []
    for gx, gy in SCREW_GUESS:
        for _ in range(3):
            p = cv2.GaussianBlur(c[gy - 70:gy + 71, gx - 70:gx + 71].astype(np.float32), (0, 0), 1.5)
            mag = cv2.magnitude(cv2.Sobel(p, cv2.CV_32F, 1, 0), cv2.Sobel(p, cv2.CV_32F, 0, 1))
            mag[mag < .35 * mag.max()] = 0
            yy, xx = np.mgrid[-70:71, -70:71]
            gx, gy = gx + int(round(float((mag * xx).sum() / mag.sum()))), gy + int(round(float((mag * yy).sum() / mag.sum())))
        screws.append([gx, gy])
    print("나사 중심", screws)

    # 바: 칸마다 기준 행
    proto = json.loads((SEMES / "line_protocol.json").read_text(encoding="utf-8"))
    sha = hashlib.sha256((OUT / "center.bmp").read_bytes()).hexdigest()
    proto["center_sha256"] = sha
    proto["scope"] += " (rig 20261112: same x ranges as SEMES, rows re-located on the rig center)"
    regions, blocks = [], []
    for name, line, x_start, y_guess, pol in BARS:
        rows = []
        for b in range(12):
            x0 = x_start + 64 * b
            prof = cv2.GaussianBlur(c[y_guess - 40:y_guess + 41, x0:x0 + 64].astype(np.float64).mean(1).reshape(-1, 1), (0, 0), 1.0).ravel()
            g = np.gradient(prof) * pol
            lo = 40 - 14 if not rows else rows[-1] - (y_guess - 40) - 3   # 이웃 칸에서 3행 안
            hi = 40 + 15 if not rows else rows[-1] - (y_guess - 40) + 4
            ref = y_guess - 40 + lo + int(np.argmax(g[lo:hi]))
            rows.append(ref)
            blocks.append({"id": f"{name}_{b:02d}", "line": line, "side": name.split("_")[1], "x0": x0, "x1": x0 + 64,
                           "reference_y": ref, "read_box": [x0, ref - 24, x0 + 64, ref + 25], "edge_polarity": pol})
        regions.append({"name": name, "line": line, "x_range": [x_start, x_start + 768], "reference_edge_search_y": [min(rows), max(rows)], "polarity": pol})
        print(name, rows)
    proto["regions"], proto["blocks"] = regions, blocks
    (OUT / "line_protocol.json").write_bytes(json.dumps(proto, ensure_ascii=False, indent=1).encode("utf-8"))

    mean = np.mean([[b[0] - q[0], b[1] - q[1]] for b, q in zip(boxes, QR)], 0)
    crop = [FULL_CROP[0] + int(round(mean[0])), FULL_CROP[1] + int(round(mean[1])), FULL_CROP[2], FULL_CROP[3]]
    site = {"center_sha256": sha, "full_crop": crop, "qr": [list(b) for b in boxes], "screws": screws, "search_x": SEARCH_X, "search_y": SEARCH_Y,
            "note": "rig 20261112. center.bmp = rig_1st/center.bmp. Made by make_rig_assets.py"}
    (OUT / "site.json").write_bytes(json.dumps(site, indent=1).encode())
    print("site", site)


if __name__ == "__main__":
    main()
