"""역순 세트 자기 센터(rig_grid_rev/center.bmp)에 맞춘 기준 자료를 만든다. 자료 준비만 한다.

왜: 장비 모듈 자료(00000_final_Y_T_ssafy/assets, 센터 = 11-12 rig_1st/center.bmp)로 역순 사진을 재면 1698장이 모두
    바 측정에서 거부된다. 11-13 아침 원점을 다시 잡은 뒤 판과 배경판 사이가 밀려, 바가 위 약 8 px, 아래 약 12 px 세로로
    옮겨 있고 바 탐색 범위(±11.5 px)를 넘는다. README 권고대로 세트 자기 센터를 기준으로 쓴다.

무엇을 바꾸나 (방법은 앞 실험 make_rig_assets.py 와 같고, 출발점이 SEMES 가 아니라 장비 자료다)
  center.bmp          rig_grid_rev/center.bmp 사본. 이 사진과 화면이 같은 자세가 (0, 0)
  QR 상자, 630점      장비 센터 → 역순 센터: QR 마다 NCC 정수 이동 + ECC homography 로 옮긴다
  나사 중심           장비 값에서 출발해 역순 센터의 기울기 크기 무게중심
  바 측정 칸 48개      가로 범위 그대로. 세로 기준 행만 역순 센터에서 가장 뚜렷한 가장자리로 다시 잡는다
  model7.tsv          그대로(1959장 L2 학습 계수). provenance.json 은 바뀐 파일의 해시만 고친다
출력: <이 실험>/실험/assets_rev/   (도구에 줄 때는 ASCII 경로로 복사한다)
실행: 저장소 루트에서  python "<이 폴더>/make_rev_assets.py"
"""
import hashlib, json, pathlib, shutil
import cv2, numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SRC = ROOT / "experiments/hye/00000_final_Y_T_ssafy/assets"
CENTER = ROOT / "data/clean/rig_20261113_rev/rig_grid_rev/center.bmp"
OUT = HERE / "assets_rev"


def read(path):
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_GRAYSCALE)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    shutil.copyfile(CENTER, OUT / "center.bmp"); shutil.copyfile(SRC / "model7.tsv", OUT / "model7.tsv")
    s, c = read(SRC / "center.bmp"), read(OUT / "center.bmp")
    site0 = json.loads((SRC / "site.json").read_text(encoding="utf-8"))

    boxes, H = [], []
    for x, y, w, h in site0["qr"]:
        pad = 60
        res = cv2.matchTemplate(c[y - pad:y + h + pad, x - pad:x + w + pad], s[y:y + h, x:x + w], cv2.TM_CCOEFF_NORMED)
        _, ncc, _, loc = cv2.minMaxLoc(res); dx, dy = loc[0] - pad, loc[1] - pad
        boxes.append([x + dx, y + dy, w, h]); m = 12
        tpl = s[y - m:y + h + m, x - m:x + w + m]; cur = c[y + dy - m:y + dy + h + m, x + dx - m:x + dx + w + m]
        cc, warp = cv2.findTransformECC(tpl, cur, np.eye(3, dtype=np.float32), cv2.MOTION_HOMOGRAPHY, (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 500, 1e-9), None, 5)
        T0 = np.array([[1, 0, -(x - m)], [0, 1, -(y - m)], [0, 0, 1]], float); T1 = np.array([[1, 0, x + dx - m], [0, 1, y + dy - m], [0, 0, 1]], float)
        H.append(T1 @ warp.astype(float) @ T0)
        print(f"QR {len(boxes)}: 정수 이동 ({dx:+d}, {dy:+d}) NCC {ncc:.3f}  ECC 상관 {cc:.4f}"); assert ncc > .5 and cc > .9

    def move(k, x, y):
        v = H[k] @ np.array([x, y, 1.0]); return float(v[0] / v[2]), float(v[1] / v[2])
    pts = json.loads((SRC / "points.json").read_text(encoding="utf-8"))["points"]; new = []
    for x, y, k, sel in pts:
        nx, ny = move(int(k), x, y); bx, by, bw, bh = boxes[int(k)]
        assert bx <= int(nx) < bx + bw and by <= int(ny) < by + bh; new.append([nx, ny, int(k), int(sel)])
    (OUT / "points.json").write_bytes(json.dumps({"points": new, "source": "rig assets points moved by per-QR ECC homography to the rig_grid_rev center"}, indent=1).encode())
    geo = json.loads((SRC / "geometry.json").read_text(encoding="utf-8"))
    for n in geo["corners"]: n["cx"], n["cy"] = move(int(n["qr"]) - 1, n["cx"], n["cy"])
    (OUT / "geometry.json").write_bytes(json.dumps(geo).encode())

    screws = []
    for gx, gy in site0["screws"]:
        for _ in range(3):
            p = cv2.GaussianBlur(c[gy - 70:gy + 71, gx - 70:gx + 71].astype(np.float32), (0, 0), 1.5)
            mag = cv2.magnitude(cv2.Sobel(p, cv2.CV_32F, 1, 0), cv2.Sobel(p, cv2.CV_32F, 0, 1)); mag[mag < .35 * mag.max()] = 0
            yy, xx = np.mgrid[-70:71, -70:71]
            gx, gy = gx + int(round(float((mag * xx).sum() / mag.sum()))), gy + int(round(float((mag * yy).sum() / mag.sum())))
        screws.append([gx, gy])
    print("나사 중심", site0["screws"], "→", screws)

    # 바: 칸마다 장비 기준 행 주변 ±20 행에서 같은 부호의 가장 뚜렷한 가장자리
    proto = json.loads((SRC / "line_protocol.json").read_text(encoding="utf-8")); shifts = []
    for b in proto["blocks"]:
        x0, x1, ref0, pol = b["x0"], b["x1"], b["reference_y"], b["edge_polarity"]
        prof = cv2.GaussianBlur(c[ref0 - 40:ref0 + 41, x0:x1].astype(np.float64).mean(1).reshape(-1, 1), (0, 0), 1.0).ravel()
        g = np.gradient(prof) * pol; ref = ref0 - 40 + 20 + int(np.argmax(g[20:61]))
        shifts.append(ref - ref0); b["reference_y"] = ref; b["read_box"] = [x0, ref - 24, x1, ref + 25]
    sh = np.array(shifts); print("바 기준 행 이동(px): 위", np.median(sh[:24]), sh[:24].min(), sh[:24].max(), " 아래", np.median(sh[24:]), sh[24:].min(), sh[24:].max())
    for r in proto["regions"]:
        rows = [b["reference_y"] for b in proto["blocks"] if b["id"].startswith(r["name"])]; r["reference_edge_search_y"] = [min(rows), max(rows)]
    proto["center_sha256"] = sha(OUT / "center.bmp"); proto["scope"] += " (rig_grid_rev 20261113: rows re-located on the rev center)"
    (OUT / "line_protocol.json").write_bytes(json.dumps(proto, ensure_ascii=False, indent=1).encode("utf-8"))

    mean = np.mean([[b[0] - q[0], b[1] - q[1]] for b, q in zip(boxes, site0["qr"])], 0)
    crop = [site0["full_crop"][0] + int(round(mean[0])), site0["full_crop"][1] + int(round(mean[1])), site0["full_crop"][2], site0["full_crop"][3]]
    site = dict(site0, center_sha256=sha(OUT / "center.bmp"), full_crop=crop, qr=boxes, screws=screws,
                note="rig_grid_rev 20261113. center.bmp = rig_grid_rev/center.bmp. Made by make_rev_assets.py from the ssafy module assets")
    (OUT / "site.json").write_bytes(json.dumps(site, indent=1).encode())
    prov = json.loads((SRC / "provenance.json").read_text(encoding="utf-8"))
    for f in ("center.bmp", "points.json", "geometry.json", "line_protocol.json", "model7.tsv"): prov[f + "_sha256"] = sha(OUT / f)
    prov["scope"] += " Reference re-made on the rig_grid_rev center (experiment 20261113_1345); coefficients unchanged."
    (OUT / "provenance.json").write_bytes(json.dumps(prov, indent=1).encode())
    print("site", site)


if __name__ == "__main__":
    main()
