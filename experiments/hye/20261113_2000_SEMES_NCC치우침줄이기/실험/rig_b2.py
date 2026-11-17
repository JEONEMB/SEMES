"""우리 장비 사진에서 NCC(흐림 σ2 + 포물선) QR 값을 잰다. SEMES 에서 고른 방법이 장비 사진에 해가 없는지 보기 위한 것이다.

기준 자료: experiments/hye/00000_final_Y_T_ssafy/assets (center.bmp, site.json 의 QR 상자와 탐색 반폭). 역순 사진도 같은 기준으로 바로 잰다.
입력: photos.tsv 여러 개 (id<TAB>경로). 출력: <출력 tsv> id, P3, B2
실행: python rig_b2.py <출력 tsv> <photos.tsv> [photos.tsv ...]
"""
import json, pathlib, sys
from multiprocessing import Pool
import cv2, numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[4]
A = ROOT / "experiments/hye/00000_final_Y_T_ssafy/assets"
SITE = json.loads((A / "site.json").read_text())
BOX = [tuple(b) for b in SITE["qr"]]; SX, SY = SITE["search_x"], SITE["search_y"]


def read(p):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_GRAYSCALE)


CENTER = read(A / "center.bmp"); CB = cv2.GaussianBlur(CENTER, (0, 0), 2.0)


def parab(c, x, y):
    l, m, r = float(c[y, x - 1]), float(c[y, x]), float(c[y, x + 1]); den = l - 2 * m + r
    return .5 * (l - r) / den if den < 0 else 0.0


def one(item):
    i, p = item
    img = read(p)
    if img is None: return i, np.nan, np.nan
    out = []
    for im, ce in ((img, CENTER), (cv2.GaussianBlur(img, (0, 0), 2.0), CB)):
        dx = []
        for x, y, w, h in BOX:
            c = cv2.matchTemplate(im[y - SY:y + h + SY, x - SX:x + w + SX], ce[y:y + h, x:x + w], cv2.TM_CCOEFF_NORMED)
            _, _, _, (px, py) = cv2.minMaxLoc(c); dx.append(px - SX + parab(c, px, py))
        out.append(float(np.mean(dx)))
    return i, out[0], out[1]


def main():
    items = []
    for f in sys.argv[2:]:
        for l in open(f, encoding="utf-8"):
            if l.strip(): a, b = l.rstrip("\n").split("\t"); items.append((a, b))
    with Pool(8) as pool: res = pool.map(one, items, chunksize=8)
    with open(sys.argv[1], "w", encoding="utf-8") as fo:
        fo.write("id\tP3\tB2\n")
        for i, a, b in res: fo.write(f"{i}\t{a:.6f}\t{b:.6f}\n")
    print("done", len(res))


if __name__ == "__main__":
    main()
