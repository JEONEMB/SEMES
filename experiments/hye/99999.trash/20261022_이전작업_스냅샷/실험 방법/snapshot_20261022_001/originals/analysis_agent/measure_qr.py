from pathlib import Path
import csv
import re
import cv2
import numpy as np

ROOT = Path(r"C:\Users\SSAFY\Downloads\[기업연계_세메스] 교보재\[기업연계_세메스] 교보재\extracted\SSAFY Test Image Set")
OUT = Path(__file__).parent
BOXES = [(1554, 1734, 972, 1164), (1815, 1995, 972, 1164), (2073, 2253, 972, 1164)]

def read_gray(path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

center = read_gray(ROOT / "center.bmp")
paths = [("center", 0., ROOT / "center.bmp")]
for axis, folder, pat in [("T", "Theta", r"T([+-]\d+\.\d+)\.bmp"), ("Y", "Y", r"Y([+-]\d+\.\d+)\.bmp")]:
    for p in (ROOT / folder).glob("*.bmp"):
        m = re.fullmatch(pat, p.name)
        if m:
            paths.append((axis, float(m.group(1)), p))

rows = []
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 150, 1e-7)
for idx, (axis, val, path) in enumerate(paths):
    im = center if axis == "center" else read_gray(path)
    row = {"axis": axis, "value": val, "file": path.name}
    for qi, (x0, x1, y0, y1) in enumerate(BOXES):
        template = center[y0:y1, x0:x1]
        # Warp coordinates refer to the entire search image, so initialize
        # with the template's location plus a coarse normalized match.
        pad = 130
        sx0, sy0 = x0-pad, y0-pad
        search = im[sy0:y1+pad, sx0:x1+pad]
        match = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(match)
        dx0, dy0 = loc[0]-pad, loc[1]-pad
        # Use only the QR template support after coarse localization.
        target = im[y0+dy0:y1+dy0, x0+dx0:x1+dx0]
        ref = template
        W = np.float32([[1, 0, 0], [0, 1, 0]])
        try:
            cc, W = cv2.findTransformECC(ref, target, W, cv2.MOTION_AFFINE, criteria, None, 3)
        except cv2.error:
            cc = float("nan")
        row.update({f"q{qi}_{k}": v for k, v in {
            "dx": float(dx0 + W[0,2]), "dy": float(dy0 + W[1,2]),
            "a00": float(W[0,0]), "a01": float(W[0,1]),
            "a10": float(W[1,0]), "a11": float(W[1,1]),
            "cc": float(cc), "match": float(score),
        }.items()})
    rows.append(row)
    if idx % 30 == 0:
        print(idx, len(paths), axis, val, flush=True)

with (OUT / "qr_measurements.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(rows)
print("WROTE", len(rows), OUT / "qr_measurements.csv")
