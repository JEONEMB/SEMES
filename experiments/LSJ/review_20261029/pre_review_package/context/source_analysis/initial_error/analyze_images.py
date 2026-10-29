"""Exploratory image-space consistency analysis of the initial SEMES dataset.

Usage: python analyze_images.py --data-root <folder containing center.bmp>
Reads original BMPs without modifying them. Units inferred from labels are
command-equivalent units, not externally verified physical positioning errors.
There is only one center image; it is never counted as two observations.
Template initialization uses zero-mean NCC (ZNCC / TM_CCOEFF_NORMED).
Existing JSON keys named 'ncc' are legacy labels for these ZNCC measurements.
"""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


# Manually located from center.bmp, in native 3840 x 2160 pixel coordinates.
REGIONS = {
    "L": (1545, 963, 1740, 1173),
    "C": (1803, 963, 1998, 1173),
    "R": (2062, 965, 2257, 1175),
    "background_left": (921, 1313, 1007, 1402),
    "background_right": (2823, 1312, 2909, 1401),
    "background_upper": (2448, 585, 2528, 663),
}


def read_gray(path):
    raw = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
    if img is None or img.shape != (2160, 3840):
        raise ValueError(f"Unexpected image: {path}, shape={getattr(img, 'shape', None)}")
    return img, hashlib.sha256(raw).hexdigest()


def quadratic_offset(a, b, c):
    denominator = a - 2 * b + c
    return float(np.clip(0.5 * (a - c) / denominator, -1, 1)) if denominator < -1e-10 else 0.0


def measure(reference, current, roi, search_margin):
    x0, y0, x1, y1 = roi
    template = reference[y0:y1, x0:x1]
    search = current[y0-search_margin:y1+search_margin, x0-search_margin:x1+search_margin]
    # Subtracts each patch's mean before normalization: this is ZNCC.
    scores = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, (mx, my) = cv2.minMaxLoc(scores)
    if mx == 0 or my == 0 or mx == scores.shape[1]-1 or my == scores.shape[0]-1:
        raise ValueError("Match reached search boundary")
    qx = quadratic_offset(scores[my, mx-1], scores[my, mx], scores[my, mx+1])
    qy = quadratic_offset(scores[my-1, mx], scores[my, mx], scores[my+1, mx])
    coarse = np.array([mx-search_margin, my-search_margin], dtype=np.float64)
    patch = current[y0+int(coarse[1]):y1+int(coarse[1]), x0+int(coarse[0]):x1+int(coarse[0])]
    center = np.array([(x1-x0-1)/2, (y1-y0-1)/2, 1.0])
    result = {"ncc": float(score), "ncc_dx": float(coarse[0]+qx), "ncc_dy": float(coarse[1]+qy)}
    criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 60, 1e-6)
    for name, model in [("translation", cv2.MOTION_TRANSLATION), ("affine", cv2.MOTION_AFFINE)]:
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            correlation, warp = cv2.findTransformECC(template, patch, warp, model, criteria, None, 5)
        except cv2.error as error:
            result[name] = {"error": str(error)}
            continue
        delta = coarse + warp @ center - center[:2]
        result[name] = {
            "correlation": float(correlation), "dx": float(delta[0]), "dy": float(delta[1]),
            "warp": warp.tolist(),
        }
    result["brightness"] = float(patch.mean())
    result["laplacian_variance"] = float(cv2.Laplacian(patch, cv2.CV_32F).var())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    candidates = list(args.data_root.rglob("center.bmp"))
    if len(candidates) != 1:
        raise ValueError(f"Expected one center.bmp, found {len(candidates)}")
    root = candidates[0].parent
    reference, center_hash = read_gray(root / "center.bmp")
    files = [("center", 0.0, root / "center.bmp")]
    for axis, prefix in [("Y", "Y"), ("Theta", "T")]:
        files.extend((axis, float(p.stem[len(prefix):]), p) for p in (root / axis).glob("*.bmp"))
    files.sort(key=lambda item: (item[0], item[1]))
    rows = []
    cv2.setNumThreads(2)
    for index, (axis, position, path) in enumerate(files):
        current, digest = (reference, center_hash) if axis == "center" else read_gray(path)
        regions = {name: measure(reference, current, roi, 110)
                   for name, roi in REGIONS.items()}
        row = {"file": str(path.relative_to(root)), "axis": axis, "label": position,
               "sha256": digest, "regions": regions}
        rows.append(row)
        if (index+1) % 25 == 0 or index+1 == len(files):
            print(f"Measured {index+1}/{len(files)}", flush=True)
    result = {
        "data_root": str(root), "opencv_version": cv2.__version__,
        "image_shape": list(reference.shape), "camera_qr_distance_mm": 385,
        "regions_xyxy": REGIONS,
        "method": "Integer ZNCC (TM_CCOEFF_NORMED) initialization; quadratic ZNCC peak interpolation and ECC translation/affine refinement. Legacy ncc keys contain ZNCC values. Methods share images and initialization; they are not statistically independent.",
        "limitations": [
            "Single image per labeled coordinate; one shared center image; no repeatability estimate.",
            "No acquisition sequence, approach direction, encoder readback, or camera calibration supplied.",
            "Labels are treated as nominal commanded coordinates, not verified ground truth.",
            "Background regions are visual control points; their physical rigidity is unverified.",
        ],
        "measurements": rows,
    }
    (args.output / "measurements.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(rows)} measurements to {args.output}", flush=True)


if __name__ == "__main__":
    main()
