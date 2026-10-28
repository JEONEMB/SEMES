"""Compare affine and homography registration on representative Theta images.

This checks sensitivity to the warp model, not true localization accuracy.
Both estimates share pixels, initialization, and the reference image.
Original measurements are preserved; output is a separate JSON file.
"""

import json
from pathlib import Path

import cv2
import numpy as np

from analyze_images import REGIONS, read_gray


def main():
    root = Path(__file__).resolve().parent
    data = json.loads((root / "measurements.json").read_text(encoding="utf-8"))
    source = Path(data["data_root"])
    reference, _ = read_gray(source / "center.bmp")
    labels = {-1.0, -0.5, -0.1, -0.01, 0.01, 0.1, 0.5, 1.0}
    samples = [r for r in data["measurements"] if r["axis"] == "Theta" and r["label"] in labels]
    output = []
    cv2.setNumThreads(2)
    for row in samples:
        current, _ = read_gray(source / row["file"])
        for name in ["L", "C", "R"]:
            x0, y0, x1, y1 = REGIONS[name]
            template = reference[y0:y1, x0:x1]
            search = current[y0-110:y1+110, x0-110:x1+110]
            score = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
            _, _, _, (mx, my) = cv2.minMaxLoc(score)
            dx0, dy0 = mx-110, my-110
            patch = current[y0+dy0:y1+dy0, x0+dx0:x1+dx0]
            initial = np.eye(3, dtype=np.float32)
            initial[:2] = np.array(row["regions"][name]["affine"]["warp"], dtype=np.float32)
            record = {"file": row["file"], "theta_label_degree": row["label"], "region": name,
                      "zncc_initial_score": row["regions"][name]["ncc"],
                      "affine_dx_px": row["regions"][name]["affine"]["dx"],
                      "affine_dy_px": row["regions"][name]["affine"]["dy"]}
            try:
                correlation, warp = cv2.findTransformECC(
                    template, patch, initial, cv2.MOTION_HOMOGRAPHY,
                    (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 150, 1e-7), None, 5)
                center = np.array([(x1-x0-1)/2, (y1-y0-1)/2, 1.0])
                projected = warp @ center
                delta = np.array([dx0, dy0]) + projected[:2]/projected[2] - center[:2]
                record.update(homography_dx_px=float(delta[0]), homography_dy_px=float(delta[1]),
                              homography_minus_affine_dx_px=float(delta[0]-record["affine_dx_px"]),
                              homography_ecc=float(correlation), homography=warp.tolist())
            except cv2.error as error:
                record["error"] = str(error)
            output.append(record)
    valid = [r for r in output if "error" not in r]
    differences = np.array([r["homography_minus_affine_dx_px"] for r in valid])
    frame_mean = {}
    for label in sorted(labels):
        group = [r for r in valid if r["theta_label_degree"] == label]
        if len(group) == 3:
            frame_mean[str(label)] = float(np.mean([r["homography_minus_affine_dx_px"] for r in group]))
    summary = {"sample_images": len(samples), "measured_regions": len(output),
               "failed_regions": len(output)-len(valid),
               "per_region_dx_difference_rms_px": float(np.sqrt(np.mean(differences**2))) if len(valid) else None,
               "per_region_dx_difference_max_abs_px": float(np.max(np.abs(differences))) if len(valid) else None,
               "frame_mean_dx_difference_px": frame_mean,
               "limitations": ["No externally known pixel correspondences: agreement does not establish accuracy.",
                               "Only 8 Theta images, 24 local QR patches were checked.",
                               "Homography adds flexibility and can fit noise; it is not automatically the correct model.",
                               "No lens-distortion calibration, large-angle search, or illumination stress test included."]}
    (root / "perspective_sensitivity.json").write_text(
        json.dumps({"summary": summary, "measurements": output}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
