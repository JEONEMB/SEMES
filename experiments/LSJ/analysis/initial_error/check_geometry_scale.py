"""Compare nominal optics, static plate geometry, and labeled Y motion.

Uses center.bmp only for the static check: no template matching, motion labels,
or ECC are used to locate the three bright square patches. The physical scale
is conditional on the stated 13 mm patch width and 5 mm edge-to-edge gap.
This is a scale consistency check, not an absolute accuracy calibration.
"""

import hashlib
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    out = Path(__file__).resolve().parent
    measurements = json.loads((out / "measurements.json").read_text(encoding="utf-8"))
    raw = np.fromfile(Path(measurements["data_root"]) / "center.bmp", dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
    if image is None or image.shape != (2160, 3840):
        raise ValueError("Expected the original 3840 x 2160 center.bmp")
    crop_origin = np.array([1520, 940], dtype=float)
    crop = image[940:1190, 1520:2280]
    scans = []
    for threshold in (110, 130, 150, 170):
        binary = (crop > threshold).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        patches = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Broad size filter selects the three white patches in this crop.
            if not (130 < w < 220 and 130 < h < 230):
                continue
            rect = cv2.minAreaRect(contour)
            corners = cv2.boxPoints(rect).astype(float) + crop_origin
            patches.append({
                "center_xy_px": (np.array(rect[0]) + crop_origin).tolist(),
                "corners_xy_px": corners.tolist(),
                "horizontal_span_px": float(np.ptp(corners[:, 0])),
            })
        patches.sort(key=lambda patch: patch["center_xy_px"][0])
        if len(patches) != 3:
            raise ValueError(f"Threshold {threshold}: expected 3 patches, got {len(patches)}")
        gaps = np.diff([patch["center_xy_px"][0] for patch in patches])
        scans.append({
            "threshold": threshold,
            "patches": patches,
            "adjacent_center_dx_px": gaps.tolist(),
            "mean_pitch_px": float(gaps.mean()),
            "conditional_scale_px_per_mm": float(gaps.mean() / 18),
        })

    selected = next(scan for scan in scans if scan["threshold"] == 130)
    focal_px = 12 / 0.0028 * (3840 / 4096)
    theoretical = focal_px / 385
    y_rows = [row for row in measurements["measurements"] if row["axis"] == "Y"]
    labels = np.array([row["label"] for row in y_rows])
    design = np.column_stack([np.ones_like(labels), labels])
    fits = {}
    for method in ("ncc", "translation", "affine"):
        displacement = np.array([
            np.mean([row["regions"][name]["ncc_dx"] if method == "ncc"
                     else row["regions"][name][method]["dx"] for name in ("L", "C", "R")])
            for row in y_rows
        ])
        coefficients = np.linalg.lstsq(design, displacement, rcond=None)[0]
        fits["zncc_quadratic" if method == "ncc" else method] = {
            "intercept_px": float(coefficients[0]),
            "slope_px_per_nominal_mm": float(coefficients[1]),
        }

    result = {
        "reference": "center.bmp",
        "reference_sha256": hashlib.sha256(raw).hexdigest(),
        "method": "Threshold bright patches; external contours; minimum-area rectangle centers; horizontal center spacing. No inter-frame registration used for static geometry.",
        "settings_used": {"distance_mm": 385, "focal_length_mm": 12,
                          "assumed_sensor_pitch_um": 2.8, "assumed_horizontal_resize": 3840/4096,
                          "assumed_patch_width_mm": 13, "assumed_edge_gap_mm": 5},
        "nominal_pinhole_px_per_mm": theoretical,
        "nominal_pinhole_expected_pitch_px": theoretical * 18,
        "static_geometry_px_per_mm": selected["conditional_scale_px_per_mm"],
        "threshold_sensitivity": scans,
        "all_y_global_linear_fits_excluding_center": fits,
        "limitations": [
            "13 mm must refer to the detected outer bright patches, with a 5 mm edge gap; no independent physical measurement supplied.",
            "Static horizontal scale assumes approximately frontal plate orientation and comparable plate-horizontal and robot-Y directions.",
            "Threshold variation is a robustness check, not a confidence interval or a subpixel accuracy guarantee.",
            "Pinhole estimate assumes 2.8 um is the actual sensor pitch, horizontal resampling rather than cropping, and 385 mm as optical depth.",
            "Agreement of the static scale and motion slope does not verify positioning accuracy, zero repeatability, or simultaneous Y/theta separation.",
        ],
    }
    (out / "geometry_scale_check.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [2.1, 1]})
    axes[0].imshow(crop, cmap="gray", vmin=0, vmax=255,
                   extent=(1520, 2280, 1190, 940))
    for patch in selected["patches"]:
        corners = np.array(patch["corners_xy_px"])
        outline = np.vstack([corners, corners[0]])
        axes[0].plot(outline[:, 0], outline[:, 1], color="cyan", lw=1)
        axes[0].plot(*patch["center_xy_px"], "+", color="yellow", ms=10)
    for index, pitch in enumerate(selected["adjacent_center_dx_px"]):
        left = selected["patches"][index]["center_xy_px"][0]
        right = selected["patches"][index+1]["center_xy_px"][0]
        axes[0].annotate("", (left, 952), (right, 952),
                         arrowprops={"arrowstyle": "<->", "color": "yellow"})
        axes[0].text((left+right)/2, 957, f"{pitch:.1f} px", ha="center", va="top",
                     color="yellow", fontsize=10,
                     bbox={"facecolor": "black", "alpha": 0.7, "edgecolor": "none", "pad": 2})
    axes[0].set_title("Static reference image: spacing of the three bright QR patches")
    axes[0].set_xlabel("Original image x (px)")
    values = [theoretical, result["static_geometry_px_per_mm"], fits["affine"]["slope_px_per_nominal_mm"]]
    names = ["Nominal optics model", "Static pitch / assumed 18 mm", "Labeled Y motion + affine"]
    axes[1].barh(names, values, color=["#94a3b8", "#0d9488", "#2563eb"])
    for index, value in enumerate(values):
        axes[1].text(value+0.12, index, f"{value:.2f}", va="center")
    axes[1].set_xlim(0, 16)
    axes[1].set_xlabel("Horizontal scale (px/mm); motion uses nominal labeled mm")
    axes[1].invert_yaxis()
    fig.text(0.5, 0.025, "Static estimate assumes 13 mm outer patch width + 5 mm gap and near-frontal geometry.\nScale consistency only; this does not establish absolute measurement accuracy.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out / "geometry_scale_check.png", dpi=150)
    plt.close(fig)
    print(json.dumps({key: result[key] for key in (
        "nominal_pinhole_px_per_mm", "nominal_pinhole_expected_pitch_px",
        "static_geometry_px_per_mm", "all_y_global_linear_fits_excluding_center")}, indent=2))
    print("Threshold scale range:", min(scan["conditional_scale_px_per_mm"] for scan in scans),
          max(scan["conditional_scale_px_per_mm"] for scan in scans))


if __name__ == "__main__":
    main()
