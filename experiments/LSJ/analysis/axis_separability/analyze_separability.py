"""Stage 3: observed minimum-step separation and joint Y/theta feasibility.

Uses stage-2 measurements without new registrations or acceptance thresholds.
The linear inverse is a diagnostic baseline, not a deployed motion controller.
"""

from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "increment_consistency" / "second_measurements.json"
MARKERS = ("L", "C", "R")
MIN_STEPS = np.array([.05, .01])


def features(row, rois, kind, method="affine"):
    values = []
    for marker in MARKERS:
        result = row["regions"][marker][method]
        values.extend([result["dx"], result["dy"]])
        if kind == "centers_and_affine_geometry_18":
            # Convert each linear affine component to a displacement in pixels
            # over the ROI width/height. These are fitted patch geometry values,
            # NOT independently detected physical QR corners.
            linear_delta = np.asarray(result["warp"])[:, :2]-np.eye(2)
            x0, y0, x1, y1 = rois[marker]
            values.extend((linear_delta*np.array([x1-x0-1, y1-y0-1])).ravel())
    return np.asarray(values)


def mean_x(row, method):
    return float(np.mean([row["regions"][m][method]["dx"] for m in MARKERS]))


def observed_minimum_steps(grouped):
    cases = [("Y", theta, .05, .1) for theta in (.01, .1, 1., 2.)]
    cases += [("Theta", .05, .01, .02), ("Theta", .05, .02, .03)]
    result = []
    for axis, fixed, lo, hi in cases:
        start = (lo, fixed) if axis == "Y" else (fixed, lo)
        end = (hi, fixed) if axis == "Y" else (fixed, hi)
        row = {"axis": axis, "fixed_other_label": fixed, "start_label": lo, "end_label": hi,
               "label_increment": round(hi-lo, 6),
               "start_files": [r["file"] for r in grouped[start]],
               "end_files": [r["file"] for r in grouped[end]], "methods": {}}
        for method in ("affine", "translation"):
            a = np.array([mean_x(r, method) for r in grouped[start]])
            b = np.array([mean_x(r, method) for r in grouped[end]])
            separation = float(b.min()-a.max())
            pairwise = (b[None, :]-a[:, None]).ravel()
            row["methods"][method] = {
                "start_positions_x_px": a.tolist(), "end_positions_x_px": b.tolist(),
                "mean_change_x_px": float(b.mean()-a.mean()),
                "signed_gap_between_observed_ranges_px": separation,
                "all_observed_end_positions_greater_than_start": bool(separation > 0),
                "count_of_9_combinations_with_positive_dx": int((pairwise > 0).sum()),
                "start_observed_range_width_px": float(np.ptp(a)),
                "end_observed_range_width_px": float(np.ptp(b)),
                "range_overlap_width_px": float(max(0, min(a.max(), b.max())-max(a.min(), b.min()))),
                "note": "Gap zero only denotes touching observed intervals. This is not a selected engineering pass threshold, a population confidence limit, or a classification score.",
            }
        result.append(row)
    return result


def error_metrics(errors):
    errors = np.asarray(errors)
    return {"sample_count": len(errors), "mean_absolute_y_mm_theta_degree": np.abs(errors).mean(axis=0).tolist(),
            "rmse_y_mm_theta_degree": np.sqrt((errors**2).mean(axis=0)).tolist(),
            "max_absolute_y_mm_theta_degree": np.abs(errors).max(axis=0).tolist()}


def linear_diagnostic(grouped, rois, kind, method):
    keys = sorted(grouped)
    labels = np.array(keys)
    data = np.array([[features(r, rois, kind, method) for r in grouped[key]] for key in keys])
    means = data.mean(axis=1)
    design = np.column_stack([np.ones(len(keys)), labels])
    coefficients = np.linalg.lstsq(design, means, rcond=None)[0]
    jacobian = coefficients[1:].T
    scaled = jacobian*MIN_STEPS
    cosine = np.dot(scaled[:, 0], scaled[:, 1])/np.prod(np.linalg.norm(scaled, axis=0))
    angle = float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))
    predictions, errors, mean_errors, folds = [], [], [], []
    for held_out, key in enumerate(keys):
        keep = np.arange(len(keys)) != held_out
        trained = np.linalg.lstsq(design[keep], means[keep], rcond=None)[0]
        estimated = np.linalg.lstsq(trained[1:].T, (data[held_out]-trained[0]).T, rcond=None)[0].T
        errors.extend(estimated-labels[held_out])
        mean_errors.append(estimated.mean(axis=0)-labels[held_out])
        folds.append({"held_out_condition": list(key), "training_condition_count": int(keep.sum()),
                      "intercept": trained[0].tolist(), "sensitivity_columns_y_theta": trained[1:].T.tolist()})
        for row, predicted in zip(grouped[key], estimated):
            predictions.append({"file": row["file"], "condition": row["condition"],
                "label_y_mm_theta_degree": list(key), "prediction_y_mm_theta_degree": predicted.tolist(),
                "error_vs_label_y_mm_theta_degree": (predicted-labels[held_out]).tolist()})
    residual = means-design@coefficients
    return {"feature_kind": kind, "registration_method": method, "feature_count": data.shape[-1],
        "fit": "Unweighted least squares: feature vector = intercept + J_y*label_Y + J_theta*label_theta. Infer labels with least squares using the fitted sensitivity matrix.",
        "feature_units": "pixels; affine geometry components multiplied by ROI width or height in pixels",
        "global_descriptive_intercept": coefficients[0].tolist(),
        "global_descriptive_sensitivity_columns_y_theta": jacobian.tolist(),
        "response_vector_angle_degree": angle,
        "sensitivity_condition_number_in_minimum_step_units": float(np.linalg.cond(scaled)),
        "condition_number_scaling": "Y 0.05 mm and theta 0.01 degree: user-specified minimum increments",
        "global_descriptive_feature_fit_residual_rms_px": float(np.sqrt(np.mean(residual**2))),
        "validation": "Leave one complete label condition (all three images) out. Train on means of the other 18 conditions, infer the held-out three images. Reference images excluded.",
        "validation_individual_images": error_metrics(errors),
        "validation_condition_means": error_metrics(mean_errors),
        "predictions": predictions, "fold_models": folds}


def plot(result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
    fig = plt.figure(figsize=(14, 11))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.1])
    ax = fig.add_subplot(grid[0, :])
    labels = []
    for i, row in enumerate(result["minimum_steps"]):
        m = row["methods"]["affine"]
        a, b = np.array(m["start_positions_x_px"]), np.array(m["end_positions_x_px"])
        origin = a.mean()
        for values, offset, color, label in ((a, -.12, "#2477ab", "변경 전 3장"), (b, .12, "#d5742b", "변경 후 3장")):
            ax.plot([values.min()-origin, values.max()-origin], [i+offset]*2, color=color, lw=2)
            ax.scatter(values-origin, [i+offset]*3, color=color, s=30, label=label if i == 0 else None)
        axis = "Y" if row["axis"] == "Y" else "θ"
        other = "θ" if row["axis"] == "Y" else "Y"
        unit = "°" if row["axis"] == "Y" else "mm"
        labels.append(f"{axis} {row['start_label']:g}→{row['end_label']:g} / {other}={row['fixed_other_label']:g}{unit}")
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set(title="최소 이동 전후의 관측 위치 범위 · QR 3개 평균 Δx", xlabel="각 구간의 변경 전 3장 평균을 0으로 놓은 위치 (px)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.grid(alpha=.2)
    for coordinate, name, unit in ((0, "Y", "mm"), (1, "θ", "°")):
        ax = fig.add_subplot(grid[1, coordinate])
        all_values = []
        for key, label, color in (("centers_affine", "QR 위치 6개 성분", "#2477ab"),
                                  ("geometry_affine", "위치 + 아핀 형상 18개 성분", "#d5742b")):
            pred = result["linear_diagnostics"][key]["predictions"]
            x = [p["label_y_mm_theta_degree"][coordinate] for p in pred]
            y = [p["prediction_y_mm_theta_degree"][coordinate] for p in pred]
            ax.scatter(x, y, color=color, label=label, alpha=.6, s=23)
            all_values.extend(x+y)
        lo, hi = min(all_values), max(all_values)
        ax.plot([lo, hi], [lo, hi], "k--", lw=.9, label="예측 = 라벨")
        ax.set(title=f"{name} 라벨 재현 · 조건 1개씩 통째로 제외한 검증", xlabel=f"파일 라벨 ({unit})", ylabel=f"진단용 선형 모델 추정 ({unit})")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("3번 분석 · 최소 이동 관측 구분과 Y·θ 동시 추정 검토", fontsize=16)
    fig.text(.04, .025, "위: 3장씩의 실제 관측 범위이며, 미관측 촬영까지의 통과 기준이 아님. 아래: 19조건·57장의 라벨 재현 결과.\n"
             "진단용 선형 모델의 한계를 확인한 결과이며, 실제 기계 오차나 모든 영상 알고리즘의 성능을 뜻하지 않음.", fontsize=10)
    fig.subplots_adjust(left=.15, right=.84, top=.9, bottom=.12, hspace=.35, wspace=.3)
    fig.savefig(HERE / "separability_comparison.png", dpi=140)
    plt.close(fig)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    grouped = defaultdict(list)
    for row in source["measurements"]:
        if not row["is_reference"]:
            grouped[(row["label_y_mm"], row["label_theta_degree"])].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: row["file"])
    assert len(grouped) == 19 and all(len(rows) == 3 for rows in grouped.values())
    diagnostics = {
        "centers_affine": linear_diagnostic(grouped, source["regions_xyxy"], "centers_6", "affine"),
        "centers_translation": linear_diagnostic(grouped, source["regions_xyxy"], "centers_6", "translation"),
        "geometry_affine": linear_diagnostic(grouped, source["regions_xyxy"], "centers_and_affine_geometry_18", "affine"),
    }
    result = {
        "purpose": "Stage 3 exploratory check completed; not a physical accuracy certification or a production model.",
        "source": "../increment_consistency/second_measurements.json",
        "condition_count": 19, "image_count": 57,
        "minimum_steps": observed_minimum_steps(grouped), "linear_diagnostics": diagnostics,
        "interpretation_limits": [
            "Three observed images per label cannot establish future detection probability or a population noise limit.",
            "Minimum Y only tests 0.05 -> 0.10 mm at four theta settings; minimum theta only tests 0.01 -> 0.02 -> 0.03 degree at Y=0.05 mm.",
            "The same-label spread includes acquisition variation and image-estimation effects; its physical origin is not identified.",
            "Nominal labels, not independently measured physical camera motion, are the validation target.",
            "A scalar horizontal shift alone cannot identify two simultaneously unknown axes under this continuous linear model.",
            "Response angles and condition numbers depend on the feature representation; no arbitrary accept/reject boundary is assigned.",
            "QR-center and affine-geometry baselines were explored here; condition-holdout results do not constitute an untouched external test set.",
            "Affine geometry is taken from fitted warp parameters, not independently detected physical QR corners.",
            "No feature noise whitening, nonlinear calibration, camera intrinsic calibration, or production-model optimization was performed.",
            "No augmentation, clipping to known label grids, pass/fail tolerance, or physical cause attribution was applied.",
        ]}
    (HERE / "separability_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    plot(result)
    concise = {"minimum_steps": [{k: row[k] for k in ("axis", "fixed_other_label", "start_label", "end_label")} |
               {"mean_dx_px": row["methods"]["affine"]["mean_change_x_px"],
                "observed_range_gap_px": row["methods"]["affine"]["signed_gap_between_observed_ranges_px"]}
               for row in result["minimum_steps"]],
               "diagnostics": {key: {k: value[k] for k in ("response_vector_angle_degree", "validation_individual_images", "validation_condition_means")}
                               for key, value in diagnostics.items()}}
    print(json.dumps(concise, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
