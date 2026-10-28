"""Recover nominal Y=0 image-x candidates from the saved Y slope fits.

No raw images are registered and no new fit is made here.
"""

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def main():
    fit_path = HERE / "y_slope_results.json"
    measurement_path = ROOT / "analysis/initial_error/measurements.json"
    fits = json.loads(fit_path.read_text(encoding="utf-8"))
    measurements = json.loads(measurement_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(measurement_path.read_bytes()).hexdigest()
    if digest != fits["sources"]["measurements"]["sha256"]:
        raise ValueError("Measurements have changed since the slope fits were saved.")
    indexed = {row["file"]: row for row in measurements["measurements"]}
    candidates = []
    for group in fits["groups"]:
        first = indexed[group["first_file"]]
        markers = {}
        for marker in ("L", "C", "R"):
            roi = measurements["regions_xyxy"][marker]
            reference_x = (roi[0] + roi[2] - 1) / 2
            first_dx = first["regions"][marker]["affine"]["dx"]
            fit = group["markers"][marker]
            slope = fit["slope_px_per_nominal_mm"]
            relative_intercept = fit["nuisance_intercept_in_relative_coordinates_px"]
            intercept = (
                reference_x + first_dx + relative_intercept
                - slope * group["label_start_mm"]
            )
            # Algebraic check against every saved relative-coordinate prediction.
            for label, relative_y in zip(
                group["labels_in_time_order_mm"], group["delta_y_from_first_mm"]
            ):
                absolute_prediction = slope * label + intercept
                relative_prediction = (
                    reference_x + first_dx + slope * relative_y + relative_intercept
                )
                if abs(absolute_prediction - relative_prediction) > 1e-9:
                    raise ValueError("Coordinate conversion failed.")
            markers[marker] = {
                "slope_px_per_nominal_mm": slope,
                "x_at_nominal_y_zero_px": intercept,
                "offset_from_center_bmp_roi_x_px": intercept - reference_x,
                "rms_fit_residual_px": fit["rms_fit_residual_px"],
                "max_abs_fit_residual_px": fit["max_abs_fit_residual_px"],
            }
        candidates.append({
            "capture_group_number": group["capture_group_number"],
            "image_count": group["image_count"],
            "label_start_mm": group["label_start_mm"],
            "label_end_mm": group["label_end_mm"],
            "markers": markers,
        })

    result = {
        "purpose": "Nominal Y=0 horizontal QR tracking-region position candidates",
        "sources": {
            "slope_results": {"path": str(fit_path), "sha256": hashlib.sha256(fit_path.read_bytes()).hexdigest()},
            "measurements": {"path": str(measurement_path), "sha256": digest},
        },
        "method": {
            "saved_fit": "u = a * (Y - Y_first) + c; u = dx - dx_first",
            "absolute_image_fit": "x = a * Y + b",
            "zero_formula": "b = reference_roi_center_x + dx_first + c - a * Y_first",
            "coordinate": "Zero-based image x, positive to the right, pixels",
            "metadata_role": "Existing acquisition groups/order only, not position or probability evidence",
        },
        "limitations": [
            "Y=0 is the filename-label origin of each run, not a verified physical alignment origin.",
            "These are tracked QR ROI-center x coordinates, not independently detected geometric QR centers.",
            "No common target is selected by averaging runs with potentially different origins.",
            "No theta calibration, vertical-coordinate fit, confidence interval, or acceptance test is performed.",
        ],
        "candidates": candidates,
    }
    (HERE / "y_zero_candidates.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "기존 Y 기울기로 복원한 라벨 Y=0의 영상 위치 후보",
        "",
        "입력: 각 촬영 구간 20장으로 이미 적합한 직선과 기존 ZNCC → ECC 아핀 측정값.",
        "이번 작업에서는 원본 영상 재정합이나 새로운 직선 적합을 하지 않았습니다.",
        "",
        "영상 가로 좌표 x = a × 파일명 Y 라벨 + b; Y=0에서 x=b.",
        "기존 상대좌표 직선 u=a×(Y-Y_first)+c를 복원하면",
        "b = center.bmp ROI 중심 x + 첫 사진의 dx + c - a×첫 사진의 Y 라벨.",
        "메타데이터는 촬영 구간/순서 구분에만 사용합니다.",
        "",
        "구간 | Y 촬영 범위(mm) | 좌 QR x(px) | 중앙 QR x(px) | 우 QR x(px)",
    ]
    for group in candidates:
        values = [group["markers"][key]["x_at_nominal_y_zero_px"] for key in ("L", "C", "R")]
        lines.append(
            f'{group["capture_group_number"]} | {group["label_start_mm"]:+.2f} → '
            f'{group["label_end_mm"]:+.2f} | ' + " | ".join(f"{v:.6f}" for v in values)
        )
    lines += [
        "",
        "좌표는 이미지 왼쪽 x=0 기준의 가로 좌표이며, 추적한 QR 영역 중심의 위치입니다.",
        "정확한 QR 도형 중심을 새로 검출한 값은 아닙니다.",
        "촬영 구간별 라벨 Y=0의 후보이므로 실제 기계 정렬점이라고 확정할 수 없습니다.",
        "서로 다른 촬영 구간의 영점을 하나로 평균내지는 않았습니다. θ 기준점은 이번 범위에 포함하지 않았습니다.",
        "",
        "20개 측정점과 적합 직선의 차이 (px)",
        "구간 | QR | RMS | 최대 절댓값",
    ]
    for group in candidates:
        for key, marker in group["markers"].items():
            lines.append(
                f'{group["capture_group_number"]} | {key} | '
                f'{marker["rms_fit_residual_px"]:.6f} | {marker["max_abs_fit_residual_px"]:.6f}'
            )
    lines += ["", "잔차는 이 측정값이 직선에서 벗어난 정도이며, 실제 기계 위치 오차나 정렬 합격 판정값이 아닙니다."]
    (HERE / "y_zero_candidates.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Saved y_zero_candidates.json and y_zero_candidates.txt; checked 240 coordinate conversions.")


if __name__ == "__main__":
    main()
