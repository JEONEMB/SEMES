"""Investigate acquisition-group hypotheses using file modification times.

File timestamps are not verified exposure times or controller motion logs.
Group offsets are image-space fit parameters, never automatic corrections.
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
KST = timezone(timedelta(hours=9))
MARKERS = ["L", "C", "R"]


def mean_dx(row, method="affine"):
    return float(np.mean([row["regions"][m]["ncc_dx"] if method == "ncc"
                          else row["regions"][m][method]["dx"] for m in MARKERS]))


def main():
    measurements = json.loads((ROOT / "measurements.json").read_text(encoding="utf-8"))
    rows = measurements["measurements"]
    data_root = Path(measurements["data_root"])
    metadata = []
    for row in rows:
        modified = (data_root / row["file"]).stat().st_mtime
        row["modified_epoch"] = modified
        row["modified_kst"] = datetime.fromtimestamp(modified, KST).isoformat()
        metadata.append({key: row[key] for key in ["file", "axis", "label", "modified_epoch", "modified_kst"]})
    center = next(r for r in rows if r["axis"] == "center")
    yrows = sorted([r for r in rows if r["axis"] == "Y"], key=lambda r: r["label"])
    theta = sorted([r for r in rows if r["axis"] == "Theta"], key=lambda r: r["label"])
    theta_scale = np.polyfit([r["label"] for r in theta], [mean_dx(r) for r in theta], 1)[0]
    result = {
        "timestamp_provenance": "Filesystem last-modified time in Asia/Seoul (UTC+09:00); NOT verified camera capture time.",
        "center_modified_kst": center["modified_kst"],
        "metadata": metadata, "Y_groups": {}, "Y_offset_models": {}, "Theta_zero_gap_sensitivity": {},
        "interpretation_limits": [
            "Even/odd Y labels were grouped after observing timestamp clusters; grouping is descriptive.",
            "Monotonic label order by file mtime does not prove stage approach direction or absence of intermediate resets.",
            "No repeated zero images or encoder logs: return-to-zero repeatability and backlash cannot be measured directly.",
            "Command-equivalent units use fitted image/label slopes, not calibrated true physical displacement.",
            "Y/Theta contributions to a baseline shift cannot be uniquely separated using horizontal displacement alone.",
            "Residual reduction after grouping is an in-sample descriptive fit, not validated compensation performance.",
        ],
        "method_sources": {
            "ECC": "https://docs.opencv.org/4.x/dc/d6b/group__video__track.html",
            "motion_terms": "https://www.newport.com/n/linear-translation-stage-terminology/resources",
        },
    }
    groups = {}
    for sign in [-1, 1]:
        side = [r for r in yrows if r["label"] * sign > 0]
        for parity in [0, 1]:
            key = f"{'negative' if sign < 0 else 'positive'}_{'0.10_grid' if parity == 0 else '0.05_offset_grid'}"
            selected = [r for r in side if round(abs(r["label"]) / 0.05) % 2 == parity]
            ordered = sorted(selected, key=lambda r: r["modified_epoch"])
            groups[key] = selected
            x = np.array([r["label"] for r in selected])
            y = np.array([mean_dx(r) for r in selected])
            slope, intercept = np.polyfit(x, y, 1)
            residual = y - (slope*x + intercept)
            result["Y_groups"][key] = {
                "count": len(selected), "first_modified_kst": ordered[0]["modified_kst"],
                "last_modified_kst": ordered[-1]["modified_kst"],
                "labels_in_modification_order": [r["label"] for r in ordered],
                "slope_px_per_nominal_mm": float(slope), "intercept_px": float(intercept),
                "within_group_rms_px": float(np.sqrt(np.mean(residual**2))),
            }
        x = np.array([r["label"] for r in side])
        parity = (-1.) ** np.rint(np.abs(x) / 0.05)
        design = np.column_stack([np.ones(len(x)), x, parity])
        model = {}
        for method in ["affine", "translation", "ncc"]:
            y = np.array([mean_dx(r, method) for r in side])
            coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
            residual = y - design @ coefficients
            model[method] = {
                "offset_px": float(coefficients[0]), "slope_px_per_nominal_mm": float(coefficients[1]),
                "even_minus_odd_offset_px": float(2*coefficients[2]),
                "even_minus_odd_command_equivalent_mm": float(2*coefficients[2]/coefficients[1]),
                "within_group_rms_px": float(np.sqrt(np.mean(residual**2))),
            }
        y = np.array([mean_dx(r) for r in side])
        step = np.diff(y)
        destination_parity = np.rint(np.abs(x[1:]) / 0.05).astype(int) % 2
        model["adjacent_label_step_mean_px_by_destination_parity"] = {
            str(p): float(step[destination_parity == p].mean()) for p in [0, 1]
        }
        result["Y_offset_models"][str(sign)] = model
    for limit in [0.05, 0.1, 0.25, 0.5, 1.0]:
        fits = {}
        for sign in [-1, 1]:
            selected = [r for r in theta if r["label"]*sign > 0 and abs(r["label"]) <= limit]
            x = np.array([r["label"] for r in selected])
            y = np.array([mean_dx(r) for r in selected])
            slope, intercept = np.polyfit(x, y, 1)
            fits[str(sign)] = {"slope_px_per_nominal_degree": float(slope), "intercept_px": float(intercept)}
        gap = fits["1"]["intercept_px"] - fits["-1"]["intercept_px"]
        fits["positive_minus_negative_zero_gap_px"] = gap
        fits["gap_command_equivalent_degree"] = gap / float(theta_scale)
        result["Theta_zero_gap_sensitivity"][str(limit)] = fits

    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    colors = ["#2373b9", "#e17024", "#23915b", "#8e54b8"]
    labels = ["Y 음수 · 0.10mm 격자", "Y 음수 · 0.05mm 오프셋 격자",
              "Y 양수 · 0.10mm 격자", "Y 양수 · 0.05mm 오프셋 격자"]
    all_x = np.array([r["label"] for r in yrows])
    all_y = np.array([mean_dx(r) for r in yrows])
    global_fit = np.polyfit(all_x, all_y, 1)
    for (key, selected), color, label in zip(groups.items(), colors, labels):
        x = np.array([r["label"] for r in selected])
        y = np.array([mean_dx(r) for r in selected])
        residual = y - np.polyval(global_fit, x)
        axes[0, 0].plot(x, residual, ".-", color=color, lw=.8, label=label)
        ordered = sorted(selected, key=lambda r: r["modified_epoch"])
        minutes = [(r["modified_epoch"]-center["modified_epoch"])/60 for r in ordered]
        axes[0, 1].plot(minutes, [r["label"] for r in ordered], ".-", color=color, lw=.8, label=label)
        group_fit = result["Y_groups"][key]
        grouped_residual = y - (group_fit["slope_px_per_nominal_mm"]*x + group_fit["intercept_px"])
        axes[1, 0].plot(x, grouped_residual, ".-", color=color, lw=.8, label=label)
    axes[0, 0].set(title="Y 전체 추세에서의 편차: 두 시각 묶음이 교대로 배치됨", xlabel="파일명 Y 좌표 (mm)", ylabel="수평 이동량의 선형 추세 잔차 (px)")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set(title="파일 수정시각 순서: 음수 묶음은 좌표 진행 순서가 반대", xlabel="center.bmp 수정시각 이후 경과 시간 (분)", ylabel="파일명 Y 좌표 (mm)")
    axes[1, 0].set(title="묶음별 직선 적합 후 잔차 (보정 성능 평가가 아님)", xlabel="파일명 Y 좌표 (mm)", ylabel="묶음별 추세 잔차 (px)")
    tx = np.array([r["label"] for r in theta])
    ty = np.array([mean_dx(r) for r in theta])
    tf = np.polyfit(tx, ty, 1)
    axes[1, 1].plot(tx, ty-np.polyval(tf,tx), ".", color="#2373b9", ms=4)
    axes[1, 1].scatter([0], [-tf[1]], marker="*", s=100, color="#c63b49", label="공유 center 기준 이미지")
    axes[1, 1].set(title="θ 양·음수 구간의 기준 위치 차이도 관찰됨", xlabel="파일명 θ 좌표 (°)", ylabel="전체 선형 추세 잔차 (px)")
    axes[1, 1].legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=.2)
        ax.axhline(0, lw=.6, color="gray")
    fig.suptitle("초기 데이터 오차 분석 — 영점 복귀 및 취득 묶음 간 편차 가설\n파일 수정시각은 촬영시각·실제 접근 방향을 보증하지 않음", fontsize=14)
    fig.savefig(ROOT / "batch_diagnostics.png", dpi=160)
    plt.close(fig)
    (ROOT / "batch_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"Y_groups": result["Y_groups"], "Y_offset_models": result["Y_offset_models"],
                      "Theta_zero_gap_sensitivity": result["Theta_zero_gap_sensitivity"]}, indent=2))


if __name__ == "__main__":
    main()
