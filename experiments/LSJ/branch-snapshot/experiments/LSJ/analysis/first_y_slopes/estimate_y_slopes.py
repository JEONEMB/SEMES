"""Estimate per-QR horizontal image/nominal-Y slopes in four first-data runs.

Reuses existing ZNCC-initialized affine ECC coordinates. Does not perform new
image registration, infer a true zero, fit theta, or establish acceptance limits.
"""

import hashlib
from itertools import groupby
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parent
MARKERS = ("L", "C", "R")
COLORS = ("#bf4d45", "#267da1", "#35965c")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fit_slope(q, u):
    centered_q = q - q.mean()
    centered_u = u - u.mean()
    slope = float(centered_q @ centered_u / (centered_q @ centered_q))
    intercept = float(u.mean() - slope * q.mean())
    predicted = slope * q + intercept
    residual = u - predicted
    adjacent = np.diff(u) / np.diff(q)
    # Independent numerical check against polynomial least squares.
    np.testing.assert_allclose(np.polyfit(q, u, 1), [slope, intercept], atol=1e-10, rtol=0)
    return {
        "slope_px_per_nominal_mm": slope,
        "nuisance_intercept_in_relative_coordinates_px": intercept,
        "rms_fit_residual_px": float(np.sqrt(np.mean(residual ** 2))),
        "max_abs_fit_residual_px": float(np.max(np.abs(residual))),
        "residuals_px": residual.tolist(),
        "adjacent_slopes_px_per_nominal_mm": adjacent.tolist(),
        "adjacent_slope_min": float(adjacent.min()),
        "adjacent_slope_max": float(adjacent.max()),
        "adjacent_slope_mean": float(adjacent.mean()),
        "adjacent_slope_sample_sd": float(adjacent.std(ddof=1)),
    }


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    paths = {
        "measurements": ANALYSIS / "initial_error/measurements.json",
        "capture_order": ANALYSIS / "initial_error/capture_order.json",
        "previous_batch_summary": ANALYSIS / "initial_error/batch_summary.json",
    }
    loaded = {key: json.loads(path.read_text(encoding="utf-8")) for key, path in paths.items()}
    measurements = loaded["measurements"]
    lookup = {row["file"]: row for row in measurements["measurements"]}
    ordered = sorted(loaded["capture_order"]["files_in_modification_order"],
                     key=lambda row: (row["modified_ns"], row["file"]))
    data_root = Path(measurements["data_root"])
    center = lookup["center.bmp"]
    if sha256(data_root / center["file"]) != center["sha256"]:
        raise ValueError("Reference image hash differs from saved measurements")
    groups = []
    verified_files = []
    mean_slope_check_errors = []
    for group_number, (name, meta_iter) in enumerate(groupby(ordered, lambda row: row["group"]), 1):
        meta_rows = list(meta_iter)
        rows = [lookup[meta["file"]] for meta in meta_rows]
        if rows[0]["axis"] != "Y":
            continue
        assert all(row["axis"] == "Y" for row in rows)
        labels = np.array([row["label"] for row in rows], dtype=float)
        np.testing.assert_allclose(np.abs(np.diff(labels)), 0.10, atol=1e-12, rtol=0)
        assert len(rows) == 20
        for row, meta in zip(rows, meta_rows):
            path = data_root / row["file"]
            if path.stat().st_mtime_ns != meta["modified_ns"]:
                raise ValueError("Original modification time changed: " + row["file"])
            if sha256(path) != row["sha256"]:
                raise ValueError("Original image content changed: " + row["file"])
            verified_files.append(row["file"])
        q = labels - labels[0]
        group = {
            "capture_group_number": group_number,
            "group_name": name,
            "image_count": len(rows),
            "adjacent_pair_count": len(rows) - 1,
            "first_file": rows[0]["file"],
            "last_file": rows[-1]["file"],
            "first_modified_kst": meta_rows[0]["modified_kst"],
            "last_modified_kst": meta_rows[-1]["modified_kst"],
            "label_start_mm": float(labels[0]),
            "label_end_mm": float(labels[-1]),
            "labels_in_time_order_mm": labels.tolist(),
            "delta_y_from_first_mm": q.tolist(),
            "markers": {},
            "samples": [],
            "adjacent_pairs": [],
        }
        marker_dx = []
        for marker in MARKERS:
            affine_rows = [row["regions"][marker]["affine"] for row in rows]
            assert all("error" not in result for result in affine_rows)
            raw_dx = np.array([result["dx"] for result in affine_rows], dtype=float)
            u = raw_dx - raw_dx[0]
            fit = fit_slope(q, u)
            original_fit = fit_slope(labels, raw_dx)
            # Reanchoring must not change an intercept-including fitted slope.
            np.testing.assert_allclose(fit["slope_px_per_nominal_mm"],
                                       original_fit["slope_px_per_nominal_mm"], atol=1e-11, rtol=0)
            fit["delta_x_from_first_px"] = u.tolist()
            fit["min_saved_affine_ecc_score"] = min(result["correlation"] for result in affine_rows)
            group["markers"][marker] = fit
            marker_dx.append(u)
        marker_dx = np.array(marker_dx)
        mean_dx = marker_dx.mean(axis=0)
        group["qr_mean"] = fit_slope(q, mean_dx)
        group["qr_mean"]["delta_x_from_first_px"] = mean_dx.tolist()
        np.testing.assert_allclose(group["qr_mean"]["slope_px_per_nominal_mm"],
                                   np.mean([group["markers"][m]["slope_px_per_nominal_mm"] for m in MARKERS]),
                                   atol=1e-11, rtol=0)
        previous_name = name.removeprefix("Y_")
        previous_slope = loaded["previous_batch_summary"]["Y_groups"][previous_name]["slope_px_per_nominal_mm"]
        check_error = abs(previous_slope - group["qr_mean"]["slope_px_per_nominal_mm"])
        mean_slope_check_errors.append(check_error)
        assert check_error < 1e-10
        for index, (row, meta) in enumerate(zip(rows, meta_rows)):
            group["samples"].append({
                "file": row["file"], "modification_order": meta["modification_order"],
                "modified_kst": meta["modified_kst"], "label_y_mm": float(labels[index]),
                "delta_y_from_first_mm": float(q[index]),
                "delta_x_from_first_px_by_qr": {m: float(marker_dx[j, index]) for j, m in enumerate(MARKERS)},
                "mean_delta_x_from_first_px": float(mean_dx[index]),
            })
        for index in range(len(rows) - 1):
            delta_q = float(labels[index + 1] - labels[index])
            group["adjacent_pairs"].append({
                "from_file": rows[index]["file"], "to_file": rows[index + 1]["file"],
                "from_label_y_mm": float(labels[index]), "to_label_y_mm": float(labels[index + 1]),
                "delta_y_mm": delta_q,
                "delta_x_px_by_qr": {m: float(marker_dx[j, index + 1] - marker_dx[j, index]) for j, m in enumerate(MARKERS)},
                "slope_px_per_nominal_mm_by_qr": {m: group["markers"][m]["adjacent_slopes_px_per_nominal_mm"][index] for m in MARKERS},
                "mean_qr_slope_px_per_nominal_mm": group["qr_mean"]["adjacent_slopes_px_per_nominal_mm"][index],
            })
        groups.append(group)
    assert len(groups) == 4 and len(verified_files) == len(set(verified_files)) == 80
    mean_slopes = np.array([group["qr_mean"]["slope_px_per_nominal_mm"] for group in groups])
    all_slopes = np.array([group["markers"][m]["slope_px_per_nominal_mm"] for group in groups for m in MARKERS])
    result = {
        "analysis_date_kst": "2026-10-27",
        "purpose": "First-data nominal-Y / horizontal QR image slope, per uninterrupted timestamp-defined run and per QR",
        "sources": {key: {"path": str(path), "sha256": sha256(path)} for key, path in paths.items()},
        "method": {
            "input_measurements": "Saved per-QR ROI-center horizontal displacement from center.bmp using ZNCC initialization and affine ECC",
            "reanchoring": "Within each run: q_i = label_i - label_first; u_i = saved_dx_i - saved_dx_first",
            "regression": "u_i = a*q_i + c + residual_i; unweighted least squares with freely fitted c; not forced through the first sample",
            "slope_formula": "a = sum((q-mean(q))*(u-mean(u))) / sum((q-mean(q))**2)",
            "adjacent_formula": "a_i = (saved_dx[i+1]-saved_dx[i]) / (label[i+1]-label[i])",
            "qr_mean": "Arithmetic mean of the three QR displacement measurements; per-QR results retained",
            "sign": "Positive image x is right; signed filename-Y differences retained; no physical axis direction inferred",
            "theta_condition": "Nominally fixed theta during each Y-only acquisition run; no independent pose verification",
            "time_role": "Group and order only; original filesystem mtime is not a verified exposure timestamp",
        },
        "verification": {
            "original_y_image_sha256_matches": len(verified_files),
            "original_y_mtime_matches": len(verified_files),
            "reference_center_sha256_matches": True,
            "slope_reanchoring_invariance_checked": True,
            "closed_form_vs_polyfit_checked": True,
            "max_difference_vs_previous_group_mean_slopes": max(mean_slope_check_errors),
        },
        "summary": {
            "run_count": 4, "image_count": 80, "qr_fit_count": 12, "adjacent_image_pair_count": 76,
            "group_mean_slope_min": float(mean_slopes.min()), "group_mean_slope_max": float(mean_slopes.max()),
            "group_mean_slope_range": float(np.ptp(mean_slopes)),
            "descriptive_mean_of_four_group_mean_slopes": float(mean_slopes.mean()),
            "group_mean_slope_range_percent_of_mean": float(100*np.ptp(mean_slopes)/mean_slopes.mean()),
            "all_twelve_qr_slope_min": float(all_slopes.min()), "all_twelve_qr_slope_max": float(all_slopes.max()),
            "per_qr_fit_rms_residual_min_px": min(group["markers"][m]["rms_fit_residual_px"] for group in groups for m in MARKERS),
            "per_qr_fit_rms_residual_max_px": max(group["markers"][m]["rms_fit_residual_px"] for group in groups for m in MARKERS),
        },
        "groups": groups,
        "limitations": [
            "No new raw-image registration: this analysis reuses existing measured positions and their biases.",
            "Adjacent changes are differences of center-referenced positions, not fresh direct pair registrations.",
            "Slopes refer to nominal filename millimetres, not independently measured true stage displacement.",
            "No common zero or pooled cross-run straight line is imposed. The descriptive slope mean is not a calibrated universal conversion.",
            "Adjacent pairs share images; their spread is descriptive, not independent repeated-motion precision or a confidence interval.",
            "RMS is an in-sample linear-fit residual, not physical positioning error or proof of meeting the alignment tolerances.",
            "Results concern the observed theta condition only; no theta-dependent model, theta slope, or mixed-axis decomposition was fitted.",
        ],
    }
    (HERE / "y_slope_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    write_report(result)
    plot_result(result)
    print(json.dumps({"summary": result["summary"], "verification": result["verification"],
                      "groups": [{"number": g["capture_group_number"], "range": [g["label_start_mm"], g["label_end_mm"]],
                                  "slopes": {m: g["markers"][m]["slope_px_per_nominal_mm"] for m in MARKERS},
                                  "mean_slope": g["qr_mean"]["slope_px_per_nominal_mm"],
                                  "mean_adjacent_min_max": [g["qr_mean"]["adjacent_slope_min"], g["qr_mean"]["adjacent_slope_max"]]}
                                 for g in groups]}, ensure_ascii=False, indent=2))


def write_report(result):
    lines = [
        "1차 데이터 Y 이동량당 영상 변화량 (기울기) — 2026-10-27", "",
        "범위: θ가 고정된 것으로 제공된 Y 연속 촬영 4구간, 각 20장, QR 3개별 직선 12개.",
        "기존 ZNCC 초기화 → ECC 아핀 정합의 QR ROI 중심 가로 좌표 측정값을 재사용했습니다.",
        "원본 Y 이미지 80장과 center.bmp의 SHA-256, Y 이미지 80장의 수정 시각이 기존 기록과 일치합니다.",
        "각 구간 첫 사진을 기준으로 Y 라벨과 영상 좌표를 차분하고, 자유 절편을 포함한 최소제곱 직선을 맞췄습니다.",
        "a = Σ[(Y-평균Y)(영상좌표-평균영상좌표)] / Σ[(Y-평균Y)^2]. 단위: px/파일명 mm.",
        "첫 사진이 (0,0)이 되더라도 직선을 원점에 강제로 통과시키지 않았습니다.",
        "인접 기울기 = 인접 영상 좌표 차이 / 인접 Y 라벨 차이. 네 구간 모두 인접 라벨 간격의 크기는 0.10mm입니다.",
        "공통 center.bmp 측정값에서 차분한 결과이며, 인접 원본 사진끼리 새로 정합한 결과는 아닙니다.",
        "좌표 x는 영상 오른쪽이 양수입니다. 로봇 보정 명령의 부호를 확정한 결과는 아닙니다.", "",
        "구간 | 촬영 순서 Y(mm) | 장수 | 좌 QR a | 중앙 QR a | 우 QR a | QR 평균 a (px/mm)",
    ]
    for g in result["groups"]:
        values = " | ".join(f'{g["markers"][m]["slope_px_per_nominal_mm"]:.6f}' for m in MARKERS)
        lines.append(f'{g["capture_group_number"]} | {g["label_start_mm"]:+.2f} → {g["label_end_mm"]:+.2f} | {g["image_count"]} | {values} | {g["qr_mean"]["slope_px_per_nominal_mm"]:.6f}')
    s = result["summary"]
    lines += ["", f'구간별 QR 평균 기울기 범위: {s["group_mean_slope_min"]:.6f} ~ {s["group_mean_slope_max"]:.6f} px/mm',
              f'위 4개 기울기의 단순 평균: {s["descriptive_mean_of_four_group_mean_slopes"]:.6f} px/mm (최종 공통 환산계수로 채택한 것은 아님)',
              f'구간 평균 기울기의 최대-최소: {s["group_mean_slope_range"]:.6f} px/mm, 평균 대비 {s["group_mean_slope_range_percent_of_mean"]:.3f}%',
              "", "직선 잔차 및 인접 구간 기울기", "구간 | QR | 직선 잔차 RMS(px) | 최대 절대 잔차(px) | 인접 기울기 최소~최대(px/mm)"]
    for g in result["groups"]:
        for m in (*MARKERS, "mean"):
            r = g["qr_mean"] if m == "mean" else g["markers"][m]
            lines.append(f'{g["capture_group_number"]} | {m} | {r["rms_fit_residual_px"]:.6f} | {r["max_abs_fit_residual_px"]:.6f} | {r["adjacent_slope_min"]:.6f} ~ {r["adjacent_slope_max"]:.6f}')
    lines += ["", "주의: 범위와 RMS는 관측값의 요약입니다. 확률 구간, 물리적 오차, 정렬 허용오차 합격 판정이 아닙니다.",
              "이번 결과에서 절대 영점, θ별 배율 변화, θ 기울기, Y·θ 분리는 추정하지 않았습니다.",
              "기존 batch_summary.json의 구간 평균 기울기와 수치 오차 범위 내에서 일치합니다.",
              "", "인접 사진 76쌍 상세 (파일명은 촬영 순서)",
              "구간 | 이전 파일 → 다음 파일 | ΔY(mm) | 좌 QR a | 중앙 QR a | 우 QR a | QR 평균 a (px/mm)"]
    for g in result["groups"]:
        for pair in g["adjacent_pairs"]:
            slopes = " | ".join(f'{pair["slope_px_per_nominal_mm_by_qr"][m]:.6f}' for m in MARKERS)
            lines.append(f'{g["capture_group_number"]} | {pair["from_file"]} → {pair["to_file"]} | {pair["delta_y_mm"]:+.2f} | {slopes} | {pair["mean_qr_slope_px_per_nominal_mm"]:.6f}')
    (HERE / "y_slope_results.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_result(result):
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False,
                         "font.size": 10, "figure.facecolor": "white", "savefig.facecolor": "white",
                         "svg.fonttype": "path"})
    fig, axes = plt.subplots(3, 4, figsize=(17, 10.5))
    fig.subplots_adjust(left=.065, right=.98, bottom=.095, top=.84, hspace=.55, wspace=.30)
    for column, group in enumerate(result["groups"]):
        q = np.array(group["delta_y_from_first_mm"])
        labels = np.array(group["labels_in_time_order_mm"])
        order = np.argsort(q)
        for marker, color in zip(MARKERS, COLORS):
            fit = group["markers"][marker]
            u = np.array(fit["delta_x_from_first_px"])
            slope = fit["slope_px_per_nominal_mm"]
            intercept = fit["nuisance_intercept_in_relative_coordinates_px"]
            axes[0, column].scatter(q, u, s=12, color=color, alpha=.7)
            axes[0, column].plot(q[order], (slope*q+intercept)[order], color=color, linewidth=.8,
                                 label=f'{marker}: {slope:.4f} px/mm')
            axes[1, column].plot(labels, fit["residuals_px"], '.-', color=color, markersize=4, linewidth=.8)
            axes[2, column].plot(np.arange(1, len(labels)), fit["adjacent_slopes_px_per_nominal_mm"],
                                 '.-', color=color, markersize=4, linewidth=.8)
        mean = group["qr_mean"]["slope_px_per_nominal_mm"]
        axes[2, column].axhline(mean, color="#283442", linestyle="--", linewidth=1, label=f'전체 적합 {mean:.4f}')
        axes[0, column].set_title(f'촬영 구간 {group["capture_group_number"]}\nY {group["label_start_mm"]:+.2f} → {group["label_end_mm"]:+.2f} mm · 20장', fontsize=12)
        axes[0, column].set_xlabel('첫 사진 대비 Y 라벨 변화 (mm)')
        axes[1, column].set_xlabel('파일명 Y 라벨 (mm)')
        axes[2, column].set_xlabel('시간순 인접 사진쌍 번호 (1-19)')
        axes[0, column].legend(fontsize=8, loc='best')
        axes[2, column].legend(fontsize=8, loc='lower right')
        axes[1, column].axhline(0, color="#8c98a3", linewidth=.6)
        axes[1, column].set_ylim(-.12, .12)
        axes[2, column].set_ylim(12.6, 15.3)
        axes[2, column].set_xticks([1, 5, 10, 15, 19])
        for row in range(3):
            axes[row, column].grid(alpha=.2)
    axes[0, 0].set_ylabel('첫 사진 대비 QR 가로 변위 (px)')
    axes[1, 0].set_ylabel('측정값 - 직선 예측값 (px)')
    axes[2, 0].set_ylabel('인접 Δx / ΔY (px/mm)')
    fig.text(.055, .95, '1차 데이터 | Y 이동량당 QR 영상 변화량', fontsize=23, weight='bold', color='#1d344b')
    fig.text(.055, .90, '위: 구간 전체 기울기  ·  가운데: 직선에서 벗어난 정도  ·  아래: 인접 0.10mm 구간별 기울기', fontsize=12)
    fig.text(.055, .035, '기존 ZNCC → 아핀 ECC 측정값 재사용 · θ 고정 가정 · 단위는 파일명 라벨 기준 · 잔차는 물리적 오차/합격 기준이 아님', fontsize=10, color='#52657b')
    fig.savefig(HERE / "y_slope_diagnostics.png", dpi=160)
    fig.savefig(HERE / "y_slope_diagnostics.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
