"""Explain saved Y-zero candidates without refitting data or estimating intervals."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SOURCES = {
    "slopes": ROOT / "analysis/first_y_slopes/y_slope_results.json",
    "zeros": ROOT / "analysis/first_y_slopes/y_zero_candidates.json",
    "measurements": ROOT / "analysis/initial_error/measurements.json",
}
COLORS = ["#2967A8", "#C45925", "#26806C", "#8852A0"]
SHAPES = ["o", "s", "^", "D"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color("#AAB4BF")
    ax.tick_params(length=3, color="#AAB4BF")
    ax.set_axisbelow(True)


def main():
    plt.rcParams.update({
        "font.family": "Malgun Gothic", "font.size": 14,
        "axes.unicode_minus": False, "axes.labelsize": 14,
        "axes.titlesize": 16, "text.color": "#263444",
        "axes.labelcolor": "#263444", "xtick.color": "#263444",
        "ytick.color": "#263444", "figure.facecolor": "white",
        "savefig.facecolor": "white", "svg.fonttype": "none",
    })
    saved = {key: json.loads(path.read_text(encoding="utf-8")) for key, path in SOURCES.items()}
    assert digest(SOURCES["slopes"]) == saved["zeros"]["sources"]["slope_results"]["sha256"]
    assert digest(SOURCES["measurements"]) == saved["zeros"]["sources"]["measurements"]["sha256"]
    points = {row["file"]: row for row in saved["measurements"]["measurements"]}
    candidates = {row["capture_group_number"]: row for row in saved["zeros"]["candidates"]}
    roi = saved["measurements"]["regions_xyxy"]["C"]
    ref_x = (roi[0] + roi[2] - 1) / 2

    fig, (ax, zero_ax) = plt.subplots(
        2, 1, figsize=(10.4, 10.0), gridspec_kw={"height_ratios": [1.35, 1]}
    )
    fig.subplots_adjust(left=.14, right=.95, bottom=.17, top=.82, hspace=.70)
    fig.suptitle("구간별 직선으로 기준 위치 후보를 구하는 과정", fontsize=20, y=.985, weight="bold")
    fig.text(.5, .921, "실제 저장된 결과 · 중앙 QR 하나를 예로 표시 · 평균 기울기 사용 안 함",
             ha="center", fontsize=13)
    ax.set_title("① 각 촬영의 20개 점에 따로 맞춘 직선: x = aY + b", loc="left", pad=15)
    ax.axvline(0, color="#7E8996", linewidth=1.3, linestyle="--", zorder=0)
    zero_values = []
    for row_index, (group, color, marker) in enumerate(zip(saved["slopes"]["groups"], COLORS, SHAPES)):
        number = group["capture_group_number"]
        candidate = candidates[number]["markers"]["C"]
        a = candidate["slope_px_per_nominal_mm"]
        b = candidate["x_at_nominal_y_zero_px"]
        labels = np.asarray(group["labels_in_time_order_mm"])
        x_pixels = np.asarray([
            ref_x + points[sample["file"]]["regions"]["C"]["affine"]["dx"]
            for sample in group["samples"]
        ])
        order = np.argsort(labels)
        ax.plot(labels[order], a * labels[order] + b, color=color, linewidth=1.6)
        ax.scatter(labels, x_pixels, s=25, color=color, marker=marker, alpha=.8)
        near_zero = labels[np.argmin(np.abs(labels))]
        ax.plot([near_zero, 0], [a * near_zero + b, b], color=color, linestyle=":", linewidth=2)
        ax.scatter([0], [b], color=color, marker=marker, s=50, zorder=5)
        ax.plot([], [], color=color, marker=marker,
                label=f"촬영 {number}: a = {a:.4f} px/mm")
        y_row = 3 - row_index
        zero_ax.scatter([b], [y_row], s=90, color=color, marker=marker, zorder=3)
        zero_ax.annotate(f"{b:.4f} px", (b, y_row), xytext=(0, 12),
                         textcoords="offset points", ha="center", fontsize=13)
        zero_values.append(b)
    ax.set_xlabel("파일명 Y 라벨 (mm)")
    ax.set_ylabel("중앙 QR 영역 중심의\n영상 가로 위치 x (px)")
    ax.set_xlim(-2.12, 2.12)
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.grid(alpha=.22)
    ax.legend(loc="upper left", fontsize=11, frameon=False)
    ax.text(.99, .025, "점 = 측정값   선 = 기존 적합 직선", transform=ax.transAxes,
            ha="right", fontsize=11)
    style(ax)

    zero_ax.set_title("② Y = 0일 때의 위치 b를 촬영별로 꺼내 비교", loc="left", pad=19)
    zero_ax.set_yticks([3, 2, 1, 0], ["촬영 2", "촬영 3", "촬영 6", "촬영 7"])
    zero_ax.set_ylim(-.5, 3.65)
    span = max(zero_values) - min(zero_values)
    zero_ax.set_xlim(min(zero_values) - .17 * span, max(zero_values) + .17 * span)
    zero_ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    zero_ax.grid(axis="x", alpha=.24)
    zero_ax.set_xlabel("Y = 0 위치 후보 b (px, 이미지 왼쪽이 x = 0)", labelpad=11)
    style(zero_ax)
    fig.text(.14, .058, "현재 계산된 것은 점(b)까지입니다. 각 점 주변의 추정 구간은 아직 계산하지 않았습니다.",
             fontsize=12)
    fig.text(.14, .028, "이 그림은 원영상의 가로 위치를 표시합니다. 이전 변위 그래프와 좌표 기준이 다릅니다.",
             fontsize=11, color="#526172")
    fig.savefig(HERE / "01_saved_candidates.png", dpi=160)
    fig.savefig(HERE / "01_saved_candidates.svg")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 7), sharey=True)
    fig.subplots_adjust(left=.10, right=.97, bottom=.29, top=.70, wspace=.26)
    fig.suptitle("추정 구간을 붙인 다음, 같은 QR의 촬영 결과끼리 비교", fontsize=20, y=.985, weight="bold")
    fig.text(.5, .885, "원리 설명용 가상 예시 — 실제 데이터·실제 추정 구간이 아닙니다", ha="center", fontsize=14, color="#A34025")
    fig.legend(handles=[
        Line2D([0], [0], marker="o", linestyle="none", color="#526172", label="점: 기준 위치 후보 b"),
        Line2D([0], [0], color="#526172", linewidth=3, label="가로 막대: b - r ~ b + r"),
    ], loc="upper center", bbox_to_anchor=(.5, .845), frameon=False, ncol=2, fontsize=12)

    intervals = [
        [(3.0, 5.5), (4.0, 6.0), (4.4, 6.5), (3.8, 6.2)],
        [(3.0, 5.5), (7.2, 9.0), (4.4, 6.5), (3.8, 6.2)],
    ]
    titles = ["A. 네 구간에 겹치는 부분이 있음", "B. 일부 촬영의 구간이 떨어져 있음"]
    for plot_index, (axis, bounds, title) in enumerate(zip(axes, intervals, titles)):
        if plot_index == 0:
            axis.axvspan(4.4, 5.5, color="#7E8996", alpha=.15, linewidth=0)
        for index, ((left, right), color, shape) in enumerate(zip(bounds, COLORS, SHAPES)):
            row = 3 - index
            midpoint = (left + right) / 2
            axis.plot([left, right], [row, row], color=color, linewidth=3)
            axis.plot([left, left], [row - .11, row + .11], color=color, linewidth=2)
            axis.plot([right, right], [row - .11, row + .11], color=color, linewidth=2)
            axis.scatter([midpoint], [row], color=color, marker=shape, s=70, zorder=3)
        axis.set_title(title, fontsize=15, pad=13)
        axis.set_yticks([3, 2, 1, 0], ["가상 촬영 A", "가상 촬영 B", "가상 촬영 C", "가상 촬영 D"])
        axis.set_xticks([])
        axis.set_xlim(1.7, 9.8)
        axis.set_ylim(-.5, 3.6)
        axis.set_xlabel("영상 가로 위치 →  (축척 없는 개념도)", fontsize=12, labelpad=12)
        style(axis)
        axis.tick_params(axis="y", length=0, labelsize=12)
    fig.text(.305, .14, "겹침은 일치 여부를 살펴볼 단서\n그 자체가 정답 위치를 보장하지는 않음",
             ha="center", fontsize=12, linespacing=1.65)
    fig.text(.765, .14, "촬영별 기준 차이를 남겨 두고 확인\n평균이나 넓힌 구간으로 강제로 합치지 않음",
             ha="center", fontsize=12, linespacing=1.65)
    fig.text(.5, .035, "실제 r은 회귀의 불확실성으로 계산해야 합니다. 여기의 막대 길이에는 통계적 의미가 없습니다.",
             ha="center", fontsize=12)
    fig.savefig(HERE / "02_interval_concept.png", dpi=160)
    fig.savefig(HERE / "02_interval_concept.svg")
    plt.close(fig)

    provenance = {
        "purpose": "Explain existing per-run zero candidates and the concept of uncertainty intervals",
        "sources": {key: {"path": str(path), "sha256": digest(path)} for key, path in SOURCES.items()},
        "01_saved_candidates": {
            "status": "Previously computed results only; central QR shown as an example",
            "coordinates": "Original-image tracked ROI-center x and nominal Y labels, not first-frame differences",
            "new_regression_or_registration": False,
        },
        "02_interval_concept": {
            "status": "Fully synthetic schematic, not derived uncertainty or actual data",
            "schematic_intervals": intervals,
            "no_numeric_scale": True,
            "no_claim_of_coverage_or_true_alignment": True,
        },
        "confidence_intervals_computed": False,
        "mean_slopes_used": False,
    }
    (HERE / "figure_provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Created 2 PNGs, 2 SVGs, and provenance. Reused existing fits; no interval estimation performed.")


if __name__ == "__main__":
    main()
