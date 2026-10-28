"""Plot saved stage-2 descriptive results; no acceptance bands."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
r = json.loads((HERE / "increment_results.json").read_text(encoding="utf-8"))
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
colors = ["#2268a2", "#ca6331", "#238066", "#9153a0"]

for ax, axis, step, unit in [(axs[0, 0], "Y", .1, "mm"), (axs[0, 1], "Theta", .01, "°")]:
    edges = [e for e in r["first_edges"] if e["axis"] == axis and e["label_increment"] == step]
    for i, run in enumerate(sorted({e["run"] for e in edges})):
        selected = [e for e in edges if e["run"] == run]
        name = run.replace("Theta_", "θ ").replace("Y_", "Y ").replace("negative", "음수").replace("positive", "양수").replace("_0.10_grid", " 0.10 격자").replace("_0.05_offset_grid", " 0.05 오프셋 격자")
        ax.plot([e["start_label"] for e in selected],
                [e["methods"]["affine"]["delta_xy_px"][0] for e in selected],
                ".-", ms=3, lw=.7, color=colors[i], label=name)
    ax.set(title=f"1차 · {'Y' if axis == 'Y' else 'θ'} +{step:g}{unit}마다 영상 변화 ({len(edges)}쌍)",
           xlabel=f"시작 라벨 ({unit})", ylabel="가로 변화량 Δx (px)")
    ax.legend(fontsize=8)

ax = axs[1, 0]
steps = [.05, .4, .5]
for i, fixed in enumerate([.01, .1, 1., 2.]):
    edges = [e for e in r["second_edges"] if e["axis"] == "Y" and e["fixed_other_label"] == fixed]
    ax.plot([steps.index(e["label_increment"]) for e in edges],
            [e["methods"]["affine"]["dx_px_per_label_unit"] for e in edges],
            "o-", color=colors[i], label=f"θ 고정 {fixed:g}°")
ax.set_xticks(range(3), ["0.05→0.10\nΔY=0.05", "0.10→0.50\nΔY=0.40", "0.50→1.00\nΔY=0.50"])
ax.set(title="2차 · θ를 고정하고 Y 변경", xlabel="Y 라벨 구간 (mm)", ylabel="Δx / ΔY 라벨 (px/mm)")
ax.legend(fontsize=8)

ax = axs[1, 1]
intervals = sorted({(e["start_label"], e["end_label"]) for e in r["second_edges"] if e["axis"] == "Theta"})
for i, fixed in enumerate([.05, .1, .5, 1.]):
    edges = [e for e in r["second_edges"] if e["axis"] == "Theta" and e["fixed_other_label"] == fixed]
    ax.scatter([intervals.index((e["start_label"], e["end_label"]))+(i-1.5)*.1 for e in edges],
               [e["methods"]["affine"]["dx_px_per_label_unit"] for e in edges],
               color=colors[i], label=f"Y 고정 {fixed:g}mm", s=35)
ax.set_xticks(range(len(intervals)), [f"{a:g}→{b:g}" for a, b in intervals], rotation=40, ha="right")
ax.set(title="2차 · Y를 고정하고 θ 변경", xlabel="θ 라벨 구간 (°)", ylabel="Δx / Δθ 라벨 (px/°)")
ax.legend(fontsize=8)

for ax in axs.flat:
    ax.grid(alpha=.2)
fig.suptitle("2번 분석 · 같은 라벨 변화량에 대한 영상 변화\nQR 3개 평균 변위 · ECC 아핀 정합", fontsize=16)
fig.text(.05, .025, "1차: 파일 시각으로 추정한 촬영 묶음 내부만 비교. 2차: 조건별 사진 3장의 평균 위치 차이.\n"
         "아래 두 그래프는 변화량 비교를 위한 라벨 단위 환산이며, 실제 기계 이동량의 보정 배율이나 합격 판정이 아님.", fontsize=10)
fig.subplots_adjust(left=.08, right=.98, top=.88, bottom=.14, hspace=.4, wspace=.25)
fig.savefig(HERE / "increment_comparison.png", dpi=140)
plt.close(fig)
