"""Draw saved six-run x-target intervals as overlapping rectangles.

Display only: no regression, registration, uncertainty estimation, or new tests.
Rectangle y coordinates are layout, not image y or statistical uncertainty.
"""

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Patch
from matplotlib.ticker import MaxNLocator, FormatStrFormatter


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "equivalence_results.json"
GROUPS = (2, 3, 4, 5, 6, 7)
COLORS = {2: "#D44949", 3: "#1677BB", 4: "#D28B00", 5: "#7150B5", 6: "#008D80", 7: "#9A5A2E"}
LABELS = {
    2: "2번  Y +0.10 → +2.00 mm", 3: "3번  Y -0.10 → -2.00 mm",
    4: "4번  θ +0.01 → +1.00°", 5: "5번  θ -0.01 → -1.00°",
    6: "6번  Y +0.05 → +1.95 mm", 7: "7번  Y -1.95 → -0.05 mm",
}
METHODS = ("point_only", "OLS_iid", "HAC_sensitivity")
TITLES = ("불확실성 반영 전", "기본 회귀 불확실성 반영", "잔차 연관성 반영 · 보조")
SUBTITLES = ("영점 점추정 + 허용오차 환산", "회귀 가정 아래 동시 신뢰영역", "HAC 민감도 계산 · 소표본 근사")
NAMES = {"L": "좌측 QR", "C": "중앙 QR", "R": "우측 QR"}

plt.rcParams.update({
    "font.family": "Malgun Gothic", "font.size": 10.5,
    "axes.unicode_minus": False, "figure.facecolor": "white",
    "savefig.facecolor": "white", "svg.fonttype": "path",
    "axes.labelcolor": "#283543", "text.color": "#283543",
    "xtick.color": "#536273", "axes.edgecolor": "#526170",
})


def interval(row, method):
    if method == "point_only":
        return row["point_only_interval"]
    return row["methods"][method]["acceptable_target_interval"]


def draw_panel(ax, rows, summary, method, domain, zoom=False):
    for index, row in enumerate(rows):
        data = interval(row, method)
        lo, hi = data["low_px"], data["high_px"]
        assert hi > lo
        bottom, top = 0.10+index*0.055, 0.92-index*0.055
        color = COLORS[row["group"]]
        ax.add_patch(Rectangle((lo, bottom), hi-lo, top-bottom,
                               fill=False, edgecolor=color, linewidth=1.9, zorder=3))
        if not zoom:
            ax.text((lo+hi)/2, top+0.012, str(row["group"]), color=color,
                    ha="center", va="bottom", fontsize=10, fontweight="bold",
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.1}, zorder=5)
    common = summary[method]
    low, high = common["low_px"], common["high_px"]
    if common["nonempty"]:
        ax.add_patch(Rectangle((low, 0.375), high-low, 0.270,
                               facecolor="#FFCC43", edgecolor="none", alpha=0.72, zorder=2))
        ax.axvline(low, color="#4A515A", linewidth=0.8, linestyle=":", zorder=1)
        ax.axvline(high, color="#4A515A", linewidth=0.8, linestyle=":", zorder=1)
        text = f"공통 {high-low:.4f}px  |  {low:.4f} ~ {high:.4f}"
        color = "#695108"
    else:
        assert high < low
        ax.add_patch(Rectangle((high, 0.375), low-high, 0.270,
                               facecolor="#FCE0E0", edgecolor="#BD3C44", hatch="////",
                               linewidth=0.5, zorder=2))
        ax.axvline(high, color="#BD3C44", linewidth=0.8, linestyle=":", zorder=1)
        ax.axvline(low, color="#BD3C44", linewidth=0.8, linestyle=":", zorder=1)
        text = f"공통 없음  |  경계 간격 {low-high:.4f}px"
        color = "#A42633"
    ax.set_xlim(domain)
    ax.set_ylim(0, 1.06)
    ax.set_yticks([])
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f" if zoom else "%.1f"))
    ax.grid(axis="x", color="#DDE3EA", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=9)
    ax.set_xlabel("원본 영상의 가로 좌표 x* (px)", fontsize=10, labelpad=6)
    ax.text(0.5, -0.32, text, transform=ax.transAxes, ha="center", va="top", fontsize=10, color=color)
    return text


def overview(data):
    fig, axes = plt.subplots(3, 3, figsize=(16.6, 12.0))
    fig.subplots_adjust(left=0.065, right=0.98, bottom=0.155, top=0.79, hspace=0.75, wspace=0.20)
    fig.text(0.065, 0.965, "6개 촬영 구간의 목표 좌표 범위", fontsize=23, weight="bold")
    fig.text(0.065, 0.927, "색 사각형 1개 = 촬영 구간 1개의 허용 목표 범위   ·   Y ±0.045 mm / θ ±0.005°", fontsize=12)
    handles = [Line2D([], [], color=COLORS[g], lw=2, label=LABELS[g]) for g in GROUPS]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.06, 0.903),
               ncol=3, frameon=False, fontsize=10.5, columnspacing=3.6, handlelength=2.4)
    for row_index, marker in enumerate(("L", "C", "R")):
        rows = sorted([r for r in data["fits"] if r["marker"] == marker], key=lambda r: r["group"])
        assert [r["group"] for r in rows] == list(GROUPS)
        lo = min(interval(r, "point_only")["low_px"] for r in rows)
        hi = max(interval(r, "point_only")["high_px"] for r in rows)
        pad = 0.04*(hi-lo)
        for col, method in enumerate(METHODS):
            ax = axes[row_index, col]
            draw_panel(ax, rows, data["summaries"][marker], method, (lo-pad, hi+pad))
            ax.text(0.015, 0.02, NAMES[marker], transform=ax.transAxes,
                    fontsize=11, weight="bold", bbox={"facecolor": "white", "edgecolor": "none", "pad": 2})
            if row_index == 0:
                ax.set_title(TITLES[col]+"\n"+SUBTITLES[col], fontsize=12, pad=13)
    meaning = [Patch(facecolor="#FFCC43", edgecolor="none", label="6개 구간의 공통 범위"),
               Patch(facecolor="#FCE0E0", edgecolor="#BD3C44", hatch="////", label="겹치지 않는 경계 사이의 틈")]
    fig.legend(handles=meaning, loc="lower left", bbox_to_anchor=(0.06, 0.034),
               ncol=2, frameon=False, fontsize=10.5, columnspacing=3)
    fig.text(0.065, 0.023, "세로 높이·배치는 구분용이며 영상 y좌표가 아닙니다. 사각형은 실제 QR 외곽이나 2차원 신뢰영역이 아닙니다.", fontsize=10, color="#596674")
    fig.text(0.065, 0.005, "번호 2~7은 메타데이터순 촬영 구간입니다. 저장된 계산 결과를 표시했으며 새 분석·검정은 수행하지 않았습니다.", fontsize=9.5, color="#596674")
    fig.savefig(HERE/"six_run_range_rectangles.png", dpi=175)
    fig.savefig(HERE/"six_run_range_rectangles.svg")
    plt.close(fig)


def center_detail(data):
    fig, axes = plt.subplots(2, 3, figsize=(16.6, 8.0))
    fig.subplots_adjust(left=0.065, right=0.98, bottom=0.18, top=0.75, hspace=0.82, wspace=0.20)
    fig.text(0.065, 0.958, "중앙 QR | 6개 사각형의 겹침과 경계 확대", fontsize=22, weight="bold")
    fig.text(0.065, 0.912, "위: 6개 범위 전체   ·   아래: 겹침을 결정하는 부분 확대   ·   세로 위치는 구분용", fontsize=12)
    handles = [Line2D([], [], color=COLORS[g], lw=2, label=LABELS[g]) for g in GROUPS]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.06, 0.885), ncol=3,
               frameon=False, fontsize=10.5, columnspacing=3.6)
    rows = sorted([r for r in data["fits"] if r["marker"] == "C"], key=lambda r: r["group"])
    summary = data["summaries"]["C"]
    lo = min(interval(r, "point_only")["low_px"] for r in rows)
    hi = max(interval(r, "point_only")["high_px"] for r in rows)
    span = hi-lo
    # The zoom shows all three previously saved common/gap endpoints on one scale.
    edge_lo = min(min(summary[k]["low_px"], summary[k]["high_px"]) for k in METHODS)
    edge_hi = max(max(summary[k]["low_px"], summary[k]["high_px"]) for k in METHODS)
    pad = (edge_hi-edge_lo)*0.35
    for col, method in enumerate(METHODS):
        axes[0, col].set_title(TITLES[col], fontsize=12, pad=12)
        draw_panel(axes[0, col], rows, summary, method, (lo-0.04*span, hi+0.04*span))
        draw_panel(axes[1, col], rows, summary, method, (edge_lo-pad, edge_hi+pad), zoom=True)
        # Direct labels identify the two limiting edges without interpreting y as data.
        lower = summary[method]["low_px"]
        upper = summary[method]["high_px"]
        axes[1, col].annotate("3번 Y의 하한", xy=(lower, 0.18), xytext=(0.04, 0.04),
                              textcoords="axes fraction", color=COLORS[3], fontsize=9,
                              arrowprops={"arrowstyle": "-", "color": COLORS[3], "lw": 0.8})
        axes[1, col].annotate("4번 θ의 상한", xy=(upper, 0.8), xytext=(0.98, 0.98),
                              textcoords="axes fraction", ha="right", va="top", color=COLORS[4], fontsize=9,
                              arrowprops={"arrowstyle": "-", "color": COLORS[4], "lw": 0.8})
    fig.text(0.065, 0.066, "노란색: 6개 구간이 모두 허용하는 x*   /   빨간 빗금: 3번 하한이 4번 상한을 넘어 생긴 틈", fontsize=11)
    fig.text(0.065, 0.027, "가로 범위만 실제 계산값입니다. 사각형 세로 높이는 표시용이며 실제 QR 영역·영상 y좌표를 뜻하지 않습니다.", fontsize=10, color="#596674")
    fig.savefig(HERE/"center_range_rectangles_zoom.png", dpi=175)
    fig.savefig(HERE/"center_range_rectangles_zoom.svg")
    plt.close(fig)


def main():
    source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    assert len(data["fits"]) == 18
    overview(data)
    center_detail(data)
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == source_hash
    provenance = {
        "source": str(SOURCE), "source_sha256": source_hash,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "figures": ["six_run_range_rectangles.png", "six_run_range_rectangles.svg",
                    "center_range_rectangles_zoom.png", "center_range_rectangles_zoom.svg"],
        "x": "Saved admissible target-coordinate interval, original image horizontal pixels",
        "y": "Artificial layout only; no image-y measurement or statistical bound",
        "rectangles": "6 acquisition runs per QR, group numbers 2 through 7",
        "yellow": "Intersection of the 6 saved x intervals",
        "red_hatch": "Empty-intersection gap: minimum upper bound to maximum lower bound",
        "new_image_registration": False, "new_regression": False,
        "new_statistical_test": False, "saved_source_modified": False,
    }
    (HERE/"range_figures_provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"figures": provenance["figures"], "source_unchanged": True}))


if __name__ == "__main__":
    main()
