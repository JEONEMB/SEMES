"""Plot recorded label coordinates in metadata order; no image registration."""

from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent
BASE = HERE.parent
INK = "#182b43"
MUTED = "#52657b"
COLORS = ["#283747", "#d04b45", "#b78313", "#36854c", "#8653aa", "#168aa6", "#275ac2"]
plt.rcParams.update({"font.family": "Malgun Gothic", "font.size": 11,
                     "axes.unicode_minus": False, "figure.facecolor": "white",
                     "savefig.facecolor": "white", "svg.fonttype": "path"})


def load_data():
    first = json.loads((BASE / "initial_error/capture_order.json").read_text(encoding="utf-8"))
    second = json.loads((BASE / "second_data_inspection/inventory.json").read_text(encoding="utf-8"))
    group_ids = {g["group"]: i for i, g in enumerate(first["chronological_groups"])}
    a = []
    for index, r in enumerate(sorted(first["files_in_modification_order"], key=lambda r: (r["modified_ns"], r["file"])), 1):
        label = float(r["label"]) if r["label"] is not None else 0.0
        a.append({"order": index, "file": r["file"], "time": r["modified_kst"],
                  "y": label if r["group"].startswith("Y_") else 0.0,
                  "theta": label if r["group"].startswith("Theta_") else 0.0,
                  "group": group_ids[r["group"]], "color": COLORS[group_ids[r["group"]]]})
    b = []
    y_colors = {0.0: COLORS[0], .05: COLORS[1], .1: COLORS[4], .5: COLORS[5], 1.0: COLORS[6]}
    for index, r in enumerate(sorted(second["images"], key=lambda r: (r["archive_modified_time_no_timezone"], r["file"])), 1):
        b.append({"order": index, "file": r["file"], "time": r["archive_modified_time_no_timezone"],
                  "y": r["label_y_mm"], "theta": r["label_theta_degree"],
                  "group": r["condition"], "color": y_colors[r["label_y_mm"]]})
    assert len(a) == 281 and len(b) == 60
    assert a[0]["file"] == "center.bmp"
    assert [r["order"] for r in b if r["y"] == r["theta"] == 0] == [1, 2, 24]
    return a, b


def decorate(ax, limits, title):
    ax.set(xlim=limits[:2], ylim=limits[2:], xlabel="Y 라벨 (mm)", ylabel="θ 라벨 (°)")
    ax.set_title(title, loc="left", color=INK, fontsize=14, pad=15, weight="bold")
    ax.grid(color="#e5ebf0", lw=.7)
    ax.set_axisbelow(True)
    ax.axhline(0, color="#a8b7c5", lw=.9, zorder=0)
    ax.axvline(0, color="#a8b7c5", lw=.9, zorder=0)
    for spine in ax.spines.values():
        spine.set_color("#cbd5df")
    ax.tick_params(labelsize=10, colors=MUTED)


def callout(ax, r, offset, text=None):
    ax.annotate(text or f"#{r['order']}", (r["y"], r["theta"]),
                xytext=offset, textcoords="offset points", color=r["color"], fontsize=10,
                weight="bold", ha="left", va="center",
                bbox={"boxstyle": "round,pad=.25", "fc": "white", "ec": "none", "alpha": .94},
                arrowprops={"arrowstyle": "-", "color": r["color"], "lw": .7})


def save(fig, stem):
    fig.savefig(HERE / f"{stem}.png", dpi=150)
    fig.savefig(HERE / f"{stem}.svg")
    plt.close(fig)


def plot_first(rows):
    fig = plt.figure(figsize=(15, 9))
    ax = fig.add_axes([.075, .19, .57, .65])
    decorate(ax, (-2.35, 2.35, -1.19, 1.19), "Y–θ 명목 좌표 · 점 = 이미지 1장")
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_yticks([-1, -.5, 0, .5, 1])
    fig.text(.055, .945, "1차 데이터 | 시간순 촬영 좌표", fontsize=23, weight="bold", color=INK)
    fig.text(.055, .895, "2026-10-04 · 원본 BMP 수정 시각(KST) · 281장 · #번호는 전체 이미지 순번", fontsize=12, color=MUTED)
    groups = defaultdict(list)
    for r in rows:
        groups[r["group"]].append(r)
    descriptions = ["기준 이미지", "Y +0.10 → +2.00 mm", "Y -0.10 → -2.00 mm",
                    "θ +0.01 → +1.00°", "θ -0.01 → -1.00°",
                    "Y +0.05 → +1.95 mm", "Y -1.95 → -0.05 mm"]
    offsets = {1: [(24, 65), (-8, 42)], 2: [(-92, 42), (-8, 42)],
               3: [(32, 25), (14, 8)], 4: [(32, -25), (14, -5)],
               5: [(20, -48), (-10, -48)], 6: [(-10, -48), (-85, -48)]}
    for gid, points in groups.items():
        color = COLORS[gid]
        if gid == 0:
            ax.scatter([0], [0], marker="*", s=140, color=color, zorder=8)
            callout(ax, points[0], (-110, 8), "#1 center")
        else:
            ax.plot([r["y"] for r in points], [r["theta"] for r in points], color=color, lw=1.3, alpha=.7)
            ax.scatter([r["y"] for r in points], [r["theta"] for r in points], s=15, color=color, zorder=4)
            for fraction in (.35, .70):
                i = int((len(points) - 1) * fraction)
                j = min(i + max(2, len(points) // 12), len(points)-1)
                ax.annotate("", xy=(points[j]["y"], points[j]["theta"]), xytext=(points[i]["y"], points[i]["theta"]),
                            arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.5, "mutation_scale": 14})
            callout(ax, points[0], offsets[gid][0])
            callout(ax, points[-1], offsets[gid][1])
        pos = .815 - gid * .087
        fig.text(.69, pos, f"{gid+1:02d}  {descriptions[gid]}", fontsize=13, weight="bold", color=color)
        rank = str(points[0]["order"]) if len(points) == 1 else f"{points[0]['order']}–{points[-1]['order']}"
        time = points[0]["time"][11:19]
        if len(points) > 1:
            time += "–" + points[-1]["time"][11:19]
        fig.text(.72, pos-.029, f"#{rank}  |  {len(points)}장  |  {time}", fontsize=10, color=MUTED)
    fig.text(.055, .095, "좌표는 파일명 라벨입니다. 파일명에 없는 반대 축은 명목상 0으로 표시했습니다.", fontsize=11, color=INK)
    fig.text(.055, .055, "실제 영점 복귀·축 위치를 확인한 그래프가 아닙니다. 화살표는 같은 촬영 구간 안의 순서이며 중간 이동 경로는 알 수 없습니다.", fontsize=10, color=MUTED)
    save(fig, "first_capture_order")


def grouped_coordinates(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["y"], r["theta"])].append(r)
    return list(groups.values())


def plot_second(rows):
    fig = plt.figure(figsize=(16, 9))
    left = fig.add_axes([.065, .22, .40, .60])
    right = fig.add_axes([.56, .22, .40, .60])
    decorate(left, (-.07, 1.16, -.12, 2.17), "전체 범위")
    decorate(right, (-.07, 1.16, -.018, .225), "작은 θ 영역 확대 · 0–0.20°")
    left.set_xticks([0, .05, .1, .5, 1.0])
    right.set_xticks([0, .05, .1, .5, 1.0])
    for ax in (left, right):
        ax.tick_params(axis="x", labelrotation=40)
    right.set_yticks([0, .01, .02, .03, .1, .2])
    fig.text(.045, .945, "2차 데이터 | 시간순 촬영 좌표", fontsize=23, weight="bold", color=INK)
    fig.text(.045, .895, "2026-10-15 · ZIP 내 BMP 수정 시각(시간대 미기재) · 60장 · #번호는 전체 이미지 순번", fontsize=12, color=MUTED)
    for ax in (left, right):
        for before, after in zip(rows, rows[1:]):
            if (before["y"], before["theta"]) == (after["y"], after["theta"]):
                continue
            if ax is right and (before["theta"] > .2 or after["theta"] > .2):
                continue
            ax.annotate("", (after["y"], after["theta"]), (before["y"], before["theta"]),
                        arrowprops={"arrowstyle": "-|>", "color": "#8697a8", "lw": .85,
                                    "linestyle": (0, (4, 3)), "alpha": .6, "shrinkA": 6, "shrinkB": 6}, zorder=1)
        for points in grouped_coordinates(rows):
            r = points[0]
            if ax is right and r["theta"] > .2:
                continue
            reference = r["y"] == r["theta"] == 0
            ax.scatter(r["y"], r["theta"], color=r["color"], marker="*" if reference else "o",
                       s=125 if reference else 48, zorder=5, edgecolors="white", linewidths=.8)
            if reference:
                callout(ax, r, (30, -18), "#1, 2, 24  center")
            elif ax is right or r["theta"] >= 1:
                if r["y"] == .05:
                    offset = (-49, 3)
                else:
                    offset = (9, 3)
                callout(ax, r, offset, f"#{points[0]['order']}–{points[-1]['order']}")
    handles = [Line2D([], [], marker="o", ls="", color=c, label=f"Y={y:.2f} mm")
               for y,c in [(.05,COLORS[1]),(.1,COLORS[4]),(.5,COLORS[5]),(1.0,COLORS[6])]]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5,.115), ncol=4, frameon=False, fontsize=11)
    fig.text(.045, .075, "같은 라벨 3장은 같은 좌표에 겹쳐 있으며, #3–5처럼 이미지 순번 범위로 표시했습니다. center3는 전체 #24입니다.", fontsize=11, color=INK)
    fig.text(.045, .035, "점선 화살표는 기록된 좌표 사이의 촬영 순서만 연결합니다. 실제 이동 경로·중간 영점 복귀·노출 시각을 뜻하지 않습니다.", fontsize=10, color=MUTED)
    save(fig, "second_capture_order")


def build_html(first, second):
    template = (HERE / "capture_order_template.html").read_text(encoding="utf-8")
    payload = json.dumps({"first": first, "second": second}, ensure_ascii=False).replace("<", "\\u003c")
    (HERE / "capture_order_interactive.html").write_text(template.replace("__DATA_JSON__", payload), encoding="utf-8")


def main():
    first, second = load_data()
    plot_first(first)
    plot_second(second)
    build_html(first, second)
    provenance = {"sources": ["analysis/initial_error/capture_order.json", "analysis/second_data_inspection/inventory.json"],
                  "first_count": len(first), "second_count": len(second),
                  "coordinates": "filename labels, not measured image displacements or independently measured robot poses",
                  "first_unlabelled_axis": "displayed as nominal zero; actual zero return unknown",
                  "time_sources": ["original filesystem modification time, KST", "ZIP BMP modification time, timezone unspecified"],
                  "connections": "chronological recorded positions, not actual intermediate robot paths",
                  "image_analysis_performed": False}
    (HERE / "plot_provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"first": len(first), "second": len(second), "output": str(HERE)}, ensure_ascii=True))


if __name__ == "__main__":
    main()
