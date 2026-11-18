"""Stage 2: image response to equal label increments, without a pass threshold.

Reuse first-data ECC translation/affine measurements. Measure second-data
images using the same registration function, relative to its own center1.
Only within-run differences (first data) and fixed-other-label differences
(second data) are compared. Labels are not verified physical displacement.
"""

import argparse
from collections import defaultdict
from itertools import combinations, product
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parent
sys.path.insert(0, str(ANALYSIS / "initial_error"))
from analyze_images import measure, read_gray

MARKERS = ("L", "C", "R")
METHODS = ("affine", "translation")
SECOND_MEASUREMENTS = HERE / "second_measurements.json"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, result):
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def measure_second():
    inventory = read_json(ANALYSIS / "second_data_inspection" / "inventory.json")
    repeats = read_json(ANALYSIS / "second_repeatability" / "repeatability_results.json")
    root = Path(inventory["summary"]["extracted_root"])
    reference_row = next(r for r in inventory["images"] if Path(r["file"]).name == "center1.bmp")
    reference, digest = read_gray(root / reference_row["file"])
    assert digest == reference_row["sha256"]
    reference_group = next(g for g in repeats["groups"] if g["is_reference"])
    rois = {m: reference_group["rois"][m]["xyxy"] for m in MARKERS}
    rows = []
    for i, row in enumerate(inventory["images"]):
        current, digest = read_gray(root / row["file"])
        if digest != row["sha256"]:
            raise ValueError(f"Source hash mismatch: {row['file']}")
        regions = {m: measure(reference, current, rois[m], 230) for m in MARKERS}
        for marker in MARKERS:
            for method in METHODS:
                if "error" in regions[marker][method]:
                    raise ValueError(f"Registration failed: {row['file']}, {marker}, {method}")
        rows.append({k: row[k] for k in ("file", "condition", "is_reference", "label_y_mm",
                     "label_theta_degree", "sha256", "archive_modified_time_no_timezone")} | {"regions": regions})
        if (i + 1) % 10 == 0:
            print(f"Second-data images measured: {i+1}/60", flush=True)
    save_json(SECOND_MEASUREMENTS, {
        "data_root": str(root), "reference_file": reference_row["file"], "regions_xyxy": rois,
        "method": "Reuse initial_error.analyze_images.measure: integer ZNCC initialization, ECC translation and affine; displacement at reference ROI center.",
        "parameters": {"search_margin_px": 230, "ecc_max_iterations": 60, "ecc_epsilon": 1e-6, "gaussian_size": 5},
        "parameters_note": "Existing first-data implementation settings; not an acceptance criterion or a new accuracy claim.",
        "opencv_version": cv2.__version__, "measurements": rows})
    return rows


def position(row, method):
    return np.mean([[row["regions"][m][method][k] for k in ("dx", "dy")] for m in MARKERS], axis=0)


def stats(values):
    values = np.asarray(values, float)
    return {"count": len(values), "mean": float(values.mean()),
            "sample_sd": float(values.std(ddof=1)) if len(values) > 1 else None,
            "min": float(values.min()), "max": float(values.max()), "range": float(np.ptp(values))}


def second_edges(rows):
    groups = defaultdict(list)
    for row in rows:
        if not row["is_reference"]:
            groups[(row["label_y_mm"], row["label_theta_degree"])].append(row)
    assert len(groups) == 19 and all(len(v) == 3 for v in groups.values())
    edges = []
    for axis, moving, fixed in (("Y", 0, 1), ("Theta", 1, 0)):
        for fixed_value in sorted({key[fixed] for key in groups}):
            keys = sorted((key for key in groups if key[fixed] == fixed_value), key=lambda key: key[moving])
            for start, end in zip(keys, keys[1:]):
                step = round(end[moving] - start[moving], 6)
                edge = {"axis": axis, "fixed_other_label": fixed_value, "start_label": start[moving],
                        "end_label": end[moving], "label_increment": step,
                        "start_condition": groups[start][0]["condition"], "end_condition": groups[end][0]["condition"],
                        "start_files": [r["file"] for r in groups[start]], "end_files": [r["file"] for r in groups[end]],
                        "methods": {}}
                for method in METHODS:
                    a = np.array([position(r, method) for r in groups[start]])
                    b = np.array([position(r, method) for r in groups[end]])
                    differences = np.array([pb-pa for pa, pb in product(a, b)])
                    delta = b.mean(axis=0)-a.mean(axis=0)
                    edge["methods"][method] = {"mean_delta_xy_px": delta.tolist(),
                        "dx_px_per_label_unit": float(delta[0]/step),
                        "nine_combinations_delta_xy_px": differences.tolist(),
                        "nine_combinations_dx_stats_px": stats(differences[:, 0])}
                edges.append(edge)
    return edges


def first_edges(rows, second):
    runs = defaultdict(list)
    for row in rows:
        if row["axis"] == "center":
            continue
        sign = "positive" if row["label"] > 0 else "negative"
        run = f"{row['axis']}_{sign}"
        if row["axis"] == "Y":
            run += "_0.10_grid" if round(abs(row["label"])/.05) % 2 == 0 else "_0.05_offset_grid"
        runs[run].append(row)
    requested = {axis: {e["label_increment"] for e in second if e["axis"] == axis} for axis in ("Y", "Theta")}
    requested["Y"].add(.1)
    requested["Theta"].add(.01)
    edges, run_info = [], []
    for run, rr in sorted(runs.items()):
        rr.sort(key=lambda r: r["label"])
        run_info.append({"run": run, "axis": rr[0]["axis"], "count": len(rr),
                         "label_min": rr[0]["label"], "label_max": rr[-1]["label"]})
        for a, b in combinations(rr, 2):
            step = round(b["label"]-a["label"], 6)
            if step not in requested[a["axis"]]:
                continue
            edges.append({"run": run, "axis": a["axis"], "start_label": a["label"], "end_label": b["label"],
                "label_increment": step, "start_file": a["file"], "end_file": b["file"],
                "methods": {method: {"delta_xy_px": (position(b, method)-position(a, method)).tolist()}
                            for method in METHODS}})
    return edges, run_info


def aggregate(first, second):
    result = []
    for axis in ("Y", "Theta"):
        steps = sorted({e["label_increment"] for e in first+second if e["axis"] == axis})
        for step in steps:
            a = [e for e in first if e["axis"] == axis and e["label_increment"] == step]
            b = [e for e in second if e["axis"] == axis and e["label_increment"] == step]
            item = {"axis": axis, "label_increment": step, "methods": {}}
            for method in METHODS:
                av = [e["methods"][method]["delta_xy_px"][0] for e in a]
                bv = [e["methods"][method]["mean_delta_xy_px"][0] for e in b]
                item["methods"][method] = {"first_dx_px": stats(av) if av else None,
                    "second_condition_mean_dx_px": stats(bv) if bv else None,
                    "second_minus_first_mean_dx_px": float(np.mean(bv)-np.mean(av)) if av and bv else None}
            result.append(item)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--remeasure-second", action="store_true")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    cv2.setNumThreads(2)
    start = time.perf_counter()
    first_rows = read_json(ANALYSIS / "initial_error" / "measurements.json")["measurements"]
    second_rows = (measure_second() if args.remeasure_second or not SECOND_MEASUREMENTS.exists()
                   else read_json(SECOND_MEASUREMENTS)["measurements"])
    second = second_edges(second_rows)
    first, runs = first_edges(first_rows, second)
    combined = aggregate(first, second)
    method_differences = [e["methods"]["affine"]["mean_delta_xy_px"][0]-e["methods"]["translation"]["mean_delta_xy_px"][0] for e in second]
    result = {
        "purpose": "Stage 2: describe image displacement consistency for equal nominal label changes. No pass/fail or stage-3 axis separability claim.",
        "primary_method": "ECC affine, mean displacement of three QR ROI centers; translation ECC retained as a model comparison.",
        "first_source": "../initial_error/measurements.json", "second_source": "second_measurements.json",
        "rules": [
            "First: six capture-run hypotheses from batch_summary.json; no pair crosses sign/grid groups, and center.bmp is excluded.",
            "First: only increments present among second-data adjacent levels, plus the first-data native within-run increments (Y 0.10, Theta 0.01).",
            "All differences are oriented toward increasing label value, not necessarily actual capture or approach direction.",
            "Second: adjacent existing levels along one label while the other label is fixed; reference condition excluded because both labels change.",
            "Second: each condition uses the mean of all three image positions; retain all 3x3 start/end differences. Same suffix is not assumed to be a matched trial.",
            "Differences use each dataset's own reference positions, so a shared additive reference offset cancels. No cross-dataset absolute position comparison.",
            "Edges sharing images and the nine start/end combinations are not independent measurements; SD/ranges are descriptive, not confidence intervals.",
            "Changing pose can change projection sensitivity; a differing slope does not by itself establish mechanical error.",
            "No synthetic augmentation, compensation fitting, automatic image exclusion, physical-error attribution, or acceptance threshold.",
        ],
        "first_runs": runs, "first_edges": first, "second_edges": second, "equal_increment_summary": combined,
        "quality": {
            "second_min_affine_correlation": min(r["regions"][m]["affine"]["correlation"] for r in second_rows for m in MARKERS),
            "second_min_translation_correlation": min(r["regions"][m]["translation"]["correlation"] for r in second_rows for m in MARKERS),
            "second_max_abs_affine_minus_translation_increment_dx_px": float(np.max(np.abs(method_differences))),
            "first_edge_count": len(first), "second_condition_edge_count": len(second),
            "second_image_position_count": len(second_rows), "elapsed_seconds": time.perf_counter()-start,
        }}
    save_json(HERE / "increment_results.json", result)
    print(json.dumps({"quality": result["quality"], "equal_increment_summary": combined}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
