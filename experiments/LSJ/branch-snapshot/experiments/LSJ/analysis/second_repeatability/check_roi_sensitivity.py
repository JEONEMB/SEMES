"""Check ROI dependence of the same two estimators; preserve primary results."""

from pathlib import Path
import json
import sys

import numpy as np

from analyze_repeats import (HERE, INVENTORY, BASE_BOXES, METHODS, PAIRS,
                             read_image, crop, measure_pair, vector, vector_summary)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    original = json.loads((HERE / "repeatability_results.json").read_text(encoding="utf-8"))
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    root = Path(inventory["summary"]["extracted_root"])
    row_by_file = {r["file"]: r for r in inventory["images"]}
    selected = {(0., 0.), (.05, .01), (.1, .1), (.5, 2.)}
    records = []
    for group in original["groups"]:
        if (group["label_y_mm"], group["label_theta_degree"]) not in selected:
            continue
        images = [read_image(root, row_by_file[f["file"]]) for f in group["files"]]
        for change in (-6, 8):
            for pi, (i, j) in enumerate(PAIRS):
                for marker in BASE_BOXES:
                    x0, y0, x1, y1 = group["rois"][marker]["xyxy"]
                    roi = [x0-change, y0-change, x1+change, y1+change]
                    measured = measure_pair(crop(images[i], roi), crop(images[j], roi))
                    base = group["pairs"][pi]["regions"][marker]
                    records.append({"condition": group["condition"], "i": i, "j": j, "marker": marker,
                                    "margin_px": 12+change, "roi": roi, "measurements": measured,
                                    "difference_from_primary_xy_px": {m: (vector(measured[m])-vector(base[m])).tolist() for m in METHODS}})
    summary = {m: vector_summary([r["difference_from_primary_xy_px"][m] for r in records]) for m in METHODS}
    summary["primary_vs_changed_roi_count"] = len(records)
    per_condition = {}
    for condition in sorted({r["condition"] for r in records}):
        per_condition[condition] = {m: vector_summary([r["difference_from_primary_xy_px"][m] for r in records if r["condition"] == condition]) for m in METHODS}
    result = {"description": "Primary margin 12 px versus 6 and 20 px; four selected groups, all three pairs and QR markers.",
              "selection": "reference; small-repeat variation; largest observed displacement; largest method difference",
              "summary": summary, "per_condition": per_condition, "records": records}
    (HERE / "roi_sensitivity.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "per_condition": per_condition}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
