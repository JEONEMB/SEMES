"""Inventory the extracted second dataset without estimating image motion.

Checks archive CRCs, BMP decoding, duplicate hashes, and filename/sidecar
consistency. Y_Actual/T_Actual provenance is not specified by the sidecars.
ZIP timestamps are recorded as naive member timestamps, not exposure times.
"""

import collections
import hashlib
import json
from pathlib import Path
import re
import sys
from zipfile import ZipFile
import zlib

import cv2
import numpy as np


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    out = Path(__file__).resolve().parent
    project = out.parents[1]
    links = (project / "Links.md").read_text(encoding="utf-8")
    line = next(line for line in links.splitlines() if "second image data folder path" in line)
    source = Path(re.search(r'"([^"]+)"', line).group(1))
    archive = next(source.glob("*.zip"))
    extracted = out / "extracted"
    initial = json.loads((out.parent / "initial_error" / "measurements.json").read_text(encoding="utf-8"))
    old_hashes = {row["sha256"] for row in initial["measurements"]}
    rows, mismatches, decode_errors = [], [], []
    sidecars = {}
    archive_bytes = sum(path.stat().st_size for path in source.iterdir() if path.is_file())
    with ZipFile(archive) as z:
        entries = [info for info in z.infolist() if not info.is_dir()]
        entry_map = {info.filename: info for info in entries}
        for info in entries:
            data = (extracted / info.filename).read_bytes()
            if len(data) != info.file_size or zlib.crc32(data) != info.CRC:
                mismatches.append(info.filename)
            if info.filename.lower().endswith(".txt"):
                text = data.decode("utf-8-sig")
                tokens = text.split()
                if tokens[:2] != ["Y_Actual", "T_Actual"] or len(tokens) != 4:
                    raise ValueError(f"Unexpected sidecar: {info.filename}")
                sidecars[info.filename] = {"Y_Actual": float(tokens[2]), "T_Actual": float(tokens[3])}
                continue
            if not info.filename.lower().endswith(".bmp"):
                continue
            image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
            if image is None:
                decode_errors.append(info.filename)
                continue
            path = Path(info.filename)
            candidate = str(path.with_suffix(".txt")).replace("\\", "/")
            if candidate not in entry_map:
                candidate += ".txt"
            if candidate not in entry_map:
                raise ValueError(f"No sidecar: {info.filename}")
            label = re.fullmatch(r"Y([+-]\d+(?:\.\d+)?) T([+-]\d+(?:\.\d+)?)", path.parent.name)
            name_label = re.fullmatch(r"Y([+-]\d+(?:\.\d+)?) T([+-]\d+(?:\.\d+)?) \((\d+)\)", path.stem)
            if label and (not name_label or tuple(map(float, label.groups())) != tuple(map(float, name_label.groups()[:2]))):
                raise ValueError(f"Folder/filename label mismatch: {info.filename}")
            sha = hashlib.sha256(data).hexdigest()
            rows.append({
                "file": info.filename, "sidecar": candidate,
                "condition": path.parent.name,
                "is_reference": label is None,
                "label_y_mm": float(label[1]) if label else 0.0,
                "label_theta_degree": float(label[2]) if label else 0.0,
                "shape_hwc": list(image.shape), "dtype": str(image.dtype),
                "bits_per_pixel": int.from_bytes(data[28:30], "little"),
                "bmp_compression": int.from_bytes(data[30:34], "little"),
                "all_rgb_channels_equal": bool(image.ndim == 3 and np.array_equal(image[:,:,0], image[:,:,1]) and np.array_equal(image[:,:,0], image[:,:,2])),
                "sha256": sha, "identical_file_in_first_dataset": sha in old_hashes,
                "archive_modified_time_no_timezone": "%04d-%02d-%02d %02d:%02d:%02d" % info.date_time,
            })
    for row in rows:
        row.update(sidecars[row["sidecar"]])
    references = [row for row in rows if row["is_reference"]]
    base = references[0]
    for row in rows:
        row["actual_y_minus_reference"] = row["Y_Actual"] - base["Y_Actual"]
        row["actual_theta_minus_reference"] = row["T_Actual"] - base["T_Actual"]
        row["y_label_difference"] = row["actual_y_minus_reference"] - row["label_y_mm"]
        row["theta_label_difference"] = row["actual_theta_minus_reference"] - row["label_theta_degree"]
    mixed = [row for row in rows if not row["is_reference"]]
    conditions = collections.Counter(row["condition"] for row in rows)
    grid = {}
    for row in mixed:
        grid.setdefault(str(row["label_y_mm"]), set()).add(row["label_theta_degree"])
    # Mentor description supplied by the user on 2026-10-20.
    requested_grid = {(y, theta) for y in (0.05, 0.1, 0.5, 1.0) for theta in (0.01, 0.1, 1.0, 2.0)}
    requested = requested_grid | {(0.05, 0.02), (0.05, 0.03)}
    observed = {(row["label_y_mm"], row["label_theta_degree"]) for row in mixed}
    summary = {
        "source_archive": str(archive), "extracted_root": str(extracted),
        "compressed_total_bytes": archive_bytes,
        "uncompressed_total_bytes": sum(info.file_size for info in entries),
        "archive_file_count": len(entries), "crc_or_size_mismatches": mismatches,
        "image_count": len(rows), "sidecar_count": len(sidecars), "decode_errors": decode_errors,
        "unique_image_hashes": len({row["sha256"] for row in rows}),
        "identical_images_in_first_dataset": sum(row["identical_file_in_first_dataset"] for row in rows),
        "image_shapes_hwc": sorted({str(row["shape_hwc"]) for row in rows}),
        "bits_per_pixel": sorted({row["bits_per_pixel"] for row in rows}),
        "bmp_compression": sorted({row["bmp_compression"] for row in rows}),
        "all_images_rgb_channels_equal": all(row["all_rgb_channels_equal"] for row in rows),
        "reference_count": len(references), "mixed_image_count": len(mixed),
        "mixed_condition_count": len({row["condition"] for row in mixed}),
        "mentor_described_condition_count": len(requested),
        "mentor_grid_conditions_present": len(requested_grid & observed),
        "missing_mentor_described_y_theta_pairs": sorted(requested - observed),
        "additional_y_theta_pairs": sorted(observed - requested),
        "images_per_condition": dict(conditions),
        "theta_values_by_y": {key: sorted(values) for key, values in sorted(grid.items(), key=lambda x:float(x[0]))},
        "reference_recorded_coordinates": [{key: row[key] for key in ("file", "Y_Actual", "T_Actual", "archive_modified_time_no_timezone")} for row in references],
        "max_abs_y_label_difference": max(abs(row["y_label_difference"]) for row in rows),
        "max_abs_theta_label_difference": max(abs(row["theta_label_difference"]) for row in rows),
        "image_time_range_no_timezone": [min(row["archive_modified_time_no_timezone"] for row in rows), max(row["archive_modified_time_no_timezone"] for row in rows)],
        "limitations": [
            "No provenance explaining whether Actual fields are encoder readbacks or commanded coordinates.",
            "Three images per condition do not specify whether the robot remained stationary or returned to that pose.",
            "Archive file modification timestamps are not verified camera exposure timestamps and have no timezone.",
            "The requested positive-positive 4x4 grid is complete, with three additional mixed conditions; negative-direction combinations and bidirectional sweeps are not present.",
            "This inventory does not measure subpixel motion or validate correction accuracy.",
        ],
    }
    (out / "inventory.json").write_text(json.dumps({"summary": summary, "images": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
