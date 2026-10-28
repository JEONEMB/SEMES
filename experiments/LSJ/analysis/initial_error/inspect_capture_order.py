"""Inspect original BMP timestamps; filesystem order is not verified exposure order."""

from collections import Counter
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from itertools import groupby
import json
from pathlib import Path
import re
import struct
import sys

from PIL import Image


HERE = Path(__file__).resolve().parent
KST = timezone(timedelta(hours=9))


def classify(path):
    if path.name.lower() == "center.bmp":
        return "center", None
    match = re.fullmatch(r"([YT])([+-]\d+(?:\.\d+)?)", path.stem)
    if not match:
        raise ValueError(f"Unexpected filename: {path}")
    axis, label_text = match.groups()
    label = Decimal(label_text)
    sign = "positive" if label > 0 else "negative"
    if axis == "T":
        return f"Theta_{sign}", label_text
    grid = "0.10_grid" if abs(label) % Decimal("0.10") == 0 else "0.05_offset_grid"
    return f"Y_{sign}_{grid}", label_text


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    existing = json.loads((HERE / "measurements.json").read_text(encoding="utf-8"))
    data_root = Path(existing["data_root"])
    rows = []
    for path in data_root.rglob("*.bmp"):
        stat = path.stat()
        with path.open("rb") as stream:
            head = stream.read(138)
        pixel_offset = struct.unpack_from("<I", head, 10)[0]
        dib_size = struct.unpack_from("<I", head, 14)[0]
        width, height = struct.unpack_from("<ii", head, 18)
        bpp = struct.unpack_from("<H", head, 28)[0]
        compression = struct.unpack_from("<I", head, 30)[0]
        pixel_bytes = ((width * bpp + 31) // 32) * 4 * abs(height)
        with Image.open(path) as image:
            exif = dict(image.getexif())
            metadata = {key: repr(value) for key, value in image.info.items()}
        group, label = classify(path)
        # Python 3.12 on Windows exposes the file creation timestamp explicitly.
        rows.append({
            "file": str(path.relative_to(data_root)),
            "group": group,
            "label": label,
            "modified_kst": datetime.fromtimestamp(stat.st_mtime, KST).isoformat(),
            "modified_ns": stat.st_mtime_ns,
            "created_kst": datetime.fromtimestamp(stat.st_birthtime, KST).isoformat(),
            "created_ns": stat.st_birthtime_ns,
            "embedded_exif_tag_count": len(exif),
            "pillow_metadata": metadata,
            "dib_header_size": dib_size,
            "pixel_offset": pixel_offset,
            "bits_per_pixel": bpp,
            "compression": compression,
            "bytes_after_pixel_array": stat.st_size - pixel_offset - pixel_bytes,
        })

    rows.sort(key=lambda row: (row["modified_ns"], row["file"]))
    for rank, row in enumerate(rows, 1):
        row["modification_order"] = rank
    created_order = sorted(rows, key=lambda row: (row["created_ns"], row["file"]))
    center = next(row for row in rows if row["file"].lower() == "center.bmp")
    groups = []
    for name, iterator in groupby(rows, key=lambda row: row["group"]):
        selected = list(iterator)
        labels = [Decimal(row["label"]) for row in selected if row["label"] is not None]
        groups.append({
            "group": name,
            "count": len(selected),
            "first_file": selected[0]["file"],
            "last_file": selected[-1]["file"],
            "first_modified_kst": selected[0]["modified_kst"],
            "last_modified_kst": selected[-1]["modified_kst"],
            "labels_in_modification_order": [str(label) for label in labels],
            "adjacent_label_steps": sorted({str(b - a) for a, b in zip(labels, labels[1:])}),
        })

    summary = {
        "data_root": str(data_root),
        "file_count": len(rows),
        "timezone": "Asia/Seoul (UTC+09:00)",
        "sort_key": "Filesystem last-modified time, not verified camera exposure time",
        "center_modification_rank": center["modification_order"],
        "center_creation_rank": next(i for i, row in enumerate(created_order, 1) if row is center),
        "center_modified_kst": center["modified_kst"],
        "center_created_kst": center["created_kst"],
        "second_file": rows[1]["file"],
        "second_file_modified_kst": rows[1]["modified_kst"],
        "center_to_second_file_seconds": (rows[1]["modified_ns"] - center["modified_ns"]) / 1e9,
        "duplicate_modification_timestamp_groups": sum(
            count > 1 for count in Counter(row["modified_ns"] for row in rows).values()
        ),
        "files_with_exif_tags": sum(row["embedded_exif_tag_count"] > 0 for row in rows),
        "all_bmp_headers_40_bytes_with_pixel_offset_54": all(
            row["dib_header_size"] == 40 and row["pixel_offset"] == 54 for row in rows
        ),
        "all_uncompressed_24_bit_without_trailing_bytes": all(
            row["bits_per_pixel"] == 24
            and row["compression"] == 0
            and row["bytes_after_pixel_array"] == 0
            for row in rows
        ),
        "interpretation": [
            "center.bmp is earliest by both filesystem modification and creation time.",
            "No embedded capture timestamp was found in the inspected BMP metadata.",
            "The ordered labels support a plausible acquisition sequence if modification times preserve image-save times.",
            "Timestamps do not establish intermediate robot movements or returns to zero.",
        ],
    }
    output = {"summary": summary, "chronological_groups": groups, "files_in_modification_order": rows}
    destination = HERE / "capture_order.json"
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "chronological_groups": groups}, ensure_ascii=False, indent=2))
    print(f"Saved: {destination}")


if __name__ == "__main__":
    main()
