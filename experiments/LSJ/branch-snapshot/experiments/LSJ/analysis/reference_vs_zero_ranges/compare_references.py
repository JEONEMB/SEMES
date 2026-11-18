"""Compare supplied reference images with six saved first-data target intervals.

Only new image measurements: first center.bmp -> second center1/2/3, three
original QR tracking patches each. No new fit, confidence interval or test.
"""

import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE / "initial_error"))
from analyze_images import measure, read_gray

MARKERS = ("L", "C", "R")
METHODS = ("point_only", "OLS_iid", "HAC_sensitivity")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interval(row, method):
    return (row["point_only_interval"] if method == "point_only" else
            row["methods"][method]["acceptable_target_interval"])


def compare_point(x, low, high):
    if low > high:
        return {"inside": False, "empty_interval": True}
    signed = x-low if x < low else x-high if x > high else 0.0
    return {"inside": low <= x <= high, "empty_interval": False,
            "signed_outside_distance_px": signed,
            "outside_distance_px": abs(signed),
            "inside_margin_px": min(x-low, high-x) if low <= x <= high else None}


def main():
    cv2.setNumThreads(2)
    first_path = BASE / "initial_error/measurements.json"
    second_path = BASE / "increment_consistency/second_measurements.json"
    range_path = BASE / "zero_equivalence/equivalence_results.json"
    first, second, ranges = (read(p) for p in (first_path, second_path, range_path))
    first_row = next(r for r in first["measurements"] if r["axis"] == "center")
    first_file = Path(first["data_root"]) / first_row["file"]
    reference, digest = read_gray(first_file)
    assert digest == first_row["sha256"]
    rois = {m: first["regions_xyxy"][m] for m in MARKERS}
    centers = {m: [(roi[0]+roi[2]-1)/2, (roi[1]+roi[3]-1)/2] for m, roi in rois.items()}
    references = [{"dataset": "first", "file": str(first_file), "name": first_file.name,
                   "sha256": digest, "position_xy_px": centers,
                   "measurement": "Definition of the first-data template coordinates; displacement exactly zero"}]
    ref_rows = sorted([r for r in second["measurements"] if r["is_reference"]], key=lambda r: r["file"])
    assert len(ref_rows) == 3
    for row in ref_rows:
        path = Path(second["data_root"]) / row["file"]
        current, digest = read_gray(path)
        assert digest == row["sha256"]
        measured, positions = {}, {}
        for marker in MARKERS:
            result = measure(reference, current, rois[marker], 110)
            for model in ("affine", "translation"):
                assert "error" not in result[model], result[model]
            assert np.isfinite(result["affine"]["dx"])
            measured[marker] = result
            positions[marker] = [centers[marker][0]+result["affine"]["dx"],
                                 centers[marker][1]+result["affine"]["dy"]]
        references.append({"dataset": "second", "file": str(path), "name": path.name,
                           "sha256": digest, "position_xy_px": positions, "registration": measured})
    for ref in references:
        ref["comparisons"] = {}
        for marker in MARKERS:
            x = ref["position_xy_px"][marker][0]
            rows = [r for r in ranges["fits"] if r["marker"] == marker]
            ref["comparisons"][marker] = {}
            for method in METHODS:
                checks = []
                for row in rows:
                    bounds = interval(row, method)
                    checks.append({"group": row["group"], "axis": row["axis"],
                                   "low_px": bounds["low_px"], "high_px": bounds["high_px"],
                                   **compare_point(x, bounds["low_px"], bounds["high_px"])})
                common = ranges["summaries"][marker][method]
                ref["comparisons"][marker][method] = {
                    "inside_groups": [r["group"] for r in checks if r["inside"]],
                    "outside_groups": [r["group"] for r in checks if not r["inside"]],
                    "inside_count": sum(r["inside"] for r in checks),
                    "checks": checks,
                    "six_way_common_interval": {"low_px": common["low_px"], "high_px": common["high_px"],
                        **compare_point(x, common["low_px"], common["high_px"])},
                }
    result = {
        "purpose": "Position containment of first/second supplied references in six saved first-data target-coordinate ranges",
        "source_sha256": {str(p.relative_to(BASE)): sha(p) for p in (first_path, second_path, range_path)},
        "script_sha256": sha(Path(__file__)),
        "method": {"registration": "Reuse initial_error.analyze_images.measure: ZNCC integer initialization -> ECC affine, 60 iterations, epsilon 1e-6, Gaussian 5, search +/-110 pixels",
                   "coordinate": "Track the SAME first-data ROI center in every reference image; x right, y down, pixels",
                   "new_registrations": 9, "new_regressions": 0, "new_statistical_tests": 0,
                   "primary_position": "affine dx/dy at first-data ROI center; translation retained only as implementation output",
                   "matching_scores": "Image registration diagnostics, not ground-truth accuracy certificates",
                   "comparison": "Each reference supplies a point coordinate. Report containment and pixel distance, not rectangle area or percentage overlap.",
                   "reference_roi_xyxy": rois},
        "limitations": [
            "Six intervals all originate from first data; second references are comparison targets, not members of those fits.",
            "This comparison uses estimated image positions without a new uncertainty estimate for the cross-dataset registration.",
            "Horizontal tracking-center coordinates only; not full QR area overlap or a physical Y/theta correction estimate.",
            "Reference images are reported separately; no three-image or three-QR averaging.",
            "An empty HAC intersection cannot contain any reference point; distance to an empty set is not reported as a finite gap.",
        ],
        "references": references,
    }
    (HERE/"reference_comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    write_report(result)
    print(json.dumps([{ "name": ref["name"], "positions": ref["position_xy_px"],
                       "inside_groups": {m:{k: ref["comparisons"][m][k]["inside_groups"] for k in METHODS} for m in MARKERS},
                       "registration_quality": {m:{"zncc": ref["registration"][m]["ncc"], "ecc":ref["registration"][m]["affine"]["correlation"]} for m in MARKERS} if "registration" in ref else None,
                       "point_only_distance_to_common": {m:ref["comparisons"][m]["point_only"]["six_way_common_interval"] for m in MARKERS}}
                      for ref in references], ensure_ascii=True, indent=2))


def write_report(result):
    lines = ["제공된 1/2차 기준 이미지와 1차 6개 촬영 구간의 목표 좌표 범위 비교", "",
             "대상: 1차 center.bmp 및 2차 center1.bmp, center2.bmp, center3.bmp.",
             "1차 템플릿의 동일한 QR 추적 영역 중심을 2차 기준 이미지 3장에서 ZNCC -> ECC 아핀으로 찾음.",
             "기존 2차 ROI 중심은 1차와 정의가 달라 그대로 비교하지 않음.",
             "비교 대상은 가로 좌표 1개이므로, 면적 겹침률이 아닌 포함 구간 수와 벗어난 픽셀 거리를 표시.",
             "6개 범위는 모두 1차 데이터에서 추정한 것. 회귀식/통계 검정을 새로 수행하거나 원본 파일을 수정하지 않음.",
             "기준 이미지/QR을 평균내지 않고 각각 비교. 포함 판정은 교차 데이터셋 정합의 불확실성을 추가한 통계 검정이 아님.", "",
             "기준 파일 | QR | x(px) | 점추정 범위 포함 구간 | 기본 불확실성 범위 포함 구간 | HAC 보조 범위 포함 구간"]
    for ref in result["references"]:
        for marker in MARKERS:
            comps = ref["comparisons"][marker]
            counts = [str(comps[k]["inside_groups"])+f" ({comps[k]['inside_count']}/6)" for k in METHODS]
            lines.append(f"{ref['dataset']}/{ref['name']} | {marker} | {ref['position_xy_px'][marker][0]:.9f} | "+" | ".join(counts))
    lines += ["", "구간별 상세 비교: 외부 거리의 부호 +는 범위 오른쪽, -는 왼쪽. 포함 시 0.",
              "파일 | QR | 방법 | 촬영 구간 | 포함 여부 | 외부 거리(px)"]
    for ref in result["references"]:
        for marker in MARKERS:
            for method in METHODS:
                for check in ref["comparisons"][marker][method]["checks"]:
                    lines.append(f"{ref['name']} | {marker} | {method} | {check['group']} | {check['inside']} | {check['signed_outside_distance_px']:.9f}")
    lines += ["", "등록 품질 지표 (정확도 보증 수치가 아님)"]
    for ref in result["references"][1:]:
        for marker in MARKERS:
            reg = ref["registration"][marker]
            lines.append(f"{ref['name']} | {marker} | ZNCC={reg['ncc']:.8f} | ECC={reg['affine']['correlation']:.8f} | dx={reg['affine']['dx']:.9f} | dy={reg['affine']['dy']:.9f}")
    lines += ["", "원본 무결성: 기준 BMP 4개 모두 기존 SHA256과 대조.",
              "재현: python analysis/reference_vs_zero_ranges/compare_references.py", ""]
    (HERE/"reference_comparison.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
