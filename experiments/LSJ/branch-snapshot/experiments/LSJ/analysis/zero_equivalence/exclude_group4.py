"""Repeat the saved equivalence assessment with acquisition group 4 excluded.

Keep the original 18-fit simultaneous multipliers so that removing the three
group-4 constraints is the only change. Do not refit or re-register images.
"""

import copy
import json
from pathlib import Path

import numpy as np

from check_zero_equivalence import (
    MARKERS, METHODS, acceptable_interval, digest, intersect,
    inverse_ellipse_interval,
)


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "without_group4"
RETAINED = {2, 3, 5, 6, 7}


def main():
    source_path = HERE / "equivalence_results.json"
    original_hash = digest(source_path)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    assert source["family_size"] == 18
    assert source["tolerances"] == {"Y": 0.045, "Theta": 0.005}
    for relative, expected in source["data"]["source_sha256"].items():
        assert digest(HERE.parent / relative) == expected, relative
    assert digest(HERE / "check_zero_equivalence.py") == source["data"]["script_sha256"]
    rows = copy.deepcopy([r for r in source["fits"] if r["group"] != 4])
    assert len(rows) == 15 and {r["group"] for r in rows} == RETAINED
    summaries = {}
    for marker in MARKERS:
        selected = [r for r in rows if r["marker"] == marker]
        assert len(selected) == 5
        assert sum(r["n"] for r in selected) == 180
        summary = {"point_only": intersect([r["point_only_interval"] for r in selected])}
        for method in METHODS:
            # Recompute endpoints from saved coefficients/covariances, retaining
            # exactly the original multipliers and tolerances.
            for row in selected:
                entry = row["methods"][method]
                recomputed = acceptable_interval(
                    np.array(row["beta_b_a"]),
                    np.array(entry["covariance_beta_order_b_a"]),
                    entry["critical_multiplier"], row["tolerance"],
                )
                saved = entry["acceptable_target_interval"]
                for key in ("low_px", "high_px", "width_px"):
                    assert abs(recomputed[key] - saved[key]) < 1e-10
            summary[method] = intersect([
                r["methods"][method]["acceptable_target_interval"] for r in selected
            ])
            summary[method]["lower_limiting_group"] = max(
                selected, key=lambda r: r["methods"][method]["acceptable_target_interval"]["low_px"]
            )["group"]
            summary[method]["upper_limiting_group"] = min(
                selected, key=lambda r: r["methods"][method]["acceptable_target_interval"]["high_px"]
            )["group"]
        common = intersect([summary[m] for m in METHODS])
        summary["intersection_of_primary_and_sensitivity"] = common
        target = (common["low_px"] + common["high_px"]) / 2 if common["nonempty"] else None
        summary["illustrative_target_px"] = target
        for row in selected:
            row["illustrative_target_px"] = target
            row["point_equivalent_offset"] = None
            for method in METHODS:
                row["methods"][method].pop("equivalent_offset_bounds", None)
                row["methods"][method].pop("inside_tolerance", None)
            if target is None:
                continue
            beta = np.array(row["beta_b_a"])
            row["point_equivalent_offset"] = float((target - beta[0]) / beta[1])
            for method in METHODS:
                entry = row["methods"][method]
                bounds = inverse_ellipse_interval(
                    target, beta, np.array(entry["covariance_beta_order_b_a"]),
                    entry["critical_multiplier"],
                )
                entry["equivalent_offset_bounds"] = bounds
                entry["inside_tolerance"] = -row["tolerance"] < bounds[0] <= bounds[1] < row["tolerance"]
                interval = entry["acceptable_target_interval"]
                assert entry["inside_tolerance"] == (interval["low_px"] < target < interval["high_px"])
                assert entry["inside_tolerance"]
        summaries[marker] = summary

    result = {
        "purpose": "User-requested subset assessment excluding acquisition group 4 only",
        "excluded_groups": [4], "retained_groups": sorted(RETAINED),
        "data": {"images": 180, "runs": 5, "fits": 15, "raw_images_reregistered": False},
        "tolerances": source["tolerances"], "alpha_family": source["alpha_family"],
        "multiplicity_family_size_retained": source["family_size"],
        "method": source["method"],
        "subset_notes": [
            "Original 18-fit confidence multipliers retained, not recalculated for 15 fits. Only group 4 constraints removed.",
            "The original simultaneous confidence event covers any retained subset under its assumptions; this does not validate a data-driven exclusion as a physical decision.",
            "Subset was selected after inspecting the six-group result. Findings apply only to groups 2,3,5,6,7, with only the negative-theta run remaining.",
            "HAC remains an approximate sensitivity calculation, without exact finite-sample 95% coverage.",
            "Separate horizontal tracking-center ranges for L/C/R, not full QR areas or a joint physical Y/theta pose.",
        ],
        "assumption_status": source["assumption_status"],
        "summaries": summaries, "fits": rows,
        "original_six_group_summaries": source["summaries"],
        "conclusion": {
            "all_primary_model_conditional_pass": all(s["OLS_iid"]["nonempty"] for s in summaries.values()),
            "all_hac_sensitivity_nonempty": all(s["HAC_sensitivity"]["nonempty"] for s in summaries.values()),
            "all_primary_and_sensitivity_intersections_nonempty": all(s["intersection_of_primary_and_sensitivity"]["nonempty"] for s in summaries.values()),
            "target_configuration_changed": False,
        },
        "provenance": {
            "input_file": str(source_path.relative_to(HERE.parent)),
            "input_sha256": original_hash, "script_sha256": digest(Path(__file__)),
            "math_module_sha256": digest(HERE / "check_zero_equivalence.py"),
            "upstream_hashes": source["data"]["source_sha256"],
        },
        "verification": {
            "upstream_hashes_match": True, "retained_images_180_fits_15": True,
            "saved_endpoints_reproduced": 30, "original_multipliers_retained": True,
            "inverse_interval_acceptance_agrees_with_endpoint_acceptance": True,
        },
    }
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / "equivalence_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    write_report(result)
    assert digest(source_path) == original_hash
    print(json.dumps({"summaries": summaries, "conclusion": result["conclusion"]}, ensure_ascii=False, indent=2))


def write_report(result):
    lines = [
        "4번 촬영 구간 제외: 영점 후보 동등성 분석", "",
        "대상: 2, 3, 5, 6, 7번. 이미지 180장, QR별 회귀식 총 15개.",
        "제외: 4번 theta +0.01 ~ +1.00도, 100장. theta는 5번 음수 구간만 남음.",
        "허용범위 유지: Y +/-0.045 mm, theta +/-0.005도.",
        "기존 ZNCC -> ECC 아핀 측정값/직선/공분산을 재사용. 이미지 재정합이나 평균은 없음.",
        "기존 18개 회귀식 기준의 동시 신뢰영역 임계값을 그대로 유지하여 제외 효과만 비교.",
        "기울기/절편의 동시 신뢰영역을 역산하는 기존의 보수적 동등성 판정 방식 유지.",
        "일반적인 고정 목표의 90% CI TOST를 새로 실행한 것은 아님.", "",
        "결과",
        f"기본 OLS 가정하 좌/중/우 모두 조건부 충족: {result['conclusion']['all_primary_model_conditional_pass']}",
        f"HAC 보조 계산에서도 좌/중/우 모두 공통 구간 존재: {result['conclusion']['all_hac_sensitivity_nonempty']}",
        f"두 계산을 모두 만족하는 공통 구간 존재: {result['conclusion']['all_primary_and_sensitivity_intersections_nonempty']}",
        "", "QR | 계산 | 공통 하한(px) | 공통 상한(px) | 폭(px) | 하한 결정 촬영 | 상한 결정 촬영",
    ]
    for marker, summary in result["summaries"].items():
        for method in ("point_only", *METHODS, "intersection_of_primary_and_sensitivity"):
            item = summary[method]
            lines.append(f"{marker} | {method} | {item['low_px']:.9f} | {item['high_px']:.9f} | {item['width_px']:.9f} | {item.get('lower_limiting_group', '-')} | {item.get('upper_limiting_group', '-')}")
    lines += ["", "공통 구간 중점의 역산 확인 (표시용 예시이며 기준값 확정 아님)",
              "QR | 촬영 | 축 | 예시 x*(px) | OLS 역산 범위 | HAC 역산 범위 | 허용반경"]
    for row in result["fits"]:
        if row["illustrative_target_px"] is None:
            continue
        parts = [f"{row['marker']} | {row['group']} | {row['axis']} | {row['illustrative_target_px']:.9f}"]
        for method in METHODS:
            lo, hi = row["methods"][method]["equivalent_offset_bounds"]
            parts.append(f"[{lo:.9f}, {hi:.9f}]")
        parts.append(f"+/-{row['tolerance']} {row['unit']}")
        lines.append(" | ".join(parts))
    lines += ["", "해석",
        "기본 계산은 선형 평균 및 구간 내 독립/정규/등분산 오차 가정하의 조건부 결과.",
        "HAC는 이분산과 촬영 순서상 잔차 연관성을 반영한 근사 보조 계산. 정확한 95% 보장을 뜻하지 않음.",
        "4번을 포함한 결과를 확인한 후 제외한 부분집합 분석이므로, 결론은 남은 5개 구간에 한정.",
        "이번 결과로 4번 데이터가 잘못되었다거나 전체 6구간 동등성이 성립한다고 판단하지 않음.",
        "각 QR 추적 중심의 가로좌표 범위이며, QR 전체 면적이나 동일한 실제 Y/theta 자세를 보장하는 범위가 아님.",
        "영점 후보 평균의 일치 여부에 관한 계산이며 최종 제어 성능/신규 영상의 예측 정확도 검증이 아님.",
        "기존 6구간 결과 보존. 목표 좌표 설정 파일 변경 없음.", "",
        "재현: python analysis/zero_equivalence/exclude_group4.py",
        "확인: 원자료 해시 일치, 15개 회귀식/180장 유지, 30개 기존 구간 경계 재현, 역산 허용범위 판정 일치.",
        "",
    ]
    (OUTPUT / "equivalence_report.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
