"""Conditional equivalence assessment of first-data nominal-zero targets.

Uses the 18 saved straight-line fits; does not register images or average QR/run
coefficients. The primary procedure inverts simultaneous coefficient confidence
ellipses. It is conservative relative to an ordinary fixed-target TOST.
HAC is a sensitivity calculation, not an exact small-sample confidence claim.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import stats


HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ALPHA = 0.05
FAMILY_SIZE = 18  # six runs, three QR tracking patches; no independence assumed
MARKERS = ("L", "C", "R")
METHODS = ("OLS_iid", "HAC_sensitivity")
TOLERANCES = {"Y": 0.045, "Theta": 0.005}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hac_covariance(design, residual, bread, lags):
    """Newey-West, Bartlett weights, n/(n-p) finite-sample scale correction."""
    scores = design * residual[:, None]
    meat = scores.T @ scores
    for lag in range(1, lags + 1):
        cross = scores[lag:].T @ scores[:-lag]
        meat += (1 - lag / (lags + 1)) * (cross + cross.T)
    covariance = bread @ meat @ bread * len(design) / (len(design) - 2)
    return (covariance + covariance.T) / 2


def acceptable_interval(beta, covariance, critical, tolerance):
    """All positive-slope lines in an ellipse place this x inside +/- tolerance."""
    b, a = beta
    slope_lo = a - critical * np.sqrt(covariance[1, 1])
    if slope_lo <= 0:
        raise ValueError("A positive slope is not supported throughout the ellipse")
    left = np.array([1.0, -tolerance])
    right = np.array([1.0, tolerance])
    low_margin = critical * np.sqrt(left @ covariance @ left)
    high_margin = critical * np.sqrt(right @ covariance @ right)
    low = float(left @ beta + low_margin)
    high = float(right @ beta - high_margin)
    return {"low_px": low, "high_px": high,
            "width_px": high - low, "nonempty": low < high,
            "lower_endpoint_uncertainty_px": float(low_margin),
            "upper_endpoint_uncertainty_px": float(high_margin),
            "slope_lower_bound": float(slope_lo)}


def inverse_ellipse_interval(target, beta, covariance, critical):
    """Exact range of (target-b)/a over the given positive-slope ellipse."""
    b, a = beta
    delta = target - b
    c2 = critical**2
    aa = a*a - c2*covariance[1, 1]
    bb = -2*a*delta - 2*c2*covariance[0, 1]
    cc = delta*delta - c2*covariance[0, 0]
    discriminant = bb*bb - 4*aa*cc
    if aa <= 0 or discriminant < 0:
        raise ValueError("Unbounded or invalid inverse confidence set")
    root = np.sqrt(discriminant)
    return [float((-bb-root)/(2*aa)), float((-bb+root)/(2*aa))]


def intersect(intervals):
    lo = max(x["low_px"] for x in intervals)
    hi = min(x["high_px"] for x in intervals)
    return {"low_px": lo, "high_px": hi, "width_px": hi-lo,
            "nonempty": lo < hi}


def check_math(beta, covariance, critical, tolerance):
    """Validate inversion against the ellipse boundary, including covariance."""
    interval = acceptable_interval(beta, covariance, critical, tolerance)
    if interval["nonempty"]:
        target = (interval["low_px"] + interval["high_px"]) / 2
        inv = inverse_ellipse_interval(target, beta, covariance, critical)
        angles = np.linspace(0, 2*np.pi, 50001)
        directions = np.vstack([np.cos(angles), np.sin(angles)])
        boundary = beta[:, None] + critical*np.linalg.cholesky(covariance)@directions
        offsets = (target-boundary[0])/boundary[1]
        assert abs(offsets.min()-inv[0]) < 1e-8
        assert abs(offsets.max()-inv[1]) < 1e-8
        assert inv[0] > -tolerance and inv[1] < tolerance
    low_inverse = inverse_ellipse_interval(interval["low_px"], beta, covariance, critical)
    high_inverse = inverse_ellipse_interval(interval["high_px"], beta, covariance, critical)
    assert abs(low_inverse[0]+tolerance) < 1e-8
    assert abs(high_inverse[1]-tolerance) < 1e-8


def main():
    source_path = BASE / "first_zero_regions/zero_region_results.json"
    measurement_path = BASE / "initial_error/measurements.json"
    metadata_path = BASE / "initial_error/capture_order.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    for path in (measurement_path, metadata_path):
        assert digest(path) == source["sources"]["source_json_sha256"][path.name]
    measurements = json.loads(measurement_path.read_text(encoding="utf-8"))
    lookup = {r["file"]: r for r in measurements["measurements"]}
    records = []
    for group in source["groups"]:
        q = np.asarray(group["labels_in_time_order"], dtype=float)
        design = np.column_stack([np.ones(len(q)), q])
        bread = np.linalg.inv(design.T @ design)
        n = len(q)
        tolerance = TOLERANCES[group["axis"]]
        critical = float(np.sqrt(2*stats.f.ppf(1-ALPHA/FAMILY_SIZE, 2, n-2)))
        lags = int(np.floor(4*(n/100)**(2/9)))
        for marker in MARKERS:
            saved = group["markers"][marker]
            beta = np.asarray(saved["linear_fit"]["coefficients_ascending"], dtype=float)[:, 0]
            positions = (np.asarray(saved["observed_affine_displacements_xy_px"])[:, 0]
                         + saved["reference_roi_center_xy_px"][0])
            points = sorted((p for p in source["source_points"]
                             if p["group_number"] == group["capture_group_number"]
                             and p["marker"] == marker), key=lambda p: p["order"])
            assert len(points) == n
            for i, point in enumerate(points):
                original = lookup[point["file"]]
                assert original["label"] == q[i] == point["label"]
                original_x = saved["reference_roi_center_xy_px"][0] + original["regions"][marker]["affine"]["dx"]
                assert abs(positions[i]-original_x) < 1e-9
            recomputed = np.linalg.lstsq(design, positions, rcond=None)[0]
            assert np.allclose(beta, recomputed, rtol=0, atol=1e-8)
            residual = positions - design @ beta
            variance = float(residual @ residual / (n-2))
            covariance = variance * bread
            robust_covariance = hac_covariance(design, residual, bread, lags)
            # Independent matrix form of the Newey-West sandwich checks indexing.
            index_lags = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
            kernel = np.maximum(0, 1-index_lags/(lags+1))
            direct = bread@design.T@(kernel*np.outer(residual, residual))@design@bread*n/(n-2)
            assert np.allclose(robust_covariance, direct, rtol=1e-9, atol=1e-12)
            method_results = {}
            for name, cov in zip(METHODS, (covariance, robust_covariance)):
                assert np.linalg.eigvalsh(cov).min() > 0
                check_math(beta, cov, critical, tolerance)
                method_results[name] = {
                    "covariance_beta_order_b_a": cov.tolist(),
                    "critical_multiplier": critical,
                    "acceptable_target_interval": acceptable_interval(beta, cov, critical, tolerance),
                }
            records.append({
                "group": group["capture_group_number"], "name": group["name"],
                "axis": group["axis"], "unit": group["label_unit"], "marker": marker,
                "n": n, "beta_b_a": beta.tolist(), "tolerance": tolerance,
                "point_only_interval": {"low_px": float(beta[0]-beta[1]*tolerance),
                                        "high_px": float(beta[0]+beta[1]*tolerance)},
                "residual_sd_px": np.sqrt(variance),
                "residual_lag1_correlation": float(np.corrcoef(residual[:-1], residual[1:])[0, 1]),
                "durbin_watson": float(np.sum(np.diff(residual)**2)/(residual@residual)),
                "existing_quadratic_minus_linear_intercept_px": saved["quadratic_minus_linear_zero_center_xy_px"][0],
                "hac_lags": lags, "methods": method_results,
            })
    assert len(records) == FAMILY_SIZE
    summaries = {}
    for marker in MARKERS:
        rows = [r for r in records if r["marker"] == marker]
        summary = {"point_only": intersect([r["point_only_interval"] for r in rows])}
        for name in METHODS:
            summary[name] = intersect([r["methods"][name]["acceptable_target_interval"] for r in rows])
            summary[name]["lower_limiting_group"] = max(rows, key=lambda r: r["methods"][name]["acceptable_target_interval"]["low_px"])["group"]
            summary[name]["upper_limiting_group"] = min(rows, key=lambda r: r["methods"][name]["acceptable_target_interval"]["high_px"])["group"]
        common = intersect([summary[name] for name in METHODS])
        summary["intersection_of_primary_and_sensitivity"] = common
        # A diagnostic candidate only, never saved into a device/configuration.
        chosen_interval = common if common["nonempty"] else summary["OLS_iid"]
        target = ((chosen_interval["low_px"]+chosen_interval["high_px"])/2
                  if chosen_interval["nonempty"] else None)
        summary["illustrative_target_px"] = target
        if target is not None:
            for row in rows:
                beta = np.asarray(row["beta_b_a"])
                row["illustrative_target_px"] = target
                row["point_equivalent_offset"] = float((target-beta[0])/beta[1])
                for name in METHODS:
                    method = row["methods"][name]
                    bounds = inverse_ellipse_interval(target, beta,
                        np.asarray(method["covariance_beta_order_b_a"]), method["critical_multiplier"])
                    method["equivalent_offset_bounds"] = bounds
                    method["inside_tolerance"] = bounds[0] > -row["tolerance"] and bounds[1] < row["tolerance"]
                    expected = (method["acceptable_target_interval"]["low_px"] < target
                                < method["acceptable_target_interval"]["high_px"])
                    assert method["inside_tolerance"] == expected
        summaries[marker] = summary
    result = {
        "purpose": "Assess nominal-zero target equivalence with coefficient uncertainty and data-dependent target selection",
        "alpha_family": ALPHA, "family_size": FAMILY_SIZE, "tolerances": TOLERANCES,
        "data": {"images": 280, "runs": 6, "fits": 18, "raw_images_reregistered": False,
                 "source_sha256": {str(p.relative_to(BASE)): digest(p) for p in (source_path, measurement_path, metadata_path)},
                 "script_sha256": digest(Path(__file__))},
        "method": {
            "primary": "OLS joint coefficient ellipses: c=sqrt(2 F_(2,n-2)(1-alpha/18)); Working-Hotelling band; Bonferroni across 18 fits",
            "coverage": "At least 95% joint coverage conditional on correct linear mean and independent, normal, homoscedastic residuals within each fit; cross-fit independence is NOT required",
            "equivalence": "For every line in each ellipse, -T < (x*-b)/a < T. Positive slope verified on each ellipse. A conservative confidence-set equivalence assessment, not ordinary 90%-CI TOST",
            "target_search": "Intersect simultaneous admissible target sets, then select a midpoint for display; coverage event holds for all choices, so no unadjusted post-selection pointwise test",
            "endpoint_formula": "lower=b-a*T+c*sqrt([1,-T] V [1,-T]^T); upper=b+a*T-c*sqrt([1,T] V [1,T]^T)",
            "inverse": "Invert (x*-b_hat-a_hat*q)^2 <= c^2*[1,q] V [1,q]^T; includes covariance between intercept and slope",
            "sensitivity": "Newey-West/Bartlett HAC covariance with lags=floor(4*(n/100)^(2/9)) and n/(n-2) scale; same F multiplier as primary. This is a sensitivity calculation without exact finite-sample 95% coverage",
            "hac_order": "Consecutive acquisition index within each run, not equal elapsed capture time; index-stationarity is an additional approximation",
            "reference": "Conditional on the saved center.bmp/template coordinate system; no new independent reference-image uncertainty experiment",
            "scope": "Mean-response nominal-zero agreement only; not future-image prediction, mixed-axis observability, or final control accuracy",
        },
        "assumption_status": "Residual serial structure observed; primary exact-model coverage not established for these data. HAC is approximate with only 20 images per Y run. Report model-conditional/sensitivity results, not unconditional validation.",
        "sources": [
            "https://arxiv.org/html/2004.01757v8#S2",
            "https://www.itl.nist.gov/div898/handbook/pmd/section4/pmd431.htm",
            "https://www.stat.purdue.edu/~bacraig/notes525/topic3.pdf",
            "https://www.itl.nist.gov/div898/handbook/prc/section4/prc473.htm",
            "https://www.statsmodels.org/stable/generated/statsmodels.stats.sandwich_covariance.cov_hac.html",
        ],
        "summaries": summaries, "fits": records,
        "conclusion": {
            "all_primary_model_conditional_pass": all(s["OLS_iid"]["nonempty"] for s in summaries.values()),
            "all_hac_sensitivity_nonempty": all(s["HAC_sensitivity"]["nonempty"] for s in summaries.values()),
            "interpretation": "Primary model-conditional equivalence holds, but does not survive the HAC sensitivity calculation. Unconditional equivalence is not established; this does not prove actual non-equivalence.",
            "target_configuration_changed": False,
        },
        "verification": {"source_hashes_match": True, "source_values_checked": 840,
                         "saved_coefficients_reproduced": 18, "hac_two_formulas_match": True,
                         "inverse_intervals_match_ellipse_boundary": True,
                         "endpoint_and_inverse_acceptance_agree": True},
    }
    (HERE/"equivalence_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    write_report(result)
    print(json.dumps({"summaries": summaries, "residual_lag1_max": max(r["residual_lag1_correlation"] for r in records),
                      "all_primary_pass": all(s["OLS_iid"]["nonempty"] for s in summaries.values()),
                      "all_sensitivity_pass": all(s["HAC_sensitivity"]["nonempty"] for s in summaries.values())}, indent=2))


def write_report(result):
    lines = [
        "1차 데이터 영점 후보: 불확실성을 반영한 동등성 검토", "",
        "계산 결과: 기본 회귀 가정 아래에서는 좌/중/우 모두 조건부 통과.",
        "촬영 순서상 잔차 연관성을 반영한 HAC 보조 계산에서는 좌/중/우 모두 공통 구간이 사라짐.",
        "따라서 현재 결과로 영점 기준의 동등성을 확정하지 않음. 실제로 비동등하다고 입증한 것도 아님.", "",
        "대상: Y 4구간 x 20장 + theta 2구간 x 100장, QR 3개를 각각 계산한 회귀식 18개.",
        "허용범위: Y +/-0.045 mm, theta +/-0.005 degree.",
        "기존 ZNCC -> ECC 아핀 측정값과 직선을 사용. QR/구간 평균, 이미지 재정합, 새 영상 분석은 하지 않음.",
        "기본 검정: 기울기/절편의 공분산을 포함한 동시 신뢰영역을 역산하는 보수적 동등성 판정.",
        "일반적인 고정 x*의 90% CI TOST를 그대로 실행한 것이 아님.",
        "전체 오류수준 0.05를 분석 설정으로 사용. 장비 허용오차는 사용자가 지정한 값 그대로 유지.",
        "각 회귀식의 Working-Hotelling 신뢰대 + 18개 회귀식에 Bonferroni 보정.",
        "이로써 같은 데이터에서 목표 좌표를 선택하는 과정을 포함. QR 3개를 독립 반복으로 가정하지 않음.",
        "기본 신뢰수준은 선형 평균과 구간 내 독립/정규/등분산 잔차 가정 아래에서만 유효.",
        "보조 계산: 촬영 순서상 잔차 연관을 반영한 HAC 공분산, 기본 검정과 같은 임계값.",
        "HAC는 표본이 작은 이 자료에서 정확한 95% 보장을 주는 검정이 아니므로 민감도 계산으로 구분.",
        "HAC 지연수는 문서 기본식 floor(4*(n/100)^(2/9)): Y=2, theta=4. 결과에 맞춰 조정하지 않음.", "",
        "QR | 점추정만 | 기본 동시 신뢰영역 반영 | HAC 보조 계산 | 둘 다 만족하는 구간",
    ]
    fmt = lambda z: f"{z['low_px']:.9f} ~ {z['high_px']:.9f} (폭 {z['width_px']:.9f} px; {'존재' if z['nonempty'] else '없음'})"
    for marker, summary in result["summaries"].items():
        lines.append(marker+" | "+" | ".join(fmt(summary[k]) for k in ("point_only", "OLS_iid", "HAC_sensitivity", "intersection_of_primary_and_sensitivity")))
    lines += ["", "두 계산 모두 공통 구간의 하한은 3번 Y 음수 촬영, 상한은 4번 theta 양수 촬영이 결정.",
              "HAC 보조 계산에서 요구 하한이 허용 상한보다 커짐. 이는 특정 예시 x*만 실패한 것이 아니라 해당 보조 계산의 공통 구간 자체가 없다는 뜻."]
    lines += ["", "예시 목표 좌표에서의 구간별 역산 결과 (목표 좌표를 설정 파일에 채택한 것은 아님)",
              "QR | 촬영 구간 | 축 | 예시 x* (px) | 점추정 | 기본 동시 신뢰영역 역산 | HAC 역산 | 허용반경"]
    for row in result["fits"]:
        if "illustrative_target_px" not in row:
            continue
        vals = [f"{row['marker']} | {row['group']} | {row['axis']} | {row['illustrative_target_px']:.9f} | {row['point_equivalent_offset']:.9f}"]
        for method in METHODS:
            item = row["methods"][method]
            lo, hi = item["equivalent_offset_bounds"]
            vals.append(f"[{lo:.9f}, {hi:.9f}] {'조건 내' if item['inside_tolerance'] else '조건 미충족'}")
        vals.append(str(row["tolerance"])+" "+row["unit"])
        lines.append(" | ".join(vals))
    lines += ["", "잔차와 모델 가정 점검", "구간 | QR | lag-1 잔차 상관 | DW | 기존 이차식-직선 영점 차이(px)"]
    for row in result["fits"]:
        lines.append(f"{row['group']} | {row['marker']} | {row['residual_lag1_correlation']:.6f} | {row['durbin_watson']:.6f} | {row['existing_quadratic_minus_linear_intercept_px']:.6f}")
    lines += ["", "판정 범위와 한계",
        "기본 계산에서 구간이 남으면 '선형/독립오차 모형 아래 조건부 동등성 충족'으로 해석.",
        "잔차에 촬영 순서상 구조가 있어 이 데이터에 정확한 95% 신뢰보장이 성립한다고 확정하지 않음.",
        "HAC 결과는 이 연관성을 고려했을 때 결론이 유지되는지 보는 근사적인 보조 근거.",
        "HAC는 직선 자체의 편향/곡률을 고치지 않음. 기존 이차식 민감도 수치는 이를 구별하기 위해 첨부.",
        "대상은 고정된 기준 이미지 좌표계에서 회귀 평균이 가리키는 라벨 영점. 새 이미지의 예측구간이나 제어 정밀도 검증이 아님.",
        "기계적 원인 분석은 수행하지 않음. 좌/중/우 목표의 물리적 동시 구현이나 Y/theta 분리 여부도 이번 범위 밖.",
        "", "재현 및 확인",
        "python analysis/zero_equivalence/check_zero_equivalence.py",
        "원본 측정 JSON/메타데이터 SHA256 일치, 840개 원자료 좌표 및 18개 기존 계수 재현 확인.",
        "HAC 공분산 두 수식 대조, 신뢰타원 경계와 역산 구간 대조, 경계값/역산 판정 일치 확인.",
        "", "근거 자료", *result["sources"], ""]
    (HERE/"equivalence_report.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
