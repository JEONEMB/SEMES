// Known-input checks for the line conventions used by line_methods (no image data).
#include "common.hpp"
#include "io_utils.hpp"

#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>

using namespace lm;

static int failures = 0;
static void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}

int main() {
    const double pi = 3.14159265358979323846;
    // Endpoint order must not change the angle (no 180 deg flip).
    Line a = normalize_line(0, 0, 100, 1), b = normalize_line(100, 1, -100, -1);
    check(std::fabs(line_angle_deg(a) - line_angle_deg(b)) < 1e-12, "reversed endpoints give the same angle");
    check(std::fabs(line_angle_deg(a) - std::atan(0.01) * 180 / pi) < 1e-12, "slope 0.01 -> atan(0.01) deg");
    Line c = normalize_line(0, 0, 100, -1);
    check(line_angle_deg(c) < 0, "upward-right line (dy<0) has a negative image angle");
    Line v = normalize_line(5, 5, 0, 10);
    json vj = line_json(v);
    check(std::fabs(line_angle_deg(v) + 90.0) < 1e-12, "vertical line -> -90 deg (range [-90, 90))");
    check(vj["slope_dy_dx"].is_null() && vj["intercept_y_px"].is_null(), "vertical line -> slope/intercept null");
    auto g = vj["general_form_abc"];
    check(std::fabs(g[0].get<double>() * 5 + g[1].get<double>() * 7 + g[2].get<double>()) < 1e-12, "vertical general form passes (5, 7)");
    Line h = normalize_line(10, 20, 1, 0);
    json hj = line_json(h);
    check(std::fabs(hj["slope_dy_dx"].get<double>()) < 1e-15 && std::fabs(hj["intercept_y_px"].get<double>() - 20) < 1e-12, "horizontal y=20");
    // Exact points on y = 700 + 0.0047 x must be recovered by the robust fit.
    std::vector<cv::Point2d> pts;
    for (int x = 24; x <= 3816; x += 2) pts.emplace_back(x, 700 + 0.0047 * x);
    pts.emplace_back(2000, 760);  // outlier
    FitParams fp; fp.inlier_px = 2.0; fp.min_points = 20;
    FitResult f = robust_fit(pts, fp);
    json fj = line_json(f.line);
    check(f.ok && std::fabs(fj["slope_dy_dx"].get<double>() - 0.0047) < 1e-12 && std::fabs(fj["intercept_y_px"].get<double>() - 700) < 1e-8,
          "robust fit recovers slope/intercept with one outlier");
    check(f.n_inlier == (int)pts.size() - 1, "outlier trimmed");
    check(f.irls_runs.size() == 3 && f.irls_runs[0].converged && f.irls_runs[1].converged && f.irls_runs[2].converged,
          "initial and both trimming solves record convergence");
    check(f.irls_runs.back().has_last_change && f.irls_runs.back().iterations >= 2 &&
          f.irls_runs.back().last_angle_change_rad < fp.irls_tol &&
          f.irls_runs.back().last_center_x_change_px < 1e-9 && f.irls_runs.back().last_center_y_change_px < 1e-9,
          "successful solve records final changes against unchanged stopping limits");

    // With no trimming, every point really enters the solve; the final distance mask still excludes the outlier.
    FitParams no_trim = fp; no_trim.trim_iterations = 0;
    FitResult separate = robust_fit(pts, no_trim);
    check(separate.ok && separate.fit_used.back() && !separate.inlier.back() &&
          separate.n_fit_used == (int)pts.size() && separate.n_inlier == (int)pts.size() - 1,
          "actual fit inputs remain distinct from post-fit distance inliers");
    double manual_sum = 0;
    for (const auto& point : pts) {
        double d = signed_distance(separate.line, point.x, point.y); manual_sum += d*d;
    }
    check(std::fabs(separate.rms_fit_used - std::sqrt(manual_sum / pts.size())) < 1e-12 &&
          separate.rms_fit_used > separate.rms_inlier,
          "actual-input RMS uses all actual inputs, not the final distance subset");
    MethodResult published;
    finish_with_points(published, pts, no_trim, "test_points");
    check(published.status == "ok" && published.has_line && published.has_rms &&
          published.intermediates["fit_points_used_flag"].back() == 1 &&
          published.intermediates["fit_points_final_inlier_flag"].back() == 0 &&
          published.intermediates["fit_points_inlier_flag"] == published.intermediates["fit_points_final_inlier_flag"],
          "published masks preserve actual-input provenance and the legacy mask meaning");
    check(published.diagnostics["fit_rms_point_set"] == "actual_last_fit_input" &&
          published.diagnostics["irls_runs"][0]["termination"] == "converged" &&
          std::fabs(published.fit_rms - separate.rms_fit_used) < 1e-12,
          "published RMS and convergence metadata identify their basis");

    // The first iteration has no preceding estimate for the convergence test.
    FitParams one_iteration = fp; one_iteration.irls_max_iter = 1;
    FitResult limited = robust_fit(pts, one_iteration);
    check(!limited.ok && limited.failure_code == "max_iterations" && limited.irls_runs.size() == 1 &&
          limited.irls_runs[0].iterations == 1 && !limited.irls_runs[0].has_last_change,
          "iteration-limit exit is not reported as convergence");
    finish_with_points(published, pts, one_iteration, "test_points");
    check(published.status == "failed" && !published.has_line && !published.has_rms && !published.has_endpoints &&
          published.diagnostics["fit_failure_code"] == "max_iterations" &&
          published.diagnostics["irls_runs"][0]["last_angle_change_rad"].is_null() &&
          !published.diagnostics.contains("fit_rms_fit_used_px"),
          "unconfirmed convergence publishes no normal line or stale RMS on reused output");

    std::vector<cv::Point2d> bad = pts;
    bad[0].x = std::numeric_limits<double>::quiet_NaN();
    FitResult invalid = robust_fit(bad, fp);
    check(!invalid.ok && invalid.failure_code == "invalid_input", "NaN coordinate is rejected, not silently dropped");
    bad[0].x = std::numeric_limits<double>::infinity();
    check(!robust_fit(bad, fp).ok, "infinite coordinate is rejected");
    std::vector<char> exclude_bad(bad.size(), 1); exclude_bad[0] = 0;
    check(huber_line(bad, &exclude_bad, fp).termination == "invalid_input", "nonfinite excluded input is still an input error");
    std::vector<char> wrong_size(1, 1);
    check(huber_line(pts, &wrong_size, fp).termination == "invalid_input", "mask-length mismatch is rejected");
    check(robust_fit(std::vector<cv::Point2d>(30, cv::Point2d(10,20)), fp).failure_code == "degenerate_geometry",
          "coincident points do not become a horizontal line");
    std::vector<cv::Point2d> isotropic;
    for (int i=0;i<5;++i) { isotropic.emplace_back(-1,-1); isotropic.emplace_back(-1,1); isotropic.emplace_back(1,-1); isotropic.emplace_back(1,1); }
    check(robust_fit(isotropic, fp).failure_code == "degenerate_geometry", "exactly isotropic scatter has no unique line direction");
    std::vector<cv::Point2d> overflow(30, cv::Point2d(1e308,1e308));
    check(robust_fit(overflow, fp).failure_code == "numerical_error", "overflow in finite input arithmetic is rejected");
    bool threw = false;
    try { normalize_line(0,0,0,0); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "zero direction does not silently normalize to horizontal");

    // Symmetric parallel rows fit y=0, yet neither row lies within 0.5px of it.
    std::vector<cv::Point2d> two_rows;
    for (int x=-20;x<=20;++x) { two_rows.emplace_back(x,-1); two_rows.emplace_back(x,1); }
    FitParams narrow = no_trim; narrow.inlier_px = 0.5;
    FitResult few_final = robust_fit(two_rows, narrow);
    check(!few_final.ok && few_final.failure_code == "too_few_final_inliers" && few_final.n_inlier == 0 &&
          few_final.irls_runs[0].converged, "final minimum count is checked even after a converged solve");
    finish_with_points(published, two_rows, narrow, "test_points");
    check(published.status == "failed" && !published.has_line && !published.has_endpoints &&
          published.diagnostics["fit_failure_code"] == "too_few_final_inliers",
          "too few final inliers cannot publish a line or artificial endpoints");
    check(json::parse(published.diagnostics.dump()) == published.diagnostics, "failure diagnostics serialize without nonfinite numeric outputs");
    std::vector<cv::Point2d> duplicate_final(20, cv::Point2d(0,0));
    duplicate_final.emplace_back(-10,-2); duplicate_final.emplace_back(-10,2);
    duplicate_final.emplace_back(10,-2); duplicate_final.emplace_back(10,2);
    FitParams duplicate_params = no_trim; duplicate_params.inlier_px = 1.0;
    FitResult collapsed = robust_fit(duplicate_final, duplicate_params);
    check(!collapsed.ok && collapsed.n_inlier == 20 && collapsed.failure_code == "degenerate_final_support",
          "enough final points at one location are not valid line support");
    FitParams bad_params = fp; bad_params.irls_tol = std::numeric_limits<double>::quiet_NaN();
    check(robust_fit(pts, bad_params).failure_code == "invalid_input", "invalid convergence parameter is rejected");
    check(robust_fit(std::vector<cv::Point2d>(2,cv::Point2d(0,0)), fp).failure_code == "too_few_points",
          "existing minimum point count is enforced");
    // Signed distance sign: +normal is (-dy, dx), i.e. +y for a horizontal line.
    check(signed_distance(h, 0, 25) > 0, "signed distance positive below a horizontal line");
    std::string abc = "abc";
    check(sha256_hex(std::vector<uint8_t>(abc.begin(), abc.end())) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "sha256('abc')");
    std::printf("%d failure(s)\n", failures);
    return failures ? 1 : 0;
}
