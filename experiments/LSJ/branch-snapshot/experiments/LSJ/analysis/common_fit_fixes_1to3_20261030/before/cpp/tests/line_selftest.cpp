// Known-input checks for the line conventions used by line_methods (no image data).
#include "common.hpp"
#include "io_utils.hpp"

#include <cmath>
#include <cstdio>

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
    // Signed distance sign: +normal is (-dy, dx), i.e. +y for a horizontal line.
    check(signed_distance(h, 0, 25) > 0, "signed distance positive below a horizontal line");
    std::string abc = "abc";
    check(sha256_hex(std::vector<uint8_t>(abc.begin(), abc.end())) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "sha256('abc')");
    std::printf("%d failure(s)\n", failures);
    return failures ? 1 : 0;
}
