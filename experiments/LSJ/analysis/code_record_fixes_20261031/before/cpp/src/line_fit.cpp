// Line representation, angle convention and the common robust line fit.
#include "common.hpp"
#include "io_utils.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace lm {

static const double kPi = 3.14159265358979323846;

cv::Rect TargetSpec::detector_region(const cv::Size& image) const {
    int ax = x0 - margin, bx = x1 + margin, ay = y0 - margin, by = y1 + margin;
    if (kind == "qr") {
        ax = roi.x - margin; bx = roi.x + roi.width - 1 + margin;
        ay = roi.y - margin; by = roi.y + roi.height - 1 + margin;
    }
    ax = std::max(ax, 0); ay = std::max(ay, 0);
    bx = std::min(bx, image.width - 1); by = std::min(by, image.height - 1);
    return cv::Rect(ax, ay, bx - ax + 1, by - ay + 1);
}

Line normalize_line(double px, double py, double dx, double dy) {
    double n = std::hypot(dx, dy);
    if (!std::isfinite(px) || !std::isfinite(py) || !std::isfinite(n) || n <= 0)
        throw std::invalid_argument("line requires finite coordinates and a nonzero finite direction");
    Line l;
    l.px = px; l.py = py;
    dx /= n; dy /= n;
    // x-increasing direction; a vertical line points to -y so the angle is -90 (range [-90, 90)).
    if (dx < 0 || (dx == 0 && dy > 0)) { dx = -dx; dy = -dy; }
    if (dx == 0) dy = -1;
    l.dx = dx; l.dy = dy;
    return l;
}

double line_angle_deg(const Line& l) {
    double a = std::atan2(l.dy, l.dx) * 180.0 / kPi;
    if (a >= 90.0) a -= 180.0;
    if (a < -90.0) a += 180.0;
    return a;
}

double signed_distance(const Line& l, double x, double y) {
    return -(x - l.px) * l.dy + (y - l.py) * l.dx;
}

json line_json(const Line& l) {
    json j;
    j["point_xy"] = {l.px, l.py};
    j["direction_xy"] = {l.dx, l.dy};
    if (std::fabs(l.dx) > 1e-12) {
        double m = l.dy / l.dx;
        j["slope_dy_dx"] = m;
        j["intercept_y_px"] = l.py - m * l.px;
    } else {
        j["slope_dy_dx"] = nullptr;
        j["intercept_y_px"] = nullptr;
        j["slope_null_reason"] = "vertical line: dx=0";
    }
    // a*x + b*y + c = 0 with (a, b) the unit normal (-dy, dx).
    double a = -l.dy, b = l.dx, c = -(a * l.px + b * l.py);
    j["general_form_abc"] = {a, b, c};
    j["image_angle_deg"] = line_angle_deg(l);
    return j;
}

json line_draw_endpoints(const Line& l, double xa, double xb) {
    if (std::fabs(l.dx) < 1e-12) return nullptr;
    double m = l.dy / l.dx, b = l.py - m * l.px;
    return json::array({json::array({xa, m * xa + b}), json::array({xb, m * xb + b})});
}

static bool finite_line(const Line& l) {
    if (!std::isfinite(l.px) || !std::isfinite(l.py) || !std::isfinite(l.dx) || !std::isfinite(l.dy)) return false;
    double n = std::hypot(l.dx, l.dy);
    if (!std::isfinite(n) || n <= 0 || !std::isfinite(l.dy * l.px - l.dx * l.py)) return false;
    if (std::fabs(l.dx) > 1e-12) {
        double m = l.dy / l.dx;
        if (!std::isfinite(m) || !std::isfinite(l.py - m * l.px)) return false;
    }
    return true;
}

IrlsResult huber_line(const std::vector<cv::Point2d>& pts, const std::vector<char>* use, const FitParams& p) {
    IrlsResult result;
    auto fail = [&](const char* code, const char* reason) {
        result.termination = code; result.reason = reason; return result;
    };
    if ((use && use->size() != pts.size()) || p.irls_max_iter < 1 ||
        !std::isfinite(p.irls_tol) || p.irls_tol <= 0 || !std::isfinite(p.huber_c) || p.huber_c <= 0)
        return fail("invalid_input", "invalid IRLS parameters or point-mask length");
    // Invalid coordinates are errors, not points to silently discard.
    for (const auto& q : pts)
        if (!std::isfinite(q.x) || !std::isfinite(q.y))
            return fail("invalid_input", "non-finite input point");
    std::vector<double> w(pts.size(), 1.0);
    if (use)
        for (size_t i = 0; i < pts.size(); ++i) w[i] = (*use)[i] ? 1.0 : 0.0;
    std::vector<char> active(pts.size());
    for (size_t i = 0; i < pts.size(); ++i) active[i] = w[i] > 0;
    // Shift for conditioning; the result is shifted back.
    double ox = 0, oy = 0; int n = 0;
    for (size_t i = 0; i < pts.size(); ++i) if (active[i]) { ox += pts[i].x; oy += pts[i].y; ++n; }
    result.n_used = n;
    if (n < 2) return fail("degenerate_geometry", "fewer than two active points");
    ox /= n; oy /= n;
    if (!std::isfinite(ox) || !std::isfinite(oy))
        return fail("numerical_error", "non-finite initial centroid");
    Line line;
    double prev_angle = std::numeric_limits<double>::quiet_NaN(), prev_cx = 0, prev_cy = 0;
    for (int it = 0; it < p.irls_max_iter; ++it) {
        result.iterations = it + 1;
        double sw = 0, sx = 0, sy = 0;
        for (size_t i = 0; i < pts.size(); ++i) {
            if (!active[i]) continue;
            sw += w[i]; sx += w[i] * (pts[i].x - ox); sy += w[i] * (pts[i].y - oy);
        }
        if (!std::isfinite(sw) || sw <= 0 || !std::isfinite(sx) || !std::isfinite(sy))
            return fail("numerical_error", "invalid sum of weights or weighted coordinates");
        double cx = sx / sw, cy = sy / sw;
        if (!std::isfinite(cx) || !std::isfinite(cy))
            return fail("numerical_error", "non-finite weighted centroid");
        double sxx = 0, syy = 0, sxy = 0;
        for (size_t i = 0; i < pts.size(); ++i) {
            if (!active[i]) continue;
            double u = pts[i].x - ox - cx, v = pts[i].y - oy - cy;
            sxx += w[i] * u * u; syy += w[i] * v * v; sxy += w[i] * u * v;
        }
        double trace = sxx + syy, anisotropy = std::hypot(2 * sxy, sxx - syy);
        if (!std::isfinite(trace) || !std::isfinite(anisotropy) || !std::isfinite(sxy))
            return fail("numerical_error", "non-finite weighted scatter matrix");
        // Exact degeneracy only: no new empirical geometric tolerance is introduced.
        if (trace <= 0 || anisotropy == 0)
            return fail("degenerate_geometry", "coincident points or no unique principal line direction");
        double angle = 0.5 * std::atan2(2 * sxy, sxx - syy);
        if (!std::isfinite(cx + ox) || !std::isfinite(cy + oy))
            return fail("numerical_error", "non-finite line position");
        line = normalize_line(cx + ox, cy + oy, std::cos(angle), std::sin(angle));
        if (!finite_line(line)) return fail("numerical_error", "non-finite line representation");
        result.line = line; result.has_line = true;
        result.has_last_change = it > 0;
        if (result.has_last_change) {
            result.last_angle_change_rad = std::fabs(angle - prev_angle);
            result.last_center_x_change_px = std::fabs(cx - prev_cx);
            result.last_center_y_change_px = std::fabs(cy - prev_cy);
            if (!std::isfinite(result.last_angle_change_rad) || !std::isfinite(result.last_center_x_change_px) ||
                !std::isfinite(result.last_center_y_change_px)) {
                result.has_last_change = false;
                return fail("numerical_error", "non-finite iteration change");
            }
        }
        bool converged = it > 0 && result.last_angle_change_rad < p.irls_tol &&
                         result.last_center_x_change_px < 1e-9 && result.last_center_y_change_px < 1e-9;
        prev_angle = angle; prev_cx = cx; prev_cy = cy;
        if (converged) {
            result.converged = true; result.termination = "converged";
            return result;
        }
        for (size_t i = 0; i < pts.size(); ++i) {
            if (!active[i]) continue;
            double r = std::fabs(signed_distance(line, pts[i].x, pts[i].y));
            if (!std::isfinite(r)) return fail("numerical_error", "non-finite point residual");
            w[i] = r <= p.huber_c ? 1.0 : p.huber_c / r;
        }
    }
    return fail("max_iterations", "IRLS iteration limit reached; convergence not confirmed");
}

FitResult robust_fit(const std::vector<cv::Point2d>& pts, const FitParams& p) {
    FitResult f;
    f.n_all = static_cast<int>(pts.size());
    if (p.min_points < 2 || p.trim_iterations < 0 || !std::isfinite(p.inlier_px) || p.inlier_px < 0) {
        f.failure_code = "invalid_input"; f.reason = "invalid minimum count, trimming count or distance";
        return f;
    }
    if (f.n_all < p.min_points) {
        f.failure_code = "too_few_points";
        f.reason = "too few points for fit: " + std::to_string(f.n_all) + " < " + std::to_string(p.min_points);
        return f;
    }
    f.fit_used.assign(pts.size(), 1); f.n_fit_used = f.n_all;
    auto solve = [&](const std::vector<char>* mask) {
        f.irls_runs.push_back(huber_line(pts, mask, p));
        const auto& run = f.irls_runs.back();
        if (!run.converged) { f.failure_code = run.termination; f.reason = run.reason; return false; }
        f.line = run.line;
        return true;
    };
    if (!solve(nullptr)) return f;
    Line line = f.line;
    f.initial_angle_deg = line_angle_deg(line);
    std::vector<char> in(pts.size(), 1);
    for (int it = 0; it < p.trim_iterations; ++it) {
        int count = 0;
        for (size_t i = 0; i < pts.size(); ++i) {
            double d = signed_distance(line, pts[i].x, pts[i].y);
            if (!std::isfinite(d)) {
                f.failure_code = "numerical_error"; f.reason = "non-finite trimming residual"; return f;
            }
            in[i] = std::fabs(d) <= p.inlier_px;
            count += in[i];
        }
        if (count < p.min_points) {
            f.failure_code = "too_few_trim_inliers";
            f.reason = "too few inliers after trimming: " + std::to_string(count);
            return f;
        }
        f.fit_used = in; f.n_fit_used = count;
        if (!solve(&f.fit_used)) return f;
        line = f.line;
    }
    double s_in = 0, s_all = 0, s_fit = 0, mx = 0; int n_in = 0;
    for (size_t i = 0; i < pts.size(); ++i) {
        double r = signed_distance(line, pts[i].x, pts[i].y);
        if (!std::isfinite(r) || !std::isfinite(r * r)) {
            f.failure_code = "numerical_error"; f.reason = "non-finite final residual"; return f;
        }
        s_all += r * r;
        if (f.fit_used[i]) s_fit += r * r;
        in[i] = std::fabs(r) <= p.inlier_px;
        if (in[i]) { s_in += r * r; ++n_in; mx = std::max(mx, std::fabs(r)); }
    }
    f.line = line;
    f.inlier = in;
    f.n_inlier = n_in;
    if (!finite_line(line) || !std::isfinite(s_in) || !std::isfinite(s_all) || !std::isfinite(s_fit)) {
        f.failure_code = "numerical_error"; f.reason = "non-finite final line or residual summary"; return f;
    }
    if (n_in < p.min_points) {
        f.failure_code = "too_few_final_inliers";
        f.reason = "too few points after final distance classification: " + std::to_string(n_in) + " < " + std::to_string(p.min_points);
        return f;
    }
    // A large duplicate count is not geometric support for a line segment.
    double support_min = std::numeric_limits<double>::infinity(), support_max = -support_min;
    for (size_t i = 0; i < pts.size(); ++i) if (in[i]) {
        double projection = (pts[i].x - line.px) * line.dx + (pts[i].y - line.py) * line.dy;
        if (!std::isfinite(projection)) {
            f.failure_code = "numerical_error"; f.reason = "non-finite final support extent"; return f;
        }
        support_min = std::min(support_min, projection); support_max = std::max(support_max, projection);
    }
    if (support_max <= support_min) {
        f.failure_code = "degenerate_final_support";
        f.reason = "final distance inliers have zero extent along the fitted line"; return f;
    }
    f.ok = true;
    f.rms_inlier = n_in ? std::sqrt(s_in / n_in) : 0;
    f.rms_fit_used = std::sqrt(s_fit / f.n_fit_used);
    f.rms_all = std::sqrt(s_all / pts.size());
    f.max_abs_inlier = mx;
    return f;
}

FitParams fit_params_for(const json& common, const std::string& kind) {
    const json& f = common.at("fit");
    FitParams p;
    p.huber_c = f.at("huber_c_px").get<double>();
    p.irls_max_iter = f.at("irls_max_iter").get<int>();
    p.irls_tol = f.at("irls_tol").get<double>();
    p.trim_iterations = f.at("trim_iterations").get<int>();
    p.inlier_px = f.at("inlier_px").at(kind).get<double>();
    p.min_points = f.at("min_points").at(kind).get<int>();
    return p;
}

void finish_with_points(MethodResult& r, const std::vector<cv::Point2d>& pts, const FitParams& fp,
                        const std::string& support_type) {
    // Clear published outputs on every attempt, including reused MethodResult objects.
    r.status = "failed"; r.has_line = false; r.has_endpoints = false; r.has_rms = false;
    r.n_selected = 0; r.failure_category.clear(); r.failure_reason.clear();
    r.rms_null_reason.clear();
    for (const char* key : {"fit_rms_point_set", "fit_rms_fit_used_px", "fit_rms_final_inlier_px",
                           "fit_rms_all_input_px", "fit_max_abs_inlier_residual_px", "fit_inlier_points",
                           "n_selected_point_set", "huber_all_points_angle_before_trim_deg"}) r.diagnostics.erase(key);
    r.support_type = support_type;
    FitResult f = robust_fit(pts, fp);
    r.diagnostics["fit_input_points"] = f.n_all;
    r.diagnostics["fit_used_points"] = f.n_fit_used;
    r.diagnostics["fit_final_inlier_points"] = f.n_inlier;
    r.diagnostics["fit_failure_code"] = f.ok ? json(nullptr) : json(f.failure_code);
    r.diagnostics["fit_status"] = f.ok ? "converged_and_valid" : "failed";
    r.diagnostics["irls_limits"] = {{"max_iterations", fp.irls_max_iter}, {"angle_change_rad", fp.irls_tol},
                                    {"center_x_change_px", 1e-9}, {"center_y_change_px", 1e-9}};
    json runs = json::array();
    for (size_t i = 0; i < f.irls_runs.size(); ++i) {
        const auto& run = f.irls_runs[i];
        runs.push_back({{"stage", i == 0 ? std::string("initial") : "trim_" + std::to_string(i)},
            {"termination", run.termination}, {"reason", run.reason}, {"iterations", run.iterations},
            {"n_used", run.n_used}, {"converged", run.converged},
            {"last_angle_change_rad", run.has_last_change ? json(run.last_angle_change_rad) : json(nullptr)},
            {"last_center_x_change_px", run.has_last_change ? json(run.last_center_x_change_px) : json(nullptr)},
            {"last_center_y_change_px", run.has_last_change ? json(run.last_center_y_change_px) : json(nullptr)}});
    }
    r.diagnostics["irls_runs"] = runs;
    auto flags_json = [](const std::vector<char>& flags) {
        json out = json::array(); for (char c : flags) out.push_back(c ? 1 : 0); return out;
    };
    r.intermediates["fit_points_used_flag"] = flags_json(f.fit_used);
    r.intermediates["fit_points_final_inlier_flag"] = flags_json(f.inlier);
    // Compatibility alias retains its historical final-classification meaning.
    r.intermediates["fit_points_inlier_flag"] = flags_json(f.inlier);
    r.diagnostics["fit_mask_note"] = "used_flag is the last solve input; final_inlier_flag and legacy inlier_flag are post-fit distance classifications";
    if (!f.ok) {
        r.fail("fit_failed", f.reason);
        return;
    }
    r.line = f.line;
    r.n_selected = f.n_inlier;
    r.fit_rms = f.rms_fit_used;
    r.diagnostics["n_selected_point_set"] = "final_distance_inliers";
    r.diagnostics["fit_rms_point_set"] = "actual_last_fit_input";
    r.diagnostics["fit_rms_fit_used_px"] = f.rms_fit_used;
    r.diagnostics["fit_rms_final_inlier_px"] = f.rms_inlier;
    r.diagnostics["fit_inlier_points"] = f.n_inlier;
    r.diagnostics["fit_rms_all_input_px"] = f.rms_all;
    r.diagnostics["fit_max_abs_inlier_residual_px"] = f.max_abs_inlier;
    r.diagnostics["huber_all_points_angle_before_trim_deg"] = f.initial_angle_deg;
    double tmin = 1e300, tmax = -1e300;
    for (size_t i = 0; i < pts.size(); ++i) {
        if (!f.inlier[i]) continue;
        double t = (pts[i].x - f.line.px) * f.line.dx + (pts[i].y - f.line.py) * f.line.dy;
        tmin = std::min(tmin, t); tmax = std::max(tmax, t);
    }
    r.ex1 = f.line.px + tmin * f.line.dx; r.ey1 = f.line.py + tmin * f.line.dy;
    r.ex2 = f.line.px + tmax * f.line.dx; r.ey2 = f.line.py + tmax * f.line.dy;
    if (!std::isfinite(r.ex1) || !std::isfinite(r.ey1) || !std::isfinite(r.ex2) || !std::isfinite(r.ey2)) {
        r.diagnostics["fit_failure_code"] = "numerical_error";
        r.diagnostics["fit_status"] = "failed";
        r.fail("fit_failed", "non-finite final drawing endpoints"); return;
    }
    r.status = "ok"; r.has_line = true; r.has_rms = true; r.has_endpoints = true;
}

double bilinear(const cv::Mat& f, double x, double y) {
    int x0 = (int)std::floor(x), y0 = (int)std::floor(y);
    x0 = std::max(0, std::min(x0, f.cols - 2));
    y0 = std::max(0, std::min(y0, f.rows - 2));
    double ax = x - x0, ay = y - y0;
    auto at = [&](int yy, int xx) {
        return f.depth() == CV_64F ? f.at<double>(yy, xx) : (double)f.at<float>(yy, xx);
    };
    return (1 - ay) * ((1 - ax) * at(y0, x0) + ax * at(y0, x0 + 1)) + ay * ((1 - ax) * at(y0 + 1, x0) + ax * at(y0 + 1, x0 + 1));
}

double rnd(double v, int digits) {
    double s = std::pow(10.0, digits);
    return std::round(v * s) / s;
}

json round_pt(double x, double y, int digits) { return json::array({rnd(x, digits), rnd(y, digits)}); }

std::string Sink::save_png(const std::string& name, const cv::Mat& image, const cv::Point& origin, json& images) {
    std::string file = prefix + "__" + name + ".png";
    write_png(join_path(abs_dir, file), image);
    std::string rel = rel_dir + "/" + file;
    images[name] = {{"path", rel}, {"origin_xy", {origin.x, origin.y}}, {"size_wh", {image.cols, image.rows}}};
    return rel;
}

}  // namespace lm
