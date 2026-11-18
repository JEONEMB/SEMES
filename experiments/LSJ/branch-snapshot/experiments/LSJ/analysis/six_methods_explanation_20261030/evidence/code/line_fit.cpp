// Line representation, angle convention and the common robust line fit.
#include "common.hpp"
#include "io_utils.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

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
    Line l;
    l.px = px; l.py = py;
    if (n <= 0) { l.dx = 1; l.dy = 0; return l; }
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

Line huber_line(const std::vector<cv::Point2d>& pts, const std::vector<char>* use, const FitParams& p) {
    std::vector<double> w(pts.size(), 1.0);
    if (use)
        for (size_t i = 0; i < pts.size(); ++i) w[i] = (*use)[i] ? 1.0 : 0.0;
    std::vector<char> active(pts.size());
    for (size_t i = 0; i < pts.size(); ++i) active[i] = w[i] > 0;
    // Shift for conditioning; the result is shifted back.
    double ox = 0, oy = 0; int n = 0;
    for (size_t i = 0; i < pts.size(); ++i) if (active[i]) { ox += pts[i].x; oy += pts[i].y; ++n; }
    if (n == 0) return Line();
    ox /= n; oy /= n;
    Line line;
    double prev_angle = std::numeric_limits<double>::quiet_NaN(), prev_cx = 0, prev_cy = 0;
    for (int it = 0; it < p.irls_max_iter; ++it) {
        double sw = 0, sx = 0, sy = 0;
        for (size_t i = 0; i < pts.size(); ++i) {
            if (!active[i]) continue;
            sw += w[i]; sx += w[i] * (pts[i].x - ox); sy += w[i] * (pts[i].y - oy);
        }
        double cx = sx / sw, cy = sy / sw;
        double sxx = 0, syy = 0, sxy = 0;
        for (size_t i = 0; i < pts.size(); ++i) {
            if (!active[i]) continue;
            double u = pts[i].x - ox - cx, v = pts[i].y - oy - cy;
            sxx += w[i] * u * u; syy += w[i] * v * v; sxy += w[i] * u * v;
        }
        double angle = 0.5 * std::atan2(2 * sxy, sxx - syy);
        line = normalize_line(cx + ox, cy + oy, std::cos(angle), std::sin(angle));
        bool converged = it > 0 && std::fabs(angle - prev_angle) < p.irls_tol &&
                         std::fabs(cx - prev_cx) < 1e-9 && std::fabs(cy - prev_cy) < 1e-9;
        prev_angle = angle; prev_cx = cx; prev_cy = cy;
        if (converged) break;
        for (size_t i = 0; i < pts.size(); ++i) {
            if (!active[i]) continue;
            double r = std::fabs(signed_distance(line, pts[i].x, pts[i].y));
            w[i] = r <= p.huber_c ? 1.0 : p.huber_c / r;
        }
    }
    return line;
}

FitResult robust_fit(const std::vector<cv::Point2d>& pts, const FitParams& p) {
    FitResult f;
    f.n_all = static_cast<int>(pts.size());
    if (f.n_all < p.min_points) {
        f.reason = "too few points for fit: " + std::to_string(f.n_all) + " < " + std::to_string(p.min_points);
        return f;
    }
    Line line = huber_line(pts, nullptr, p);
    f.initial_angle_deg = line_angle_deg(line);
    std::vector<char> in(pts.size(), 1);
    for (int it = 0; it < p.trim_iterations; ++it) {
        int count = 0;
        for (size_t i = 0; i < pts.size(); ++i) {
            in[i] = std::fabs(signed_distance(line, pts[i].x, pts[i].y)) <= p.inlier_px;
            count += in[i];
        }
        if (count < p.min_points) {
            f.reason = "too few inliers after trimming: " + std::to_string(count);
            return f;
        }
        line = huber_line(pts, &in, p);
    }
    double s_in = 0, s_all = 0, mx = 0; int n_in = 0;
    for (size_t i = 0; i < pts.size(); ++i) {
        double r = signed_distance(line, pts[i].x, pts[i].y);
        s_all += r * r;
        in[i] = std::fabs(r) <= p.inlier_px;
        if (in[i]) { s_in += r * r; ++n_in; mx = std::max(mx, std::fabs(r)); }
    }
    f.ok = true;
    f.line = line;
    f.inlier = in;
    f.n_inlier = n_in;
    f.rms_inlier = n_in ? std::sqrt(s_in / n_in) : 0;
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
    r.support_type = support_type;
    FitResult f = robust_fit(pts, fp);
    r.diagnostics["fit_input_points"] = f.n_all;
    if (!f.ok) {
        r.fail("fit_failed", f.reason);
        return;
    }
    r.status = "ok";
    r.has_line = true;
    r.line = f.line;
    r.n_selected = f.n_inlier;
    r.has_rms = true;
    r.fit_rms = f.rms_inlier;
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
    r.has_endpoints = true;
    r.ex1 = f.line.px + tmin * f.line.dx; r.ey1 = f.line.py + tmin * f.line.dy;
    r.ex2 = f.line.px + tmax * f.line.dx; r.ey2 = f.line.py + tmax * f.line.dy;
    // Inlier flags let the report colour fitted support points.
    json flags = json::array();
    for (char c : f.inlier) flags.push_back(c ? 1 : 0);
    r.intermediates["fit_points_inlier_flag"] = flags;
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
