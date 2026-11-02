// QR-only methods.
//   Q10 Otsu binarisation -> outer contour of the bright QR square -> top-side contour points -> line fit
//   Q11 outer-square corner detection (Shi-Tomasi) -> cornerSubPix -> line through the two top corners
//   Q12 finder patterns -> cornerSubPix -> diagonal-intersection centres -> line through A and B
//       (port of analysis/reference_qr_finder_20261029/measure_finders.py)
#include "common.hpp"
#include "effective_settings.hpp"
#include "methods.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>

namespace lm {

namespace {
const char* kTwoPointRms = "two-point line: residual is 0 by construction and is not accuracy evidence";

bool touches_border(const cv::Rect& b, const cv::Size& s) {
    return b.x <= 0 || b.y <= 0 || b.x + b.width >= s.width || b.y + b.height >= s.height;
}

void two_point_result(MethodResult& r, const cv::Point2d& a, const cv::Point2d& b, const std::string& support) {
    r.status = "ok";
    r.has_line = true;
    r.line = normalize_line(a.x, a.y, b.x - a.x, b.y - a.y);
    r.has_endpoints = true;
    r.ex1 = a.x; r.ey1 = a.y; r.ex2 = b.x; r.ey2 = b.y;
    r.n_selected = 2;
    r.support_type = support;
    r.has_rms = false;
    r.rms_null_reason = kTwoPointRms;
}
}  // namespace

MethodResult method_qr_contour(MethodContext& c) {
    validate_method_parameters("Q10",c.params);
    MethodResult r;
    const TargetSpec& t = c.target;
    const json& p = c.params;
    cv::Mat crop = c.gray(t.roi).clone(), bin;
    double thr = cv::threshold(crop, bin, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    c.sink.save_png("otsu_binary", bin, t.roi.tl(), r.images);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bin.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    r.diagnostics["otsu_threshold"] = thr;
    r.n_candidates = (int)contours.size();
    int best = -1; double best_area = 0;
    json cj = json::array();
    for (size_t i = 0; i < contours.size(); ++i) {
        cv::Rect b = cv::boundingRect(contours[i]);
        double area = std::fabs(cv::contourArea(contours[i]));
        bool border = touches_border(b, crop.size());
        cj.push_back({{"index", (int)i}, {"area", area}, {"bbox_xywh_original", {b.x + t.roi.x, b.y + t.roi.y, b.width, b.height}}, {"touches_roi_border", border}});
        if (!border && area > best_area) { best_area = area; best = (int)i; }
    }
    r.intermediates["external_contours"] = cj;
    if (best < 0) { r.fail("detection_failed", "no external bright contour inside the ROI (all touch the ROI border)"); return r; }
    const auto& ct = contours[best];
    std::vector<cv::Point> poly;
    cv::approxPolyDP(ct, poly, p.at("approx_eps_frac").get<double>() * cv::arcLength(ct, true), true);
    json pj = json::array();
    for (auto& q : poly) pj.push_back({q.x + t.roi.x, q.y + t.roi.y});
    r.intermediates["polygon_vertices"] = pj;
    r.intermediates["selected_contour_index"] = best;
    json full = json::array();
    for (auto& q : ct) full.push_back({q.x + t.roi.x, q.y + t.roi.y});
    r.intermediates["selected_contour"] = full;
    if (poly.size() != 4) {
        r.fail("corner_not_identified", "approxPolyDP gave " + std::to_string(poly.size()) + " vertices, expected 4");
        return r;
    }
    auto tl = *std::min_element(poly.begin(), poly.end(), [](const cv::Point& a, const cv::Point& b) { return a.x + a.y < b.x + b.y; });
    auto tr = *std::max_element(poly.begin(), poly.end(), [](const cv::Point& a, const cv::Point& b) { return a.x - a.y < b.x - b.y; });
    auto nearest = [&](const cv::Point& v) {
        int k = 0; double d = 1e300;
        for (size_t i = 0; i < ct.size(); ++i) { double e = cv::norm(ct[i] - v); if (e < d) { d = e; k = (int)i; } }
        return k;
    };
    int itl = nearest(tl), itr = nearest(tr), n = (int)ct.size();
    // Two contour paths join TL and TR; the top side is the one with the smaller mean y.
    std::vector<cv::Point> p1, p2;
    for (int k = itl;; k = (k + 1) % n) { p1.push_back(ct[k]); if (k == itr) break; }
    for (int k = itl;; k = (k - 1 + n) % n) { p2.push_back(ct[k]); if (k == itr) break; }
    auto mean_y = [](const std::vector<cv::Point>& v) { double s = 0; for (auto& q : v) s += q.y; return s / std::max<size_t>(1, v.size()); };
    const auto& top = mean_y(p1) < mean_y(p2) ? p1 : p2;
    double trim = p.at("corner_trim_frac").get<double>() * (tr.x - tl.x);
    std::vector<cv::Point2d> pts;
    json tj = json::array();
    for (auto& q : top) {
        bool keep = q.x >= tl.x + trim && q.x <= tr.x - trim;
        tj.push_back({q.x + t.roi.x, q.y + t.roi.y, keep ? 1 : 0});
        if (keep) pts.emplace_back(q.x + t.roi.x, q.y + t.roi.y);
    }
    r.intermediates["top_run_fields"] = {"x", "y", "used_after_corner_trim"};
    r.intermediates["top_run"] = tj;
    r.diagnostics["top_corners_from_polygon"] = {{tl.x + t.roi.x, tl.y + t.roi.y}, {tr.x + t.roi.x, tr.y + t.roi.y}};
    r.diagnostics["contour_offset_note"] = "contour points are centres of the outermost bright pixels; the physical boundary lies about 0.5 px above (constant offset, no slope effect)";
    json fitpts = json::array();
    for (auto& q : pts) fitpts.push_back(round_pt(q.x, q.y, 1));
    r.intermediates["fit_points"] = fitpts;
    finish_with_points(r, pts, c.fit, "contour_integer_pixels: top side of the Otsu outer contour, corners trimmed");
    return r;
}

MethodResult method_qr_corners(MethodContext& c) {
    MethodResult r;
    const TargetSpec& t = c.target;
    const json& p = c.params;
    cv::Mat crop = c.gray(t.roi).clone();
    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(crop, corners, p.at("max_corners").get<int>(), p.at("quality_level").get<double>(),
                            p.at("min_distance").get<double>(), cv::noArray(), p.at("block_size").get<int>(),
                            p.at("use_harris").get<bool>());
    r.n_candidates = (int)corners.size();
    const int d = p.at("quad_offset_px").get<int>(), q = p.at("quad_half_px").get<int>();
    const double minc = p.at("min_quad_contrast").get<double>();
    auto box = [&](double cx, double cy) {
        int x0 = (int)std::lround(cx) - q, y0 = (int)std::lround(cy) - q;
        if (x0 < 0 || y0 < 0 || x0 + 2 * q + 1 > crop.cols || y0 + 2 * q + 1 > crop.rows) return -1.0;
        return cv::mean(crop(cv::Rect(x0, y0, 2 * q + 1, 2 * q + 1)))[0];
    };
    json cj = json::array();
    int itl = -1, itr = -1;
    std::vector<std::string> kind(corners.size());
    for (size_t i = 0; i < corners.size(); ++i) {
        double x = corners[i].x, y = corners[i].y;
        double ul = box(x - d, y - d), ur = box(x + d, y - d), ll = box(x - d, y + d), lr = box(x + d, y + d);
        std::string k = "other";
        if (ul >= 0 && ur >= 0 && ll >= 0 && lr >= 0) {
            if (lr - std::max({ul, ur, ll}) >= minc) k = "outer_top_left_pattern";
            else if (ll - std::max({ul, ur, lr}) >= minc) k = "outer_top_right_pattern";
        } else k = "too_close_to_roi_border";
        kind[i] = k;
        if (k == "outer_top_left_pattern" && (itl < 0 || y < corners[itl].y - 0.5 || (std::fabs(y - corners[itl].y) <= 0.5 && x < corners[itl].x))) itl = (int)i;
        if (k == "outer_top_right_pattern" && (itr < 0 || y < corners[itr].y - 0.5 || (std::fabs(y - corners[itr].y) <= 0.5 && x > corners[itr].x))) itr = (int)i;
        cj.push_back({{"xy", {x + t.roi.x, y + t.roi.y}}, {"quadrant_means_ul_ur_ll_lr", {ul, ur, ll, lr}}, {"pattern", k}});
    }
    for (size_t i = 0; i < corners.size(); ++i) {
        std::string st = (int)i == itl ? "selected_top_left" : (int)i == itr ? "selected_top_right" : "not_selected";
        cj[i]["state"] = st;
    }
    r.intermediates["corner_candidates"] = cj;
    if (itl < 0 || itr < 0) {
        r.fail("corner_not_identified", std::string("outer border corner pattern not found: ") + (itl < 0 ? "top-left " : "") + (itr < 0 ? "top-right" : ""));
        return r;
    }
    cv::Point2f a0 = corners[itl], b0 = corners[itr];
    if (b0.x - a0.x < 0.5 * crop.cols || std::fabs(b0.y - a0.y) > 15) {
        r.fail("corner_not_identified", "identified top corners are not a plausible outer-border top pair");
        r.intermediates["rejected_pair"] = {{a0.x + t.roi.x, a0.y + t.roi.y}, {b0.x + t.roi.x, b0.y + t.roi.y}};
        return r;
    }
    std::vector<cv::Point2f> ref = {a0, b0};
    int hw = p.at("subpix_half_win").get<int>();
    cv::cornerSubPix(crop, ref, cv::Size(hw, hw), cv::Size(-1, -1),
                     cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, p.at("subpix_max_iter").get<int>(), p.at("subpix_eps").get<double>()));
    double s1 = cv::norm(ref[0] - a0), s2 = cv::norm(ref[1] - b0);
    r.intermediates["top_corners"] = {
        {{"name", "top_left"}, {"initial_xy", {a0.x + t.roi.x, a0.y + t.roi.y}}, {"refined_xy", {ref[0].x + t.roi.x, ref[0].y + t.roi.y}}, {"shift_px", s1}},
        {{"name", "top_right"}, {"initial_xy", {b0.x + t.roi.x, b0.y + t.roi.y}}, {"refined_xy", {ref[1].x + t.roi.x, ref[1].y + t.roi.y}}, {"shift_px", s2}}};
    double max_shift = p.at("max_refine_shift_px").get<double>();
    if (s1 > max_shift || s2 > max_shift) {
        r.fail("corner_not_identified", "cornerSubPix moved a corner farther than max_refine_shift_px");
        return r;
    }
    two_point_result(r, cv::Point2d(ref[0].x + t.roi.x, ref[0].y + t.roi.y), cv::Point2d(ref[1].x + t.roi.x, ref[1].y + t.roi.y),
                     "two refined outer-border corners (top-left, top-right)");
    return r;
}

namespace {
std::vector<cv::Point2d> quad_order(const std::vector<cv::Point>& q) {
    std::vector<cv::Point2d> p(q.begin(), q.end());
    cv::Point2d c(0, 0);
    for (auto& v : p) c += v;
    c *= 0.25;
    std::sort(p.begin(), p.end(), [&](const cv::Point2d& a, const cv::Point2d& b) {
        return std::atan2(a.y - c.y, a.x - c.x) < std::atan2(b.y - c.y, b.x - c.x);
    });
    int k = 0;
    for (int i = 1; i < 4; ++i) if (p[i].x + p[i].y < p[k].x + p[k].y) k = i;
    std::rotate(p.begin(), p.begin() + k, p.end());
    return p;
}
bool diag_intersection(const std::vector<cv::Point2d>& q, cv::Point2d& out) {
    cv::Point2d d1 = q[2] - q[0], d2 = q[3] - q[1], rhs = q[1] - q[0];
    double det = d1.x * (-d2.y) - (-d2.x) * d1.y;
    if (std::fabs(det) < 1e-12) return false;
    double tt = (rhs.x * (-d2.y) - (-d2.x) * rhs.y) / det;
    double ss = (d1.x * rhs.y - d1.y * rhs.x) / det;
    if (!(tt > 0 && tt < 1 && ss > 0 && ss < 1)) return false;
    out = q[0] + tt * d1;
    return true;
}
}  // namespace

MethodResult method_qr_finders(MethodContext& c) {
    MethodResult r;
    const TargetSpec& t = c.target;
    const json& p = c.params;
    cv::Mat crop = c.gray(t.roi).clone(), mask;
    double thr = cv::threshold(crop, mask, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
    c.sink.save_png("otsu_binary_inv", mask, t.roi.tl(), r.images);
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hier;
    cv::findContours(mask.clone(), contours, hier, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
    r.diagnostics["otsu_threshold"] = thr;
    r.n_candidates = (int)contours.size();
    const double amin = p.at("min_outer_area_px2").get<double>(), afrac = p.at("max_outer_area_roi_fraction").get<double>();
    const auto wr = p.at("white_outer_area_ratio"), ir = p.at("inner_white_area_ratio");
    const double eps = p.at("polygon_epsilon_perimeter_fraction").get<double>(), sidemin = p.at("min_side_length_ratio").get<double>();
    int hw = p.at("subpix_half_window").get<int>();
    cv::TermCriteria crit(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, p.at("subpix_max_iter").get<int>(), p.at("subpix_eps").get<double>());
    struct Pat { cv::Point2d center; std::vector<cv::Point2d> init, refined; double area; };
    std::vector<Pat> pats;
    for (size_t i = 0; i < contours.size(); ++i) {
        int child = hier[i][2];
        if (child < 0) continue;
        int grand = hier[child][2];
        if (grand < 0) continue;
        double oa = std::fabs(cv::contourArea(contours[i]));
        if (!(oa > amin && oa < afrac * crop.total())) continue;
        double ca = std::fabs(cv::contourArea(contours[child])), ia = std::fabs(cv::contourArea(contours[grand]));
        if (!(ca / oa > wr[0].get<double>() && ca / oa < wr[1].get<double>() && ia / ca > ir[0].get<double>() && ia / ca < ir[1].get<double>())) continue;
        std::vector<cv::Point> quad;
        cv::approxPolyDP(contours[i], quad, eps * cv::arcLength(contours[i], true), true);
        if (quad.size() != 4 || !cv::isContourConvex(quad)) continue;
        auto init = quad_order(quad);
        double smin = 1e300, smax = 0;
        for (int k = 0; k < 4; ++k) { double s = cv::norm(init[(k + 1) % 4] - init[k]); smin = std::min(smin, s); smax = std::max(smax, s); }
        if (smin / smax < sidemin) continue;
        std::vector<cv::Point2f> rf(init.begin(), init.end());
        cv::cornerSubPix(crop, rf, cv::Size(hw, hw), cv::Size(-1, -1), crit);
        Pat pt; pt.area = oa;
        for (int k = 0; k < 4; ++k) {
            pt.init.emplace_back(init[k].x + t.roi.x, init[k].y + t.roi.y);
            pt.refined.emplace_back(rf[k].x + t.roi.x, rf[k].y + t.roi.y);
        }
        if (!diag_intersection(pt.refined, pt.center)) continue;
        pats.push_back(pt);
    }
    json pj = json::array();
    for (auto& q : pats) {
        json a = json::array(), b = json::array();
        for (int k = 0; k < 4; ++k) { a.push_back({q.init[k].x, q.init[k].y}); b.push_back({q.refined[k].x, q.refined[k].y}); }
        pj.push_back({{"center_xy", {q.center.x, q.center.y}}, {"initial_corners_xy", a}, {"refined_corners_xy", b}, {"outer_area_px2", q.area}});
    }
    r.intermediates["finder_candidates"] = pj;
    if (pats.size() != 3) {
        r.fail("detection_failed", "expected 3 nested finder patterns, found " + std::to_string(pats.size()));
        return r;
    }
    std::vector<int> idx = {0, 1, 2};
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return pats[a].center.y < pats[b].center.y; });
    int A = idx[0], B = idx[1], C = idx[2];
    if (pats[B].center.x < pats[A].center.x) std::swap(A, B);
    cv::Point2d a = pats[A].center, b = pats[B].center, cc = pats[C].center;
    r.intermediates["assignment"] = {{"A_top_left", A}, {"B_top_right", B}, {"C_bottom_left", C}};
    if (!(b.x > a.x && cc.y > a.y && std::fabs(cc.x - a.x) < 0.3 * cv::norm(b - a))) {
        r.fail("detection_failed", "finder layout check failed (A/B/C arrangement)");
        return r;
    }
    two_point_result(r, a, b, "two finder centres (diagonal intersections of refined outer finder corners), A_top_left -> B_top_right");
    r.diagnostics["baseline_length_px"] = cv::norm(b - a);
    return r;
}

}  // namespace lm
