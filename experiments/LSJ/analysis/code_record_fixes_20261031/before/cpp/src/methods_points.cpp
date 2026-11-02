// Edge-point methods: each produces subpixel boundary points, then the common robust line fit.
//   01 profile model (Gaussian on Sobel derivative, S01)
//   02 Canny-Devernay (author code, S02)
//   05 Zernike moments, ramp edge (S05)
//   06 Partial area effect, second order (S06)
#include "common.hpp"
#include "effective_settings.hpp"
#include "methods.hpp"

extern "C" {
#include "devernay.h"
}

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <map>
#include <mutex>

namespace lm {

namespace {

const double kPi = 3.14159265358979323846;

// Visible columns of a target: x0..x1 with the configured step, occluded columns skipped.
std::vector<int> visible_columns(const TargetSpec& t, int* occluded) {
    std::vector<int> cols;
    int occ = 0;
    for (int x = t.x0; x <= t.x1; x += t.column_step) {
        if (t.excluded(x)) { ++occ; continue; }
        cols.push_back(x);
    }
    if (occluded) *occluded = occ;
    return cols;
}

// Float copy of the band plus a fixed pad (for windows/derivatives around band rows).
struct BandPatch {
    cv::Mat f;  // CV_64F
    int ox = 0, oy = 0;
    double at(int x, int y) const { return f.at<double>(y - oy, x - ox); }
};

BandPatch band_patch(const cv::Mat& gray, const TargetSpec& t, int pad) {
    BandPatch p;
    int ax = std::max(0, t.x0 - pad), bx = std::min(gray.cols - 1, t.x1 + pad);
    int ay = std::max(0, t.y0 - pad), by = std::min(gray.rows - 1, t.y1 + pad);
    gray(cv::Rect(ax, ay, bx - ax + 1, by - ay + 1)).convertTo(p.f, CV_64F);
    p.ox = ax; p.oy = ay;
    return p;
}

// Nelder-Mead minimiser (fixed standard coefficients).
template <class F>
std::array<double, 3> nelder_mead3(F f, std::array<double, 3> x0, const std::array<double, 3>& step,
                                   int max_iter, double tol, int* iters, bool* converged) {
    std::array<std::array<double, 3>, 4> s;
    std::array<double, 4> v;
    s[0] = x0;
    for (int i = 0; i < 3; ++i) { s[i + 1] = x0; s[i + 1][i] += step[i]; }
    for (int i = 0; i < 4; ++i) v[i] = f(s[i]);
    *converged = false;
    int it = 0;
    for (; it < max_iter; ++it) {
        std::array<int, 4> idx = {0, 1, 2, 3};
        std::sort(idx.begin(), idx.end(), [&](int a, int b) { return v[a] < v[b]; });
        auto s2 = s; auto v2 = v;
        for (int i = 0; i < 4; ++i) { s[i] = s2[idx[i]]; v[i] = v2[idx[i]]; }
        if (std::fabs(v[3] - v[0]) <= tol * (std::fabs(v[0]) + 1e-30)) { *converged = true; break; }
        std::array<double, 3> c = {0, 0, 0};
        for (int i = 0; i < 3; ++i) for (int k = 0; k < 3; ++k) c[k] += s[i][k] / 3.0;
        auto mix = [&](double a) { std::array<double, 3> r; for (int k = 0; k < 3; ++k) r[k] = c[k] + a * (s[3][k] - c[k]); return r; };
        auto xr = mix(-1.0); double fr = f(xr);
        if (fr < v[0]) {
            auto xe = mix(-2.0); double fe = f(xe);
            if (fe < fr) { s[3] = xe; v[3] = fe; } else { s[3] = xr; v[3] = fr; }
        } else if (fr < v[2]) {
            s[3] = xr; v[3] = fr;
        } else {
            auto xc = fr < v[3] ? mix(-0.5) : mix(0.5);
            double fc = f(xc);
            if (fc < std::min(fr, v[3])) { s[3] = xc; v[3] = fc; }
            else {
                for (int i = 1; i < 4; ++i) { for (int k = 0; k < 3; ++k) s[i][k] = s[0][k] + 0.5 * (s[i][k] - s[0][k]); v[i] = f(s[i]); }
            }
        }
    }
    *iters = it;
    int best = (int)(std::min_element(v.begin(), v.end()) - v.begin());
    return s[best];
}

std::string excluded(const std::string& why) { return "excluded:" + why; }

}  // namespace

// ---------------------------------------------------------------------------------------------
// 01: S01 Fabijanska (2014) Sec. III.B. Gaussian A*exp(-(x-mu)^2/(2 s^2)) fitted with Nelder-Mead to
// Sobel gradient samples along the edge normal (vertical here). Coarse edge: per-column argmax of
// the polarity-signed Sobel response inside the band (replaces the paper's ISODATA + skeleton step).
MethodResult method_profile_gaussian(MethodContext& c) {
    validate_method_parameters(c.target.kind=="qr"?"Q01":"E01",c.params);
    MethodResult r;
    const TargetSpec& t = c.target;
    const json& p = c.params;
    const int hw = p.at("half_window").get<int>();
    const double min_peak = p.at("min_peak").get<double>();
    const int max_iter = p.at("nm_max_iter").get<int>();
    const double tol = p.at("nm_tol").get<double>();
    const double max_shift = p.at("max_center_shift_px").get<double>();
    const int ex_every = p.at("profile_examples_every").get<int>();

    BandPatch bp = band_patch(c.gray, t, hw + 4);
    cv::Mat gy;
    cv::Sobel(bp.f, gy, CV_64F, 0, 1, 3, 1.0, 0.0, cv::BORDER_REFLECT101);
    int occluded = 0;
    auto cols = visible_columns(t, &occluded);
    std::vector<cv::Point2d> pts;
    json cand = json::array(), examples = json::array();
    int ci = 0, nm_not_conv = 0;
    for (int x : cols) {
        int best_y = -1; double best = -1e300;
        for (int y = t.y0; y <= t.y1; ++y) {
            double v = t.polarity * gy.at<double>(y - bp.oy, x - bp.ox);
            if (v > best) { best = v; best_y = y; }
        }
        json row = {x, best_y};
        std::string state;
        double A = 0, mu = 0, s = 0; int iters = 0; bool conv = false;
        if (best < min_peak) state = excluded("weak_peak");
        else if (best_y == t.y0 || best_y == t.y1) state = excluded("peak_at_band_edge");
        else {
            std::vector<double> ys, fs;
            for (int k = -hw; k <= hw; ++k) {
                ys.push_back(best_y + k);
                fs.push_back(t.polarity * gy.at<double>(best_y + k - bp.oy, x - bp.ox));
            }
            auto sse = [&](const std::array<double, 3>& q) {
                double e = 0, ss = std::fabs(q[2]) + 1e-12;
                for (size_t i = 0; i < ys.size(); ++i) {
                    double d = ys[i] - q[1];
                    double m = q[0] * std::exp(-d * d / (2 * ss * ss)) - fs[i];
                    e += m * m;
                }
                return e;
            };
            std::array<double, 3> x0 = {best, (double)best_y, (double)(2 * hw + 1)};
            std::array<double, 3> st = {0.2 * best, 1.0, 0.5 * (2 * hw + 1)};
            auto q = nelder_mead3(sse, x0, st, max_iter, tol, &iters, &conv);
            A = q[0]; mu = q[1]; s = std::fabs(q[2]);
            if (!conv) ++nm_not_conv;
            if (!conv) state = excluded("nelder_mead_not_converged");
            else if (A <= 0) state = excluded("non_positive_amplitude");
            else if (std::fabs(mu - best_y) > max_shift) state = excluded("center_shift_exceeds_limit");
            else { state = "selected"; pts.emplace_back((double)x, mu); }
            if (ex_every > 0 && ci % ex_every == 0) {
                examples.push_back({{"x", x}, {"y_samples", ys}, {"sobel_samples", fs}, {"A", A}, {"mu", mu},
                                    {"sigma", s}, {"nm_iterations", iters}, {"converged", conv}, {"state", state}});
            }
        }
        row.push_back(state); row.push_back(rnd(mu, 5)); row.push_back(rnd(A, 3)); row.push_back(rnd(s, 4));
        cand.push_back(row);
        ++ci;
    }
    r.n_candidates = (int)cols.size();
    r.intermediates["columns_fields"] = {"x", "coarse_peak_y", "state", "mu", "A", "sigma"};
    r.intermediates["columns"] = cand;
    r.intermediates["profile_examples"] = examples;
    json fitpts = json::array();
    for (auto& q : pts) fitpts.push_back(round_pt(q.x, q.y, 5));
    r.intermediates["fit_points"] = fitpts;
    r.diagnostics["visible_columns"] = (int)cols.size();
    r.diagnostics["occluded_columns_skipped"] = occluded;
    r.diagnostics["nelder_mead_not_converged"] = nm_not_conv;
    finish_with_points(r, pts, c.fit, "subpixel_edge_points: Gaussian mu on Sobel derivative profile (1 per column)");
    return r;
}

// ---------------------------------------------------------------------------------------------
// 02: S02 author implementation devernay() (Canny with Devernay sub-pixel correction, chaining,
// hysteresis). Candidate filtering by polarity/orientation and per-column-bin strongest point.
MethodResult method_canny_devernay(MethodContext& c) {
    MethodResult r;
    const TargetSpec& t = c.target;
    const json& p = c.params;
    const double sigma = p.at("sigma").get<double>();
    const double th_h = p.at("th_h").get<double>(), th_l = p.at("th_l").get<double>();

    cv::Rect reg = t.detector_region(c.gray.size());
    cv::Mat f;
    c.gray(reg).convertTo(f, CV_64F);
    if (!f.isContinuous()) f = f.clone();
    double *xs = nullptr, *ys = nullptr; int N = 0, M = 0; int* limits = nullptr;
    devernay(&xs, &ys, &N, &limits, &M, f.ptr<double>(), f.cols, f.rows, sigma, th_h, th_l);

    // Gradient of the same Gaussian-smoothed patch, used only for polarity/orientation/strength.
    int off = (int)std::ceil(sigma * std::sqrt(2.0 * 3.0 * std::log(10.0)));
    cv::Mat g; cv::GaussianBlur(f, g, cv::Size(2 * off + 1, 2 * off + 1), sigma, sigma, cv::BORDER_REPLICATE);
    cv::Mat gx = cv::Mat::zeros(g.size(), CV_64F), gyy = cv::Mat::zeros(g.size(), CV_64F);
    for (int y = 1; y < g.rows - 1; ++y)
        for (int x = 1; x < g.cols - 1; ++x) {
            gx.at<double>(y, x) = g.at<double>(y, x + 1) - g.at<double>(y, x - 1);
            gyy.at<double>(y, x) = g.at<double>(y + 1, x) - g.at<double>(y - 1, x);
        }

    struct Cand { double x, y, strength; int curve; std::string state; };
    std::vector<Cand> cands;
    int in_region_total = N;
    std::map<int, int> best_in_bin;
    for (int k = 0; k < M; ++k) {
        for (int i = limits[k]; i < limits[k + 1]; ++i) {
            double lx = xs[i], ly = ys[i];
            double X = lx + reg.x, Y = ly + reg.y;
            if (!t.in_band(Y)) continue;  // only band points are candidates
            Cand cd{X, Y, 0, k, ""};
            double vx = bilinear(gx, lx, ly), vy = bilinear(gyy, lx, ly);
            cd.strength = std::hypot(vx, vy);
            if (!t.in_x(X)) cd.state = excluded("outside_x_range");
            else if (t.excluded(X)) cd.state = excluded("occluded_mask");
            else if (t.polarity * vy <= 0) cd.state = excluded("wrong_polarity");
            else if (std::fabs(vy) <= std::fabs(vx)) cd.state = excluded("not_near_horizontal_gradient");
            else cd.state = "candidate";
            cands.push_back(cd);
            if (cd.state == "candidate") {
                int bin = (int)std::floor((std::round(X) - t.x0) / (double)t.column_step);
                auto it = best_in_bin.find(bin);
                int idx = (int)cands.size() - 1;
                if (it == best_in_bin.end()) best_in_bin[bin] = idx;
                else if (cands[it->second].strength < cd.strength) {
                    cands[it->second].state = excluded("weaker_in_column_bin");
                    it->second = idx;
                } else cands[idx].state = excluded("weaker_in_column_bin");
            }
        }
    }
    std::free(xs); std::free(ys); std::free(limits);
    std::vector<cv::Point2d> pts;
    json arr = json::array();
    for (auto& cd : cands) {
        if (cd.state == "candidate") { cd.state = "selected"; pts.emplace_back(cd.x, cd.y); }
        arr.push_back({rnd(cd.x, 4), rnd(cd.y, 4), (int)std::lround(cd.x), (int)std::lround(cd.y), cd.curve, rnd(cd.strength, 3), cd.state});
    }
    r.n_candidates = (int)cands.size();
    r.intermediates["band_points_fields"] = {"x_subpixel", "y_subpixel", "x_rounded", "y_rounded", "curve", "strength", "state"};
    r.intermediates["band_points"] = arr;
    json fitpts = json::array();
    for (auto& q : pts) fitpts.push_back(round_pt(q.x, q.y, 5));
    r.intermediates["fit_points"] = fitpts;
    r.diagnostics["devernay_points_in_region"] = in_region_total;
    r.diagnostics["devernay_curves_in_region"] = M;
    r.diagnostics["detector_region_xywh"] = {reg.x, reg.y, reg.width, reg.height};
    r.diagnostics["integer_pixel_note"] = "x_rounded/y_rounded = nearest pixel of each Devernay point (the NMS pixel; the correction offset is within +-0.5 px along one axis).";
    finish_with_points(r, pts, c.fit, "subpixel_edge_points: Canny-Devernay points, strongest per column bin");
    return r;
}

// ---------------------------------------------------------------------------------------------
// 05: S05 Renshaw & Christian (2020). Moments A11 (complex) and A20 from pixel-integrated masks over the
// inscribed unit circle; psi = atan2(Im A11, Re A11) (Eq. 31); Q_E = A20 / (Re A11 cos psi + Im A11 sin psi)
// (Eq. 42); ramp-edge distance l = (1 - w^2 - sqrt((w^2-1)^2 - 2 w^2 Q_E)) / w^2 (Eq. 41), w = k_edge*sigma
// (Eq. 33) in normalised units; location = guess + Np*l/2 (cos psi, sin psi) (Eq. 25).
namespace {
struct ZernikeMasks { int Np; cv::Mat re11, im11, m20; };
const ZernikeMasks& zernike_masks(int Np, int ss) {
    static std::mutex mu;
    static std::map<std::pair<int, int>, ZernikeMasks> cache;
    std::lock_guard<std::mutex> lock(mu);
    auto key = std::make_pair(Np, ss);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    ZernikeMasks z; z.Np = Np;
    z.re11 = cv::Mat::zeros(Np, Np, CV_64F); z.im11 = z.re11.clone(); z.m20 = z.re11.clone();
    int h = Np / 2;
    double scale = 2.0 / Np;                       // u_bar = (2/Np)(u - u~), Eq. 1
    double da = (scale / ss) * (scale / ss);       // sub-sample area in normalised units
    for (int j = -h; j <= h; ++j)
        for (int i = -h; i <= h; ++i) {
            double sre = 0, sim = 0, s20 = 0;
            for (int b = 0; b < ss; ++b)
                for (int a = 0; a < ss; ++a) {
                    double u = scale * (i - 0.5 + (a + 0.5) / ss);
                    double v = scale * (j - 0.5 + (b + 0.5) / ss);
                    double r2 = u * u + v * v;
                    if (r2 > 1.0) continue;
                    sre += u * da; sim += v * da; s20 += (2 * r2 - 1) * da;
                }
            z.re11.at<double>(j + h, i + h) = sre;
            z.im11.at<double>(j + h, i + h) = sim;  // = transpose of Re mask (Eq. 23)
            z.m20.at<double>(j + h, i + h) = s20;
        }
    return cache.emplace(key, z).first->second;
}
}  // namespace

MethodResult method_zernike(MethodContext& c) {
    MethodResult r;
    const TargetSpec& t = c.target;
    const json& p = c.params;
    const double cb = p.at("coarse_blur_sigma").get<double>();
    const double min_g = p.at("min_coarse_grad").get<double>();
    const int Np = p.at("Np").get<int>();
    const double k_edge = p.at("k_edge").get<double>();
    const double blur_px = p.at("blur_sigma_px").get<double>();
    const int ss = p.at("supersample").get<int>();
    const double max_shift = p.at("max_shift_px").get<double>();
    const double max_dev = p.at("max_orientation_dev_deg").get<double>();
    const int ex_every = p.at("window_examples_every").get<int>();
    const ZernikeMasks& Z = zernike_masks(Np, ss);
    const int h = Np / 2;
    const double wbar = 2.0 * k_edge * blur_px / Np;

    BandPatch bp = band_patch(c.gray, t, 12);
    cv::Mat sm; cv::GaussianBlur(bp.f, sm, cv::Size(0, 0), cb, cb, cv::BORDER_REFLECT101);
    int occluded = 0;
    auto cols = visible_columns(t, &occluded);
    std::vector<cv::Point2d> pts;
    json cand = json::array(), examples = json::array();
    int ci = 0;
    for (int x : cols) {
        int best_y = -1; double best = -1e300;
        for (int y = t.y0; y <= t.y1; ++y) {
            int yy = y - bp.oy, xx = x - bp.ox;
            double g = 0.5 * (sm.at<double>(yy + 1, xx) - sm.at<double>(yy - 1, xx));
            if (t.polarity * g > best) { best = t.polarity * g; best_y = y; }
        }
        std::string state;
        double psi = 0, Q = 0, l = 0, shift = 0, u = 0, v = 0, are = 0, aim = 0, a20 = 0;
        if (best < min_g) state = excluded("weak_coarse_gradient");
        else if (best_y == t.y0 || best_y == t.y1) state = excluded("peak_at_band_edge");
        else {
            cv::Mat win = bp.f(cv::Rect(x - bp.ox - h, best_y - bp.oy - h, Np, Np));
            double mean = cv::mean(win)[0];
            for (int j = 0; j < Np; ++j)
                for (int i = 0; i < Np; ++i) {
                    double I = win.at<double>(j, i) - mean;
                    are += Z.re11.at<double>(j, i) * I;
                    aim += Z.im11.at<double>(j, i) * I;
                    a20 += Z.m20.at<double>(j, i) * I;
                }
            psi = std::atan2(aim, are);
            double a11p = are * std::cos(psi) + aim * std::sin(psi);
            Q = a20 / a11p;
            double dev = std::acos(std::max(-1.0, std::min(1.0, t.polarity * std::sin(psi)))) * 180.0 / kPi;
            double disc = (wbar * wbar - 1) * (wbar * wbar - 1) - 2 * wbar * wbar * Q;
            if (dev > max_dev) state = excluded("orientation_not_matching_target");
            else if (disc < 0) state = excluded("ramp_model_no_real_solution");
            else {
                l = wbar > 1e-9 ? (1 - wbar * wbar - std::sqrt(disc)) / (wbar * wbar) : Q;
                shift = Np * l / 2.0;
                u = x + shift * std::cos(psi);
                v = best_y + shift * std::sin(psi);
                if (std::fabs(shift) > max_shift) state = excluded("shift_exceeds_limit");
                else { state = "selected"; pts.emplace_back(u, v); }
            }
            if (ex_every > 0 && ci % ex_every == 0) {
                json w = json::array();
                for (int j = 0; j < Np; ++j) { json row = json::array(); for (int i = 0; i < Np; ++i) row.push_back(win.at<double>(j, i)); w.push_back(row); }
                examples.push_back({{"x", x}, {"coarse_y", best_y}, {"window", w}, {"A11_re", are}, {"A11_im", aim},
                                    {"A20", a20}, {"psi_deg", psi * 180 / kPi}, {"Q_E", Q}, {"l_ramp", l},
                                    {"l_step_model", Q}, {"shift_px", shift}, {"u", u}, {"v", v}, {"state", state}});
            }
        }
        cand.push_back({x, best_y, state, rnd(psi * 180 / kPi, 4), rnd(Q, 6), rnd(l, 6), rnd(shift, 5), rnd(u, 5), rnd(v, 5),
                        rnd(best_y + Np * Q / 2.0 * std::sin(psi), 5)});
        ++ci;
    }
    r.n_candidates = (int)cols.size();
    r.intermediates["columns_fields"] = {"x", "coarse_y", "state", "psi_deg", "Q_E", "l_ramp", "shift_px", "u", "v", "v_if_step_model"};
    r.intermediates["columns"] = cand;
    r.intermediates["window_examples"] = examples;
    json masks;
    auto m2j = [](const cv::Mat& m) { json a = json::array(); for (int j = 0; j < m.rows; ++j) { json row = json::array(); for (int i = 0; i < m.cols; ++i) row.push_back(m.at<double>(j, i)); a.push_back(row); } return a; };
    masks["M11_re"] = m2j(Z.re11); masks["M11_im"] = m2j(Z.im11); masks["M20"] = m2j(Z.m20);
    r.intermediates["zernike_masks"] = masks;
    json fitpts = json::array();
    for (auto& q : pts) fitpts.push_back(round_pt(q.x, q.y, 5));
    r.intermediates["fit_points"] = fitpts;
    r.diagnostics["w_bar_normalised"] = wbar;
    r.diagnostics["visible_columns"] = (int)cols.size();
    r.diagnostics["occluded_columns_skipped"] = occluded;
    finish_with_points(r, pts, c.fit, "subpixel_edge_points: Zernike ramp-edge location (2D, 1 per column)");
    return r;
}

// ---------------------------------------------------------------------------------------------
// 06: S06 Trujillo-Pino et al. (2013), partial area effect, second-order (y = a + b x + c x^2) basic
// model on the raw image. Window 3 x (2m+1). Column sums S_j = B(2m+1) + (A-B)(a + b j + c(j^2+1/12) + m + 1/2)
// give c = (S_L+S_R-2S_M)/(2(A-B)), b = (S_R-S_L)/(2(A-B)), a = (2S_M-(A+B)(2m+1))/(2(A-B)) - c/12.
// A/B: mean of the 3 pixels of the window's first/last row (A = side of smaller y).
MethodResult method_pae(MethodContext& c) {
    MethodResult r;
    const TargetSpec& t = c.target;
    const json& p = c.params;
    const int m = p.at("half_height").get<int>();
    const double min_g = p.at("min_grad").get<double>();
    const double min_contrast = p.at("min_contrast").get<double>();
    const double max_a = p.at("max_abs_a_px").get<double>();

    BandPatch bp = band_patch(c.gray, t, m + 4);
    int occluded = 0;
    auto cols = visible_columns(t, &occluded);
    std::vector<cv::Point2d> pts;
    json cand = json::array(), examples = json::array();
    auto I = [&](int x, int y) { return bp.at(x, y); };
    auto fy = [&](int x, int y) { return 0.5 * (I(x, y + 1) - I(x, y - 1)); };
    auto fx = [&](int x, int y) { return 0.5 * (I(x + 1, y) - I(x - 1, y)); };
    int ci = 0;
    for (int x : cols) {
        int best_y = -1; double best = -1;
        for (int y = t.y0; y <= t.y1; ++y) {
            double gy = fy(x, y), gx = fx(x, y);
            // Edge pixel rule for horizontal edges (paper / Method.md): |fy|>|fx| and local max of |fy| in y.
            if (t.polarity * gy <= 0 || std::fabs(gy) <= std::fabs(gx)) continue;
            if (std::fabs(fy(x, y - 1)) > std::fabs(gy) || std::fabs(fy(x, y + 1)) > std::fabs(gy)) continue;
            if (std::fabs(gy) > best) { best = std::fabs(gy); best_y = y; }
        }
        std::string state;
        double a = 0, b = 0, cc = 0, A = 0, B = 0;
        if (best_y < 0) state = excluded("no_edge_pixel");
        else if (best < min_g) state = excluded("weak_gradient");
        else {
            double S[3] = {0, 0, 0};
            for (int j = -1; j <= 1; ++j) for (int k = -m; k <= m; ++k) S[j + 1] += I(x + j, best_y + k);
            A = (I(x - 1, best_y - m) + I(x, best_y - m) + I(x + 1, best_y - m)) / 3.0;
            B = (I(x - 1, best_y + m) + I(x, best_y + m) + I(x + 1, best_y + m)) / 3.0;
            if (t.polarity * (B - A) < min_contrast) state = excluded("low_contrast_A_B");
            else {
                double den = 2 * (A - B);
                cc = (S[0] + S[2] - 2 * S[1]) / den;
                b = (S[2] - S[0]) / den;
                a = (2 * S[1] - (A + B) * (2 * m + 1)) / den - cc / 12.0;
                if (std::fabs(a) > max_a) state = excluded("a_exceeds_limit");
                else { state = "selected"; pts.emplace_back((double)x, best_y + a); }
            }
            if (ci % 100 == 0) {
                json w = json::array();
                for (int k = -m; k <= m; ++k) w.push_back({I(x - 1, best_y + k), I(x, best_y + k), I(x + 1, best_y + k)});
                examples.push_back({{"x", x}, {"edge_pixel_y", best_y}, {"window_rows_top_to_bottom", w}, {"S_L", S[0]}, {"S_M", S[1]},
                                    {"S_R", S[2]}, {"A", A}, {"B", B}, {"a", a}, {"b", b}, {"c", cc}, {"state", state}});
            }
        }
        cand.push_back({x, best_y, state, rnd(a, 5), rnd(b, 5), rnd(cc, 5), rnd(A, 2), rnd(B, 2)});
        ++ci;
    }
    r.n_candidates = (int)cols.size();
    r.intermediates["columns_fields"] = {"x", "edge_pixel_y", "state", "a", "b", "c", "A", "B"};
    r.intermediates["columns"] = cand;
    r.intermediates["window_examples"] = examples;
    json fitpts = json::array();
    for (auto& q : pts) fitpts.push_back(round_pt(q.x, q.y, 5));
    r.intermediates["fit_points"] = fitpts;
    r.diagnostics["window"] = "3x" + std::to_string(2 * m + 1);
    r.diagnostics["visible_columns"] = (int)cols.size();
    r.diagnostics["occluded_columns_skipped"] = occluded;
    finish_with_points(r, pts, c.fit, "subpixel_edge_points: PAE second-order a at column centre (1 per column)");
    return r;
}

}  // namespace lm
