// =====================================================================================
// sim_draw.cpp — 시뮬레이터 그림
// =====================================================================================
#include "sim_draw.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace sim {

namespace {
const cv::Scalar kBg(250, 250, 250), kAxis(90, 90, 90), kGrid(225, 225, 225), kRed(40, 40, 220), kGreen(60, 160, 40),
    kBlue(190, 110, 30), kOrange(30, 140, 230), kGray(150, 150, 150), kTol(200, 230, 200);

std::string sf(const char* f, double a, double b = 0, double c = 0, double d = 0) {
    char buf[256];
    std::snprintf(buf, sizeof(buf), f, a, b, c, d);
    return buf;
}
// 글자 배율 (설정 display.plot_font_scale). 글자 크기·굵기와 글자에 딸린 여백(P)을 함께 키움
double g_fs = 1.0;
int fp(double px) { return int(std::lround(px * g_fs)); }   // 글자에 딸린 여백
int thick(int th) { return std::max(1, int(th * g_fs + 0.5)); }
void text(cv::Mat& m, const std::string& s, cv::Point p, double scale, const cv::Scalar& c, int th = 1) {
    cv::putText(m, s, p, cv::FONT_HERSHEY_SIMPLEX, scale * g_fs, c, thick(th), cv::LINE_AA);
}
// 오른쪽 끝을 x 에 맞춤
void textRight(cv::Mat& m, const std::string& s, cv::Point p, double scale, const cv::Scalar& c, int th = 1) {
    int base = 0;
    const cv::Size z = cv::getTextSize(s, cv::FONT_HERSHEY_SIMPLEX, scale * g_fs, thick(th), &base);
    text(m, s, cv::Point(p.x - z.width, p.y), scale, c, th);
}
void outlined(cv::Mat& m, const std::string& s, cv::Point p, double scale, const cv::Scalar& c, int th) {
    cv::putText(m, s, p, cv::FONT_HERSHEY_SIMPLEX, scale * g_fs, cv::Scalar(0, 0, 0), thick(th) + 3, cv::LINE_AA);
    cv::putText(m, s, p, cv::FONT_HERSHEY_SIMPLEX, scale * g_fs, c, thick(th), cv::LINE_AA);
}

// 선형 축 사상
struct Axis {
    double lo, hi; int p0, p1;
    int map(double v) const { return int(std::lround(p0 + (v - lo) / (hi - lo) * (p1 - p0))); }
};
void frameBox(cv::Mat& m, const cv::Rect& r) { cv::rectangle(m, r, kAxis, 1); }
} // namespace

void setPlotFontScale(double s) { g_fs = s > 0 ? s : 1.0; }
double plotFontScale() { return g_fs; }

// =====================================================================================
cv::Mat drawCamera(const cv::Mat& img8, const StepRecord* step, const AlignmentInfo* info, int width_px) {
    cv::Mat v;
    if (img8.empty()) return cv::Mat(cv::Size(width_px, width_px * 9 / 16), CV_8UC3, cv::Scalar(40, 40, 40));
    if (img8.channels() == 1) cv::cvtColor(img8, v, cv::COLOR_GRAY2BGR); else v = img8.clone();
    const double s = double(width_px) / v.cols;
    cv::resize(v, v, cv::Size(width_px, int(std::lround(v.rows * s))), 0, 0, cv::INTER_AREA);
    if (info) {
        for (const auto& r : info->rois) {
            const cv::Rect2d rr(r.ref_rect_px.x + r.shift_px.x, r.ref_rect_px.y + r.shift_px.y, r.ref_rect_px.width, r.ref_rect_px.height);
            const cv::Rect rs(int(rr.x * s), int(rr.y * s), std::max(2, int(rr.width * s)), std::max(2, int(rr.height * s)));
            const cv::Scalar c = r.ok ? cv::Scalar(60, 220, 60) : kRed;
            cv::rectangle(v, rs, c, 2);
            outlined(v, r.name, rs.tl() + cv::Point(0, -fp(4)), 0.45, c, 1);
        }
    }
    if (step) {
        const double sc = width_px / 1280.0;
        const int th = std::max(1, int(2 * sc));
        const auto row = [&](int k) { return cv::Point(fp(12 * sc), fp((32 + 32 * k) * sc)); };
        outlined(v, sf("step %.0f   true  Y %+.4f mm  T %+.4f deg", step->k, step->truth.y_mm, step->truth.t_deg), row(0), 0.75 * sc, cv::Scalar(255, 255, 255), th);
        if (step->est_ok)
            outlined(v, sf("model est  Y %+.4f mm  T %+.4f deg", step->est.y_mm, step->est.t_deg), row(1), 0.75 * sc, cv::Scalar(120, 230, 255), th);
        else
            outlined(v, "model FAILED", row(1), 0.75 * sc, kRed, th);
        const std::string st = step->stop ? (step->truth_in_tol ? "STOP - truth in tolerance" : "STOP - truth OUT of tolerance")
                                          : sf("move  Y %+.4f mm  T %+.4f deg", step->cmd.y_mm, step->cmd.t_deg);
        outlined(v, st, row(2), 0.75 * sc, step->stop ? (step->truth_in_tol ? cv::Scalar(90, 240, 90) : kRed) : cv::Scalar(255, 255, 255), th);
    }
    return v;
}

// =====================================================================================
cv::Mat drawPanorama(const World& w, const TrialResult* trial, int width_px) {
    if (!w.ready()) return cv::Mat(cv::Size(width_px, width_px / 2), CV_8UC3, cv::Scalar(40, 40, 40));
    cv::Mat v, cov;
    cv::cvtColor(w.panorama(), v, cv::COLOR_GRAY2BGR);
    // 실제 사진으로 채우지 못한 곳(가장자리 복제)은 붉게 어둡게
    cv::Mat tint(v.size(), CV_8UC3, cv::Scalar(60, 60, 140));
    cv::Mat mask = w.covered() == 0;
    cv::addWeighted(v, 0.5, tint, 0.5, 0, tint);
    tint.copyTo(v, mask);
    const double s = double(width_px) / v.cols;
    cv::resize(v, v, cv::Size(width_px, int(std::lround(v.rows * s))), 0, 0, cv::INTER_AREA);
    const auto poly = [&](const std::vector<cv::Point2f>& q, const cv::Scalar& c, int th) {
        std::vector<cv::Point> p;
        for (const auto& x : q) p.emplace_back(int(x.x * s), int(x.y * s));
        cv::polylines(v, p, true, c, th, cv::LINE_AA);
    };
    poly(w.footprint(Pose()), kGreen, 2);
    if (trial && !trial->steps.empty()) {
        std::vector<cv::Point> path;
        for (size_t i = 0; i < trial->steps.size(); ++i) {
            const auto fp = w.footprint(trial->steps[i].truth);
            if (i + 1 < trial->steps.size()) poly(fp, kGray, 1);
            cv::Point2f c(0, 0);
            for (const auto& x : fp) c += x * 0.25f;
            path.emplace_back(int(c.x * s), int(c.y * s));
        }
        poly(w.footprint(trial->steps.back().truth), kRed, 2);
        for (size_t i = 1; i < path.size(); ++i) cv::line(v, path[i - 1], path[i], cv::Scalar(0, 200, 255), 1, cv::LINE_AA);
    }
    outlined(v, "panorama (green = target view, red = current view, red tint = edge fill)", cv::Point(fp(8), fp(22)), 0.5, cv::Scalar(255, 255, 255), 1);
    return v;
}

// =====================================================================================
cv::Mat drawPlane(const TrialResult& t, const SimConfig& cfg, cv::Size size) {
    cv::Mat m(size, CV_8UC3, kBg);
    const int W = size.width, H = size.height, pad = fp(52), gap = fp(56), top = fp(30);
    const int pw = (W - 2 * pad - gap) / 2, ph = H - top - fp(50);
    const cv::Rect L(pad, top, pw, ph), R(pad + pw + gap, top, pw, ph);
    // 왼쪽 범위 : 시작 범위와 궤적을 모두 담음
    double ylo = -cfg.start_range_y_mm, yhi = cfg.start_range_y_mm, tlo = -cfg.start_range_t_deg, thi = cfg.start_range_t_deg;
    for (const auto& s : t.steps) {
        ylo = std::min(ylo, s.truth.y_mm); yhi = std::max(yhi, s.truth.y_mm);
        tlo = std::min(tlo, s.truth.t_deg); thi = std::max(thi, s.truth.t_deg);
    }
    const double ty = cfg.tol_y_um / 1000.0, tt = cfg.tol_t_deg;
    struct Panel { cv::Rect r; Axis ax, ay; const char* title; };
    Panel P[2] = {{L, {ylo * 1.05, yhi * 1.05, L.x, L.x + L.width}, {tlo * 1.05, thi * 1.05, L.y + L.height, L.y}, "all (Y mm, T deg)"},
                  {R, {-3 * ty, 3 * ty, R.x, R.x + R.width}, {-3 * tt, 3 * tt, R.y + R.height, R.y}, "zoom: tolerance box"}};
    for (auto& p : P) {
        cv::rectangle(m, p.r, cv::Scalar(255, 255, 255), cv::FILLED);
        // 허용 범위 상자
        cv::Rect tol(cv::Point(p.ax.map(-ty), p.ay.map(tt)), cv::Point(p.ax.map(ty), p.ay.map(-tt)));
        cv::rectangle(m, tol & p.r, kTol, cv::FILLED);
        cv::line(m, cv::Point(p.ax.map(0), p.r.y), cv::Point(p.ax.map(0), p.r.y + p.r.height), kGrid, 1);
        cv::line(m, cv::Point(p.r.x, p.ay.map(0)), cv::Point(p.r.x + p.r.width, p.ay.map(0)), kGrid, 1);
        frameBox(m, p.r);
        text(m, p.title, cv::Point(p.r.x, p.r.y - fp(8)), 0.5, kAxis);
        text(m, sf("%+.3f", p.ax.lo), cv::Point(p.r.x, p.r.y + p.r.height + fp(18)), 0.4, kAxis);
        textRight(m, sf("%+.3f", p.ax.hi), cv::Point(p.r.x + p.r.width, p.r.y + p.r.height + fp(18)), 0.4, kAxis);
        textRight(m, sf("%+.3f", p.ay.hi), cv::Point(p.r.x - fp(3), p.r.y + fp(10)), 0.38, kAxis);
        textRight(m, sf("%+.3f", p.ay.lo), cv::Point(p.r.x - fp(3), p.r.y + p.r.height), 0.38, kAxis);
        cv::Mat roi = m(p.r);
        std::vector<cv::Point> pts;
        for (const auto& s : t.steps) pts.emplace_back(p.ax.map(s.truth.y_mm) - p.r.x, p.ay.map(s.truth.t_deg) - p.r.y);
        for (size_t i = 1; i < pts.size(); ++i) cv::arrowedLine(roi, pts[i - 1], pts[i], kBlue, 1, cv::LINE_AA, 0, 0.08);
        for (size_t i = 0; i < pts.size(); ++i) {
            const auto& s = t.steps[i];
            cv::circle(roi, pts[i], i == 0 ? 6 : 4, i == 0 ? kOrange : (s.truth_in_tol ? kGreen : kRed), cv::FILLED, cv::LINE_AA);
            if (s.est_ok) {   // 모델 추정 위치 (참값과의 차 = 추정 오차)
                const cv::Point e(p.ax.map(s.est.y_mm) - p.r.x, p.ay.map(s.est.t_deg) - p.r.y);
                cv::drawMarker(roi, e, kGray, cv::MARKER_TILTED_CROSS, 7, 1);
            }
        }
    }
    text(m, "orange = start, dot = true (green = in tol), x = estimate", cv::Point(pad, H - fp(8)), 0.42, kAxis);
    return m;
}

// =====================================================================================
cv::Mat drawConvergence(const TrialResult& t, const SimConfig& cfg, cv::Size size) {
    cv::Mat m(size, CV_8UC3, kBg);
    const cv::Rect r(fp(60), fp(30), size.width - fp(60) - fp(20), size.height - fp(30) - fp(36));
    cv::rectangle(m, r, cv::Scalar(255, 255, 255), cv::FILLED);
    const int K = std::max(cfg.control_max_moves, int(t.steps.size()));
    const Axis ax{0, double(K), r.x, r.x + r.width}, ay{-2, 3, r.y + r.height, r.y};   // log10(|오차| / 허용)
    for (int e = -2; e <= 3; ++e) {
        cv::line(m, cv::Point(r.x, ay.map(e)), cv::Point(r.x + r.width, ay.map(e)), e == 0 ? cv::Scalar(80, 170, 80) : kGrid, e == 0 ? 2 : 1);
        textRight(m, e == 0 ? "tol" : sf("1e%+.0f", e), cv::Point(r.x - fp(4), ay.map(e) + fp(5)), 0.4, kAxis);
    }
    for (int k = 0; k <= K; k += std::max(1, K / 10)) text(m, sf("%.0f", k), cv::Point(ax.map(k) - fp(6), r.y + r.height + fp(20)), 0.4, kAxis);
    frameBox(m, r);
    const auto lg = [](double v) { return std::log10(std::max(v, 1e-6)); };
    for (int axis = 0; axis < 2; ++axis) {
        std::vector<cv::Point> pts;
        for (const auto& s : t.steps) {
            const double e = axis == 0 ? std::abs(s.truth.y_mm) * 1000.0 / cfg.tol_y_um : std::abs(s.truth.t_deg) / cfg.tol_t_deg;
            pts.emplace_back(ax.map(s.k), ay.map(std::max(-2.0, std::min(3.0, lg(e)))));
        }
        const cv::Scalar c = axis == 0 ? kBlue : kOrange;
        cv::polylines(m, pts, false, c, 2, cv::LINE_AA);
        for (const auto& p : pts) cv::circle(m, p, 3, c, cv::FILLED, cv::LINE_AA);
    }
    text(m, "|true error| / tol  (blue = Y, orange = Theta)  vs step", cv::Point(r.x, fp(20)), 0.45, kAxis);
    return m;
}

// =====================================================================================
cv::Mat drawMonte(const MonteSummary& ms, const SimConfig& cfg, cv::Size size) {
    cv::Mat m(size, CV_8UC3, kBg);
    const int W = size.width, H = size.height, gap = fp(40), top = fp(34), side = fp(50);
    const cv::Rect L(side, top, (W - 2 * side - gap) / 2, H - top - fp(40)), R(L.x + L.width + gap, top, L.width, H - top - fp(40));
    // 왼쪽 : 이동 횟수 분포
    cv::rectangle(m, L, cv::Scalar(255, 255, 255), cv::FILLED);
    frameBox(m, L);
    const int nb = int(ms.moves_hist.size());
    int hmax = 1;
    for (int v : ms.moves_hist) hmax = std::max(hmax, v);
    for (int i = 0; i < nb; ++i) {
        const int x0 = L.x + i * L.width / nb + 2, x1 = L.x + (i + 1) * L.width / nb - 2;
        const int h = ms.moves_hist[i] * (L.height - 20) / hmax;
        cv::rectangle(m, cv::Point(x0, L.y + L.height - h), cv::Point(x1, L.y + L.height), i == nb - 1 ? kRed : kBlue, cv::FILLED);
        if (ms.moves_hist[i]) text(m, std::to_string(ms.moves_hist[i]), cv::Point(x0, L.y + L.height - h - fp(4)), 0.38, kAxis);
        if ((i % 2 == 0 && i < nb - 2) || i == nb - 1) text(m, i == nb - 1 ? "fail" : std::to_string(i), cv::Point(x0, L.y + L.height + fp(18)), 0.38, kAxis);
    }
    text(m, sf("moves until stop (n=%.0f, success %.1f%%)", ms.n, ms.n ? 100.0 * ms.success / ms.n : 0.0), cv::Point(L.x, fp(22)), 0.48, kAxis);
    // 오른쪽 : 멈춘 뒤 참 오차 (허용 범위로 나눔)
    cv::rectangle(m, R, cv::Scalar(255, 255, 255), cv::FILLED);
    const Axis ax{-1.5, 1.5, R.x, R.x + R.width}, ay{-1.5, 1.5, R.y + R.height, R.y};
    cv::rectangle(m, cv::Point(ax.map(-1), ay.map(1)), cv::Point(ax.map(1), ay.map(-1)), kTol, cv::FILLED);
    cv::line(m, cv::Point(ax.map(0), R.y), cv::Point(ax.map(0), R.y + R.height), kGrid);
    cv::line(m, cv::Point(R.x, ay.map(0)), cv::Point(R.x + R.width, ay.map(0)), kGrid);
    frameBox(m, R);
    for (const auto& t : ms.trials) {
        const double ey = std::max(-1.45, std::min(1.45, t.final_truth.y_mm * 1000.0 / cfg.tol_y_um));
        const double et = std::max(-1.45, std::min(1.45, t.final_truth.t_deg / cfg.tol_t_deg));
        cv::circle(m, cv::Point(ax.map(ey), ay.map(et)), 4, t.success ? kGreen : kRed, cv::FILLED, cv::LINE_AA);
    }
    text(m, "final true error / tol (x = Y, y = Theta)", cv::Point(R.x, fp(22)), 0.48, kAxis);
    text(m, "-1", cv::Point(ax.map(-1) - fp(8), R.y + R.height + fp(18)), 0.4, kAxis);
    text(m, "+1", cv::Point(ax.map(1) - fp(8), R.y + R.height + fp(18)), 0.4, kAxis);
    return m;
}

// =====================================================================================
cv::Mat drawField(const World& w, int width_px) {
    if (!w.ready()) return cv::Mat();
    const cv::Mat& a = w.coef(0, 1);   // dx 의 Y 감도 (px/mm)
    cv::Mat n, c, v;
    a.convertTo(n, CV_8U, 255.0 / (16.0 - 9.0), -9.0 * 255.0 / (16.0 - 9.0));   // 9 ~ 16 px/mm
    cv::applyColorMap(n, c, cv::COLORMAP_JET);
    cv::resize(c, v, cv::Size(width_px, width_px * a.rows / a.cols), 0, 0, cv::INTER_NEAREST);
    // 채운 격자는 빗금 대신 어둡게
    cv::Mat inval;
    cv::resize(w.valid() == 0, inval, v.size(), 0, 0, cv::INTER_NEAREST);
    cv::Mat dark = v * 0.45;
    dark.copyTo(v, inval);
    outlined(v, "Y sensitivity map (px/mm): blue 9 -> red 16, dark = filled (no texture)", cv::Point(fp(8), fp(22)), 0.5, cv::Scalar(255, 255, 255), 1);
    return v;
}

} // namespace sim
