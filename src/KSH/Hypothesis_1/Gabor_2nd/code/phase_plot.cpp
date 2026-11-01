// =====================================================================================
// phase_plot.cpp — 시각화 함수 구현
// =====================================================================================
#include "phase_plot.hpp"
#include <cmath>
#include <cstdio>
#include <algorithm>
#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif

namespace pp {

namespace {
const cv::Scalar C_TEXT(40, 40, 40), C_GRID(230, 230, 230), C_AXIS(150, 150, 150);
const cv::Scalar C_TRAIN(200, 150, 60);   // 파랑 계열 (BGR)
const cv::Scalar C_TEST(40, 120, 230);    // 주황 계열
const cv::Scalar C_FIT(60, 60, 200);      // 빨강 계열
const cv::Scalar C_BAND(215, 240, 215);   // 연한 초록
const cv::Scalar C_EDGE[pa::NUM_QR] = { cv::Scalar(230, 160, 40), cv::Scalar(80, 200, 60), cv::Scalar(220, 100, 220) };

double niceStep(double raw) {
    if (raw <= 0) return 1;
    double e = std::pow(10.0, std::floor(std::log10(raw)));
    double f = raw / e;
    return ((f < 1.5) ? 1 : (f < 3) ? 2 : (f < 7) ? 5 : 10) * e;
}

std::string fmt(double v, int prec) {
    char b[64]; std::snprintf(b, sizeof(b), "%.*f", prec, v); return b;
}
int precFor(double step) { return step >= 1 ? 0 : (int)std::ceil(-std::log10(step) - 1e-9); }

void putTextShadow(cv::Mat& img, const std::string& s, cv::Point p, double scale, cv::Scalar color, int thick = 1) {
    cv::putText(img, s, p, cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(255, 255, 255), thick + 3, cv::LINE_AA);
    cv::putText(img, s, p, cv::FONT_HERSHEY_SIMPLEX, scale, color, thick, cv::LINE_AA);
}
} // namespace

// ---------------------------------------------------------------------------------
// Panel
// ---------------------------------------------------------------------------------
Panel::Panel(cv::Mat& canvas, cv::Rect area, double x0, double x1, double y0, double y1)
    : c_(canvas), a_(area), x0_(x0), x1_(x1), y0_(y0), y1_(y1) {
    if (x1_ - x0_ < 1e-12) x1_ = x0_ + 1;
    if (y1_ - y0_ < 1e-12) y1_ = y0_ + 1;
}

cv::Point Panel::px(double x, double y) const {
    double u = (x - x0_) / (x1_ - x0_), v = (y - y0_) / (y1_ - y0_);
    return cv::Point(a_.x + (int)std::lround(u * a_.width), a_.y + a_.height - (int)std::lround(v * a_.height));
}

void Panel::frame(const std::string& title, const std::string& xLabel, const std::string& yLabel) {
    double sx = niceStep((x1_ - x0_) / 6), sy = niceStep((y1_ - y0_) / 5);
    for (double x = std::ceil(x0_ / sx) * sx; x <= x1_ + 1e-9; x += sx) {
        cv::Point p = px(x, y0_);
        cv::line(c_, cv::Point(p.x, a_.y), cv::Point(p.x, a_.y + a_.height), C_GRID, 1);
        cv::putText(c_, fmt(std::fabs(x) < sx * 1e-6 ? 0 : x, precFor(sx)), cv::Point(p.x - 16, a_.y + a_.height + 18),
                    cv::FONT_HERSHEY_SIMPLEX, 0.42, C_TEXT, 1, cv::LINE_AA);
    }
    for (double y = std::ceil(y0_ / sy) * sy; y <= y1_ + 1e-9; y += sy) {
        cv::Point p = px(x0_, y);
        cv::line(c_, cv::Point(a_.x, p.y), cv::Point(a_.x + a_.width, p.y), C_GRID, 1);
        cv::putText(c_, fmt(std::fabs(y) < sy * 1e-6 ? 0 : y, precFor(sy)), cv::Point(a_.x - 62, p.y + 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.42, C_TEXT, 1, cv::LINE_AA);
    }
    cv::rectangle(c_, a_, C_AXIS, 1);
    cv::putText(c_, title, cv::Point(a_.x, a_.y - 12), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(10, 10, 10), 1, cv::LINE_AA);
    cv::putText(c_, xLabel, cv::Point(a_.x + a_.width / 2 - 50, a_.y + a_.height + 38), cv::FONT_HERSHEY_SIMPLEX, 0.45, C_TEXT, 1, cv::LINE_AA);
    if (!yLabel.empty())
        cv::putText(c_, yLabel, cv::Point(a_.x + a_.width - 8 * (int)yLabel.size() - 6, a_.y - 12),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, C_AXIS, 1, cv::LINE_AA);
}

void Panel::band(double yLo, double yHi, cv::Scalar color) {
    cv::Point p0 = px(x0_, std::min(std::max(yHi, y0_), y1_)), p1 = px(x1_, std::max(std::min(yLo, y1_), y0_));
    cv::rectangle(c_, cv::Rect(p0, p1), color, cv::FILLED);
}
void Panel::hline(double y, cv::Scalar color, int thick) {
    if (y < y0_ || y > y1_) return;
    cv::line(c_, px(x0_, y), px(x1_, y), color, thick, cv::LINE_AA);
}
void Panel::line(double xa, double ya, double xb, double yb, cv::Scalar color, int thick) {
    cv::Point a = px(xa, ya), b = px(xb, yb);
    cv::Rect clip(a_.x, a_.y, a_.width + 1, a_.height + 1);
    if (cv::clipLine(clip, a, b)) cv::line(c_, a, b, color, thick, cv::LINE_AA);
}
void Panel::point(double x, double y, cv::Scalar color, int r, bool filled) {
    if (x < x0_ || x > x1_ || y < y0_ || y > y1_) return;
    cv::circle(c_, px(x, y), r, color, filled ? cv::FILLED : 1, cv::LINE_AA);
}
void Panel::text(const std::string& s, int row, cv::Scalar color) {
    putTextShadow(c_, s, cv::Point(a_.x + 10, a_.y + 22 + row * 22), 0.5, color);
}
void Panel::autoRange(const std::vector<double>& v, double& lo, double& hi, double margin, double minSpan) {
    if (v.empty()) { lo = -1; hi = 1; return; }
    lo = *std::min_element(v.begin(), v.end());
    hi = *std::max_element(v.begin(), v.end());
    double mid = 0.5 * (lo + hi), span = std::max(hi - lo, minSpan);
    if (span <= 0) span = 1;
    span *= (1 + 2 * margin);
    lo = mid - span / 2; hi = mid + span / 2;
}

// ---------------------------------------------------------------------------------
// ① 특징 스윕
// ---------------------------------------------------------------------------------
cv::Mat plotFeatureSweeps(const std::vector<pa::Sample>& s, const pa::Calibration& cal) {
    const int NR = 4, W = 1400, pw = 560, ph = 230, lx[2] = { 110, 790 }, ty = 70, gap = 320;
    const int H = ty + NR * gap + 10;
    cv::Mat img(H, W, CV_8UC3, cv::Scalar::all(255));
    const int  fk[NR]  = { 2, 1, 3, 5 };     // tC, sL, sC, sR
    const char* fname[NR] = { "tC (px): shift of center QR",
                              "sL (px/100px): x-scale change of LEFT QR",
                              "sC (px/100px): x-scale change of CENTER QR",
                              "sR (px/100px): x-scale change of RIGHT QR" };
    for (int col = 0; col < 2; ++col) {
        char sw = col == 0 ? 'Y' : 'T';
        for (int r = 0; r < NR; ++r) {
            int k = fk[r];
            std::vector<double> xs, ys;
            for (const pa::Sample& q : s) {
                if (!q.feat.valid || (q.sweep != sw && q.sweep != 'C')) continue;
                xs.push_back(sw == 'Y' ? q.trueY : q.trueT);
                ys.push_back(q.feat.f[k]);
            }
            double x0, x1, y0, y1;
            Panel::autoRange(xs, x0, x1, 0.03);
            Panel::autoRange(ys, y0, y1, 0.12, k % 2 ? 0.05 : 0.0);
            Panel P(img, cv::Rect(lx[col], ty + r * gap, pw, ph), x0, x1, y0, y1);
            P.frame(std::string(sw == 'Y' ? "[Y sweep] " : "[Theta sweep] ") + fname[r],
                    sw == 'Y' ? "robot Y (mm)" : "robot Theta (deg)", "");
            for (const pa::Sample& q : s) {
                if (!q.feat.valid || (q.sweep != sw && q.sweep != 'C')) continue;
                P.point(sw == 'Y' ? q.trueY : q.trueT, q.feat.f[k], q.train ? C_TRAIN : C_TEST, 3, q.train);
            }
            if (cal.ok) {
                double slope = sw == 'Y' ? cal.a[k] : cal.b[k];
                P.line(x0, cal.c[k] + slope * x0, x1, cal.c[k] + slope * x1, C_FIT, 2);
                char buf[160];
                std::snprintf(buf, sizeof(buf), "slope %.4f /%s   noise sigma %.4f", slope, sw == 'Y' ? "mm" : "deg", cal.sigma(k));
                P.text(buf, 0, C_FIT);
            }
        }
    }
    cv::putText(img, "filled = train (even index), open = test (odd index), red line = calibrated linear model. "
                     "Theta changes sL and sR in opposite directions (perspective) - Y does not.",
                cv::Point(40, H - 14), cv::FONT_HERSHEY_SIMPLEX, 0.48, C_TEXT, 1, cv::LINE_AA);
    return img;
}

// ---------------------------------------------------------------------------------
// ② 추정 오차
// ---------------------------------------------------------------------------------
cv::Mat plotEstimation(const std::vector<pa::Sample>& s, double tolY_mm, double tolT_deg) {
    const int W = 1400, H = 860, pw = 560, ph = 290, lx[2] = { 110, 790 }, ty[2] = { 70, 480 };
    cv::Mat img(H, W, CV_8UC3, cv::Scalar::all(255));
    for (int row = 0; row < 2; ++row) {
        char sw = row == 0 ? 'Y' : 'T';
        for (int col = 0; col < 2; ++col) {          // col 0 = Y 오차(µm), col 1 = Θ 오차(deg)
            std::vector<double> xs, ys;
            int n = 0, pass = 0; double ss = 0, mx = 0;
            double tol = col == 0 ? tolY_mm * 1000 : tolT_deg;
            for (const pa::Sample& q : s) {
                if (q.train || q.sweep != sw || !q.feat.valid) continue;
                double x = sw == 'Y' ? q.trueY : q.trueT;
                double e = col == 0 ? (q.estY - q.trueY) * 1000 : (q.estT - q.trueT);
                xs.push_back(x); ys.push_back(e);
                n++; ss += e * e; mx = std::max(mx, std::fabs(e)); if (std::fabs(e) <= tol) pass++;
            }
            double x0, x1, y0, y1;
            Panel::autoRange(xs, x0, x1, 0.03);
            std::vector<double> yr = ys; yr.push_back(tol * 1.3); yr.push_back(-tol * 1.3);
            Panel::autoRange(yr, y0, y1, 0.05);
            Panel P(img, cv::Rect(lx[col], ty[row], pw, ph), x0, x1, y0, y1);
            P.band(-tol, tol, C_BAND);
            std::string title = std::string(sw == 'Y' ? "[Y sweep test] " : "[Theta sweep test] ") +
                                (col == 0 ? "Y error (um)" : "Theta error (deg)");
            P.frame(title, sw == 'Y' ? "true Y (mm)" : "true Theta (deg)", "");
            P.hline(0, C_AXIS, 1);
            for (size_t i = 0; i < xs.size(); ++i)
                P.point(xs[i], ys[i], std::fabs(ys[i]) <= tol ? C_TRAIN : cv::Scalar(40, 40, 220), 3);
            char buf[200];
            std::snprintf(buf, sizeof(buf), "RMS %.*f  max %.*f  pass %d/%d (tol +-%g)",
                          col == 0 ? 1 : 4, n ? std::sqrt(ss / n) : 0.0, col == 0 ? 1 : 4, mx, pass, n, tol);
            P.text(buf, 0, cv::Scalar(20, 20, 20));
        }
    }
    cv::putText(img, "green band = tolerance (Y 90 um, Theta 0.09 deg). Off-axis panels show crosstalk "
                     "(e.g. Theta error while only Y moved).",
                cv::Point(110, H - 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, C_TEXT, 1, cv::LINE_AA);
    return img;
}

// ---------------------------------------------------------------------------------
// ③ QR 측정 오버레이
// ---------------------------------------------------------------------------------
cv::Mat drawQROverlay(const cv::Mat& image, const pa::PhaseAligner& al, const pa::Features& f,
                      const pa::Estimate& est, const std::string& title) {
    cv::Mat bgr;
    if (image.channels() == 1) cv::cvtColor(image, bgr, cv::COLOR_GRAY2BGR); else bgr = image.clone();
    const pa::QRBox* rb = al.referenceBoxes();
    int x0 = std::max(0, rb[0].rect.x - 260), x1 = std::min(bgr.cols, rb[2].rect.br().x + 260);
    int y0 = std::max(0, rb[1].rect.y - 200), y1 = std::min(bgr.rows, rb[1].rect.br().y + 60);
    cv::Mat crop = bgr(cv::Rect(x0, y0, x1 - x0, y1 - y0)).clone();
    const double S = 1.5;
    cv::resize(crop, crop, cv::Size(), S, S, cv::INTER_LINEAR);
    auto P = [&](double x, double y) { return cv::Point((int)std::lround((x - x0) * S), (int)std::lround((y - y0) * S)); };
    const char* nm[pa::NUM_QR] = { "L", "C", "R" };
    for (int q = 0; q < pa::NUM_QR; ++q) {
        cv::Rect r = al.roi(q);
        // 기준 위치 : 빨간 점선 사각형
        for (int k = 0; k < 2; ++k) {
            double xx = k == 0 ? r.x : r.x + r.width;
            for (int yy = r.y; yy < r.y + r.height; yy += 10)
                cv::line(crop, P(xx, yy), P(xx, std::min(r.y + r.height, yy + 5)), cv::Scalar(50, 50, 230), 2);
        }
        if (!f.valid) continue;
        const pa::QRMeasure& m = f.qr[q];
        cv::Scalar col = m.valid ? C_EDGE[q] : cv::Scalar(0, 0, 255);
        // 측정된 위치 : 중심 이동 t 와 배율 s 를 반영한 좌·우 경계
        double xc = r.x + 0.5 * (r.width - 1);
        double xl = r.x + m.shift + m.scale * (r.x - xc) / 100.0;
        double xr = r.x + r.width + m.shift + m.scale * (r.x + r.width - xc) / 100.0;
        cv::line(crop, P(xl, r.y), P(xl, r.y + r.height), col, 2, cv::LINE_AA);
        cv::line(crop, P(xr, r.y), P(xr, r.y + r.height), col, 2, cv::LINE_AA);
        cv::arrowedLine(crop, P(xc, r.y - 30), P(xc + m.shift, r.y - 30), col, 3, cv::LINE_AA, 0, 0.3);
        char buf[96];
        std::snprintf(buf, sizeof(buf), "%s  t %+.3f px", nm[q], m.shift);
        putTextShadow(crop, buf, P(r.x - 10, r.y - 120), 0.75, col, 2);
        std::snprintf(buf, sizeof(buf), "s %+.4f /100px", m.scale);
        putTextShadow(crop, buf, P(r.x - 10, r.y - 85), 0.65, col, 2);
        std::snprintf(buf, sizeof(buf), "valid %.0f%%  fit %.2f px", 100 * m.validRatio, m.fitStd);
        putTextShadow(crop, buf, P(r.x - 10, r.y + r.height + 25), 0.5, C_TEXT, 1);
    }
    cv::Mat out(crop.rows + 110, crop.cols, CV_8UC3, cv::Scalar::all(255));
    crop.copyTo(out(cv::Rect(0, 110, crop.cols, crop.rows)));
    cv::putText(out, title, cv::Point(16, 34), cv::FONT_HERSHEY_SIMPLEX, 0.8, C_TEXT, 2, cv::LINE_AA);
    char buf[256];
    std::snprintf(buf, sizeof(buf), "coarse (phase correlation) dx %+.2f px, dy %+.2f px   valid QR %d/3",
                  f.coarseDx, f.coarseDy, f.numValidQR);
    cv::putText(out, buf, cv::Point(16, 66), cv::FONT_HERSHEY_SIMPLEX, 0.6, C_TEXT, 1, cv::LINE_AA);
    if (est.valid) {
        std::snprintf(buf, sizeof(buf), "estimate: Y %+.4f mm (+-%.4f)   Theta %+.4f deg (+-%.4f)", est.y, est.sdY, est.t, est.sdT);
        cv::putText(out, buf, cv::Point(16, 98), cv::FONT_HERSHEY_SIMPLEX, 0.65, C_FIT, 2, cv::LINE_AA);
    }
    return out;
}

// ---------------------------------------------------------------------------------
// ④ QR 별 측정 품질
// ---------------------------------------------------------------------------------
cv::Mat plotQRQuality(const std::vector<pa::Sample>& s) {
    const int W = 1400, H = 470;
    cv::Mat img(H, W, CV_8UC3, cv::Scalar::all(255));
    const char* nm[pa::NUM_QR] = { "L", "C", "R" };
    double vr[pa::NUM_QR] = { 0 }, fs[pa::NUM_QR] = { 0 }, gs[pa::NUM_QR] = { 0 };
    int n = 0;
    for (const pa::Sample& q : s) {
        if (!q.feat.valid) continue;
        n++;
        for (int k = 0; k < pa::NUM_QR; ++k) { vr[k] += q.feat.qr[k].validRatio; fs[k] += q.feat.qr[k].fitStd; gs[k] += q.feat.qr[k].gradSpread; }
    }
    for (int k = 0; k < pa::NUM_QR; ++k) { vr[k] *= 100.0 / std::max(1, n); fs[k] /= std::max(1, n); gs[k] /= std::max(1, n); }
    const double* vals[3] = { vr, fs, gs };
    const char* titles[3] = { "valid pixel ratio (%)", "fit residual std (px)", "phase-gradient spread (rel.)" };
    for (int p = 0; p < 3; ++p) {
        double mx = 1e-9;
        for (int k = 0; k < pa::NUM_QR; ++k) mx = std::max(mx, vals[p][k]);
        Panel P(img, cv::Rect(90 + p * 450, 70, 360, 300), -0.5, 2.5, 0, mx * 1.25);
        P.frame(titles[p], "", "");
        for (int k = 0; k < pa::NUM_QR; ++k) {
            cv::rectangle(img, P.px(k - 0.3, vals[p][k]), P.px(k + 0.3, 0), C_EDGE[k], cv::FILLED);
            char b[32];
            std::snprintf(b, sizeof(b), p == 0 ? "%.0f" : "%.3f", vals[p][k]);
            cv::putText(img, b, cv::Point(P.px(k, 0).x - 22, P.px(k, vals[p][k]).y - 6), cv::FONT_HERSHEY_SIMPLEX, 0.45, C_TEXT, 1, cv::LINE_AA);
            cv::putText(img, nm[k], cv::Point(P.px(k, 0).x - 6, 395), cv::FONT_HERSHEY_SIMPLEX, 0.6, C_TEXT, 1, cv::LINE_AA);
        }
    }
    cv::putText(img, "averages over all images. Confidence measures (paper ch.4): weak response and non-linear phase pixels are excluded.",
                cv::Point(40, H - 16), cv::FONT_HERSHEY_SIMPLEX, 0.48, C_TEXT, 1, cv::LINE_AA);
    return img;
}

// ---------------------------------------------------------------------------------
// ⑤ 2변수 vs 3변수 비교
// ---------------------------------------------------------------------------------
cv::Mat plotModelCompare(const std::vector<pa::Sample>& s2, const std::vector<pa::Sample>& s3,
                         double tolY_mm, double tolT_deg) {
    const int W = 1400, H = 860, pw = 560, ph = 290, lx[2] = { 110, 790 }, ty[2] = { 70, 480 };
    cv::Mat img(H, W, CV_8UC3, cv::Scalar::all(255));
    const cv::Scalar C2(150, 150, 150), C3(200, 110, 30);
    for (int row = 0; row < 2; ++row) {
        char sw = row == 0 ? 'Y' : 'T';
        for (int col = 0; col < 2; ++col) {
            double tol = col == 0 ? tolY_mm * 1000 : tolT_deg;
            std::vector<double> xs, ys;
            double ss[2] = { 0, 0 }; int n = 0;
            for (size_t i = 0; i < s3.size(); ++i) {
                const pa::Sample& q = s3[i];
                if (q.train || q.sweep != sw || !q.feat.valid) continue;
                double x = sw == 'Y' ? q.trueY : q.trueT;
                double e2 = col == 0 ? (s2[i].estY - q.trueY) * 1000 : s2[i].estT - q.trueT;
                double e3 = col == 0 ? (q.estY - q.trueY) * 1000 : q.estT - q.trueT;
                xs.push_back(x); ys.push_back(e2); ys.push_back(e3);
                ss[0] += e2 * e2; ss[1] += e3 * e3; n++;
            }
            double x0, x1, y0, y1;
            Panel::autoRange(xs, x0, x1, 0.03);
            std::vector<double> yr = ys; yr.push_back(tol * 1.3); yr.push_back(-tol * 1.3);
            Panel::autoRange(yr, y0, y1, 0.05);
            Panel P(img, cv::Rect(lx[col], ty[row], pw, ph), x0, x1, y0, y1);
            P.band(-tol, tol, C_BAND);
            P.frame(std::string(sw == 'Y' ? "[Y sweep test] " : "[Theta sweep test] ") + (col == 0 ? "Y error (um)" : "Theta error (deg)"),
                    sw == 'Y' ? "true Y (mm)" : "true Theta (deg)", "");
            P.hline(0, C_AXIS, 1);
            for (size_t i = 0; i < s3.size(); ++i) {
                const pa::Sample& q = s3[i];
                if (q.train || q.sweep != sw || !q.feat.valid) continue;
                double x = sw == 'Y' ? q.trueY : q.trueT;
                double e2 = col == 0 ? (s2[i].estY - q.trueY) * 1000 : s2[i].estT - q.trueT;
                double e3 = col == 0 ? (q.estY - q.trueY) * 1000 : q.estT - q.trueT;
                P.point(x, e2, C2, 4, false);
                P.point(x, e3, C3, 3, true);
            }
            char buf[160];
            std::snprintf(buf, sizeof(buf), "RMS  2-var %.*f   3-var %.*f", col == 0 ? 1 : 4, n ? std::sqrt(ss[0] / n) : 0.0,
                          col == 0 ? 1 : 4, n ? std::sqrt(ss[1] / n) : 0.0);
            P.text(buf, 0, cv::Scalar(20, 20, 20));
        }
    }
    cv::putText(img, "gray open = 2-variable [Y, Theta],  blue filled = 3-variable [Y, Theta, common magnification]",
                cv::Point(110, H - 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, C_TEXT, 1, cv::LINE_AA);
    return img;
}

// ---------------------------------------------------------------------------------
// ⑥ 추정된 공통 배율 변화
// ---------------------------------------------------------------------------------
cv::Mat plotMagnification(const std::vector<pa::Sample>& s, const std::vector<double>& mag) {
    const int W = 1400, H = 470;
    cv::Mat img(H, W, CV_8UC3, cv::Scalar::all(255));
    for (int col = 0; col < 2; ++col) {
        char sw = col == 0 ? 'Y' : 'T';
        std::vector<double> xs, ys;
        for (size_t i = 0; i < s.size(); ++i) {
            if (s[i].sweep != sw || std::isnan(mag[i])) continue;
            xs.push_back(sw == 'Y' ? s[i].trueY : s[i].trueT);
            ys.push_back(mag[i]);
        }
        double x0, x1, y0, y1, mu = 0, sd = 0;
        for (double v : ys) mu += v;
        if (!ys.empty()) mu /= ys.size();
        for (double v : ys) sd += (v - mu) * (v - mu);
        if (ys.size() > 1) sd = std::sqrt(sd / (ys.size() - 1));
        Panel::autoRange(xs, x0, x1, 0.03);
        Panel::autoRange(ys, y0, y1, 0.1);
        Panel P(img, cv::Rect(110 + col * 680, 70, 560, 320), x0, x1, y0, y1);
        P.frame(std::string(sw == 'Y' ? "[Y sweep] " : "[Theta sweep] ") + "estimated common magnification (ppm)",
                sw == 'Y' ? "robot Y (mm)" : "robot Theta (deg)", "");
        P.hline(0, C_AXIS, 1);
        for (size_t i = 0; i < xs.size(); ++i) P.point(xs[i], ys[i], C_TRAIN, 3);
        char buf[96];
        std::snprintf(buf, sizeof(buf), "mean %.1f ppm   std %.1f ppm", mu, sd);
        P.text(buf, 0, cv::Scalar(20, 20, 20));
    }
    cv::putText(img, "1 ppm magnification = scene appears 0.0001% larger (e.g. camera-plate distance change of 1e-6 x distance)",
                cv::Point(110, H - 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, C_TEXT, 1, cv::LINE_AA);
    return img;
}

void saveAndShow(const std::string& window, const cv::Mat& img, const std::string& path, bool show) {
    if (!path.empty()) cv::imwrite(path, img);
    if (!show) return;
    double s = std::min(1.0, std::min(1600.0 / img.cols, 900.0 / img.rows));
    cv::namedWindow(window, cv::WINDOW_NORMAL);
    cv::resizeWindow(window, (int)(img.cols * s), (int)(img.rows * s));
    cv::imshow(window, img);
}

void makeDir(const std::string& dir) {
    // "results/h1_split" 처럼 여러 단계 경로도 앞에서부터 하나씩 만든다
    for (size_t i = 1; i <= dir.size(); ++i) {
        if (i < dir.size() && dir[i] != '/' && dir[i] != '\\') continue;
        std::string sub = dir.substr(0, i);
#ifdef _WIN32
        _mkdir(sub.c_str());
#else
        mkdir(sub.c_str(), 0755);
#endif
    }
}

} // namespace pp
