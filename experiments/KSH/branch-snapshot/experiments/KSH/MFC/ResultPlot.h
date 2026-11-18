// =====================================================================================
// ResultPlot.h — 일괄 분석 결과(CSV)로 "모델이 얼마나 잘 맞추는지" 그래프를 그린다 (OpenCV 그리기만 사용)
//   헤더만으로 구성 (QR_OpenCV_CheckDlg.cpp 에서만 포함) → 프로젝트 파일 수정 없이 사용
//   그래프 안 글자는 OpenCV putText 가 한글을 못 그리므로 영어로 표기
//
//   ┌ 요약 : 전체 / 사진 종류별 (center, Y only, Θ only, Y+Θ) Θ·Y RMS, 최대, 통과율 ┐
//   │ ① 추정 Θ vs 정답 Θ (y = x)   ② Θ 오차 vs 정답 Θ (±허용 띠)   ③ Θ 오차 분포        │
//   │ ④ 추정 Y vs 정답 Y (y = x)   ⑤ Y 오차 vs 정답 Y (±허용 띠)   ⑥ Y 오차 vs Θ 오차   │
//   └──────────────────────────────────────────────────────────────────────────┘
// =====================================================================================
#pragma once
#include "cv_inc.h"
#include <vector>
#include <string>
#include <cmath>
#include <cstdio>
#include <algorithm>

namespace rp {

// CSV 한 줄 (정답·추정)
struct Rec {
    std::string name;
    double trueY = 0, estY = 0;   // mm
    double trueT = 0, estT = 0;   // deg
    bool   hasTruth = false, hasEst = false;
};

// 사진 종류 : 정답으로 구분
enum Kind { K_CENTER = 0, K_Y, K_T, K_MIX, K_COUNT };
inline int kindOf(const Rec& r) {
    const bool y0 = std::fabs(r.trueY) < 1e-9, t0 = std::fabs(r.trueT) < 1e-9;
    if (y0 && t0) return K_CENTER;
    if (t0) return K_Y;
    if (y0) return K_T;
    return K_MIX;
}
inline const char* kindName(int k) {
    static const char* n[K_COUNT] = { "center", "Y only", "Theta only", "Y + Theta" };
    return n[k];
}
inline cv::Scalar kindColor(int k) {            // BGR
    static const cv::Scalar c[K_COUNT] = { cv::Scalar(120, 120, 120), cv::Scalar(200, 120, 30),
                                           cv::Scalar(30, 130, 240), cv::Scalar(60, 170, 60) };
    return c[k];
}

// ---------------------------------------------------------------------------------
// 그래프 한 칸
// ---------------------------------------------------------------------------------
class Panel {
public:
    Panel(cv::Mat& c, cv::Rect a, double x0, double x1, double y0, double y1)
        : c_(c), a_(a), x0_(x0), x1_(x1), y0_(y0), y1_(y1) {
        if (x1_ - x0_ < 1e-12) x1_ = x0_ + 1;
        if (y1_ - y0_ < 1e-12) y1_ = y0_ + 1;
    }
    cv::Point px(double x, double y) const {
        return cv::Point(a_.x + (int)std::lround((x - x0_) / (x1_ - x0_) * a_.width),
                         a_.y + a_.height - (int)std::lround((y - y0_) / (y1_ - y0_) * a_.height));
    }
    void frame(const std::string& title, const std::string& xl, const std::string& yl) {
        const cv::Scalar grid(232, 232, 232), txt(50, 50, 50);
        double sx = niceStep((x1_ - x0_) / 6), sy = niceStep((y1_ - y0_) / 5);
        for (double x = std::ceil(x0_ / sx) * sx; x <= x1_ + 1e-9; x += sx) {
            cv::Point p = px(x, y0_);
            cv::line(c_, cv::Point(p.x, a_.y), cv::Point(p.x, a_.y + a_.height), grid, 1);
            cv::putText(c_, fmt(std::fabs(x) < sx * 1e-6 ? 0 : x, prec(sx)), cv::Point(p.x - 18, a_.y + a_.height + 18),
                        cv::FONT_HERSHEY_SIMPLEX, 0.42, txt, 1, cv::LINE_AA);
        }
        for (double y = std::ceil(y0_ / sy) * sy; y <= y1_ + 1e-9; y += sy) {
            cv::Point p = px(x0_, y);
            cv::line(c_, cv::Point(a_.x, p.y), cv::Point(a_.x + a_.width, p.y), grid, 1);
            cv::putText(c_, fmt(std::fabs(y) < sy * 1e-6 ? 0 : y, prec(sy)), cv::Point(a_.x - 66, p.y + 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.42, txt, 1, cv::LINE_AA);
        }
        cv::rectangle(c_, a_, cv::Scalar(150, 150, 150), 1);
        cv::putText(c_, title, cv::Point(a_.x, a_.y - 12), cv::FONT_HERSHEY_SIMPLEX, 0.58, cv::Scalar(10, 10, 10), 1, cv::LINE_AA);
        cv::putText(c_, xl, cv::Point(a_.x + a_.width / 2 - 60, a_.y + a_.height + 40), cv::FONT_HERSHEY_SIMPLEX, 0.47, txt, 1, cv::LINE_AA);
        cv::putText(c_, yl, cv::Point(a_.x + a_.width - 9 * (int)yl.size(), a_.y - 12), cv::FONT_HERSHEY_SIMPLEX, 0.45,
                    cv::Scalar(130, 130, 130), 1, cv::LINE_AA);
    }
    void axes() {                                                 // 원점을 지나는 x = 0, y = 0 축 (진한 회색)
        const cv::Scalar ax(90, 90, 90);
        if (x0_ <= 0 && 0 <= x1_) line(0, y0_, 0, y1_, ax, 1);
        if (y0_ <= 0 && 0 <= y1_) line(x0_, 0, x1_, 0, ax, 1);
    }
    void band(double lo, double hi, cv::Scalar color) {           // 가로 띠 (허용 범위)
        double a = (std::max)(lo, y0_), b = (std::min)(hi, y1_);
        if (b <= a) return;
        cv::rectangle(c_, cv::Rect(px(x0_, b), px(x1_, a)), color, cv::FILLED);
    }
    void vband(double lo, double hi, cv::Scalar color) {          // 세로 띠
        double a = (std::max)(lo, x0_), b = (std::min)(hi, x1_);
        if (b <= a) return;
        cv::rectangle(c_, cv::Rect(px(a, y1_), px(b, y0_)), color, cv::FILLED);
    }
    void line(double xa, double ya, double xb, double yb, cv::Scalar color, int thick = 1) {
        cv::Point a = px(xa, ya), b = px(xb, yb);
        cv::Rect clip(a_.x, a_.y, a_.width + 1, a_.height + 1);
        if (cv::clipLine(clip, a, b)) cv::line(c_, a, b, color, thick, cv::LINE_AA);
    }
    void point(double x, double y, cv::Scalar color, int r = 3) {
        if (x < x0_ || x > x1_ || y < y0_ || y > y1_) return;
        cv::circle(c_, px(x, y), r, color, cv::FILLED, cv::LINE_AA);
    }
    void bar(double xa, double xb, double h, cv::Scalar color) {
        if (h <= 0) return;
        cv::Point p0 = px(xa, (std::min)(h, y1_)), p1 = px(xb, y0_);
        cv::rectangle(c_, cv::Rect(p0, p1), color, cv::FILLED);
        cv::rectangle(c_, cv::Rect(p0, p1), cv::Scalar(255, 255, 255), 1);
    }
    void text(const std::string& s, int row, cv::Scalar color) {
        cv::Point p(a_.x + 10, a_.y + 22 + row * 22);
        cv::putText(c_, s, p, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 4, cv::LINE_AA);
        cv::putText(c_, s, p, cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv::LINE_AA);
    }
    static double niceStep(double raw) {
        if (raw <= 0) return 1;
        double e = std::pow(10.0, std::floor(std::log10(raw))), f = raw / e;
        return ((f < 1.5) ? 1 : (f < 3) ? 2 : (f < 7) ? 5 : 10) * e;
    }
    static std::string fmt(double v, int p) { char b[64]; std::snprintf(b, sizeof(b), "%.*f", p, v); return b; }
    static int prec(double step) { return step >= 1 ? 0 : (int)std::ceil(-std::log10(step) - 1e-9); }

private:
    cv::Mat& c_;
    cv::Rect a_;
    double x0_, x1_, y0_, y1_;
};

// 범위 : 자료 최소~최대에 여유. minSpan 보다 좁으면 넓힘
inline void autoRange(const std::vector<double>& v, double& lo, double& hi, double margin, double minSpan) {
    if (v.empty()) { lo = -1; hi = 1; return; }
    lo = *std::min_element(v.begin(), v.end());
    hi = *std::max_element(v.begin(), v.end());
    double mid = 0.5 * (lo + hi), span = (std::max)(hi - lo, minSpan);
    if (span <= 0) span = 1;
    span *= 1 + 2 * margin;
    lo = mid - span / 2; hi = mid + span / 2;
}
// 원점(0)을 가운데 둔 대칭 범위 : ±max(|값|)·1.15, 최소 ±minHalf  → 한쪽 값만 있어도 원점이 항상 가운데
inline void symRange(const std::vector<double>& v, double minHalf, double& lo, double& hi) {
    double m = minHalf;
    for (double x : v) m = (std::max)(m, std::fabs(x) * 1.15);
    lo = -m; hi = m;
}

struct Stat { int n = 0, pass = 0; double rms = 0, maxAbs = 0, mean = 0; };
inline Stat statOf(const std::vector<double>& e, double tol) {
    Stat s;
    for (double x : e) { s.n++; s.rms += x * x; s.mean += x; s.maxAbs = (std::max)(s.maxAbs, std::fabs(x)); if (std::fabs(x) <= tol) s.pass++; }
    if (s.n) { s.rms = std::sqrt(s.rms / s.n); s.mean /= s.n; }
    return s;
}

// ---------------------------------------------------------------------------------
// 보고서 그림 한 장
// ---------------------------------------------------------------------------------
inline cv::Mat drawReport(const std::vector<Rec>& recs, double tolY_mm, double tolT_deg, const std::string& title)
{
    const int W = 1820, H = 1180;
    cv::Mat img(H, W, CV_8UC3, cv::Scalar::all(255));
    cv::putText(img, title, cv::Point(24, 38), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(10, 10, 10), 2, cv::LINE_AA);

    // 정답과 추정이 모두 있는 행만
    std::vector<const Rec*> v;
    for (const Rec& r : recs) if (r.hasTruth && r.hasEst) v.push_back(&r);
    if (v.empty()) {
        cv::putText(img, "No rows with both ground truth (file name) and estimate.", cv::Point(24, 90),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(30, 30, 200), 2, cv::LINE_AA);
        return img;
    }
    std::vector<double> tT, eTv, tY, eYv, eT, eY;       // eT : deg, eY : um
    for (const Rec* r : v) {
        tT.push_back(r->trueT); eTv.push_back(r->estT); eT.push_back(r->estT - r->trueT);
        tY.push_back(r->trueY); eYv.push_back(r->estY); eY.push_back((r->estY - r->trueY) * 1000);
    }

    // ---------------------------------------------------------------- 요약 표
    {
        char b[256];
        int y = 78;
        const cv::Scalar dark(30, 30, 30);
        std::snprintf(b, sizeof(b), "%-12s %5s | %-30s | %-30s | %s", "group", "n",
                      "Theta err: RMS / max / pass", "Y err: RMS / max / pass", "both pass");
        cv::putText(img, b, cv::Point(24, y), cv::FONT_HERSHEY_PLAIN, 1.25, dark, 1, cv::LINE_AA);
        y += 26;
        for (int k = -1; k < K_COUNT; ++k) {
            std::vector<double> a, c;
            int both = 0;
            for (size_t i = 0; i < v.size(); ++i) {
                if (k >= 0 && kindOf(*v[i]) != k) continue;
                a.push_back(eT[i]); c.push_back(eY[i]);
                if (std::fabs(eT[i]) <= tolT_deg && std::fabs(eY[i]) <= tolY_mm * 1000) both++;
            }
            if (a.empty()) continue;
            Stat st = statOf(a, tolT_deg), sy = statOf(c, tolY_mm * 1000);
            char tb[64], yb[64];
            std::snprintf(tb, sizeof(tb), "%.4f / %.4f deg / %3.0f%%", st.rms, st.maxAbs, 100.0 * st.pass / st.n);
            std::snprintf(yb, sizeof(yb), "%5.1f / %5.1f um / %3.0f%%", sy.rms, sy.maxAbs, 100.0 * sy.pass / sy.n);
            std::snprintf(b, sizeof(b), "%-12s %5d | %-30s | %-30s | %d/%d (%.0f%%)", k < 0 ? "ALL" : kindName(k),
                          st.n, tb, yb, both, st.n, 100.0 * both / st.n);
            cv::putText(img, b, cv::Point(24, y), cv::FONT_HERSHEY_PLAIN, 1.25, k < 0 ? cv::Scalar(20, 20, 160) : kindColor(k),
                        k < 0 ? 2 : 1, cv::LINE_AA);
            y += 24;
        }
        char tol[160];
        std::snprintf(tol, sizeof(tol), "tolerance: Theta +-%.3f deg, Y +-%.0f um", tolT_deg, tolY_mm * 1000);
        cv::putText(img, tol, cv::Point(1180, 78), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(90, 90, 90), 1, cv::LINE_AA);
        // 범례
        for (int k = 0; k < K_COUNT; ++k) {
            cv::circle(img, cv::Point(1190 + k * 150, 108), 6, kindColor(k), cv::FILLED, cv::LINE_AA);
            cv::putText(img, kindName(k), cv::Point(1202 + k * 150, 113), cv::FONT_HERSHEY_SIMPLEX, 0.5, kindColor(k), 1, cv::LINE_AA);
        }
    }

    const int pw = 500, ph = 330, lx[3] = { 100, 700, 1300 }, ty[2] = { 270, 760 };
    const cv::Scalar okBand(215, 240, 215), diag(150, 150, 150), red(40, 40, 210);

    auto drawScatter = [&](Panel& P, const std::vector<double>& xs, const std::vector<double>& ys) {
        for (size_t i = 0; i < v.size(); ++i) P.point(xs[i], ys[i], kindColor(kindOf(*v[i])), 3);
    };

    // ---------------------------------------------------------------- ① 추정 Θ vs 정답 Θ
    {
        std::vector<double> all(tT); all.insert(all.end(), eTv.begin(), eTv.end());
        double lo, hi; symRange(all, 0.2, lo, hi);
        Panel P(img, cv::Rect(lx[0], ty[0], pw, ph), lo, hi, lo, hi);
        P.frame("(1) estimated vs true Theta", "true Theta (deg)", "est (deg)");
        P.axes();
        P.line(lo, lo, hi, hi, diag, 1);
        drawScatter(P, tT, eTv);
        P.text("on the gray line = perfect", 0, cv::Scalar(90, 90, 90));
    }
    // ---------------------------------------------------------------- ② Θ 오차 vs 정답 Θ
    {
        double xlo, xhi, ylo, yhi;
        symRange(tT, 0.2, xlo, xhi);
        symRange(eT, tolT_deg * 2.5, ylo, yhi);
        Panel P(img, cv::Rect(lx[1], ty[0], pw, ph), xlo, xhi, ylo, yhi);
        P.band(-tolT_deg, tolT_deg, okBand);
        P.frame("(2) Theta error vs true Theta", "true Theta (deg)", "error (deg)");
        P.axes();
        drawScatter(P, tT, eT);
        Stat s = statOf(eT, tolT_deg);
        char b[128];
        std::snprintf(b, sizeof(b), "RMS %.4f  max %.4f  pass %d/%d", s.rms, s.maxAbs, s.pass, s.n);
        P.text(b, 0, red);
        P.text("green band = tolerance", 1, cv::Scalar(60, 140, 60));
    }
    // ---------------------------------------------------------------- ③ Θ 오차 분포
    {
        double lo, hi; symRange(eT, tolT_deg * 2.5, lo, hi);
        const int NB = 31;
        std::vector<int> cnt(NB, 0);
        for (double e : eT) {
            int b = (int)std::floor((e - lo) / (hi - lo) * NB);
            cnt[(std::max)(0, (std::min)(NB - 1, b))]++;
        }
        int mx = *std::max_element(cnt.begin(), cnt.end());
        Panel P(img, cv::Rect(lx[2], ty[0], pw, ph), lo, hi, 0, mx * 1.15 + 1);
        P.vband(-tolT_deg, tolT_deg, okBand);
        P.frame("(3) Theta error histogram", "error (deg)", "count");
        P.axes();
        for (int b = 0; b < NB; ++b)
            P.bar(lo + (hi - lo) * b / NB, lo + (hi - lo) * (b + 1) / NB, cnt[b], cv::Scalar(30, 130, 240));
        Stat s = statOf(eT, tolT_deg);
        char t[128];
        std::snprintf(t, sizeof(t), "mean %+.4f  (bias)", s.mean);
        P.text(t, 0, red);
    }
    // ---------------------------------------------------------------- ④ 추정 Y vs 정답 Y
    {
        std::vector<double> all(tY); all.insert(all.end(), eYv.begin(), eYv.end());
        double lo, hi; symRange(all, 0.4, lo, hi);
        Panel P(img, cv::Rect(lx[0], ty[1], pw, ph), lo, hi, lo, hi);
        P.frame("(4) estimated vs true Y", "true Y (mm)", "est (mm)");
        P.axes();
        P.line(lo, lo, hi, hi, diag, 1);
        drawScatter(P, tY, eYv);
    }
    // ---------------------------------------------------------------- ⑤ Y 오차 vs 정답 Y
    {
        double xlo, xhi, ylo, yhi;
        symRange(tY, 0.4, xlo, xhi);
        symRange(eY, tolY_mm * 1000 * 1.5, ylo, yhi);
        Panel P(img, cv::Rect(lx[1], ty[1], pw, ph), xlo, xhi, ylo, yhi);
        P.band(-tolY_mm * 1000, tolY_mm * 1000, okBand);
        P.frame("(5) Y error vs true Y", "true Y (mm)", "error (um)");
        P.axes();
        drawScatter(P, tY, eY);
        Stat s = statOf(eY, tolY_mm * 1000);
        char b[128];
        std::snprintf(b, sizeof(b), "RMS %.1f  max %.1f  pass %d/%d", s.rms, s.maxAbs, s.pass, s.n);
        P.text(b, 0, red);
    }
    // ---------------------------------------------------------------- ⑥ Y 오차 vs Θ 오차 (Θ 오차가 Y 로 새는 정도)
    {
        double xlo, xhi, ylo, yhi;
        symRange(eT, tolT_deg * 2.5, xlo, xhi);
        symRange(eY, tolY_mm * 1000 * 1.5, ylo, yhi);
        Panel P(img, cv::Rect(lx[2], ty[1], pw, ph), xlo, xhi, ylo, yhi);
        P.vband(-tolT_deg, tolT_deg, okBand);
        P.frame("(6) Y error vs Theta error", "Theta error (deg)", "Y error (um)");
        P.axes();
        drawScatter(P, eT, eY);
        // 최소제곱 기울기와 상관
        double n = (double)eT.size(), sx = 0, sy = 0, sxx = 0, sxy = 0, syy = 0;
        for (size_t i = 0; i < eT.size(); ++i) { sx += eT[i]; sy += eY[i]; sxx += eT[i] * eT[i]; sxy += eT[i] * eY[i]; syy += eY[i] * eY[i]; }
        double vx = sxx - sx * sx / n, vy = syy - sy * sy / n, cxy = sxy - sx * sy / n;
        if (vx > 0 && vy > 0) {
            double slope = cxy / vx, b0 = (sy - slope * sx) / n, rho = cxy / std::sqrt(vx * vy);
            P.line(xlo, b0 + slope * xlo, xhi, b0 + slope * xhi, red, 2);
            char t[128];
            std::snprintf(t, sizeof(t), "slope %.2f mm/deg  corr %.2f", slope / 1000.0, rho);
            P.text(t, 0, red);
            P.text("strong corr = Y error comes from Theta error", 1, cv::Scalar(90, 90, 90));
        }
    }
    return img;
}

} // namespace rp
