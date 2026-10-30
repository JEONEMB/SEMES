// OpenCV 로 그리는 간단한 차트와 사진 확대 그림. 글자는 Hershey 폰트라 영문·숫자만 쓴다.
#pragma once
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdio>
#include <string>
#include <vector>
#include <fstream>
#include "dataset.hpp"
#include "report.hpp"

namespace hye {

// imwrite 는 한글 경로를 못 쓰므로 PNG 로 인코딩해 직접 쓴다
inline void write_png(const fs::path& p, const cv::Mat& m) {
    std::vector<uchar> buf;
    cv::imencode(".png", m, buf);
    std::ofstream f(p, std::ios::binary);
    f.write((const char*)buf.data(), (std::streamsize)buf.size());
}

// 색: 범주 6색 (BGR). 기본 = 파랑, 강조 = 빨강, 보조 = 회색
inline cv::Scalar color(int i) {
    static const cv::Scalar c[] = {{180, 119, 31}, {14, 127, 255}, {44, 160, 44}, {40, 39, 214}, {189, 103, 148}, {75, 86, 140}};
    return c[((i % 6) + 6) % 6];
}
const cv::Scalar kRed{40, 39, 214}, kGray{160, 160, 160}, kGreen{44, 160, 44}, kBlack{30, 30, 30};

inline void text(cv::Mat& m, const std::string& s, cv::Point p, double scale = 0.5, cv::Scalar c = kBlack, int th = 1) {
    cv::putText(m, s, p, cv::FONT_HERSHEY_SIMPLEX, scale, c, th, cv::LINE_AA);
}
inline int text_w(const std::string& s, double scale = 0.5, int th = 1) {
    int base = 0;
    return cv::getTextSize(s, cv::FONT_HERSHEY_SIMPLEX, scale, th, &base).width;
}

struct Chart {
    struct Series {
        std::vector<double> x, y;
        cv::Scalar c;
        std::string label;
        int kind;  // 0 점, 1 선, 2 선+점
        int r;
    };
    std::string title, xlabel, ylabel;
    int W = 900, H = 560, L = 80, R = 30, T = 50, B = 60;
    std::vector<Series> s;
    std::vector<std::pair<double, cv::Scalar>> hl, vl;  // 수평·수직 기준선
    std::vector<std::pair<cv::Point2d, std::string>> notes;
    bool equal_axes = false, fixed_range = false;
    double x0 = 0, x1 = 1, y0 = 0, y1 = 1;

    Chart(std::string t, std::string xl, std::string yl) : title(std::move(t)), xlabel(std::move(xl)), ylabel(std::move(yl)) {}
    Chart& scatter(const std::vector<double>& x, const std::vector<double>& y, cv::Scalar c, std::string lab = "", int r = 3) {
        s.push_back({x, y, c, lab, 0, r});
        return *this;
    }
    Chart& line(const std::vector<double>& x, const std::vector<double>& y, cv::Scalar c, std::string lab = "", bool dots = false) {
        s.push_back({x, y, c, lab, dots ? 2 : 1, 3});
        return *this;
    }
    Chart& hline(double v, cv::Scalar c = kGray) { hl.push_back({v, c}); return *this; }
    Chart& vline(double v, cv::Scalar c = kGray) { vl.push_back({v, c}); return *this; }
    Chart& note(double x, double y, std::string t) { notes.push_back({{x, y}, std::move(t)}); return *this; }
    Chart& range(double a, double b, double c, double d) { x0 = a; x1 = b; y0 = c; y1 = d; fixed_range = true; return *this; }

    static double nice(double span) {
        double e = std::pow(10, std::floor(std::log10(span / 6)));
        for (double m : {1.0, 2.0, 2.5, 5.0, 10.0}) if (span / (m * e) <= 7) return m * e;
        return 10 * e;
    }
    void autorange() {
        if (fixed_range) return;
        double a = 1e300, b = -1e300, c = 1e300, d = -1e300;
        for (auto& q : s)
            for (size_t i = 0; i < q.x.size(); ++i) {
                if (!std::isfinite(q.x[i]) || !std::isfinite(q.y[i])) continue;
                a = std::min(a, q.x[i]); b = std::max(b, q.x[i]); c = std::min(c, q.y[i]); d = std::max(d, q.y[i]);
            }
        for (auto& h : hl) { c = std::min(c, h.first); d = std::max(d, h.first); }
        for (auto& v : vl) { a = std::min(a, v.first); b = std::max(b, v.first); }
        if (a > b) { a = 0; b = 1; c = 0; d = 1; }
        if (b - a < 1e-12) { a -= 1; b += 1; }
        if (d - c < 1e-12) { c -= 1; d += 1; }
        double px = (b - a) * 0.05, py = (d - c) * 0.08;
        x0 = a - px; x1 = b + px; y0 = c - py; y1 = d + py;
    }
    cv::Point P(double x, double y) const {
        return {L + (int)std::lround((x - x0) / (x1 - x0) * (W - L - R)), T + (int)std::lround((y1 - y) / (y1 - y0) * (H - T - B))};
    }
    cv::Mat render() {
        autorange();
        cv::Mat m(H, W, CV_8UC3, cv::Scalar(255, 255, 255));
        cv::rectangle(m, {L, T}, {W - R, H - B}, kGray, 1);
        auto ticks = [&](double a, double b, bool isx) {
            double st = nice(b - a);
            for (double v = std::ceil(a / st) * st; v <= b + 1e-9; v += st) {
                char buf[32];
                std::snprintf(buf, sizeof buf, st < 0.01 ? "%.3f" : st < 0.1 ? "%.2f" : st < 1 ? "%.1f" : "%.0f", std::abs(v) < 1e-12 ? 0.0 : v);
                if (isx) {
                    auto p = P(v, y0);
                    cv::line(m, {p.x, T}, {p.x, H - B}, {235, 235, 235}, 1);
                    text(m, buf, {p.x - text_w(buf, 0.4) / 2, H - B + 18}, 0.4);
                } else {
                    auto p = P(x0, v);
                    cv::line(m, {L, p.y}, {W - R, p.y}, {235, 235, 235}, 1);
                    text(m, buf, {L - 8 - text_w(buf, 0.4), p.y + 5}, 0.4);
                }
            }
        };
        ticks(x0, x1, true);
        ticks(y0, y1, false);
        for (auto& h : hl) { auto p = P(x0, h.first); cv::line(m, {L, p.y}, {W - R, p.y}, h.second, 1, cv::LINE_AA); }
        for (auto& v : vl) { auto p = P(v.first, y0); cv::line(m, {p.x, T}, {p.x, H - B}, v.second, 1, cv::LINE_AA); }
        cv::Rect clip(L, T, W - L - R + 1, H - T - B + 1);
        for (auto& q : s) {
            if (q.kind >= 1)
                for (size_t i = 1; i < q.x.size(); ++i) {
                    auto a = P(q.x[i - 1], q.y[i - 1]), b = P(q.x[i], q.y[i]);
                    if (cv::clipLine(clip, a, b)) cv::line(m, a, b, q.c, 2, cv::LINE_AA);
                }
            if (q.kind != 1)
                for (size_t i = 0; i < q.x.size(); ++i) {
                    if (!std::isfinite(q.y[i])) continue;
                    auto p = P(q.x[i], q.y[i]);
                    if (clip.contains(p)) cv::circle(m, p, q.r, q.c, -1, cv::LINE_AA);
                }
        }
        for (auto& n : notes) {
            auto p = P(n.first.x, n.first.y);
            if (!clip.contains(p)) continue;
            cv::circle(m, p, 7, kRed, 2, cv::LINE_AA);
            text(m, n.second, {std::min(p.x + 9, W - R - text_w(n.second, 0.4)), p.y - 8}, 0.4, kRed);
        }
        text(m, title, {L, 30}, 0.65, kBlack, 2);
        text(m, xlabel, {(L + W - R) / 2 - text_w(xlabel) / 2, H - 18});
        cv::Mat yl(30, text_w(ylabel) + 10, CV_8UC3, cv::Scalar(255, 255, 255));
        text(yl, ylabel, {5, 20});
        cv::rotate(yl, yl, cv::ROTATE_90_COUNTERCLOCKWISE);
        if (yl.rows <= H) yl.copyTo(m(cv::Rect(4, std::max(0, (T + H - B) / 2 - yl.rows / 2), yl.cols, yl.rows)));
        int ly = T + 18, k = 0;
        for (auto& q : s) {
            if (q.label.empty()) continue;
            int x = W - R - 10 - text_w(q.label, 0.45) - 18;
            cv::rectangle(m, {x, ly - 9 + k * 20}, {x + 10, ly + 1 + k * 20}, q.c, -1);
            text(m, q.label, {x + 16, ly + k * 20}, 0.45);
            ++k;
        }
        return m;
    }
    void save(const std::string& file) { write_png(image_dir() / fs::u8path(file), render()); }
};

// 막대: 범주별 값
inline void bar_chart(const std::string& file, const std::string& title, const std::vector<std::string>& cats,
                      const std::vector<double>& vals, const std::string& ylabel, int highlight = -1, double ref = kNaN) {
    const int W = std::max(600, 90 * (int)cats.size() + 120), H = 520, L = 80, R = 30, T = 50, B = 110;
    cv::Mat m(H, W, CV_8UC3, cv::Scalar(255, 255, 255));
    double top = 0;
    for (double v : vals) top = std::max(top, v);
    if (std::isfinite(ref)) top = std::max(top, ref);
    top = top <= 0 ? 1 : top * 1.15;
    int n = (int)cats.size(), bw = (W - L - R) / std::max(1, n);
    cv::line(m, {L, H - B}, {W - R, H - B}, kGray, 1);
    for (int i = 0; i < n; ++i) {
        int h = (int)std::lround(vals[i] / top * (H - T - B));
        cv::Rect r(L + i * bw + bw / 6, H - B - h, bw * 2 / 3, h);
        cv::rectangle(m, r, i == highlight ? kRed : color(0), -1);
        std::string v = fmt(vals[i], vals[i] >= 100 ? 0 : vals[i] >= 10 ? 1 : 3);
        text(m, v, {r.x + r.width / 2 - text_w(v, 0.4) / 2, r.y - 6}, 0.4);
        cv::Mat lab(26, text_w(cats[i], 0.42) + 8, CV_8UC3, cv::Scalar(255, 255, 255));
        text(lab, cats[i], {4, 18}, 0.42);
        cv::rotate(lab, lab, cv::ROTATE_90_COUNTERCLOCKWISE);
        int lh = std::min(lab.rows, B - 8);
        lab(cv::Rect(0, lab.rows - lh, lab.cols, lh)).copyTo(m(cv::Rect(std::min(W - lab.cols, r.x + r.width / 2 - 13), H - B + 4, lab.cols, lh)));
    }
    if (std::isfinite(ref)) {
        int y = H - B - (int)std::lround(ref / top * (H - T - B));
        cv::line(m, {L, y}, {W - R, y}, kRed, 1, cv::LINE_AA);
    }
    text(m, title, {L, 30}, 0.65, kBlack, 2);
    text(m, ylabel, {8, T - 10}, 0.45);
    write_png(image_dir() / fs::u8path(file), m);
}

// 사진 일부를 확대해 표시. pts 는 원본 좌표, 빨간 십자 + 화살표(이동 벡터 x 배율)
struct Mark {
    cv::Point2d p;
    cv::Point2d d{0, 0};
    std::string label;
    cv::Scalar c = kRed;
};
struct Box {
    cv::Rect2d r;
    cv::Scalar c = kRed;
    std::string label;
};
inline cv::Mat zoom(const cv::Mat& img, cv::Rect roi, int out_w, const std::vector<Mark>& marks = {}, double arrow_gain = 0,
                    const std::string& caption = "", const std::vector<Box>& boxes = {}) {
    roi &= cv::Rect(0, 0, img.cols, img.rows);
    cv::Mat c = img(roi).clone();
    if (c.channels() == 1) cv::cvtColor(c, c, cv::COLOR_GRAY2BGR);
    double sc = (double)out_w / roi.width;
    cv::resize(c, c, {out_w, (int)std::lround(roi.height * sc)}, 0, 0, sc >= 1 ? cv::INTER_NEAREST : cv::INTER_AREA);
    for (auto& b : boxes) {
        cv::Point a((int)std::lround((b.r.x - roi.x) * sc), (int)std::lround((b.r.y - roi.y) * sc));
        cv::Point e((int)std::lround((b.r.x + b.r.width - roi.x) * sc), (int)std::lround((b.r.y + b.r.height - roi.y) * sc));
        cv::rectangle(c, a, e, b.c, 2, cv::LINE_AA);
        if (!b.label.empty()) text(c, b.label, {a.x + 4, e.y - 6}, 0.45, b.c);
    }
    for (auto& k : marks) {
        cv::Point p((int)std::lround((k.p.x - roi.x) * sc), (int)std::lround((k.p.y - roi.y) * sc));
        cv::drawMarker(c, p, k.c, cv::MARKER_CROSS, 14, 2, cv::LINE_AA);
        if (arrow_gain > 0 && (k.d.x != 0 || k.d.y != 0))
            cv::arrowedLine(c, p, p + cv::Point((int)std::lround(k.d.x * arrow_gain * sc), (int)std::lround(k.d.y * arrow_gain * sc)), k.c, 2,
                            cv::LINE_AA, 0, 0.2);
        if (!k.label.empty()) text(c, k.label, p + cv::Point(8, -8), 0.45, k.c);
    }
    if (!caption.empty()) {
        cv::rectangle(c, {0, 0}, {text_w(caption, 0.55) + 12, 26}, cv::Scalar(255, 255, 255), -1);
        text(c, caption, {6, 19}, 0.55);
    }
    return c;
}

// 같은 높이로 가로 이어 붙이기
inline cv::Mat hstack(const std::vector<cv::Mat>& v, int gap = 8) {
    int h = 0, w = 0;
    for (auto& m : v) { h = std::max(h, m.rows); w += m.cols + gap; }
    cv::Mat o(h, std::max(1, w - gap), CV_8UC3, cv::Scalar(255, 255, 255));
    int x = 0;
    for (auto& m : v) { m.copyTo(o(cv::Rect(x, 0, m.cols, m.rows))); x += m.cols + gap; }
    return o;
}
inline cv::Mat vstack(const std::vector<cv::Mat>& v, int gap = 8) {
    int h = 0, w = 0;
    for (auto& m : v) { w = std::max(w, m.cols); h += m.rows + gap; }
    cv::Mat o(std::max(1, h - gap), w, CV_8UC3, cv::Scalar(255, 255, 255));
    int y = 0;
    for (auto& m : v) { m.copyTo(o(cv::Rect(0, y, m.cols, m.rows))); y += m.rows + gap; }
    return o;
}
inline void save_png(const std::string& file, const cv::Mat& m) { write_png(image_dir() / fs::u8path(file), m); }

// 정답 대비 추정 그림: 왼쪽 Y, 오른쪽 T. 통과선 ±0.09, 튀는 점은 이름 표시
inline void pred_vs_truth(const std::string& file, const std::string& title, const std::vector<double>& ty, const std::vector<double>& py,
                          const std::vector<double>& tt, const std::vector<double>& pt, const std::vector<std::string>& names, size_t mark = 5) {
    std::vector<double> ey(ty.size()), et(tt.size());
    for (size_t i = 0; i < ty.size(); ++i) { ey[i] = py[i] - ty[i]; et[i] = pt[i] - tt[i]; }
    Chart a(title + " - Y error", "true Y (mm)", "pred - true (mm)");
    a.scatter(ty, ey, color(0)).hline(kPass, kRed).hline(-kPass, kRed).hline(0);
    for (auto i : worst(ey, mark)) if (std::abs(ey[i]) > 1e-9) a.note(ty[i], ey[i], names[i]);
    Chart b(title + " - T error", "true T (deg)", "pred - true (deg)");
    b.scatter(tt, et, color(1)).hline(kPass, kRed).hline(-kPass, kRed).hline(0);
    for (auto i : worst(et, mark)) if (std::abs(et[i]) > 1e-9) b.note(tt[i], et[i], names[i]);
    save_png(file, hstack({a.render(), b.render()}));
}

}  // namespace hye
