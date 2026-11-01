// =====================================================================================
// plot_utils.hpp — OpenCV 로 그래프·글자·색 지도를 그리는 작은 도구 (영상 위 글자는 영어만 가능)
// =====================================================================================
#ifndef MM_PLOT_UTILS_HPP
#define MM_PLOT_UTILS_HPP

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

namespace mm {

namespace col {
const cv::Scalar white(255, 255, 255), black(0, 0, 0), gray(150, 150, 150), light(225, 225, 225);
const cv::Scalar blue(200, 110, 30), orange(20, 130, 245), green(60, 170, 60), red(50, 50, 220);
const cv::Scalar cyan(220, 200, 0), purple(160, 60, 150), yellow(0, 210, 230);
} // namespace col

inline std::string fmt(const char* f, double a) { char b[128]; std::snprintf(b, sizeof(b), f, a); return b; }
inline std::string fmt(const char* f, double a, double b2) { char b[160]; std::snprintf(b, sizeof(b), f, a, b2); return b; }
inline std::string fmt(const char* f, double a, double b2, double c) { char b[200]; std::snprintf(b, sizeof(b), f, a, b2, c); return b; }

inline void text(cv::Mat& img, const std::string& s, cv::Point p, double scale = 0.5,
                 const cv::Scalar& c = col::black, int thick = 1) {
    cv::putText(img, s, p, cv::FONT_HERSHEY_SIMPLEX, scale, c, thick, cv::LINE_AA);
}

// 2D 선 그래프
class Plot {
public:
    Plot(cv::Mat& img, cv::Rect area, double x0, double x1, double y0, double y1)
        : img_(img), r_(area), x0_(x0), x1_(x1), y0_(y0), y1_(y1) {}

    cv::Point map(double x, double y) const {
        const double fx = (x - x0_) / (x1_ - x0_), fy = (y - y0_) / (y1_ - y0_);
        return cv::Point(r_.x + int(std::lround(fx * r_.width)), r_.y + r_.height - int(std::lround(fy * r_.height)));
    }

    void axes(const std::string& title, const std::string& xl, const std::string& yl, int nx = 5, int ny = 4) {
        cv::rectangle(img_, r_, col::white, cv::FILLED);
        for (int i = 0; i <= nx; ++i) {
            const double x = x0_ + (x1_ - x0_) * i / nx;
            cv::line(img_, map(x, y0_), map(x, y1_), col::light, 1);
            text(img_, tick(x), map(x, y0_) + cv::Point(-18, 16), 0.4, col::black);
        }
        for (int i = 0; i <= ny; ++i) {
            const double y = y0_ + (y1_ - y0_) * i / ny;
            cv::line(img_, map(x0_, y), map(x1_, y), col::light, 1);
            text(img_, tick(y), map(x0_, y) + cv::Point(-58, 4), 0.4, col::black);
        }
        if (y0_ < 0 && y1_ > 0) cv::line(img_, map(x0_, 0), map(x1_, 0), col::gray, 1);
        cv::rectangle(img_, r_, col::black, 1);
        text(img_, title, cv::Point(r_.x, r_.y - 8), 0.5, col::black, 1);
        text(img_, xl, cv::Point(r_.x + r_.width / 2 - 40, r_.y + r_.height + 34), 0.45);
        if (!yl.empty()) text(img_, yl, cv::Point(r_.x + r_.width - 8 * int(yl.size()), r_.y - 8), 0.42, col::gray);
    }

    void line(const std::vector<cv::Point2d>& pts, const cv::Scalar& c, int thick = 2) {
        std::vector<cv::Point> p;
        for (const auto& q : pts) p.push_back(map(q.x, q.y));
        cv::polylines(img_, p, false, c, thick, cv::LINE_AA);
    }
    void vline(double x, const cv::Scalar& c, const std::string& label = "") {
        cv::line(img_, map(x, y0_), map(x, y1_), c, 1, cv::LINE_AA);
        if (!label.empty()) text(img_, label, map(x, y1_) + cv::Point(3, 14), 0.4, c);
    }
    void dot(double x, double y, const cv::Scalar& c, int r = 4) { cv::circle(img_, map(x, y), r, c, cv::FILLED, cv::LINE_AA); }
    void legend(const std::vector<std::pair<std::string, cv::Scalar>>& items) {
        int yy = r_.y + 18;
        for (const auto& it : items) {
            cv::line(img_, cv::Point(r_.x + r_.width - 230, yy - 4), cv::Point(r_.x + r_.width - 200, yy - 4), it.second, 3);
            text(img_, it.first, cv::Point(r_.x + r_.width - 192, yy), 0.42);
            yy += 18;
        }
    }

private:
    static std::string tick(double v) {
        char b[32];
        const double a = std::abs(v);
        if (a >= 100 || a == 0) std::snprintf(b, sizeof(b), "%.0f", v);
        else if (a >= 1) std::snprintf(b, sizeof(b), "%.1f", v);
        else std::snprintf(b, sizeof(b), "%.3f", v);
        return b;
    }
    cv::Mat& img_;
    cv::Rect r_;
    double x0_, x1_, y0_, y1_;
};

// 부호 있는 값 지도 (파랑 − / 흰색 0 / 빨강 +), 범위 ±lim
inline cv::Mat signedColorMap(const cv::Mat& v64, double lim) {
    cv::Mat out(v64.size(), CV_8UC3);
    for (int y = 0; y < v64.rows; ++y)
        for (int x = 0; x < v64.cols; ++x) {
            const double t = std::max(-1.0, std::min(1.0, v64.at<double>(y, x) / lim));
            const int a = int(255 * (1.0 - std::abs(t)));
            out.at<cv::Vec3b>(y, x) = t >= 0 ? cv::Vec3b(uchar(a), uchar(a), 255) : cv::Vec3b(255, uchar(a), uchar(a));
        }
    return out;
}

inline void colorBar(cv::Mat& img, cv::Rect r, double lim, const std::string& unit) {
    for (int x = 0; x < r.width; ++x) {
        const double t = 2.0 * x / (r.width - 1) - 1.0;
        const int a = int(255 * (1.0 - std::abs(t)));
        const cv::Scalar c = t >= 0 ? cv::Scalar(a, a, 255) : cv::Scalar(255, a, a);
        cv::line(img, cv::Point(r.x + x, r.y), cv::Point(r.x + x, r.y + r.height), c);
    }
    cv::rectangle(img, r, col::black, 1);
    text(img, fmt("-%.3g", lim), cv::Point(r.x - 4, r.y + r.height + 14), 0.4);
    text(img, "0", cv::Point(r.x + r.width / 2 - 3, r.y + r.height + 14), 0.4);
    text(img, fmt("+%.3g ", lim) + unit, cv::Point(r.x + r.width - 30, r.y + r.height + 14), 0.4);
}

} // namespace mm

#endif
