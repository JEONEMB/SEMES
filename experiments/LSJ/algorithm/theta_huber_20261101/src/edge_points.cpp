#include "theta_pipeline.hpp"
#include <cmath>
#include <stdexcept>

namespace theta {
std::vector<cv::Point2d> selectColumnMaxima(const cv::Mat& gy, const Region& r) {
    if (gy.empty() || gy.type() != CV_32FC1 || r.x_first < 0 || r.y_first < 0 ||
        r.x_last >= gy.cols || r.y_last >= gy.rows || r.x_first > r.x_last ||
        r.y_first > r.y_last) throw std::runtime_error("invalid Sobel image or ROI");
    for (const auto& v : r.excluded_x)
        if (v[0] > v[1]) throw std::runtime_error("invalid occlusion interval");
    std::vector<cv::Point2d> points;
    for (int x = r.x_first; x <= r.x_last; ++x) {
        bool excluded = false;
        for (const auto& v : r.excluded_x)
            if (v[0] <= x && x <= v[1]) excluded = true;
        if (excluded) continue;
        int best_y = r.y_first;
        float best = gy.at<float>(best_y, x);
        if (!std::isfinite(best)) throw std::runtime_error("non-finite Sobel response");
        for (int y = r.y_first + 1; y <= r.y_last; ++y) {
            const float value = gy.at<float>(y, x);
            if (!std::isfinite(value)) throw std::runtime_error("non-finite Sobel response");
            if (value > best) { best = value; best_y = y; }
        }
        // Signed Gy maximum, first y wins ties. No response/motion rejection.
        points.emplace_back(x, best_y);
    }
    return points;
}
}
