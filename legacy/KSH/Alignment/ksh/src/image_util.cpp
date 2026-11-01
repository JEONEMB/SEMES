// image_util.cpp — 영상 공통 도구 구현
#include "ksh/image_util.hpp"

#include <opencv2/imgproc.hpp>

#include <cmath>

namespace ksh {

cv::Mat to_gray32(const cv::Mat &src) {
    cv::Mat gray, out;
    if (src.channels() == 3) cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    else if (src.channels() == 4) cv::cvtColor(src, gray, cv::COLOR_BGRA2GRAY);
    else gray = src;
    gray.convertTo(out, CV_32F);
    return out;
}

bool rect_inside(const cv::Rect &rect, const cv::Size &image_size) {
    return rect.x >= 0 && rect.y >= 0 && rect.x + rect.width <= image_size.width && rect.y + rect.height <= image_size.height;
}

cv::Point round_point(const cv::Point2d &p) {
    return cv::Point(int(std::lround(p.x)), int(std::lround(p.y)));
}

cv::Rect rect_around(const cv::Point &center_px, int half_width_px, int half_height_px) {
    return cv::Rect(center_px.x - half_width_px, center_px.y - half_height_px, 2 * half_width_px, 2 * half_height_px);
}

cv::Rect roi_rect(const RoiSpec &roi) {
    return rect_around(roi.center_px, roi.half_width_px, roi.half_width_px);
}

} // namespace ksh
