// image_util.cpp — 영상 공통 도구 구현
#include "ksh/image_util.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <vector>

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
    return rect_around(roi.center_px, roi.half_width_px, roi.half_height_px > 0 ? roi.half_height_px : roi.half_width_px);
}

cv::Mat read_image(const std::string &path, int flags) {
    if (path.empty()) return cv::Mat();
    cv::Mat img = cv::imread(path, flags);
    if (!img.empty()) return img;
    // UTF-8 경로 (Windows 의 cv::imread 는 한글 경로를 못 읽음)
    std::ifstream f(std::filesystem::u8path(path), std::ios::binary);
    if (!f) return cv::Mat();
    const std::vector<uchar> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (bytes.empty()) return cv::Mat();
    return cv::imdecode(bytes, flags);
}

} // namespace ksh
