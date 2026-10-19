#ifndef ALIGNMENT_HPP
#define ALIGNMENT_HPP
#include <opencv2/opencv.hpp>
#include <array>
#include <string>
#include <limits>
using error_t = double;
struct Alignment_t {
    // Physical alignment errors are not estimated by plate detection yet.
    error_t y = std::numeric_limits<double>::quiet_NaN();
    error_t t = std::numeric_limits<double>::quiet_NaN();
};
struct AlignmentOptions {
    bool visualize = true;
    std::string outputPrefix; // Empty disables PNG output; parent directory must exist.
};
struct PlateDetectionResult {
    bool found = false;
    double score = -1;
    cv::Rect searchRoi;
    std::array<cv::Point2f, 4> corners{}; // Full-image coordinates, cyclic order.
    cv::Mat edges, hough, visualization;
};
Alignment_t alignment(cv::Mat& src);
Alignment_t alignment(cv::Mat& src, const AlignmentOptions& options,
                      PlateDetectionResult* detection = nullptr);
#endif
