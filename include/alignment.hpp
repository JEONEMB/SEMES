#ifndef ALIGNMENT_HPP
#define ALIGNMENT_HPP
#include <opencv2/opencv.hpp>
#include <array>
#include <string>
#include <limits>
#include <vector>
using error_t = double;
struct Alignment_t {
    // Physical alignment errors are not estimated by plate detection yet.
    error_t y = std::numeric_limits<double>::quiet_NaN();
    error_t t = std::numeric_limits<double>::quiet_NaN();
};
struct AlignmentOptions {
    bool visualize = true;
    // Ratios in plate coordinates, relative to the plate center.
    // Initial layout estimated from supplied images; replace with measured layout if available.
    std::array<float, 3> qrCenterOffsetsX{{-0.265f, 0.f, 0.265f}};
    float qrCenterOffsetY = 0.f;
    float qrWidthRatio = 0.225f;
    float qrHeightRatio = 1.f;
    std::string outputPrefix; // Empty disables PNG output; parent directory must exist.
};
struct QrFeature {
    int id = -1; // Stable module-grid ID: row * (moduleCount + 1) + column.
    cv::Point2f model, initial, refined; // model in modules; image positions in original pixels.
    double shift = 0, minEigenvalue = 0, contrast = 0;
    double reprojectionError = std::numeric_limits<double>::quiet_NaN();
    bool localAccepted = false, accepted = false;
};
struct QrRoiResult {
    bool valid = false;
    cv::Point2f center;
    std::array<cv::Point2f, 4> corners{}; // TL, TR, BR, BL in source image.
    cv::Mat image; // Perspective-corrected crop used for decoding.
    std::string payload, featureStatus;
    bool decoded = false, modelMatched = false, featuresValid = false;
    double modelMismatch = 1;
    int moduleCount = 0;
    cv::Mat idealModules, decodedVisualization, candidateVisualization, featureVisualization;
    std::vector<QrFeature> features;

};
struct PlateDetectionResult {
    bool found = false;
    double score = -1;
    cv::Rect searchRoi;
    std::array<cv::Point2f, 4> corners{}; // Full-image coordinates, cyclic order.
    cv::Point2f center;
    std::array<QrRoiResult, 3> qrRois{}; // Left, center, right in image orientation.
    cv::Mat edges, hough, visualization, roiVisualization;
};
// Decode crop, reconstruct and verify module pattern, refine on original grayscale pixels.
void analyzeQrFeatures(const cv::Mat& source, const AlignmentOptions& options,
                       QrRoiResult& qr, int index);
Alignment_t alignment(cv::Mat& src);
Alignment_t alignment(cv::Mat& src, const AlignmentOptions& options,
                      PlateDetectionResult* detection = nullptr);
#endif
