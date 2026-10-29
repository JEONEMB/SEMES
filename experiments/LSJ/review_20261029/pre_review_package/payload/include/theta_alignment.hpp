#pragma once

#include <opencv2/opencv.hpp>
#include <array>
#include <string>

namespace semes {

struct LineConfig {
    std::string name;
    int x0 = 0, x1 = 0, expectedY = 0, halfHeight = 0, polarity = 0;
    double positiveSlope = 0, negativeSlope = 0; // image degrees / robot degrees
};

struct ThetaConfig {
    int width = 3840, height = 2160;
    cv::Rect locatorRoi = cv::Rect(1803, 963, 195, 210);
    int searchMargin = 110;
    double tickDeg = 0.01, toleranceDeg = 0.005, maxStepDeg = 1.0;
    std::array<LineConfig, 2> lines;
    static ThetaConfig load(const std::string& utf8Path);
    void validate() const;
};

struct LineMeasurement {
    double angleDeg = 0, edgeRmsPx = 0;
    int points = 0;
};

struct ThetaResult {
    std::array<double, 2> angles{}, targets{}, perLineTheta{};
    double estimatedThetaDeg = 0;
    double continuousCorrectionDeg = 0, commandDeg = 0;
    double fitResidualImageDeg = 0;
    int group = 0;
    bool thetaWithinTolerance = false;
    bool commandAvailable = true;
    bool outsideCalibration = false;
    std::string status;
    cv::Point2d locatorShift;
    double zncc = 0, ecc = 0;
    std::array<LineMeasurement, 2> measurements;
};

// Pure conversion: fixed center-image targets and fixed first-data slopes.
ThetaResult thetaFromAngles(const std::array<double, 2>& angles,
                           const std::array<double, 2>& targets,
                           const ThetaConfig& config);

class ThetaEstimator {
public:
    ThetaEstimator(const cv::Mat& reference, const ThetaConfig& config);
    ThetaResult estimate(const cv::Mat& image) const;
    const std::array<double, 2>& targets() const { return targets_; }
    static LineMeasurement measureLine(const cv::Mat& gray, const LineConfig& line,
                                       cv::Point2d shift = cv::Point2d());
private:
    ThetaConfig config_;
    cv::Mat locatorTemplate_;
    std::array<double, 2> targets_;
};

// UTF-8 paths, including Korean Windows paths. Raw images are read-only.
std::string readTextUtf8(const std::string& path);
void writeTextUtf8(const std::string& path, const std::string& text);
cv::Mat readGrayUtf8(const std::string& path);
void writeThetaResult(cv::FileStorage& fs, const ThetaResult& result);

} // namespace semes
