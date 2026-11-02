#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace theta {

struct Region {
    int x_first = 0, x_last = -1, y_first = 0, y_last = -1; // inclusive
    std::vector<cv::Vec2i> excluded_x;
};

// Matches the ordinary statsmodels RLM/HuberT settings already verified.
struct HuberSettings {
    double t = 1.345;
    int max_iterations = 50;  // includes the initial OLS fit
    double tolerance = 1e-8; // absolute change of statsmodels' deviance
};

struct LineResult {
    double intercept_px = 0, slope = 0, angle_deg = 0;
    double mad_scale_px = 0, tau_px = 0, deviance_delta = 0;
    int point_count = 0, iterations = 0, downweighted_points = 0;
    bool converged = false;
    std::string stop_reason;
};

struct Calibration {
    // phi = intercept + per_y_mm * Y + per_theta_deg * theta
    double upper_intercept = 0, upper_per_y_mm = 0, upper_per_theta_deg = 0;
    double lower_intercept = 0, lower_per_y_mm = 0, lower_per_theta_deg = 0;
    double target_theta_deg = 0;
};

struct Config {
    int width = 3840, height = 2160;
    Region upper, lower;
    HuberSettings huber;
    Calibration calibration;
    std::string calibration_id;
};

struct DecodedPose {
    double nuisance_y_mm;       // used internally for theta separation, NOT a Y command
    double theta_deg;
    double theta_correction_deg;
    double determinant;
};

struct Result {
    LineResult upper, lower;
    DecodedPose pose;
};

// Functions throw std::runtime_error for invalid/non-finite/undefined inputs.
// No implicit fallback to a zero command.
std::vector<cv::Point2d> selectColumnMaxima(const cv::Mat& vertical_sobel,
                                         const Region& region);
LineResult fitHuberVertical(const std::vector<cv::Point2d>& points,
                           const HuberSettings& settings = HuberSettings());
DecodedPose decodeAngles(double upper_deg, double lower_deg,
                         const Calibration& calibration);
void validateConfig(const Config& config);
Config loadConfig(const std::string& utf8_path);

class Detector {
public:
    explicit Detector(Config config);
    Result estimate(const cv::Mat& image) const;
private:
    Config config_;
};

} // namespace theta
