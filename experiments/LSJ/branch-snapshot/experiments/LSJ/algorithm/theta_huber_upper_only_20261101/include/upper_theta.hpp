#pragma once
#include "theta_pipeline.hpp" // reused Region, HuberSettings, LineResult declarations

namespace theta {
struct UpperConfig {
    int width = 3840, height = 2160;
    Region upper;
    HuberSettings huber;
    double intercept_deg = 0;
    double per_theta_deg = 0;
    double target_theta_deg = 0;
    std::string calibration_id;
};

struct UpperResult {
    LineResult upper;
    double theta_deg = 0;
    double theta_correction_deg = 0;
};

// No Y input, no Y estimate and no lower-structure fit in this model.
double decodeUpperAngle(double angle_deg, const UpperConfig& config);
UpperConfig loadUpperConfig(const std::string& utf8_path);

class UpperDetector {
public:
    explicit UpperDetector(UpperConfig config);
    UpperResult estimate(const cv::Mat& image) const;
private:
    UpperConfig config_;
};
}
