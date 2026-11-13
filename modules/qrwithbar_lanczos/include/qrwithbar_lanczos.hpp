#pragma once
#include <opencv2/core.hpp>
#include <array>
#include <memory>
#include <string>
namespace qrwithbar {
struct Estimate {
    double y_mm, yaw_deg;
    std::array<double,10> features;
    int qr_points, upper_valid, lower_valid;
    bool outside_training_range;
};
// Calibrated to the supplied reference/QR pattern. A different camera setup or
// QR pattern needs calibration; interpolation is not a replacement for it.
class Estimator {
public:
    explicit Estimator(const std::string& assets_directory);
    ~Estimator();
    Estimate estimate(const cv::Mat& source) const;
private:
    struct Implementation;
    std::shared_ptr<const Implementation> implementation_;
};
std::string default_assets_directory();
}
