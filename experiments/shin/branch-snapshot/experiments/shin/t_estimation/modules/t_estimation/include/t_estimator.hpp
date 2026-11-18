#pragma once
#include "t_model.hpp"
#include <memory>
namespace t_only {
struct Estimate { double t; double ticks; Features features; bool outside_calibration_range; bool requires_y_fixed_at_reference; };
class Estimator {
public:
    // calibration_directory contains the unchanged QRwithBar reference/templates.
    Estimator(const std::string& calibration_directory,const std::string& model_file);
    ~Estimator();
    Estimate estimate(const cv::Mat&) const;
private:
    struct Impl;
    std::shared_ptr<const Impl> impl_;
};
}
