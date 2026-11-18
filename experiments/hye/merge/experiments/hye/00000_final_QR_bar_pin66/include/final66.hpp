#pragma once
#include "observer_internal.hpp"
#include "pin_observer.hpp"
#include <string>
namespace final66 {
// QR projected displacements54, absolute bar mean/slope4, screw displacements8.
cv::Mat features(const cv::Mat& qr102, const cv::Mat& pins8,
                 const std::vector<double>& block_x);
class LinearModel {
public:
    void load(const std::string& path);
    void save(const std::string& path) const;
    void fit(const cv::Mat& x, const cv::Mat& targets);
    cv::Mat predict(const cv::Mat& x) const;
private:
    cv::Mat design(const cv::Mat& x) const;
    cv::Mat mean_, scale_, weights_;
};
struct Result {
    Alignment_t pose;
    cv::Mat raw110;
    cv::Mat pin_quality;
    double qr_correlation, line_correlation;
    bool outside_training_range;
};
class Estimator {
public:
    explicit Estimator(const std::string& assets);
    Result predict(const cv::Mat& image) const;
private:
    alignment_detail::Observer qr_;
    final66_pin::Observer pins_;
    LinearModel model_;
    std::vector<double> block_x_;
    cv::Vec2d minimum_, maximum_;
};
}
