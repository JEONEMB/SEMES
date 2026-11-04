#pragma once
#include <opencv2/core.hpp>
#include <memory>
#include <string>

namespace final_pin {
struct Measurement {
    cv::Mat displacement; // 1x8: x,y for four fixed wall screw patches, pixels.
    cv::Mat quality;      // 1x8: NCC, photometric RMS for each patch.
};
class Observer {
public:
    explicit Observer(const cv::Mat& center);
    Measurement measure(const cv::Mat& image) const;
private:
    cv::Mat templates_[4];
};
// 1x14: three QR mean displacements relative to the mean wall displacement,
// then all four wall displacements. No filenames or labels are inputs.
cv::Mat relative_features(const cv::Mat& qr102, const cv::Mat& pins8);

class Regressor {
public:
    void fit(const cv::Mat& features, const cv::Mat& targets, int degree,
             double ridge = 1e-4);
    cv::Vec2d predict(const cv::Mat& features) const;
    void save(const std::string& utf8_file) const;
    void load(const std::string& utf8_file);
private:
    cv::Mat design(const cv::Mat& features) const;
    cv::Mat mean_, scale_, weights_;
    int degree_ = 1;
    double ridge_ = 1e-4;
};
class Estimator {
public:
    // qr_assets contains fixed center/geometry/line assets in this repository.
    Estimator(const std::string& qr_assets, const std::string& trained_model);
    ~Estimator();
    Estimator(const Estimator&) = delete;
    Estimator& operator=(const Estimator&) = delete;
    cv::Vec2d predict(const cv::Mat& grayscale) const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
