#pragma once
#include <opencv2/core.hpp>
#include <memory>
#include <string>
#include <vector>

namespace final66_pin {
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
}
