#pragma once
#include <opencv2/core.hpp>
#include <memory>
#include <string>
#include <vector>

namespace final04_pin {
struct Measurement {
    cv::Mat displacement; // 1x8: x,y for four fixed wall screw patches, pixels.
    cv::Mat quality;      // 1x8: NCC, photometric RMS for each patch.
};
class Observer {
public:
    explicit Observer(const cv::Mat& center);
    Measurement measure(const cv::Mat& image) const;
    // hint: 나사가 움직였을 대략의 정수 이동(QR 평균 이동). 그 주변 ±60 px 만 NCC 로 찾고,
    // 거기서 찾지 못하면(NCC < 0.6 또는 봉우리가 창 끝) 원래처럼 ±260 px 에서 다시 찾는다. 같은 봉우리면 결과가 같다.
    Measurement measure(const cv::Mat& image, cv::Point hint) const;
private:
    bool find_peak(const cv::Mat& image, int k, cv::Point center_shift, int radius_search, cv::Point& shift, double& ncc) const;
    cv::Mat templates_[4];
};
}
