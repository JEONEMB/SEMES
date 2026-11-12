#pragma once
#include <opencv2/core.hpp>

namespace final00 {
// 나사 4개의 이동을 "기울기 영상 맞춤"으로 잰다.
// 나사 주변 141×141 조각에서 밝기 변화 크기 영상(흐림 1.5 뒤 Sobel)을 만들고,
// 센터 사진의 같은 영상과 반지름 45 px 안에서 ECC 평행이동으로 맞춘다.
class ScrewGradient {
public:
    // center: 센터 사진(8비트 회색, 3840×2160)
    explicit ScrewGradient(const cv::Mat& center);
    // gray: 현재 사진. initial8: 나사 4개의 대략 이동(x,y 순서 1×8, CV_64F). 정수로 반올림해 조각 위치를 잡는 데만 쓴다.
    // 반환: 1×8 CV_64F (x,y 순서). 맞춤에 실패한 나사는 NaN.
    cv::Mat measure(const cv::Mat& gray, const cv::Mat& initial8) const;
private:
    cv::Mat templates_[4];
    cv::Mat mask_;
};
}
