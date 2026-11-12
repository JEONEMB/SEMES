#pragma once
#include <opencv2/core.hpp>

namespace v2 {
// 00004 와 같은 학습식(표준화, 절편 제외 ridge 1e-7·n, 두 축 공유 Huber 가중 6회)을 임의의 값 개수에 쓴다.
struct LinearModel {
    cv::Mat mean, scale;   // 1×p
    cv::Mat weights;       // (p+1)×2, 첫 행은 절편. 열은 Y, T
    // x: n×p CV_64F, targets: n×2 CV_64F (Y, T)
    void fit(const cv::Mat& x, const cv::Mat& targets);
    // 반환: n×2 CV_64F
    cv::Mat predict(const cv::Mat& x) const;
};
// 통과 기준: |eY| ≤ 0.045 이고 |eT| ≤ 0.01
inline bool passes(double error_y, double error_t) { return std::abs(error_y) <= 0.045 && std::abs(error_t) <= 0.01; }
}
