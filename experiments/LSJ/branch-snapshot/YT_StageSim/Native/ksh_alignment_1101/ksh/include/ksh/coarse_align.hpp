// =====================================================================================
// ksh/coarse_align.hpp — ① Coarse Alignment (대략 정렬) : 위상 상관(FFT)
//
//  두 영상의 FFT 위상 차만 남겨 역 FFT 하면 이동량 위치에 봉우리가 선다 (Kuglin & Hines 1975).
//  가능한 모든 이동을 한 번에 평가하고 밝기·대비 변화에 강하지만 정확도는 화소 수준이다.
//  → Fine Alignment 의 출발점을 만든다 (실험계획 1 : Coarse → Fine 순서 채택)
// =====================================================================================
#ifndef KSH_COARSE_ALIGN_HPP
#define KSH_COARSE_ALIGN_HPP

#include <opencv2/core.hpp>

namespace ksh {

struct CoarseParams {
    cv::Point center_px;   // 위상 상관 창 중심 (기준 사진 좌표, 플레이트 주변)
    cv::Size window_px;    // 창 크기
};

// moved 가 base 대비 이동한 양 (px). 두 영상은 같은 크기의 CV_32F. 해닝 창을 곱해 가장자리 영향을 줄임
cv::Point2d phase_shift(const cv::Mat &base32, const cv::Mat &moved32);

// 기준·현재 영상의 같은 창에서 위상 상관 → 플레이트 전체의 대략 이동 (px). 창이 영상 밖이면 false
bool coarse_alignment(const cv::Mat &ref32, const cv::Mat &cur32, const CoarseParams &params, cv::Point2d &shift_px);

} // namespace ksh

#endif
