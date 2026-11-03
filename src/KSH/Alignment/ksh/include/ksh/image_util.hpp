// =====================================================================================
// ksh/image_util.hpp — 영상 공통 도구 (회색 변환, 영역 검사, ROI 정의)
//  모든 ksh 모듈이 함께 쓰는 가장 아래 단계. 상태 없음
// =====================================================================================
#ifndef KSH_IMAGE_UTIL_HPP
#define KSH_IMAGE_UTIL_HPP

#include <opencv2/core.hpp>

#include <string>

namespace ksh {

// 측정 영역 하나 (기준 사진 좌표). 정사각형 = 중심 ± 반폭
struct RoiSpec {
    std::string name;
    cv::Point center_px;
    int half_width_px = 0;
    int half_height_px = 0;   // 세로 반폭. 0 이면 half_width_px 와 같음 (정사각형) — 실험계획 2026-10-30 의 56×60 홀 ROI 용
};

// 컬러(3·4채널)·회색 영상 → CV_32F 회색
cv::Mat to_gray32(const cv::Mat &src);

// 사각형이 영상 안에 완전히 들어오는지
bool rect_inside(const cv::Rect &rect, const cv::Size &image_size);

// 소수 좌표 → 가장 가까운 정수 좌표
cv::Point round_point(const cv::Point2d &p);

// 중심과 반폭(가로·세로)으로 사각형 만들기
cv::Rect rect_around(const cv::Point &center_px, int half_width_px, int half_height_px);

// RoiSpec → 기준 사진 위 사각형
cv::Rect roi_rect(const RoiSpec &roi);

} // namespace ksh

#endif
