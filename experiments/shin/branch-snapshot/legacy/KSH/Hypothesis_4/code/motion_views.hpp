// =====================================================================================
// motion_views.hpp — 운동 모델을 그림으로 보여 주는 화면들 (모두 1600 × 900 BGR 영상)
//
//  1. Field      : 현재 자세 (Y, Θ) 에서 화면 전체의 변위 화살표 + 평균 이동을 뺀 나머지(원근) 색 지도
//  2. Separation : 같은 평균 이동을 만드는 Y 와 Θ 의 변위장을 나란히 — 둘의 차이가 Y·Θ 구분 단서
//  3. Profile    : QR 높이의 가로 단면에서 Δu, (Δu − 중심 이동), Δv 곡선
//  4. Depth      : 깊이별 이동량 — Y 는 1/Z, Θ 는 거의 일정 → 플레이트·배경 시차로 Y·Θ 분리
// =====================================================================================
#ifndef MM_MOTION_VIEWS_HPP
#define MM_MOTION_VIEWS_HPP

#include "camera_model.hpp"
#include <opencv2/core.hpp>

namespace mm {

// 배경으로 깔 기준 영상 (회색, 0.25 배 축소본 960 × 540). 없으면 빈 Mat
// arrowGain : 화살표 길이 배율 (1 = 자동 — 가장 긴 화살표가 일정 길이)
cv::Mat renderFieldView(const CameraModel& cam, const Pose& pose, double arrowGain, const cv::Mat& refSmall);
cv::Mat renderSeparationView(const CameraModel& cam, double yEquivMm, double arrowGain, const cv::Mat& refSmall);
cv::Mat renderProfileView(const CameraModel& cam, double yEquivMm);
cv::Mat renderDepthView(const CameraModel& cam, const Pose& pose);

} // namespace mm

#endif
