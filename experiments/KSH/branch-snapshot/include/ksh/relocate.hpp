// =====================================================================================
// ksh/relocate.hpp — 기준 사진(center) 교체 시 ROI 재배치
//
//  ROI 좌표(QR 3곳, 배경 홀 2곳)와 Coarse 창은 'ROI 템플릿 사진'(교보재 촬영의 center) 위에서 정한 픽셀 좌표다.
//  장비에서 새 center 를 찍으면 카메라 장착 위치 차이로 같은 물체가 다른 픽셀에 찍힐 수 있으므로,
//  템플릿과 새 center 를 위상 상관으로 비교해 ROI·Coarse 창을 새 center 의 해당 위치로 옮긴다.
//
//  ① 전체 화면 저해상도 위상 상관 (1/downscale 축소)  → 큰 이동 (수백 px 까지)
//  ② Coarse 창 위상 상관 (원본 해상도, ① 만큼 옮긴 창) → 화소 수준 이동
//  ③ ROI 마다 정밀 정합 (QR : ② 결과에서 IC-GN, 배경 : 넓은 창 위상 상관 → IC-GN)
//     → ROI 마다 따로 옮김 (깊이가 다른 배경은 플레이트와 이동량이 다를 수 있음)
//  ④ 검사 : 봉우리 세기, 정합 상관, 이동 크기 상한, QR 3개 이동의 차이(회전·배율 변화 의심)
//
//  보정식 계수(px/mm, px/°)는 바꾸지 않는다 — 카메라 거리·각도가 같고 위치만 옮겨진 경우에 맞는 방법
// =====================================================================================
#ifndef KSH_RELOCATE_HPP
#define KSH_RELOCATE_HPP

#include "ksh/features.hpp"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace ksh {

struct RelocationParams {
    bool enable = true;           // false 면 재배치하지 않음 (설정 ROI 그대로)
    std::string template_path;    // ROI 템플릿 사진 경로 (비우면 재배치하지 않음)
    int downscale = 4;            // ① 축소 배율 (3840 → 960 px)
    double min_response = 0.3;    // ① 봉우리 세기 하한
    double max_shift_px = 300.0;  // 허용 이동 상한 (px, 원본 해상도). 넘으면 장착이 크게 달라진 것으로 보고 실패
    double max_spread_px = 2.0;   // QR 3개 이동의 최대 − 최소 상한 (px). 넘으면 회전·배율 변화 의심 → 실패
};

struct RelocationResult {
    bool ok = false;
    std::string message;              // 실패 이유 (성공이면 빈 문자열)
    cv::Point2d global_px;            // ① 전체 화면 이동 (새 center − 템플릿)
    double global_response = 0.0;     // ① 봉우리 세기
    cv::Point2d coarse_px;            // ② Coarse 창 이동
    double coarse_response = 0.0;     // ② 봉우리 세기
    std::vector<RoiShift> rois;       // ③ ROI 별 이동 (QR → 배경)
    double qr_spread_px = 0.0;        // ④ QR 3개 이동 차이 (가로·세로 중 큰 값)
};

// 템플릿(base 좌표 기준)과 새 기준 사진을 비교해 ROI·Coarse 창을 옮긴 설정을 out 에 만든다
//  template32, new32 : 같은 크기의 CV_32F 회색. 실패하면 false (out 은 바꾸지 않음)
bool relocate_rois(const cv::Mat &template32, const cv::Mat &new32, const FeatureParams &base, const RelocationParams &rp,
                   FeatureParams &out, RelocationResult &result);

} // namespace ksh

#endif
