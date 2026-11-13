// =====================================================================================
// ksh/screw_locate.hpp — ② 나사 ROI 위치 추정 + 머리 중심 측정
//
//  나사는 뒤판 위에 있어 플레이트와 같이 움직이지만, QR(앞판)보다 카메라에서 멀어 Y 이동이 작게 보인다
//  (QR −14.6 px/mm, 나사 −11.1 px/mm, Θ 는 둘 다 약 −85 px/°)
//  ROI 위치 추정 : 센터 사진에서의 나사 위치를 이번 프레임의 QR 이동량만큼 옮김
//     x = ref.x + follow_x × (qx − qr_ref.x),  y = ref.y + follow_y × (qy − qr_ref.y)
//  ROI 안 : 크게 흐린 영상의 최소점(머리) → 머리 바깥 테두리를 rays 방향으로 잡아 원 맞춤 (3회 반복, 3σ 밖 제외)
// =====================================================================================
#ifndef KSH_SCREW_LOCATE_HPP
#define KSH_SCREW_LOCATE_HPP

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace ksh {

struct ScrewSpec {
    std::string name;
    cv::Point2d ref_center_px;    // 센터 사진에서의 머리 중심
};

struct ScrewParams {
    std::vector<ScrewSpec> screws;
    cv::Point2d qr_ref_px{1906.3, 1130.8};   // 센터 사진의 QR 중심 (ROI 이동의 기준)
    double follow_x = 0.95;       // QR 가로 이동 대비 나사 가로 이동 비율
    double follow_y = 1.0;
    int roi_half_width_px = 110, roi_half_height_px = 90;
    double blur_sigma_px = 2.5;   // 머리 안 반사광을 누르는 흐림
    double thr_frac = 0.3;        // 테두리 문턱 = 배경 − thr_frac × (배경 − 머리 중심)
    double rmax_px = 48;          // 머리 반지름 상한
    int rays = 120;
    double min_contrast = 15;     // 배경 − 머리 밝기 차 하한 (가림·조명 이상 검출)
};

struct ScrewResult {
    std::string name;
    bool ok = false;
    std::string message;
    cv::Rect roi_px;              // 이번 프레임에서 찾은 ROI
    cv::Point2d center_px;        // 머리 중심 (서브픽셀)
    double radius_px = 0, fit_rms_px = 0, contrast = 0;
};

cv::Point2d predict_screw_center(const ScrewSpec &spec, const ScrewParams &params, const cv::Point2d &qr_center_px);
ScrewResult locate_screw(const cv::Mat &gray8, const ScrewSpec &spec, const ScrewParams &params, const cv::Point2d &qr_center_px);

} // namespace ksh

#endif
