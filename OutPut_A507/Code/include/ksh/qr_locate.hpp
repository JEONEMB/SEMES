// =====================================================================================
// ksh/qr_locate.hpp — ① QR 3개 위치 찾기 (기준 사진 없이 사진 한 장에서 절대 위치)
//
//  1/4 축소 영상에서 어두운 QR 띠 → 띠 안 흰 사각형 3개(열·행 밝기 투영)
//  → 사각형마다 4변을 구간 평균 단면의 50% 교차로 서브픽셀 직선 맞춤 → 꼭짓점 → 중심
//  결과 center_px (3개 평균) 가 특징 s 와 나사 ROI 위치 추정의 기준이 된다
// =====================================================================================
#ifndef KSH_QR_LOCATE_HPP
#define KSH_QR_LOCATE_HPP

#include <opencv2/core.hpp>

#include <string>

namespace ksh {

struct QrParams {
    cv::Rect search_roi;          // 띠를 찾는 영역 (0 크기면 사진 전체)
    double strip_dark_thr = 90;   // 1/4 축소 영상에서 이보다 어두우면 띠 후보
    double strip_min_w_px = 600;  // 띠 최소 폭
    double strip_aspect_min = 2.5, strip_aspect_max = 8;
    double thr_frac = 0.25;       // 사각형 투영 문턱 = 어두움 + thr_frac × (밝음 − 어두움)
    int proj_smooth_px = 4;       // 투영 평활 반경
    int gap_merge_px = 20;        // QR 무늬 때문에 끊긴 밝은 구간을 이 간격 이하면 합침
    double size_min_px = 150, size_max_px = 260;   // 사각형 한 변 허용 범위
    double edge_margin = 0.15;    // 변 양 끝 비율은 안 씀 (둥근 모서리)
    double edge_search_px = 8;    // 변 위치 찾기 폭
    double edge_plateau_px = 4;   // 50% 교차의 밝기 기준 구간
    int edge_segments = 6;        // 변 하나를 나누는 구간 수 (구간 평균 단면 → 직선 맞춤)
};

struct QrResult {
    bool ok = false;
    std::string message;          // 실패 이유
    cv::Rect strip_px;            // 찾은 띠
    cv::Rect squares_px[3];       // 사각형 대략 위치 (왼쪽부터)
    cv::Point2d centers_px[3];    // 사각형 중심 (서브픽셀)
    cv::Point2d center_px;        // 3개 평균 (qx, qy)
    double edge_fit_rms_px = 0;   // 변 직선 맞춤 잔차 평균
};

// gray8 : CV_8UC1 (컬러는 image_util 의 to_gray8 로 먼저 바꿈)
QrResult locate_qr(const cv::Mat &gray8, const QrParams &params);

} // namespace ksh

#endif
