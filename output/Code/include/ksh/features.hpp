// =====================================================================================
// ksh/features.hpp — ③ 시차 특징 측정 : ROI 이동 → s, d
//
//  s = QR 3개의 평균 가로 이동 (px)            — 플레이트(가까움)가 얼마나 움직였나
//  d = s − 배경 ROI(하단 문 홀 W2·W3) 평균 (px) — 가까운 것과 먼 것이 얼마나 다르게 움직였나 (시차)
//  Y 이동은 깊이마다 다른 이동(시차)을, Θ 회전은 깊이와 무관한 이동을 만들어 두 특징으로 Y·Θ 를 구별한다
// =====================================================================================
#ifndef KSH_FEATURES_HPP
#define KSH_FEATURES_HPP

#include "ksh/coarse_align.hpp"
#include "ksh/fine_align.hpp"
#include "ksh/image_util.hpp"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace ksh {

struct FeatureParams {
    CoarseParams coarse;
    FineParams fine;
    std::vector<RoiSpec> qr_rois;           // 플레이트 위 QR (반복 무늬 → 넓은 창 탐색 없이 Coarse 결과에서 바로 Fine)
    std::vector<RoiSpec> background_rois;   // 깊이가 다른 배경 (넓은 창 위상 상관 → Fine)
    int wide_window_scale = 2;              // 배경 ROI 위상 상관 창 = ROI 반폭 × 이 값
    double qr_max_dev_px = 15.0;            // QR 결과가 Coarse 에서 이 이상 벗어나면 이웃 QR 에 붙은 것으로 보고 실패
    double wide_min_response = 0.0;         // 배경 ROI 넓은 창 위상 상관 봉우리 세기 하한 (0 = 검사 안 함)
};

struct RoiShift {
    std::string name;
    cv::Rect ref_rect_px;   // 기준 사진에서의 위치
    cv::Point2d shift_px;   // 측정 이동 (현재 − 기준)
    double zncc = 0.0;      // 정합 상관 ZNCC (1 에 가까울수록 신뢰)
    double response = 0.0;  // 넓은 창 위상 상관 봉우리 세기 (배경 ROI 만, QR 은 0)
    bool converged = false; // Fine Alignment 수렴 여부
    bool ok = false;
    std::string message;    // 실패 이유 (성공이면 빈 문자열)
};

struct FeatureMeasurement {
    bool ok = false;
    std::string message;          // 실패 이유
    cv::Point2d coarse_px;        // ① Coarse Alignment 결과
    double coarse_response = 0.0; // ① 위상 상관 봉우리 세기
    std::vector<RoiShift> rois;   // ② QR → 배경 순서
    double s_px = 0.0;            // ③
    double d_px = 0.0;
};

// ROI 하나 : (wide_search 이면 넓은 창 위상 상관으로 출발점을 다시 잡고) Fine Alignment
RoiShift measure_roi(const cv::Mat &ref32, const cv::Mat &cur32, const RoiSpec &roi, const cv::Point2d &guess_px, bool wide_search,
                     int wide_window_scale, const FineParams &fine, double wide_min_response = 0.0);

// ROI 이동 목록 → s, d (QR 개수 = n_qr, 그 뒤가 배경). 개수가 0 이면 false
bool compute_features(const std::vector<RoiShift> &rois, size_t n_qr, double &s_px, double &d_px);

// ①·②·③ 전체 : 기준·현재 영상(CV_32F 회색, 같은 크기) → 특징
FeatureMeasurement measure_features(const cv::Mat &ref32, const cv::Mat &cur32, const FeatureParams &params);

} // namespace ksh

#endif
