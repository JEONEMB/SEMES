// =====================================================================================
// ksh/features.hpp — ③ 시차 특징 측정 : QR 위치 · 나사 위치 → s, d
//
//  s = QR 3개 중심의 가로 위치 qx (px, 사진 좌표)                 — 앞판이 어디 있나
//  d = s − 나사 가로 위치 평균 (px)                                  — 앞판과 뒤판이 얼마나 다르게 보이나 (시차)
//  Y 이동은 깊이마다 다른 이동(시차)을, Θ 회전은 깊이와 거의 무관한 이동을 만들어 두 특징으로 Y·Θ 를 구별한다
//  (교보재 방식과 같은 s·d 구조. 다만 기준 사진 대비 이동이 아니라 사진 한 장 안의 절대 위치라 기준 사진이 필요 없다)
// =====================================================================================
#ifndef KSH_FEATURES_HPP
#define KSH_FEATURES_HPP

#include "ksh/qr_locate.hpp"
#include "ksh/screw_locate.hpp"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace ksh {

struct FeatureParams {
    QrParams qr;
    ScrewParams screws;
};

struct FeatureMeasurement {
    bool ok = false;
    std::string message;              // 실패 이유
    QrResult qr;                      // ①
    std::vector<ScrewResult> screws;  // ②
    double s_px = 0.0;                // ③
    double d_px = 0.0;
};

// ①·②·③ 전체 : 8bit 회색 영상 → 특징
FeatureMeasurement measure_features(const cv::Mat &gray8, const FeatureParams &params);

} // namespace ksh

#endif
