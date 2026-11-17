// =====================================================================================
// ksh/features.hpp — ③ 시차 특징 : QR 위치 · 뒷벽 홀 위치 → s, d (사진 한 장 안의 절대 위치, 기준 사진 없음)
//
//  s = QR 3개 중심 가로 위치 평균 (px)                  — 앞판(플레이트)이 화면 어디 있나
//  d = s − 홀 4곳 가로 위치 평균 (px)                    — 앞판과 뒷벽이 얼마나 다르게 보이나 (시차)
//  Y 이동은 깊이마다 다른 이동(시차)을, Θ 회전은 깊이와 거의 무관한 이동을 만들어 두 특징으로 Y·Θ 를 구별한다
//  교보재 부호 : +Y, +Θ 에서 플레이트가 화면 오른쪽 (s 증가)
// =====================================================================================
#ifndef KSH_FEATURES_HPP
#define KSH_FEATURES_HPP

#include "ksh/hole_locate.hpp"
#include "ksh/qr_locate.hpp"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace ksh {

struct FeatureParams {
    QrParams qr;
    HoleParams holes;
};

struct FeatureMeasurement {
    bool ok = false;
    std::string message;              // 실패 이유
    QrResult qr;                      // ①
    std::vector<HoleResult> holes;    // ②
    double s_px = 0.0;                // ③
    double d_px = 0.0;
};

// ①·②·③ 전체 : 8bit 회색 영상 → 특징
FeatureMeasurement measure_features(const cv::Mat &gray8, const FeatureParams &params);

} // namespace ksh

#endif
