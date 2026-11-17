// =====================================================================================
// ksh/setup.hpp — 초기 세팅 : 센터 사진 → ROI 박스 절대 좌표 갱신 + 영점
//
//  ① 지금 박스로 센터 사진들을 잰다 (박스가 수십 px 어긋나 있어도 ①-1 대략 위치와 search_px 여유로 찾음)
//  ② 첫 센터 사진에서 찾은 중심(반올림)으로 박스 7개(QR 3 + 홀 4)의 cx·cy 를 바꾼다 (hw 는 그대로)
//  ③ 새 박스로 다시 재서 센터 사진들의 (Y, Θ) 평균 = 영점
//  검사(경고) : 센터 사진끼리 s 차이 > max_s_spread_px, d 차이 > max_d_spread_px, QR 간격 차이 > 2 px
//  사진은 저장하지 않는다 — 결과는 숫자(박스 좌표 7쌍 + 영점 2개)뿐이고 설정 파일(텍스트)에 쓴다
// =====================================================================================
#ifndef KSH_SETUP_HPP
#define KSH_SETUP_HPP

#include "ksh/estimator.hpp"
#include "ksh/features.hpp"
#include "ksh/pose_model.hpp"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace ksh {

struct SetupResult {
    bool ok = false;
    std::string message;
    std::vector<std::string> warnings;
    FeatureParams params;                       // 박스를 고친 측정 파라미터
    ZeroOffset zero;                            // 새 영점
    std::vector<FeatureMeasurement> measured;   // 새 박스로 잰 센터 사진들
    double s_spread_px = 0.0, d_spread_px = 0.0;
};

// centers : 8bit 회색 센터 사진들 (1장 이상)
SetupResult run_setup(const std::vector<cv::Mat> &centers, const FeatureParams &params, const PoseModel &model,
                      double max_s_spread_px = 0.3, double max_d_spread_px = 0.1);

} // namespace ksh

#endif
