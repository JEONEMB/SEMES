// =====================================================================================
// ksh/estimator.hpp — 사진 한 장 → (Y, Θ) 전체 흐름 (상태 없는 함수)
//
//  ① QR 3개 → ② 나사 2개 → ③ 특징 s, d → ④ 모델 역변환 → 영점 빼기
//  설정·모델·영점을 인자로 받으므로 여러 설정을 동시에 써도 서로 영향이 없다
//  (제출 형식 alignment() 는 이 함수를 감싼 얇은 진입점)
// =====================================================================================
#ifndef KSH_ESTIMATOR_HPP
#define KSH_ESTIMATOR_HPP

#include "ksh/features.hpp"
#include "ksh/pose_model.hpp"

#include <opencv2/core.hpp>

#include <string>

namespace ksh {

// 영점 : 모델이 내는 값에서 뺄 값 (센터에서 잰 Y·Θ). 기본 0
struct ZeroOffset {
    double y_mm = 0.0, t_deg = 0.0;
};

struct PoseEstimate {
    bool ok = false;
    std::string message;             // 실패 이유
    double y_mm = 0.0, t_deg = 0.0;  // 센터 대비 어긋남 — 라벨(스테이지 명령)과 같은 부호, 보정 이동 = (−Y, −Θ)
    FeatureMeasurement features;
    double elapsed_ms = 0.0;         // 회색 변환 포함
};

// src : 현재 사진 (컬러·회색 모두 가능, 바꾸지 않음)
PoseEstimate estimate_pose(const cv::Mat &src, const FeatureParams &params, const PoseModel &model, const ZeroOffset &zero = ZeroOffset());

} // namespace ksh

#endif
