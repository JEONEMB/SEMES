// =====================================================================================
// ksh/estimator.hpp — 사진 한 장 → (Y, Θ) 전체 흐름 (상태 없는 함수)
//
//  ① Coarse → ② Fine → ③ 특징 s, d → ④ 모델 역변환
//  기준 영상·설정·모델을 인자로 받으므로 여러 기준/모델을 동시에 써도 서로 영향이 없다
//  (제출 형식 alignment() 는 이 함수를 감싼 얇은 진입점)
// =====================================================================================
#ifndef KSH_ESTIMATOR_HPP
#define KSH_ESTIMATOR_HPP

#include "ksh/features.hpp"
#include "ksh/pose_model.hpp"

#include <opencv2/core.hpp>

#include <string>

namespace ksh {

struct PoseEstimate {
    bool ok = false;
    std::string message;             // 실패 이유
    double y_mm = 0.0, t_deg = 0.0;  // 기준(티칭) 사진 대비 어긋남 — 라벨과 같은 부호, 보정 이동 = (−Y, −Θ)
    FeatureMeasurement features;
    double elapsed_ms = 0.0;         // 회색 변환 포함
};

// ref32 : 기준 사진 (to_gray32 결과). src : 현재 사진 (컬러·회색 모두 가능, 바꾸지 않음)
PoseEstimate estimate_pose(const cv::Mat &ref32, const cv::Mat &src, const FeatureParams &params, const PoseModel &model);

} // namespace ksh

#endif
