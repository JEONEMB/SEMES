// =====================================================================================
// ksh/batch.hpp — 사진 여러 장 처리 (학습·평가 도구가 함께 씀)
//  사진 읽기 → 회색 변환 → 특징 측정 / 자세 추정. 처리 시간을 사진마다 기록
// =====================================================================================
#ifndef KSH_BATCH_HPP
#define KSH_BATCH_HPP

#include "ksh/dataset.hpp"
#include "ksh/estimator.hpp"
#include "ksh/features.hpp"

#include <opencv2/core.hpp>

#include <iosfwd>
#include <map>
#include <vector>

namespace ksh {

struct PhotoFeatures {
    LabeledPhoto photo;
    FeatureMeasurement features;
    double elapsed_ms = 0.0;   // 측정 시간 (사진 읽기 제외)
};

struct PhotoEstimate {
    LabeledPhoto photo;
    PoseEstimate estimate;
};

// 한 기준 사진 대비 특징만 측정 (모델 학습용). progress 가 있으면 50장마다 진행 표시
std::vector<PhotoFeatures> measure_photos(const std::vector<LabeledPhoto> &photos, const cv::Mat &ref32, const FeatureParams &params,
                                          std::ostream *progress = nullptr);

// 세트별 기준 사진으로 자세 추정 (평가용). ref32_by_set 에 없는 세트의 사진은 건너뜀
std::vector<PhotoEstimate> estimate_photos(const std::vector<LabeledPhoto> &photos, const std::map<int, cv::Mat> &ref32_by_set,
                                           const FeatureParams &params, const PoseModel &model, std::ostream *progress = nullptr);

// 성공한 측정만 학습 표본으로
std::vector<FeatureSample> to_samples(const std::vector<PhotoFeatures> &measured);

} // namespace ksh

#endif
