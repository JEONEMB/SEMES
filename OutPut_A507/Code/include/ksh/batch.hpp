// =====================================================================================
// ksh/batch.hpp — 사진 여러 장 처리 (학습 도구가 씀)
//  사진 읽기 → 회색 → 특징 측정. 여러 스레드로 나눠 처리하고, 사진마다 시간을 기록
// =====================================================================================
#ifndef KSH_BATCH_HPP
#define KSH_BATCH_HPP

#include "ksh/dataset.hpp"
#include "ksh/estimator.hpp"
#include "ksh/features.hpp"

#include <iosfwd>
#include <vector>

namespace ksh {

struct PhotoFeatures {
    LabeledPhoto photo;
    FeatureMeasurement features;
    double elapsed_ms = 0.0;   // 측정 시간 (사진 읽기 제외)
};

// 특징 측정 (모델 학습용). threads ≤ 0 이면 CPU 수. progress 가 있으면 100장마다 진행 표시
std::vector<PhotoFeatures> measure_photos(const std::vector<LabeledPhoto> &photos, const FeatureParams &params, int threads = 0,
                                          std::ostream *progress = nullptr);

// 성공한 측정만 학습 표본으로 (기준 사진 제외)
std::vector<FeatureSample> to_samples(const std::vector<PhotoFeatures> &measured);

} // namespace ksh

#endif
