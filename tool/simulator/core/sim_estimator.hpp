// =====================================================================================
// sim_estimator.hpp — 시뮬레이터가 부르는 '추정기' (사진 한 장 → 어긋남 Y, Θ)
//
//  ksh    : 내장 — Alignment 모듈(alignment_init / alignment_set_reference / alignment / alignment_last_info)
//           모델이 본 ROI 등 진단 정보를 화면에 그릴 수 있음
//  plugin : DLL(.dll / .so) 을 실행 중에 읽음 — 경계는 sdk/sim_plugin.h 의 C 함수
//           팀원 알고리즘(팀 공통 alignment())을 exe 를 다시 빌드하지 않고 시험
// =====================================================================================
#ifndef SIM_ESTIMATOR_HPP
#define SIM_ESTIMATOR_HPP

#include "sim_config.hpp"
#include "alignment.hpp"   // AlignmentInfo (ksh 진단 정보)

#include <opencv2/core.hpp>

#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace sim {

struct EstimateResult {
    bool ok = false;
    double y_mm = std::numeric_limits<double>::quiet_NaN();
    double t_deg = std::numeric_limits<double>::quiet_NaN();
    std::string message;
};

class Estimator {
public:
    virtual ~Estimator() = default;
    virtual std::string name() const = 0;                        // 화면에 보일 이름
    virtual bool supportsReference() const = 0;                  // 기준 사진 교체 가능 여부
    virtual bool setReference(const cv::Mat& image) = 0;         // 지원 안 하면 false
    virtual EstimateResult estimate(const cv::Mat& image) = 0;
    virtual const AlignmentInfo* info() const { return nullptr; } // 마지막 추정의 진단 (없으면 nullptr)
};

// 설정(estimator.*)대로 추정기를 만듦. 실패하면 nullptr 과 이유
std::unique_ptr<Estimator> makeEstimator(const SimConfig& cfg, std::string& message);

// 고를 수 있는 추정기 목록 : "ksh" + plugins 폴더의 DLL (화면 선택 상자용)
std::vector<std::string> listPlugins(const std::string& pluginDir = "plugins");

} // namespace sim

#endif
