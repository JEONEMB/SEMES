#include "alignment.hpp"
#include "final00.hpp"
#include "joint_reference.hpp"
#include "detail/alignment_diagnostics.hpp"
#include <cmath>
#include <memory>
#include <mutex>
#include <stdexcept>

namespace {
std::unique_ptr<final00::Estimator> estimator;
std::mutex context_mutex;
bool health_check = false;  // 측정값 일치 점검에 걸리면 멈출지(기본: 멈추지 않고 경고만, 2026-11-13 사용자 지시)
}

void set_joint_health_check(bool enabled) { std::lock_guard<std::mutex> lock(context_mutex); health_check = enabled; }
bool joint_health_check() { std::lock_guard<std::mutex> lock(context_mutex); return health_check; }

void initialize_joint(const std::string& assets) {
    // 새 자료를 끝까지 준비한 뒤에 바꾼다. 준비에 실패하면 이전 상태가 그대로 남는다.
    auto next = std::make_unique<final00::Estimator>(assets);
    std::lock_guard<std::mutex> lock(context_mutex);
    estimator = std::move(next);
}

final00::ReferenceReport set_joint_reference(const std::vector<cv::Mat>& centers) {
    std::lock_guard<std::mutex> lock(context_mutex);
    if (!estimator) throw std::logic_error("Call initialize_joint(FINAL00_ASSETS) first");
    return estimator->set_reference(centers);
}

final00::ReferenceReport set_joint_reference(const cv::Mat& center) {
    return set_joint_reference(std::vector<cv::Mat>{center});
}

namespace {
final00::Estimator& ready() {
    if (!estimator) throw std::logic_error("Call initialize_joint(FINAL00_ASSETS) first");
    return *estimator;
}
}

Alignment_t set_joint_zero(const std::vector<cv::Mat>& centers) {
    std::lock_guard<std::mutex> lock(context_mutex);
    const auto z = ready().set_zero(centers);
    return {z[0], z[1]};
}
Alignment_t set_joint_zero(const cv::Mat& center) { return set_joint_zero(std::vector<cv::Mat>{center}); }
void set_joint_zero_value(double y, double t) {
    std::lock_guard<std::mutex> lock(context_mutex);
    if (!std::isfinite(y) || !std::isfinite(t)) throw std::invalid_argument("영점 값이 수가 아니다");
    ready().set_zero_value(y, t);
}
void clear_joint_zero() { std::lock_guard<std::mutex> lock(context_mutex); ready().clear_zero(); }
Alignment_t joint_zero() { std::lock_guard<std::mutex> lock(context_mutex); const auto z = ready().zero(); return {z[0], z[1]}; }

void initialize_joint_method(const std::string& assets, const std::string& qr_method) {
    auto next = std::make_unique<final00::Estimator>(assets, qr_method);   // 준비에 실패하면 이전 상태가 그대로 남는다
    std::lock_guard<std::mutex> lock(context_mutex);
    estimator = std::move(next);
}
std::string joint_qr_method() { std::lock_guard<std::mutex> lock(context_mutex); return ready().qr_method(); }
std::string joint_model_id() { std::lock_guard<std::mutex> lock(context_mutex); return ready().model_id(); }

Alignment_t joint(cv::Mat& src) {
    using namespace alignment_detail;
    reset_diagnostics("joint");
    try {
        std::lock_guard<std::mutex> lock(context_mutex);
        if (!estimator) throw std::logic_error("Call initialize_joint(FINAL00_ASSETS) first");
        const auto r = estimator->predict(src);
        auto& d = diagnostics_state();
        d.measurement_valid = true;
        d.minimum_qr_correlation = r.measurement.qr_correlation;
        d.minimum_line_correlation = r.measurement.line_correlation;
        d.selected_branch = "huber_linear7";
        d.tracked_points = 3;   // QR 3개 NCC (예전 536점 추적)
        d.valid_points = r.measurement.valid_points;
        d.model_cost = r.health;                                        // 측정값 일치 점검값 (점검식이 없으면 NaN)
        d.normalized_cost_margin = r.health / r.health_threshold;       // 1 을 넘으면 값 하나가 다른 값과 맞지 않는다
        set_diagnostics(r.outside_training_range, false,
            "QR mean dx (track536 or NCC parabola) + bar tilt2 (unless unused) + screw gradient x4 linear estimate, absolute-zero labels. "
            "Vertical displacements are not used; vertical faults, new-machine transfer and pose stability are not certified. "
            "model_cost = value-consistency check (max normalized residual).");
        if (r.health_failed()) {
            const std::string warning = std::string("WARNING 측정값이 서로 맞지 않는다(") + final00::kValueNames[r.health_worst] + ", 점검값 " + std::to_string(r.health) +
                                        " > 기준 " + std::to_string(r.health_threshold) + "). 가림, 반사, 이물을 확인하고, 화면이 바뀌었으면 영점을 다시 잡는다";
            if (health_check) throw std::runtime_error(warning);   // 멈춤을 켠 경우
            d.message = warning + " | " + d.message;               // 기본: 값은 그대로 내고 경고를 남긴다
        }
        return r.pose;
    } catch (const std::exception& e) {
        diagnostics_state().measurement_valid = false;
        diagnostics_state().message = e.what();
        throw;
    }
}
