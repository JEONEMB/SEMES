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
}

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
        set_diagnostics(r.outside_training_range, false,
            "QR3 NCC parabola mean dx + bar tilt2 + screw gradient x4 linear estimate, absolute-zero labels. "
            "Vertical displacements are not used; vertical faults, new-machine transfer and pose stability are not certified.");
        return r.pose;
    } catch (const std::exception& e) {
        diagnostics_state().measurement_valid = false;
        diagnostics_state().message = e.what();
        throw;
    }
}
