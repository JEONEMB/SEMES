#include "alignment.hpp"
#include "final00.hpp"
#include "detail/alignment_diagnostics.hpp"
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
        d.tracked_points = 536;
        d.valid_points = r.measurement.valid_points;
        set_diagnostics(r.outside_training_range, false,
            "QR536 mean dx + bar tilt2 + screw gradient x4 linear estimate, absolute-zero labels. "
            "Vertical displacements are not used; vertical faults, new-machine transfer and pose stability are not certified.");
        return r.pose;
    } catch (const std::exception& e) {
        diagnostics_state().measurement_valid = false;
        diagnostics_state().message = e.what();
        throw;
    }
}
