#include "alignment.hpp"
#include "final66.hpp"
#include "detail/alignment_diagnostics.hpp"
#include <memory>
#include <mutex>
#include <stdexcept>
namespace {
std::unique_ptr<final66::Estimator> estimator;
std::mutex context_mutex;
}
void initialize_joint(const std::string& assets) {
    // Prepare completely first: failed reload preserves the previous context.
    auto next=std::make_unique<final66::Estimator>(assets);
    std::lock_guard<std::mutex> lock(context_mutex);
    estimator=std::move(next);
}
Alignment_t joint(cv::Mat& src) {
    using namespace alignment_detail;
    reset_diagnostics("joint");
    try {
        std::lock_guard<std::mutex> lock(context_mutex);
        if(!estimator) throw std::logic_error("Call initialize_joint(FINAL66_ASSETS) first");
        const auto r=estimator->predict(src);
        auto& d=diagnostics_state();
        d.measurement_valid=true;
        d.minimum_qr_correlation=r.qr_correlation;
        d.minimum_line_correlation=r.line_correlation;
        d.selected_branch="weighted_linear66";
        // QR27 are virtual projections, not 27 independently measured corners.
        // Leave tracked_points unset rather than inventing a point-validity score.
        set_diagnostics(r.outside_training_range,false,
            "QR54+bar4+pin8 weighted linear estimate. Screw physical motion, new-machine transfer and pose stability are not certified.");
        return r.pose;
    } catch(const std::exception& e) {
        diagnostics_state().measurement_valid=false;
        diagnostics_state().message=e.what();
        throw;
    }
}
