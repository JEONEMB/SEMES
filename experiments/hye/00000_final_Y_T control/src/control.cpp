#include "control_internal.hpp"
#include "detail/alignment_diagnostics.hpp"
#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
namespace {
std::unique_ptr<legacy_control::Tracker> tracker;
std::unique_ptr<legacy_control::Model> model;
}
void initialize_control(const std::string& assets,double compatibility_limit){
    (void)compatibility_limit; // Retained API parameter; the historical model has no T gate.
    if(assets.empty()){tracker.reset();model.reset();return;}
    auto next_model=std::make_unique<legacy_control::Model>(legacy_control::fs::u8path(assets)/"model.json");
    auto next_tracker=std::make_unique<legacy_control::Tracker>(legacy_control::fs::u8path(assets));
    tracker=std::move(next_tracker);model=std::move(next_model);
}
Alignment_t control(cv::Mat& src){
    using namespace alignment_detail;
    reset_diagnostics("control");
    try {
        if(!tracker||!model)throw std::logic_error("Call initialize_control(CONTROL_ASSETS) before control()");
        auto gray=legacy_control::checked_gray(src);
        auto tracks=tracker->measure(gray);
        auto feature=tracker->features(tracks);
        std::vector<legacy_control::Choice> choices;
        auto best=model->predict(feature,&choices);
        auto& d=diagnostics_state();
        d.measurement_valid=true;d.tracked_points=630;d.valid_points=tracker->valid_selected(tracks);
        d.minimum_qr_correlation=*std::min_element(tracks.ncc.begin(),tracks.ncc.end());
        d.selected_axis=best.axis;d.selected_branch=best.branch;d.model_cost=best.cost;
        double other=std::numeric_limits<double>::infinity();
        for(const auto& q:choices)if(q.axis!=best.axis)other=std::min(other,q.cost);
        d.other_axis_cost=other;
        d.normalized_cost_margin=std::isfinite(other)?(other-best.cost)/(other+best.cost+1e-12):std::numeric_limits<double>::quiet_NaN();
        set_diagnostics(std::abs(best.value)>(best.axis=="Y"?2.:1.),false,
            "Original QR536 Y-or-T estimate; selected-point validity is diagnostic only; caller establishes stage2 pose conditions");
        return best.axis=="Y"?Alignment_t{best.value,0}:Alignment_t{0,best.value};
    }catch(const std::exception& e){diagnostics_state().measurement_valid=false;diagnostics_state().message=e.what();throw;}
}
