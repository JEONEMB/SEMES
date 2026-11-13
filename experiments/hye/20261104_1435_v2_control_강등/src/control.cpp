#include "control_internal.hpp"
#include "detail/alignment_diagnostics.hpp"
#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
namespace {
std::unique_ptr<legacy_control::Tracker> tracker;
std::unique_ptr<legacy_control::Bars> bars;
std::unique_ptr<legacy_control::ModelV2> model;
}
void initialize_control(const std::string& assets,double compatibility_limit){
    (void)compatibility_limit; // 기존 호출 호환용 인자. v2 는 쓰지 않는다.
    if(assets.empty()){tracker.reset();bars.reset();model.reset();return;}
    auto dir=legacy_control::fs::u8path(assets);
    auto next_model=std::make_unique<legacy_control::ModelV2>(dir/"control_v2_model.json");
    auto next_tracker=std::make_unique<legacy_control::Tracker>(dir);
    auto next_bars=std::make_unique<legacy_control::Bars>(dir,legacy_control::read_gray(dir/"center.bmp"));
    for(int k=0;k<48;++k)if(std::abs(next_bars->centers()[k]-next_model->block_x[k])>1e-9)throw std::runtime_error("Control v2 model block centers differ from line protocol");
    // 모두 불러온 뒤에만 바꾼다. 실패하면 이전 상태를 그대로 둔다.
    tracker=std::move(next_tracker);bars=std::move(next_bars);model=std::move(next_model);
}
Alignment_t control(cv::Mat& src){
    using namespace alignment_detail;
    reset_diagnostics("control");
    try {
        if(!tracker||!bars||!model)throw std::logic_error("Call initialize_control(CONTROL_ASSETS) before control()");
        auto gray=legacy_control::checked_gray(src);
        auto tracks=tracker->measure(gray);
        double s=tracker->mean_shift(tracks),line_corr=1;
        double d=bars->slope_difference(gray,&line_corr);
        auto e=model->predict(s,d);
        auto& g=diagnostics_state();
        g.measurement_valid=true;g.tracked_points=630;g.valid_points=tracker->valid_selected(tracks);
        g.minimum_qr_correlation=*std::min_element(tracks.ncc.begin(),tracks.ncc.end());
        g.minimum_line_correlation=line_corr;
        g.selected_axis="Y";g.selected_branch="v2";
        // 2차 전제(T≈0)를 남은 T 추정으로 확인한다: |t| <= 0.01 (T 1틱) 이면 충족.
        set_diagnostics(e.outside,std::abs(e.t)<=0.01,
            "v2: y = precise QR536 Y with gated residual-T correction; t = residual T estimate from bar slope difference");
        return Alignment_t{e.y,e.t};
    }catch(const std::exception& ex){diagnostics_state().measurement_valid=false;diagnostics_state().message=ex.what();throw;}
}
