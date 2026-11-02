// Synthetic inputs only: regression checks for the removed Gaussian rejection gates.
#include "common.hpp"
#include "methods.hpp"
#include "effective_settings.hpp"
#include "io_utils.hpp"
#include <cmath>
#include <cstdio>

using namespace lm;
int failures = 0;
void check(bool ok, const char* label) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", label);
    if (!ok) ++failures;
}
cv::Mat edge(double contrast, double center) {
    cv::Mat image(120,120,CV_64FC1);
    for(int y=0;y<image.rows;++y)
        image.row(y).setTo(100.0 + contrast * std::erf((y-center)/(std::sqrt(2.0)*2.5)));
    return image;
}
TargetSpec target(int y0=40,int y1=60) {
    TargetSpec t; t.id="synthetic";t.kind="external";
    t.x0=10;t.x1=100;t.y0=y0;t.y1=y1;t.polarity=1;t.column_step=1;
    t.roi=cv::Rect(0,0,120,120);return t;
}
MethodResult measure(const cv::Mat& image,const TargetSpec& t,const json& p,const json& common) {
    Sink sink;MethodContext c{image,t,p,common,fit_params_for(common,t.kind),sink};
    return method_profile_gaussian(c);
}
bool all_forwarded(const MethodResult& r) {
    return r.intermediates.at("columns").size()==91 &&
           r.intermediates.at("fit_points").size()==91 &&
           r.diagnostics.at("fit_input_points").get<int>()==91;
}
int main() {
    const json cfg=json::parse(read_text("configs/methods.json"));
    const auto common=cfg.at("common");
    const auto p=cfg.at("methods").at("E01/Q01").at("external");
    auto weak=measure(edge(0.5,50),target(),p,common);
    check(all_forwarded(weak) && weak.intermediates["columns"][0][6].get<double>()<8,
          "weak-response profiles are fitted and forwarded");

    auto outside=measure(edge(30,53),target(48,49),p,common);
    const auto& row=outside.intermediates["columns"][0];
    check(all_forwarded(outside) && row[1].get<int>()==49,
          "coarse maxima at search-band endpoints are retained");
    check(std::fabs(outside.intermediates["fit_points"][0][1].get<double>()-49)>2,
          "returned positions more than 2px from coarse locations reach common fit");

    auto short_run=p;short_run["nm_max_iter"]=1;
    auto unfinished=measure(edge(30,50),target(),short_run,common);
    check(all_forwarded(unfinished) && unfinished.diagnostics["nelder_mead_not_converged"].get<int>()==91 &&
          !unfinished.intermediates["columns"][0][8].get<bool>(),
          "nonconvergence is recorded without removing points");

    auto negative=measure(edge(-30,50),target(50,53),p,common);
    check(all_forwarded(negative) && negative.intermediates["columns"][0][4].get<double>()<0,
          "negative-amplitude results are retained");
    auto flat=measure(edge(0,50),target(),p,common);
    check(all_forwarded(flat) && flat.intermediates["columns"][0][4].get<double>()==0,
          "zero-amplitude results are retained");

    auto qr_target=target();qr_target.kind="qr";
    const auto qr_params=cfg.at("methods").at("E01/Q01").at("qr");
    auto qr=measure(edge(0.5,50),qr_target,qr_params,common);
    check(all_forwarded(qr),"shared Q01 path also retains weak profiles");
    for(const auto* key:{"min_peak","max_center_shift_px"}) {
        auto old=p;old[key]=2;bool rejected=false;
        try { validate_method_parameters("E01",old); } catch(const std::exception&) {rejected=true;}
        check(rejected,"removed threshold configuration cannot silently remain active");
    }
    std::printf("%d failure(s)\n",failures);
    return failures?1:0;
}
