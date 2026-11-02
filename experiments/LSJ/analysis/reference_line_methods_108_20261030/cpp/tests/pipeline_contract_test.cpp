#include "segment_support.hpp"
#include "effective_settings.hpp"
#include "io_utils.hpp"
#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
#include <stdexcept>

using namespace lm;
int failures=0;
void check(bool value,const char* label){std::printf("%s %s\n",value?"PASS":"FAIL",label);if(!value)++failures;}
bool rejects(const std::function<void()>& fn){try{fn();return false;}catch(const std::exception&){return true;}}
json parameters(const json& cfg,const std::string& id){
    std::string key=id;
    if(id.substr(1)<="09")key="E"+id.substr(1)+"/Q"+id.substr(1);
    return cfg.at("methods").at(key).at(id[0]=='E'?"external":"qr");
}
TargetSpec target(const std::string& kind="external"){
    TargetSpec t;t.id="test";t.kind=kind;t.x0=10;t.x1=90;t.y0=40;t.y1=60;
    t.exclude_x={{40,60}};t.polarity=1;t.roi=cv::Rect(5,20,100,80);t.margin=16;t.column_step=2;return t;
}
int main(){
    TargetSpec t=target();DetectedSegment s{0,30,100,70,0,{}};
    auto pieces=clip_segment_support(t,s);
    check(pieces.size()==2&&std::fabs(pieces.front().xa-23.75)<1e-12&&std::fabs(pieces.back().xb-76.25)<1e-12,
          "clip segment against y band and subtract central occlusion");
    auto points=sample_segment_support(t,s,pieces,1.0);
    bool allowed=!points.empty();for(const auto& p:points)allowed=allowed&&allowed_support_point(t,p.x,p.y);
    check(allowed,"every generated point satisfies x AND y bounds and exclusion mask");
    check(pieces[0].xb<39.5&&pieces[1].xa>60.5&&!t.excluded(pieces[0].xb)&&!t.excluded(pieces[1].xa),
          "closed occlusion endpoints are not accidentally sampled");
    auto reversed=clip_segment_support(t,{100,70,0,30,0,{}});
    check(reversed.size()==2&&std::fabs(reversed[0].xa-pieces[0].xa)<1e-12&&std::fabs(reversed[1].xb-pieces[1].xb)<1e-12,
          "reversing endpoints preserves the visible support");
    auto descending=clip_segment_support(t,{0,70,100,30,0,{}});
    check(descending.size()==2&&std::fabs(descending[0].xa-23.75)<1e-12&&std::fabs(descending[1].xb-76.25)<1e-12,
          "negative slopes are clipped on the correct side");
    check(clip_segment_support(t,{0,30,100,30,0,{}}).empty(),"horizontal segment outside band is removed");
    auto horizontal=clip_segment_support(t,{0,50,100,50,0,{}});
    check(horizontal.size()==2&&horizontal.front().xa==9.5&&horizontal.back().xb==90.5,
          "horizontal segment retains only target x support");
    TargetSpec crossing=t;crossing.x1=100;crossing.exclude_x.clear();
    auto partial=clip_segment_support(crossing,{0,30,100,40,0,{}});
    check(partial.size()==1&&std::fabs(partial[0].xa-95)<1e-12,
          "valid crossing part is retained even when original midpoint is outside band");
    TargetSpec overlap=t;overlap.exclude_x={{45,60},{40,50},{110,120}};
    auto merged=clip_segment_support(overlap,s);
    check(merged.size()==2&&merged[0].xb==pieces[0].xb&&merged[1].xa==pieces[1].xa,
          "overlapping unsorted occlusions do not duplicate support");
    TargetSpec hidden=t;hidden.exclude_x={{0,100}};
    check(clip_segment_support(hidden,s).empty(),"fully hidden segment has no support");
    check(rejects([&](){sample_segment_support(t,s,pieces,0);}),"zero sampling step is rejected");
    check(rejects([&](){clip_segment_support(t,{0,0,100,std::numeric_limits<double>::quiet_NaN(),0,{}});}),
          "non-finite segment geometry is rejected");
    TargetSpec asymmetric=t;asymmetric.x1=40;asymmetric.exclude_x.clear();
    DetectedSegment flat{0,50,100,50,0,{}};
    auto mid=visible_support_midpoint(flat,clip_segment_support(asymmetric,flat));
    check(std::fabs(mid.x-25)<1e-12,"pair hypothesis anchor uses visible support rather than original midpoint");
    TargetSpec broad=t;broad.x0=0;broad.x1=100;broad.y0=0;broad.y1=100;broad.exclude_x.clear();
    DetectedSegment tilted{0,40,100,42.5,0,{}};
    check(sample_segment_support(broad,flat,clip_segment_support(broad,flat),1).size()==
          sample_segment_support(broad,tilted,clip_segment_support(broad,tilted),1).size(),
          "x-step sampling has equal counts for equal x spans, not equal arc lengths");

    json cfg=json::parse(read_text("configs/methods.json"));json common=cfg.at("common");
    // Only this artificial gray ramp and supplied segments are used; no detector or captured image is run.
    cv::Mat ramp(120,1100,CV_8UC1);for(int y=0;y<ramp.rows;++y)ramp.row(y).setTo(y*2);
    TargetSpec narrow=t;narrow.x1=990;narrow.y0=49;narrow.y1=51;narrow.exclude_x={{480,520}};
    DetectedSegment supplied{0,40,1000,60,0,{}};
    for(int x=0;x<=1000;++x)supplied.support.emplace_back(x,(int)std::round(40+0.02*x));
    supplied.support.emplace_back(450,90);
    json ep=parameters(cfg,"E04");Sink sink;
    MethodContext ctx{ramp,narrow,ep,common,fit_params_for(common,"external"),sink};
    MethodResult measured=select_and_fit_segments(ctx,{supplied},"known_segment_fixture");
    check(measured.status=="ok"&&measured.has_line&&std::fabs(measured.line.dy/measured.line.dx-0.02)<1e-12,
          "selector fits the supplied clipped line through the real common pipeline");
    bool bounded=true;
    for(const auto& p:measured.intermediates.at("fit_points"))bounded=bounded&&allowed_support_point(narrow,p[0],p[1]);
    check(bounded&&measured.intermediates.at("fit_points").size()>=50,"selector never passes out-of-region samples to Huber");
    double expected=(150.0-41.0)*std::sqrt(1+0.02*0.02);
    check(std::fabs(measured.diagnostics.at("consensus_score_visible_len_px").get<double>()-expected)<1e-9,
          "consensus score uses clipped arc length only");
    bounded=true;for(const auto& p:measured.intermediates.at("selected_support_pixels"))bounded=bounded&&allowed_support_point(narrow,p[0],p[1]);
    check(bounded,"native support diagnostics are restricted to allowed regions too");
    check(measured.intermediates.at("visible_segment_pieces").size()==2&&
          measured.support_type.find("not exact arc-length")!=std::string::npos,
          "clipped pieces and honest sampling semantics are recorded");

    int classified=0;
    for(char prefix:{'E','Q'})for(int i=1;i<=(prefix=='E'?9:12);++i){
        std::string id=std::string(1,prefix)+(i<10?"0":"")+std::to_string(i);
        auto tt=target(prefix=='E'?"external":"qr");auto p=parameters(cfg,id);
        auto report=effective_settings(id,p,common,tt,{3840,2160});
        size_t count=report["applied"]["method"].size();
        if(report["inactive"].contains("method"))count+=report["inactive"]["method"].size();
        if(report["reporting_only"].contains("method"))count+=report["reporting_only"]["method"].size();
        if(count==p.size())++classified;
    }
    check(classified==21,"all configured method keys are classified across all 21 methods");
    auto qr=target("qr");auto p12=parameters(cfg,"Q12");auto q12=effective_settings("Q12",p12,common,qr,{3840,2160});
    check(!q12["applied"].contains("common_fit")&&q12["inactive"].contains("common_fit")&&
          !q12["applied"]["target"].contains("band_y_inclusive"),"Q12 records Huber and common band as inactive");
    auto changed_common=common;changed_common["fit"]["huber_c_px"]=9.0;
    check(effective_settings("Q12",p12,changed_common,qr,{3840,2160})["applied"]==q12["applied"],
          "unused Huber setting cannot change the Q12 effective hash payload");
    auto changed_target=qr;changed_target.y0=2;changed_target.y1=3;changed_target.margin=8;changed_target.column_step=5;
    check(effective_settings("Q12",p12,common,changed_target,{3840,2160})["applied"]==q12["applied"],
          "unused QR band/margin/step do not affect applied settings");
    auto q10=effective_settings("Q10",parameters(cfg,"Q10"),common,qr,{3840,2160});
    check(q10["applied"].contains("common_fit")&&!q10["applied"]["target"].contains("band_y_inclusive"),
          "Q10 uses Huber but searches ROI without common band");
    auto p4=parameters(cfg,"E04");auto e4=effective_settings("E04",p4,common,t,{3840,2160});
    auto inactive_sigma=p4;inactive_sigma["sigma_scale"]=0.9;
    check(e4["inactive"]["method"].contains("sigma_scale")&&
          effective_settings("E04",inactive_sigma,common,t,{3840,2160})["applied"]==e4["applied"],
          "LSD scale=1 records sigma_scale as inactive and excludes it from applied hash");
    auto p1=parameters(cfg,"E01");auto e1=effective_settings("E01",p1,common,t,{3840,2160});
    auto more_examples=p1;more_examples["profile_examples_every"]=10;
    check(effective_settings("E01",more_examples,common,t,{3840,2160})["applied"]==e1["applied"],
          "display-only example frequency is excluded from measurement settings");
    auto active_step=t;active_step.column_step=1;
    check(effective_settings("E01",p1,common,active_step,{3840,2160})["applied"]!=e1["applied"],
          "active column step changes measurement settings");
    auto bad=p1;bad["gradient"]="other";
    check(rejects([&](){validate_method_parameters("E01",bad);}),"unsupported Gaussian mode is rejected");
    bad=p1;bad["init_sigma"]="other";
    check(rejects([&](){validate_method_parameters("E01",bad);}),"unsupported initialization mode is rejected");
    bad=parameters(cfg,"Q10");bad["contour_mode"]="RETR_TREE";
    check(rejects([&](){validate_method_parameters("Q10",bad);}),"unsupported contour mode is rejected");
    bad=parameters(cfg,"Q10");bad["threshold"]="adaptive";
    check(rejects([&](){validate_method_parameters("Q10",bad);}),"unsupported threshold mode is rejected");
    bad=parameters(cfg,"E07");bad["ed_operator"]="typo";
    check(rejects([&](){validate_method_parameters("E07",bad);}),"unknown ED operator does not fall back to Sobel");
    bad=parameters(cfg,"E07");bad["ed_sigma"]=0.5;
    check(rejects([&](){validate_method_parameters("E07",bad);}),"native ED sigma clamping is not silently recorded as the requested value");
    bad=p4;bad["scale"]=0.8;
    check(rejects([&](){validate_method_parameters("E04",bad);}),"unsupported scaled LSD support mapping is explicit");
    bad=parameters(cfg,"E08");bad["do_merge"]=true;
    check(rejects([&](){validate_method_parameters("E08",bad);}),"unsupported merged FLD support mapping is explicit");
    bad=parameters(cfg,"E08");bad["distance_threshold"]=1e300;
    check(rejects([&](){validate_method_parameters("E08",bad);}),"double-to-float overflow is rejected before recording applied settings");
    bad=p1;bad["typo"]=1;
    check(rejects([&](){validate_method_parameters("E01",bad);}),"unknown method key is rejected");
    bad=p1;bad["nm_max_iter"]=2.5;
    check(rejects([&](){validate_method_parameters("E01",bad);}),"fractional integer parameter is rejected");
    changed_common=common;changed_common["segment_selection"]["external"]["sample_step_px"]=0;
    check(rejects([&](){effective_settings("E04",p4,changed_common,t,{3840,2160});}),"invalid active common sampling step is rejected before execution");
    changed_common=common;changed_common["fit"]["typo"]=1;
    check(rejects([&](){effective_settings("E01",p1,changed_common,t,{3840,2160});}),"unknown common fit key is rejected");
    std::printf("%d failure(s)\n",failures);return failures?1:0;
}
