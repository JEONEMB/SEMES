#include "alignment.hpp"
#include "common/reference.hpp"
#include "steps/02_find_screws.hpp"
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <cmath>
// main: 실제 영역 정합 실패 영상에서 3단계 값 대체와 최종 측정 지속 검사
//   입력: 모델 폴더
//   출력: 검사 수, 실패 수, 실제 경고 메시지
int main(int argc,char** argv) {
    if(argc!=2)return 2;
    cv::setNumThreads(1);
    alignment::Reference ref;std::string error;
    if(!alignment::load_reference(argv[1],ref,error)){std::cerr<<error;return 2;}
    alignment::ScrewFinder finder(ref);
    const std::array<bool,4> wanted{{true,true,true,true}};
    int count=0,failed=0;
    const auto expect=[&](bool ok,const char* label){++count;if(!ok){++failed;std::cerr<<"FAIL "<<label<<'\n';}};
    const auto normal=finder.find_screw_shifts(ref.photo,0,0,wanted);
    for(const auto& s:normal) {
        expect(s.found,"normal screw found");
#ifdef HAS_AREA
        expect(!s.area_fallback,"normal area fit retained");
#endif
    }
    // 중심 93x93을 유지하고 바깥 고리만 검게 만들어 OpenCV ECC가 실제로 수렴하지 않게 한다.
    // 반환값 조작이나 모의 함수 없이 영상 입력만 변경한다.
    auto damaged=ref.photo.clone();const auto p=ref.screw[0];
    for(int y=-75;y<=75;++y)for(int x=-75;x<=75;++x)
        if(std::max(std::abs(x),std::abs(y))>46)damaged.at<unsigned char>(p.y+y,p.x+x)=0;
    const auto shifts=finder.find_screw_shifts(damaged,0,0,wanted);
    expect(shifts[0].found,"stage three measurement retained");
    expect(std::isfinite(shifts[0].dx)&&std::isfinite(shifts[0].dy),"finite edge coordinates");
#ifdef HAS_AREA
    expect(shifts[0].area_fallback,"actual ECC failure uses fallback");
    expect(shifts[0].area_dx==shifts[0].dx,"area is exactly edge value");
    expect(shifts[0].quality_warning,"fallback emits warning");
    for(int k=1;k<4;++k)expect(!shifts[k].area_fallback && shifts[k].area_dx==normal[k].area_dx,"successful areas unchanged");
#endif
    expect(alignment::initialize(argv[1]).status==alignment::Status::Ok,"initialize");
    expect(alignment::set_zero_from_photos({ref.photo}).status!=alignment::Status::Failed,"zero calibration");
    const auto r=alignment::measure_alignment(damaged);
    expect(r.status!=alignment::Status::Failed && std::isfinite(r.y_mm)&&std::isfinite(r.t_deg),"public measurement continues");
    expect(r.screws_used=="1234","all screws remain in formula");
#ifdef HAS_AREA
    expect(r.screw_area_shift_px[0]==r.screw_shift_px[0],"public feature substitution");
    expect(r.status==alignment::Status::Warning,"public warning");
#endif
#ifndef DELIVERY_API
    expect(r.correction_allowed,"operational correction remains allowed");
#endif
    std::cout<<"Damaged ring Y="<<r.y_mm<<" T="<<r.t_deg<<" screws="<<r.screws_used<<'\n'<<r.message<<'\n';
    expect(alignment::set_zero_from_photos({damaged}).status!=alignment::Status::Failed,"zero with retained edge coordinates");
    const auto zero=alignment::measure_alignment(damaged);
    expect(zero.status!=alignment::Status::Failed && std::abs(zero.y_mm)<1e-10 && std::abs(zero.t_deg)<1e-10,"matching zero remains zero");
    auto hidden=ref.photo.clone();
    for(const auto& c:ref.screw)hidden(cv::Rect(c.x-120,c.y-120,240,240)).setTo(128);
    for(const auto& s:finder.find_screw_shifts(hidden,0,0,wanted))expect(!s.found,"real screw detection failure is not fabricated");
    std::cout<<count<<" checks, "<<failed<<" failures\n";
    return failed?1:0;
}
