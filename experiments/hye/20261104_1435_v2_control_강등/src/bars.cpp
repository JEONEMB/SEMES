#include "control_internal.hpp"
#include <stdexcept>
namespace legacy_control {
Bars::Bars(const fs::path& assets,const cv::Mat& center){
    cv::FileStorage protocol((assets/"line_protocol.json").u8string(),cv::FileStorage::READ);
    if(!protocol.isOpened())throw std::runtime_error("Missing line protocol");
    for(auto it=protocol["blocks"].begin();it!=protocol["blocks"].end();++it){
        auto b=(*it)["read_box"];cv::Rect r(int(b[0]),int(b[1]),int(b[2])-int(b[0]),int(b[3])-int(b[1]));
        if(r.height!=49||r.width!=64||(r&cv::Rect(0,0,3840,2160))!=r)throw std::runtime_error("Invalid permitted line block");
        boxes_.push_back(r);x_.push_back(r.x+r.width/2.0);reference_.push_back(line_profile(center,r));
    }
    if(boxes_.size()!=48)throw std::runtime_error("Expected48 line blocks (24 upper, 24 lower)");
}
double Bars::slope_difference(const cv::Mat& gray,double* minimum_correlation)const{
    double slope[2],mc=1;
    for(int bar=0;bar<2;++bar){
        double shift[24],mx=0,my=0,sxx=0,sxy=0;
        for(int k=0;k<24;++k){auto f=match_line(reference_[bar*24+k],line_profile(gray,boxes_[bar*24+k]));
            if(!f.valid)throw std::runtime_error("Line normal-profile match failed at block"+std::to_string(bar*24+k));
            mc=std::min(mc,f.correlation);shift[k]=f.shift;mx+=(x_[bar*24+k]-1934.)/1000./24;my+=f.shift/24;}
        for(int k=0;k<24;++k){double u=(x_[bar*24+k]-1934.)/1000.-mx;sxx+=u*u;sxy+=u*(shift[k]-my);}
        slope[bar]=sxy/sxx;
    }
    if(minimum_correlation)*minimum_correlation=mc;
    return slope[0]-slope[1];
}
}
