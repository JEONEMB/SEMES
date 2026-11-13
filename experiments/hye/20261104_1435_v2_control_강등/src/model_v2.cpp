#include "control_internal.hpp"
#include <cmath>
#include <stdexcept>
namespace legacy_control {
ModelV2::ModelV2(const fs::path& file){
    cv::FileStorage f(file.u8string(),cv::FileStorage::READ);
    if(!f.isOpened())throw std::runtime_error("Cannot load control v2 model: "+file.u8string());
    auto m=f["model"];
    if(std::string(m["form"])!="v2_precise_y_with_gated_residual_t")throw std::runtime_error("Expected control v2 model");
    f0_=double(m["y0"][0]);f1_=double(m["y0"][1]);
    c0_=double(m["t_hat"][0]);c1_=double(m["t_hat"][1]);c2_=double(m["t_hat"][2]);
    k_=double(m["k"]);theta_=double(m["theta"]);
    auto r=m["training_range"];s_min_=double(r["s_min"]);s_max_=double(r["s_max"]);d_min_=double(r["d_min"]);d_max_=double(r["d_max"]);
    auto bx=m["block_x"];for(int i=0;i<int(bx.size());++i)block_x.push_back(double(bx[i]));
    for(double v:{f0_,f1_,c0_,c1_,c2_,k_,theta_})if(!std::isfinite(v))throw std::runtime_error("Nonfinite control v2 parameter");
    if(theta_<0||block_x.size()!=48||!(s_min_<s_max_)||!(d_min_<d_max_))throw std::runtime_error("Invalid control v2 model");
}
Estimate ModelV2::predict(double s,double d)const{
    double t_hat=c0_+c1_*s+c2_*d;
    // 남은 T 추정이 잡음 문턱(theta) 안이면 보정하지 않는다. 밖이면 문턱을 넘은 만큼만 Y 에서 뺀다.
    double gated=std::abs(t_hat)<=theta_?0.0:t_hat-(t_hat>0?theta_:-theta_);
    double y=f0_+f1_*s-k_*gated;
    if(!std::isfinite(y)||!std::isfinite(t_hat))throw std::runtime_error("Nonfinite control v2 estimate");
    return {y,t_hat,t_hat,s<s_min_||s>s_max_||d<d_min_||d>d_max_};
}
}
