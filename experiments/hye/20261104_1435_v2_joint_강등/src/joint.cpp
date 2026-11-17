#include "alignment.hpp"
#include "observer_internal.hpp"
#include "detail/alignment_diagnostics.hpp"
#include <cmath>
#include <stdexcept>
#include <limits>
namespace alignment_detail {
static std::unique_ptr<Observer> observer;
static JointModel model;
void set_measurement(const Observation& o) {
    auto& diagnostics=diagnostics_state();
    diagnostics.measurement_valid=true;
    diagnostics.minimum_qr_correlation=o.qr_correlation;
    diagnostics.minimum_line_correlation=o.line_correlation;
}
Observation observe(const cv::Mat& src) {
    if(!observer)throw std::logic_error("Call initialize_alignment() before joint/control");
    auto o=observer->measure(src);set_measurement(o);return o;
}
Alignment_t predict_joint(const Observation& o) {
    bool outside=false;auto q=model.predict(o.values,&outside);
    // 학습 자료의 s, d 범위 밖이면 진단으로 알린다. 값은 자르거나 바꾸지 않는다.
    set_diagnostics(outside,false,"Image match valid. Pose stability/robot convergence is not certified.");
    return q;
}
void JointModel::load(const fs::path& path) {
    cv::FileStorage f(path.u8string(),cv::FileStorage::READ);
    if(!f.isOpened())throw std::runtime_error("Cannot load joint model: "+path.u8string());
    auto m=f["model"];
    if(std::string(m["form"])!="v2_qr_mean_shift_bar_slope_difference")throw std::runtime_error("Expected joint v2 model");
    W=node_matrix(m["W"]);
    if(W.rows!=3||W.cols!=2||!cv::checkRange(W))throw std::runtime_error("Invalid joint v2 coefficients");
    auto bx=m["block_x"];if(!bx.isSeq()||bx.size()!=48)throw std::runtime_error("Joint v2 model requires48 block centers");
    block_x.clear();for(int k=0;k<48;++k)block_x.push_back(double(bx[k]));
    auto r=m["training_range"];s_min=double(r["s_min"]);s_max=double(r["s_max"]);d_min=double(r["d_min"]);d_max=double(r["d_max"]);
    if(!(s_min<s_max)||!(d_min<d_max))throw std::runtime_error("Invalid joint v2 training range");
}
Alignment_t JointModel::predict(const cv::Mat& x,bool* outside) const {
    auto f=v2_features(x,block_x);
    Alignment_t result{W.at<double>(0,0)+W.at<double>(1,0)*f[0]+W.at<double>(2,0)*f[1],W.at<double>(0,1)+W.at<double>(1,1)*f[0]+W.at<double>(2,1)*f[1]};
    if(!std::isfinite(result.y)||!std::isfinite(result.t))throw std::runtime_error("Nonfinite joint estimate");
    if(outside)*outside=f[0]<s_min||f[0]>s_max||f[1]<d_min||f[1]>d_max;
    return result;
}
}
void initialize_joint(const std::string& assets) {
    using namespace alignment_detail;
    auto prepared=std::make_unique<Observer>(fs::u8path(assets));
    JointModel prepared_model;prepared_model.load(fs::u8path(assets)/"joint_v2_model.json");
    // 모델의 블록 중심과 측정기의 블록 중심이 같아야 한다.
    auto centers=prepared->block_centers();for(int k=0;k<48;++k)if(std::abs(centers[k]-prepared_model.block_x[k])>1e-9)throw std::runtime_error("Joint v2 model block centers differ from line protocol");
    // Commit only after the observer AND model have loaded successfully.
    // A failed attempt leaves the previous joint context usable.
    observer=std::move(prepared);model=std::move(prepared_model);
}
Alignment_t joint(cv::Mat& src) {
    using namespace alignment_detail;reset_diagnostics("joint");
    try{return predict_joint(observe(src));}
    catch(const std::exception& e){set_diagnostics(false,false,e.what());throw;}
}
