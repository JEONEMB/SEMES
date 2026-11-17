#include "alignment.hpp"
#include "observer_internal.hpp"
#include <cmath>
#include <stdexcept>
#include <limits>
namespace alignment_detail {
static std::unique_ptr<Observer> observer;
static JointModel model;
static thread_local AlignmentDiagnostics diagnostics;
void reset_diagnostics(const std::string& stage) {
    diagnostics={stage,false,false,false,std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::quiet_NaN(),"call has not completed"};
}
void set_measurement(const Observation& o) {
    diagnostics.measurement_valid=true;
    diagnostics.minimum_qr_correlation=o.qr_correlation;
    diagnostics.minimum_line_correlation=o.line_correlation;
}
void set_diagnostics(bool outside,bool precondition,const std::string& message) {
    diagnostics.outside_training_range=outside;
    diagnostics.control_precondition_met=precondition;
    diagnostics.message=message;
}
Observation observe(const cv::Mat& src) {
    if(!observer)throw std::logic_error("Call initialize_alignment() before joint/control");
    auto o=observer->measure(src);set_measurement(o);return o;
}
Alignment_t predict_joint(const Observation& o) {
    auto q=model.predict(o.values);
    // Stored optimizer bounds contain a.25 normalized margin. Diagnose the
    // actual calibration interval, without clipping/changing the estimate.
    bool outside=q.y/model.scale[0]<model.lower[0]+.25||q.y/model.scale[0]>model.upper[0]-.25||q.t/model.scale[1]<model.lower[1]+.25||q.t/model.scale[1]>model.upper[1]-.25;
    set_diagnostics(outside,false,"Image match valid. Pose stability/robot convergence is not certified.");
    return q;
}
void JointModel::load(const fs::path& path) {
    cv::FileStorage f(path.u8string(),cv::FileStorage::READ);
    if(!f.isOpened())throw std::runtime_error("Cannot load joint model: "+path.u8string());
    auto m=f["model"];
    if(int(m["degree"])!=1||std::abs(double(m["shrink"])-.01)>1e-12)throw std::runtime_error("Expected frozen degree1/shrink.01 joint model");
    origin=node_matrix(m["origin"]);projector=node_matrix(m["projector"]);initial=node_matrix(m["initial"]);
    for(int k=0;k<2;++k){scale[k]=double(m["scale"][k]);lower[k]=double(m["lower"][k]);upper[k]=double(m["upper"][k]);}
    if(origin.rows!=1||origin.cols!=204||projector.rows!=204||projector.cols!=2||initial.rows!=2||initial.cols!=2||!cv::checkRange(origin)||!cv::checkRange(projector)||!cv::checkRange(initial))throw std::runtime_error("Invalid joint parameter dimensions");
}
Alignment_t JointModel::predict(const cv::Mat& x) const {
    cv::Mat z=feature_map(x),h=(z-origin)*projector,q=h*initial;
    Alignment_t result{q.at<double>(0,0)*scale[0],q.at<double>(0,1)*scale[1]};
    if(!std::isfinite(result.y)||!std::isfinite(result.t))throw std::runtime_error("Nonfinite joint estimate");
    return result;
}
}
void initialize_alignment(const std::string& assets,const std::string& control_path,double maximum_t) {
    using namespace alignment_detail;
    auto prepared=std::make_unique<Observer>(fs::u8path(assets));
    JointModel prepared_model;prepared_model.load(fs::u8path(assets)/"both_model.json");
    load_control(control_path,maximum_t);
    observer=std::move(prepared);model=std::move(prepared_model);
}
Alignment_t joint(cv::Mat& src) {
    using namespace alignment_detail;reset_diagnostics("joint");
    try{return predict_joint(observe(src));}
    catch(const std::exception& e){set_diagnostics(false,false,e.what());throw;}
}
AlignmentDiagnostics alignment_diagnostics() {return alignment_detail::diagnostics;}
