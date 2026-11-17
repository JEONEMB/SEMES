#include "alignment.hpp"
#include "observer_internal.hpp"
#include <cmath>
#include <stdexcept>
namespace alignment_detail {
static cv::Mat origin,projector;
static double maximum_t=.09,minimum_y=-2,maximum_y=2;
static bool ready=false;
void load_control(const std::string& path,double limit){
    if(!std::isfinite(limit)||limit<=0)throw std::invalid_argument("Positive control T gate required");
    if(path.empty()){ready=false;maximum_t=limit;return;}
    cv::FileStorage f(path,cv::FileStorage::READ);if(!f.isOpened())throw std::runtime_error("Cannot load control model: "+path);
    cv::Mat o,p;f["origin"]>>o;f["projector"]>>p;
    if(o.rows!=1||o.cols!=204||p.rows!=204||p.cols!=1||o.type()!=CV_64F||p.type()!=CV_64F||!cv::checkRange(o)||!cv::checkRange(p))throw std::runtime_error("Invalid conditional-Y parameters");
    origin=o;projector=p;minimum_y=double(f["minimum_y"]);maximum_y=double(f["maximum_y"]);maximum_t=limit;ready=true;
}
}
Alignment_t control(cv::Mat& src){
    using namespace alignment_detail;reset_diagnostics("control");
    try{
        if(!ready)throw std::logic_error("Initialize a trained T=0 control model before control()");
        auto o=observe(src);auto q=predict_joint(o);
        if(std::abs(q.t)>maximum_t){set_diagnostics(false,false,"Joint T estimate exceeds configured control gate; align T and recapture first");throw std::domain_error("control() T-near-zero precondition not met");}
        cv::Mat y=(feature_map(o.values)-origin)*projector;q.y=y.at<double>(0);
        if(!std::isfinite(q.y))throw std::runtime_error("Nonfinite conditional-Y estimate");
        set_diagnostics(q.y<minimum_y||q.y>maximum_y,true,"Conditional Y estimate; T gate is an observable proxy, not true-pose certification");return q;
    }catch(const std::exception& e){auto d=alignment_diagnostics();set_diagnostics(d.outside_training_range,d.control_precondition_met,e.what());throw;}
}
