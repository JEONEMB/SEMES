#include "motion.hpp"
#include <algorithm>
using namespace semes::numeric;
namespace semes::motion {
static double nan(){return std::numeric_limits<double>::quiet_NaN();}
static Json pair_json(cv::Vec2d v){Json a=Json::list();a.push(v[0]);a.push(v[1]);return a;}
static cv::Vec2d pair(const Json& j){if(j.type!=Json::Array||j.size()!=2)throw std::runtime_error("Expected two command components");return {j.at(0).type==Json::Null?nan():j.at(0).num(),j.at(1).type==Json::Null?nan():j.at(1).num()};}
cv::Mat vector_from_json(const Json& j){if(j.type!=Json::Array)throw std::runtime_error("Expected visual vector");cv::Mat v(int(j.size()),1,CV_64F);for(int i=0;i<v.rows;++i)v.at<double>(i)=j.at(i).type==Json::Null?nan():j.at(i).num();return v;}
Json vector_json(const cv::Mat& v){Json j=Json::list();for(int i=0;i<int(v.total());++i){double x=v.ptr<double>()[i];j.push(std::isfinite(x)?Json(x):Json());}return j;}
void Settings::validate()const{
    for(double x:{probe_y,probe_t,max_step_y,max_step_t,gain,minimum_response_snr,maximum_condition,noise_rms,stop_feature_rms,stop_command})if(!std::isfinite(x))throw std::runtime_error("Finite settings required");
    if((probe_count!=2&&probe_count!=3)||max_turns<3||max_turns>20)throw std::runtime_error("Two/three probes and at most20 turns required");
    if(std::min({probe_y,probe_t,max_step_y,max_step_t,noise_rms,minimum_response_snr,maximum_condition,stop_feature_rms,stop_command})<=0)throw std::runtime_error("Positive settings required");
    if(probe_y>max_step_y||probe_t>max_step_t||gain<=0||gain>1||history_frames<4)throw std::runtime_error("Invalid probe caps, gain or history");
}
Json Settings::json()const{Json j=Json::dict();j["probe_y"]=probe_y;j["probe_t"]=probe_t;j["max_step_y"]=max_step_y;j["max_step_t"]=max_step_t;j["gain"]=gain;j["minimum_response_snr"]=minimum_response_snr;j["maximum_condition"]=maximum_condition;j["noise_rms"]=noise_rms;j["stop_feature_rms"]=stop_feature_rms;j["stop_command"]=stop_command;j["probe_count"]=probe_count;j["max_turns"]=max_turns;j["history_frames"]=history_frames;return j;}
Settings Settings::from_json(const Json& j){Settings s;auto get=[&](const char* key,double& x){if(j.has(key))x=j.at(key).num();};get("probe_y",s.probe_y);get("probe_t",s.probe_t);get("max_step_y",s.max_step_y);get("max_step_t",s.max_step_t);get("gain",s.gain);get("minimum_response_snr",s.minimum_response_snr);get("maximum_condition",s.maximum_condition);get("noise_rms",s.noise_rms);get("stop_feature_rms",s.stop_feature_rms);get("stop_command",s.stop_command);auto integer=[&](const char* k,int& x){if(j.has(k)){double v=j.at(k).num();if(!std::isfinite(v)||v!=std::floor(v))throw std::runtime_error("Integer setting required");x=int(v);}};integer("probe_count",s.probe_count);integer("max_turns",s.max_turns);integer("history_frames",s.history_frames);s.validate();return s;}
Controller::Controller(Settings settings,const cv::Mat& prior):cfg_(settings),prior_(prior.clone()),J_(prior.clone()),correction_(nan(),nan()){
    cfg_.validate();if(!prior.empty()&&(prior.type()!=CV_64F||prior.cols!=2||!cv::checkRange(prior)))throw std::runtime_error("Invalid prior response");
}
Controller& Controller::start(const cv::Mat& observation,const cv::Mat& fixed_goal){
    if(status_!="not_started")throw std::runtime_error("Already started");
    if(observation.type()!=CV_64F||observation.cols!=1||observation.rows<2||observation.size()!=fixed_goal.size()||!cv::checkRange(fixed_goal))throw std::runtime_error("Invalid visual vectors");
    if(!J_.empty()&&J_.rows!=observation.rows)throw std::runtime_error("Prior feature dimension mismatch");
    s_=observation.clone();goal_=fixed_goal.clone();history_.push_back({cumulative_,s_.clone()});status_="running";assess();return *this;
}
bool Controller::identify(){
    int start=std::max(0,int(history_.size())-cfg_.history_frames),n=int(history_.size())-start;
    cv::Mat U(n,2,CV_64F),S(n,s_.rows,CV_64F);
    for(int i=0;i<n;++i){U.at<double>(i,0)=history_[start+i].cumulative[0]/cfg_.probe_y;U.at<double>(i,1)=history_[start+i].cumulative[1]/cfg_.probe_t;cv::Mat row=history_[start+i].observation.t();row.copyTo(S.row(i));}
    cv::Mat um,sm;cv::reduce(U,um,0,cv::REDUCE_AVG);cv::reduce(S,sm,0,cv::REDUCE_AVG);
    U-=cv::repeat(um,n,1);S-=cv::repeat(sm,n,1);cv::Mat G=U.t()*U,e,V;cv::eigen(G,e,V);
    if(e.at<double>(1)<=std::max(e.at<double>(0)*1e-10,1e-12)){
        cv::Mat sv;cv::SVD::compute(U,sv);int rank=0;double cutoff=sv.at<double>(0)*std::max(U.rows,U.cols)*std::numeric_limits<double>::epsilon();for(int i=0;i<int(sv.total());++i)rank+=sv.at<double>(i)>cutoff;
        diagnostics_=Json::dict();diagnostics_["accepted"]=false;diagnostics_["reason"]="inputs_do_not_span_two_axes";diagnostics_["input_rank"]=rank;return false;
    }
    cv::Mat B=least_squares(U,S).t(),diagonal=cv::Mat::zeros(2,2,CV_64F);for(int i=0;i<2;++i)diagonal.at<double>(i,i)=std::sqrt(e.at<double>(i));cv::Mat root=V.t()*diagonal*V;
    cv::Mat sv;cv::SVD::compute(B*root,sv);double snr=sv.at<double>(int(sv.total())-1)/(cfg_.noise_rms*std::sqrt(double(s_.rows)));
    cv::SVD::compute(B,sv);double condition=sv.at<double>(0)/std::max(sv.at<double>(int(sv.total())-1),1e-20);
    bool accepted=snr>=cfg_.minimum_response_snr&&condition<=cfg_.maximum_condition;
    diagnostics_=Json::dict();diagnostics_["accepted"]=accepted;diagnostics_["reason"]=accepted?"accepted":"weak_or_ambiguous_visual_response";diagnostics_["response_snr"]=snr;diagnostics_["scaled_response_condition"]=condition;diagnostics_["input_rank"]=2;diagnostics_["fit_rms"]=rms(S-U*B.t());diagnostics_["frames"]=n;diagnostics_["reference"]="original fixed center; intercept is response regression only";
    if(accepted&&!prior_.empty()){
        cv::Mat scaled=prior_.clone();scaled.col(0)*=cfg_.probe_y;scaled.col(1)*=cfg_.probe_t;cv::Mat relative=least_squares(scaled,B,1e-10)-cv::Mat::eye(2,2,CV_64F);
        cv::SVD::compute(relative,sv);double change=sv.at<double>(0);diagnostics_["relative_response_change"]=change;
        if(change>.5){accepted=false;diagnostics_["accepted"]=false;diagnostics_["reason"]="response_change_exceeds_guard";}
        else {cv::Mat fitted;cv::solve(G+2*cv::Mat::eye(2,2,CV_64F),U.t()*S+2*scaled.t(),fitted,cv::DECOMP_LU);B=fitted.t();}
    }
    if(accepted){J_=B.clone();J_.col(0)/=cfg_.probe_y;J_.col(1)/=cfg_.probe_t;++updates_;}
    return accepted;
}
void Controller::assess(){
    if(!cv::checkRange(s_)){status_="measurement_failed";feature_rms_=std::numeric_limits<double>::infinity();correction_={nan(),nan()};return;}
    feature_rms_=rms(s_-goal_);correction_={nan(),nan()};
    if(!J_.empty()){cv::Mat c=-least_squares(J_,s_-goal_,1e-10);correction_={c.at<double>(0),c.at<double>(1)};}
    bool candidate=!J_.empty()&&feature_rms_<=cfg_.stop_feature_rms&&std::max(std::abs(correction_[0]),std::abs(correction_[1]))<=cfg_.stop_command;
    streak_=candidate?streak_+1:0;
    if(turn_>=cfg_.probe_count&&streak_>=2)status_="visual_goal_confirmed";
    else if(turn_>=cfg_.max_turns)status_="turn_limit";
    else if(turn_>=cfg_.probe_count&&J_.empty())status_="insufficient_separation";
}
std::optional<cv::Vec2d> Controller::next_action(){
    if(status_!="running")return {};if(pending_)return pending_;
    cv::Vec2d u;
    if(turn_<cfg_.probe_count){kind_="probe";u=turn_==0?cv::Vec2d(cfg_.probe_y,0):turn_==1?cv::Vec2d(0,cfg_.probe_t):cv::Vec2d(-cfg_.probe_y,-cfg_.probe_t);}
    else if(streak_){kind_="confirm";u={0,0};}
    else{kind_="feedback";u=correction_*cfg_.gain;double factor=std::max({1.,std::abs(u[0])/(cfg_.max_step_y*trust_),std::abs(u[1])/(cfg_.max_step_t*trust_)});u/=factor;}
    pending_=u;return u;
}
void Controller::observe(const cv::Mat& observation,const cv::Vec2d& u){
    if(status_!="running"||!pending_)throw std::runtime_error("Call next_action before acknowledging a move");
    if(observation.type()!=CV_64F||observation.size()!=s_.size()||!std::isfinite(u[0])||!std::isfinite(u[1]))throw std::runtime_error("Invalid observation or acknowledged increment");
    auto old=s_.clone();cv::Mat delta=observation-old,predicted;if(!J_.empty())predicted=J_*cv::Mat(u);
    auto suggested=*pending_;pending_.reset();++turn_;cumulative_+=u;s_=observation.clone();
    if(cv::checkRange(s_)){
        history_.push_back({cumulative_,s_.clone()});if(cv::norm(u)>1e-12)identify();
        if(kind_=="feedback"){double ratio=cv::norm(s_-goal_)/std::max(cv::norm(old-goal_),1e-12);if(ratio>1.05)trust_=std::max(.1,trust_*.5);else if(ratio<.8)trust_=std::min(1.,trust_*1.2);}
    }
    assess();Json event=Json::dict();event["turn"]=turn_;event["kind"]=kind_;event["suggested"]=pair_json(suggested);event["executed_relative"]=pair_json(u);event["observed_change"]=vector_json(delta);event["predicted_change"]=predicted.empty()?Json():vector_json(predicted);event["cumulative_relative"]=pair_json(cumulative_);event["identification"]=diagnostics_;event["feature_rms"]=std::isfinite(feature_rms_)?Json(feature_rms_):Json();event["correction"]=J_.empty()?Json():pair_json(correction_);event["status"]=status_;events_.push(event);
}
Json Controller::snapshot()const{
    Json j=Json::dict();j["status"]=status_;j["turns"]=turn_;j["mode"]=prior_.empty()?"online_only":"prior_guarded";j["feature_rms"]=std::isfinite(feature_rms_)?Json(feature_rms_):Json();j["correction"]=std::isfinite(correction_[0])&&std::isfinite(correction_[1])?pair_json(correction_):Json();j["jacobian"]=J_.empty()?Json():jsonmatrix(J_);j["identification"]=diagnostics_;j["identification_updates"]=updates_;j["hardware_command_sent"]=false;return j;
}
Json Controller::state()const{
    Json j=Json::dict();j["schema"]="semes.motion.v1";j["settings"]=cfg_.json();j["prior"]=prior_.empty()?Json():jsonmatrix(prior_);j["jacobian"]=J_.empty()?Json():jsonmatrix(J_);j["observation"]=vector_json(s_);j["fixed_goal"]=vector_json(goal_);j["turn"]=turn_;j["streak"]=streak_;j["updates"]=updates_;j["trust"]=trust_;j["status"]=status_;j["last_kind"]=kind_;j["feature_rms"]=std::isfinite(feature_rms_)?Json(feature_rms_):Json();j["correction"]=pair_json(correction_);j["pending"]=pending_?pair_json(*pending_):Json();j["cumulative"]=pair_json(cumulative_);j["diagnostics"]=diagnostics_;j["events"]=events_;j["history"]=Json::list();for(auto& h:history_){Json f=Json::dict();f["cumulative"]=pair_json(h.cumulative);f["observation"]=vector_json(h.observation);j["history"].push(f);}return j;
}
Controller Controller::from_state(const Json& j){
    if(j.at("schema").str()!="semes.motion.v1")throw std::runtime_error("Unsupported motion session schema");
    Controller c(Settings::from_json(j.at("settings")),j.at("prior").type==Json::Null?cv::Mat():matrix(j.at("prior")));
    c.J_=j.at("jacobian").type==Json::Null?cv::Mat():matrix(j.at("jacobian"));c.s_=vector_from_json(j.at("observation"));c.goal_=vector_from_json(j.at("fixed_goal"));c.turn_=int(j.at("turn").num());c.streak_=int(j.at("streak").num());c.updates_=int(j.at("updates").num());c.trust_=j.at("trust").num();c.status_=j.at("status").str();c.kind_=j.at("last_kind").str();c.correction_=pair(j.at("correction"));if(j.at("pending").type!=Json::Null)c.pending_=pair(j.at("pending"));c.cumulative_=pair(j.at("cumulative"));c.diagnostics_=j.at("diagnostics");c.events_=j.at("events");c.feature_rms_=j.at("feature_rms").type==Json::Null?std::numeric_limits<double>::infinity():j.at("feature_rms").num();for(auto& h:j.at("history").array)c.history_.push_back({pair(h.at("cumulative")),vector_from_json(h.at("observation"))});
    if(c.turn_<0||c.turn_>c.cfg_.max_turns||c.s_.size()!=c.goal_.size()||c.s_.rows<2||!cv::checkRange(c.goal_)||(!c.J_.empty()&&(c.J_.rows!=c.s_.rows||c.J_.cols!=2||!cv::checkRange(c.J_)))||(!c.prior_.empty()&&c.prior_.rows!=c.s_.rows)||!std::isfinite(c.trust_)||c.trust_<.1||c.trust_>1)throw std::runtime_error("Invalid persisted motion state");
    return c;
}
cv::Vec2d choose_probe_amplitudes(const cv::Mat& prior,cv::Vec2d minimum,cv::Vec2d maximum,double noise_rms){
    if(prior.type()!=CV_64F||prior.cols!=2||prior.rows<2||!cv::checkRange(prior)||!std::isfinite(noise_rms)||noise_rms<=0)throw std::runtime_error("Invalid probe response");cv::Vec2d result;
    for(int i=0;i<2;++i){if(minimum[i]<=0||maximum[i]<minimum[i])throw std::runtime_error("Invalid probe bounds");auto v=prior.col(i),other=prior.col(1-i);cv::Mat distinguish=v-other*(other.dot(v)/std::max(other.dot(other),1e-20));double required=3*noise_rms*std::sqrt(double(prior.rows))/std::max(cv::norm(distinguish),1e-12);result[i]=std::clamp(required,minimum[i],maximum[i]);}return result;
}
}
