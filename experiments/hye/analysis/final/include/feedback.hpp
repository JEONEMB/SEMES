#pragma once
#include "motion.hpp"
namespace semes::historical {
struct FeedbackController {
    cv::Mat prior,J,s,goal,correction;std::string policy,status="not_started",last_kind;
    int turn=0,streak=0,probes=0;double trust=1,feature_rms=std::numeric_limits<double>::infinity();
    std::vector<cv::Vec2d> inputs;std::vector<cv::Mat> deltas;
    FeedbackController()=default;
    FeedbackController(const cv::Mat& jacobian,const std::string& selected):prior(jacobian.clone()),J(jacobian.clone()),policy(selected){if(J.type()!=CV_64F||J.cols!=2||J.rows==0)throw std::runtime_error("Two-column feedback Jacobian required");if(policy=="fixed_jacobian")probes=0;else if(policy=="probe2_broyden")probes=2;else if(policy=="probe3_broyden")probes=3;else throw std::runtime_error("Unknown historical feedback policy");}
    void assess(){
        if(!cv::checkRange(s)){status="measurement_failed";feature_rms=std::numeric_limits<double>::infinity();correction=cv::Mat(2,1,CV_64F,cv::Scalar(std::numeric_limits<double>::quiet_NaN()));return;}
        cv::Mat error=s-goal,G=J.t()*J;double damping=std::max(cv::trace(G)[0]*1e-9,1e-8);cv::solve(G+damping*cv::Mat::eye(2,2,CV_64F),-J.t()*error,correction,cv::DECOMP_LU);feature_rms=numeric::rms(error);
        bool candidate=feature_rms<=2.5&&cv::norm(correction,cv::NORM_INF)<=.03;streak=candidate?streak+1:0;
        if(turn>=probes&&streak>=2)status="visual_goal_confirmed";else if(turn>=20)status="turn_limit";
    }
    FeedbackController& start(const cv::Mat& observation,const cv::Mat& fixed_goal){s=observation.clone();goal=fixed_goal.clone();if(s.cols!=1||s.rows!=J.rows||goal.size()!=s.size())throw std::runtime_error("Visual feature shape mismatch");status="running";assess();return *this;}
    std::optional<cv::Vec2d> next_action(){
        if(status!="running")return std::nullopt;if(turn<probes){last_kind="probe";return turn==0?cv::Vec2d(.05,0):turn==1?cv::Vec2d(0,.01):cv::Vec2d(-.05,-.01);}
        if(streak){last_kind="confirm";return cv::Vec2d(0,0);}last_kind="feedback";cv::Vec2d action(.7*correction.at<double>(0),.7*correction.at<double>(1));double scale=std::max({std::abs(action[0])/(.25*trust),std::abs(action[1])/(.25*trust),1.});return action/scale;
    }
    void observe(const cv::Mat& observation,const cv::Vec2d& executed){
        if(status!="running")throw std::runtime_error("Controller is not awaiting an observation");if(observation.size()!=s.size())throw std::runtime_error("Feedback feature shape mismatch");cv::Mat delta=observation-s;double old_norm=cv::norm(s-goal);++turn;
        if(cv::checkRange(observation)&&policy!="fixed_jacobian"){
            if(last_kind=="probe"){
                inputs.push_back(executed);deltas.push_back(delta.clone());if(turn==probes){cv::Mat U(2,int(inputs.size()),CV_64F),S(J.rows,int(inputs.size()),CV_64F);for(int i=0;i<int(inputs.size());++i){U.at<double>(0,i)=inputs[i][0];U.at<double>(1,i)=inputs[i][1];deltas[i].copyTo(S.col(i));}cv::Mat ridge=(cv::Mat_<double>(2,2)<<.25*.05*.05,0,0,.25*.01*.01);J=(S*U.t()+prior*ridge)*(U*U.t()+ridge).inv();}
            }else if(last_kind=="feedback"&&cv::norm(executed)>=.0025){cv::Mat u(executed),update=(delta-J*u)*u.t()/(executed.dot(executed)+1e-8);double scale=std::min(1.,.5*cv::norm(J)/std::max(cv::norm(update),1e-12));J+=.25*scale*update;}
        }
        if(last_kind=="feedback"&&cv::checkRange(observation)){double ratio=cv::norm(observation-goal)/std::max(old_norm,1e-12);if(ratio>1.05)trust=std::max(.1,trust*.5);else if(ratio<.8)trust=std::min(1.,trust*1.2);}s=observation.clone();assess();
    }
    Json state()const{
        Json j=Json::dict();j["prior"]=jsonmatrix(prior);j["J"]=jsonmatrix(J);j["s"]=motion::vector_json(s);j["goal"]=motion::vector_json(goal);j["correction"]=motion::vector_json(correction);j["policy"]=policy;j["status"]=status;j["last_kind"]=last_kind;j["turn"]=turn;j["streak"]=streak;j["probes"]=probes;j["trust"]=trust;j["feature_rms"]=feature_rms;j["inputs"]=Json::list();j["deltas"]=Json::list();for(const auto& v:inputs){Json a=Json::list();a.push(v[0]);a.push(v[1]);j["inputs"].push(a);}for(const auto& v:deltas)j["deltas"].push(motion::vector_json(v));return j;
    }
    static FeedbackController from_state(const Json& j){FeedbackController c(matrix(j.at("prior")),j.at("policy").str());c.J=matrix(j.at("J"));c.s=motion::vector_from_json(j.at("s"));c.goal=motion::vector_from_json(j.at("goal"));c.correction=motion::vector_from_json(j.at("correction"));c.status=j.at("status").str();c.last_kind=j.at("last_kind").str();c.turn=int(j.at("turn").num());c.streak=int(j.at("streak").num());c.probes=int(j.at("probes").num());c.trust=j.at("trust").num();c.feature_rms=j.at("feature_rms").type==Json::Null?std::numeric_limits<double>::infinity():j.at("feature_rms").num();for(const auto& v:j.at("inputs").array)c.inputs.push_back({v.at(0).num(),v.at(1).num()});for(const auto& v:j.at("deltas").array)c.deltas.push_back(motion::vector_from_json(v));return c;}
};
}
