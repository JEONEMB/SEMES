#include "motion.hpp"
#include <iostream>
#include <chrono>
using namespace semes;
using namespace semes::motion;
static void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
static cv::Mat at(const cv::Mat& J,cv::Vec2d q,const cv::Mat& goal){return goal+J*cv::Mat(q);}
int wmain(int argc,wchar_t** wide){
    Utf8Arguments args(argc,wide);
    try{
        if(argc!=2&&argc!=3)throw std::runtime_error("motion_checks new-output [--unique]");fs::path out=fs::u8path(args.data()[1]);if(argc==3){if(std::string(args.data()[2])!="--unique")throw std::runtime_error("Unknown check option");out/=std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());}if(fs::exists(out))throw std::runtime_error("New output required");
        Json checks=Json::dict();cv::Mat J=(cv::Mat_<double>(4,2)<<120,20,-80,300,50,-90,200,150),goal=(cv::Mat_<double>(4,1)<<7,-2,5,1),zero=cv::Mat::zeros(4,1,CV_64F);
        Settings cfg;cfg.probe_y=.2;cfg.probe_t=.04;cfg.probe_count=2;
        cv::Vec2d q(.65,-.4);Controller c(cfg);c.start(at(J,q,goal),goal);require(c.response().empty(),"Online-only start received a response prior");
        for(int i=0;i<2;++i){auto u=c.next_action();require(u&&c.next_action()==u,"Pending action was changed");q+=*u;c.observe(at(J,q,goal),*u);}
        require(cv::norm(c.response()-J,cv::NORM_INF)<1e-10,"Known response not identified");require(cv::norm(c.correction()+q)<1e-10,"Correction targets earlier pose");checks["response_identified_from_acknowledged_moves"]=true;
        for(int i=0;i<20;++i){auto u=c.next_action();if(!u)break;q+=*u;c.observe(at(J,q,goal),*u);}
        require(c.turns()<=20&&c.status()=="visual_goal_confirmed"&&std::max(std::abs(q[0]),std::abs(q[1]))<=.09,"Linear fixture failed");checks["known_linear_fixture"]=c.snapshot();
        q={.2,.3};Controller partial(cfg);partial.start(at(J,q,zero),zero);for(auto u:{cv::Vec2d(.13,.003),cv::Vec2d(-.01,.027)}){partial.next_action();q+=u;partial.observe(at(J,q,zero),u);}require(cv::norm(partial.response()-J,cv::NORM_INF)<1e-9,"Suggested move used instead of actual acknowledgement");checks["partial_cross_axis_acknowledgements"]=true;
        Controller refused;refused.start(cv::Mat::ones(4,1,CV_64F)*10,zero);for(int i=0;i<3;++i){refused.next_action();refused.observe(cv::Mat::ones(4,1,CV_64F)*10,{0,0});}require(refused.status()=="insufficient_separation"&&!refused.next_action(),"Unexecuted probes accepted");checks["unexecuted_probes_rejected"]=true;
        cv::Mat one=J.clone();cv::Mat second=2*J.col(0);second.copyTo(one.col(1));q={.5,.3};Controller rankone(cfg);rankone.start(at(one,q,zero),zero);for(int i=0;i<2;++i){auto u=rankone.next_action();q+=*u;rankone.observe(at(one,q,zero),*u);}require(rankone.response().empty()&&rankone.status()=="insufficient_separation","Rank-one response accepted");checks["rank_one_response_rejected"]=true;
        Controller failed(cfg);failed.start(cv::Mat::ones(4,1,CV_64F),zero);failed.next_action();cv::Mat invalid(4,1,CV_64F,cv::Scalar(std::numeric_limits<double>::quiet_NaN()));failed.observe(invalid,{0,0});require(failed.turns()==1&&failed.status()=="measurement_failed"&&!failed.next_action()&&failed.snapshot().at("feature_rms").type==Json::Null,"Failed capture was not counted");checks["failed_capture_counts_turn"]=true;
        q={100,100};Controller capped(cfg);capped.start(at(J,q,zero),zero);for(int i=0;i<20;++i){auto u=capped.next_action();require(u&&std::abs((*u)[0])<=cfg.max_step_y&&std::abs((*u)[1])<=cfg.max_step_t,"Action cap exceeded");q+=*u;capped.observe(at(J,q,zero),*u);}require(capped.status()=="turn_limit"&&capped.turns()==20&&!capped.next_action(),"Turn limit exceeded");checks["twenty_turn_limit_and_caps"]=true;
        bool rejected=false;try{capped.observe(at(J,q,zero),{0,0});}catch(const std::exception&){rejected=true;}require(rejected,"Unrequested observation accepted");checks["state_machine_rejects_unsolicited_observation"]=true;
        Controller guarded(cfg,J);q={.2,.3};cv::Mat s=at(J,q,zero);guarded.start(s,zero);for(int i=0;i<2;++i){guarded.next_action();guarded.observe(s,{0,0});}require(guarded.status()=="running"&&cv::norm(guarded.response()-J)==0,"Uninformative moves changed prior");checks["declared_prior_preserved"]=true;
        Controller changed(cfg,J);changed.start(s,zero);for(int i=0;i<2;++i){auto u=changed.next_action();s+=3*J*cv::Mat(*u);changed.observe(s,*u);}require(cv::norm(changed.response()-J)==0&&changed.snapshot().at("identification").at("reason").str()=="response_change_exceeds_guard","Large response change accepted");checks["large_response_change_rejected"]=true;
        auto amp=choose_probe_amplitudes(J,{.05,.01},{.25,.25},1);require(amp[0]>=.05&&amp[1]>=.01&&amp[0]<=.25&&amp[1]<=.25,"Adaptive probe exceeds bounds");checks["adaptive_probe_respects_caps"]=true;
        Controller original(cfg,J);original.start(at(J,{.3,-.2},zero),zero);original.next_action();auto restored=Controller::from_state(Json::parse(original.state().dump()));require(original.snapshot().dump()==restored.snapshot().dump()&&original.next_action()==restored.next_action(),"Session persistence differs");auto u=*restored.next_action();original.observe(at(J,{.4,-.1},zero),u);restored.observe(at(J,{.4,-.1},zero),u);require(cv::norm(original.response()-restored.response())<1e-12&&original.snapshot().dump()==restored.snapshot().dump(),"Session replay differs");checks["json_session_roundtrip"]=true;
        Json report=Json::dict();report["created_kst"]=now();report["scope"]="Native synthetic software checks, not physical robot trials";report["checks"]=checks;report["passed"]=true;write(out/"checks.json",report.dump());std::cout<<report.dump()<<'\n';return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
