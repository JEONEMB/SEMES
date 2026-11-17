#include "image_observer.hpp"
#include "motion.hpp"
#include "workspace.hpp"
#include <windows.h>
#include <iostream>
using namespace semes;
static void save(const fs::path& path,const Json& j){fs::create_directories(fs::absolute(path).parent_path());auto pending=path;pending+=L".pending";if(fs::exists(pending))throw std::runtime_error("Pending session file exists");write(pending,j.dump());if(!MoveFileExW(pending.c_str(),path.c_str(),MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH))throw std::runtime_error("Atomic session save failed");}
int wmain(int argc,wchar_t** wide){Utf8Arguments args(argc,wide);auto argv=args.data();try{
    if(argc<3)throw std::runtime_error("image_session start state image calibration.json settings.json online-only|prior-guarded [adaptive] | observe state image dy dt | inspect state");
    std::string operation=argv[1];fs::path state_path=fs::u8path(argv[2]);Json state=Json::dict();
    if(operation=="inspect"){if(argc!=3)throw std::runtime_error("inspect accepts state only");state=Json::parse(read(state_path));auto controller=motion::Controller::from_state(state.at("controller"));auto j=controller.snapshot();j["events"]=controller.events();j["condition"]=state.at("condition");std::cout<<j.dump()<<'\n';return 0;}
    auto root=workspace_root();ImageObserver extractor(root/"data/raw/hw/bmp/20261014/center.bmp",root/"data/reference/joint_center_630_20261022/assets/reference.npz",root/"data/reference/codex_experiments_20261014/joint_yt_visual_feedback_20261023/lines/protocol.json");motion::Controller controller;
    if(operation=="start"){
        if((argc!=7&&argc!=8)||fs::exists(state_path))throw std::runtime_error("Choose a new state and supply image, calibration, settings, mode");
        auto cal=Json::parse(read(fs::u8path(argv[4])));auto cfg=motion::Settings::from_json(Json::parse(read(fs::u8path(argv[5]))));cfg.noise_rms=std::max(cal.at("noise_rms").num(),1.);auto prior=matrix(cal.at("jacobian"));std::string mode=argv[6];if(mode!="online-only"&&mode!="prior-guarded")throw std::runtime_error("Unknown mode");
        if(argc==8){if(std::string(argv[7])!="adaptive"||mode!="prior-guarded")throw std::runtime_error("Adaptive probes require the declared training prior");auto probes=motion::choose_probe_amplitudes(prior,{cfg.probe_y,cfg.probe_t},{cfg.max_step_y,cfg.max_step_t},cfg.noise_rms);cfg.probe_y=probes[0];cfg.probe_t=probes[1];}
        auto condition=cal.at("condition").str();auto observation=extractor.extract(bytes(fs::u8path(argv[3])),condition);if(!observation.at("success").number)throw std::runtime_error("Initial image measurement failed; session not started");
        auto projection=matrix(cal.at("projection"));cv::Mat raw=matrix(observation.at("X")),projected=raw*projection;cv::Mat visual=projected.t();controller=motion::Controller(cfg,mode=="prior-guarded"?prior:cv::Mat());controller.start(visual,cv::Mat::zeros(visual.size(),CV_64F));
        state["condition"]=condition;state["projection"]=jsonmatrix(projection);state["image_hashes"]=Json::list();state["image_hashes"].push(observation.at("image_sha256"));state["calibration_sha256"]=sha256(fs::u8path(argv[4]));state["scope"]="Development image feedback suggestions using a declared training projection; no physical pose verification or hardware dispatch";
    }else if(operation=="observe"){
        if(argc!=6)throw std::runtime_error("observe requires state, image, acknowledged dy and dt");state=Json::parse(read(state_path));controller=motion::Controller::from_state(state.at("controller"));auto raw=bytes(fs::u8path(argv[3]));cv::Mat visual;
        try{auto observation=extractor.extract(raw,state.at("condition").str());if(observation.at("success").number){cv::Mat projected=matrix(observation.at("X"))*matrix(state.at("projection"));visual=projected.t();}}
        catch(const std::exception& e){if(!state.has("measurement_errors"))state["measurement_errors"]=Json::list();Json error=Json::dict();error["turn"]=controller.turns()+1;error["error"]=e.what();state["measurement_errors"].push(error);}
        if(visual.empty()){auto snapshot=controller.state();visual=cv::Mat(int(state.at("projection").at(0).size()),1,CV_64F,cv::Scalar(std::numeric_limits<double>::quiet_NaN()));}
        auto parse=[](const char* s){std::string value=s;size_t end=0;double v=std::stod(value,&end);if(end!=value.size()||!std::isfinite(v))throw std::runtime_error("Finite acknowledged command required");return v;};controller.observe(visual,{parse(argv[4]),parse(argv[5])});state["image_hashes"].push(sha256(raw));
    }else throw std::runtime_error("Unknown image session operation");
    auto action=controller.next_action();state["controller"]=controller.state();save(state_path,state);auto result=controller.snapshot();result["condition"]=state.at("condition");result["suggested_relative_command"]=Json();if(action){Json a=Json::list();a.push((*action)[0]);a.push((*action)[1]);result["suggested_relative_command"]=a;}result["notice"]="Visual confirmation is not independent physical pose verification; suggestions only";std::cout<<result.dump()<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
