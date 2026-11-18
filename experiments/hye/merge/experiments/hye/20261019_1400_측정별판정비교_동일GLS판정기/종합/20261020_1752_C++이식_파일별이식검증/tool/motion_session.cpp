#include "motion.hpp"
#include <windows.h>
#include <iostream>
using namespace semes;
using namespace semes::motion;
static cv::Vec2d command(const Json& j){if(j.type!=Json::Array||j.size()!=2)throw std::runtime_error("Two acknowledged command values required");return {j.at(0).num(),j.at(1).num()};}
static void atomic_save(const fs::path& path,const Json& j){
    fs::create_directories(fs::absolute(path).parent_path());auto temporary=path;temporary+=L".pending";if(fs::exists(temporary))throw std::runtime_error("Pending session file exists; refusing to overwrite");write(temporary,j.dump());
    if(!MoveFileExW(temporary.c_str(),path.c_str(),MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH))throw std::runtime_error("Cannot replace motion session atomically");
}
int wmain(int argc,wchar_t** wide){
    Utf8Arguments args(argc,wide);char** argv=args.data();
    try{
        if(argc<3)throw std::runtime_error("motion_session start-features state.json input.json | observe-features state.json input.json | inspect state.json");
        std::string operation=argv[1];fs::path state=fs::u8path(argv[2]);
        Controller c;
        if(operation=="start-features"){
            if(argc!=4||fs::exists(state))throw std::runtime_error("A new state file and visual input JSON are required");auto input=Json::parse(read(fs::u8path(argv[3])));auto settings=input.has("settings")?Settings::from_json(input.at("settings")):Settings();cv::Mat prior=input.has("prior")&&input.at("prior").type!=Json::Null?matrix(input.at("prior")):cv::Mat();c=Controller(settings,prior);c.start(vector_from_json(input.at("observation")),vector_from_json(input.at("fixed_goal")));
        }else{
            c=Controller::from_state(Json::parse(read(state)));
            if(operation=="inspect"){if(argc!=3)throw std::runtime_error("inspect accepts only a state path");Json j=c.snapshot();j["events"]=c.events();std::cout<<j.dump()<<'\n';return 0;}
            if(operation!="observe-features"||argc!=4)throw std::runtime_error("Unknown motion session operation");auto input=Json::parse(read(fs::u8path(argv[3])));c.observe(vector_from_json(input.at("observation")),command(input.at("executed_relative")));
        }
        auto action=c.next_action();atomic_save(state,c.state());Json result=c.snapshot();Json move=Json::list();if(action){move.push((*action)[0]);move.push((*action)[1]);result["suggested_relative_command"]=move;}else result["suggested_relative_command"]=Json();result["notice"]="Feature API only. Image extraction is a separate native input stage; no hardware commands are sent.";std::cout<<result.dump()<<'\n';return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
