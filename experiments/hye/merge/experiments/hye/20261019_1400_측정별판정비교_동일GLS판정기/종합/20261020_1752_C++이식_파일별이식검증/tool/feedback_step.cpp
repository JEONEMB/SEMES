#include "feedback.hpp"
#include <iostream>
using namespace semes;
int wmain(int argc,wchar_t** wide){Utf8Arguments args(argc,wide);auto argv=args.data();try{
    if(argc!=3)throw std::runtime_error("feedback_step input.json new-output");auto input=Json::parse(read(fs::u8path(argv[1])));fs::path out=fs::u8path(argv[2]);if(fs::exists(out))throw std::runtime_error("New output required");historical::FeedbackController c;
    if(input.has("state")){c=historical::FeedbackController::from_state(input.at("state"));const auto& command=input.at("executed");if(command.size()!=2)throw std::runtime_error("Two acknowledged increments required");c.observe(motion::vector_from_json(input.at("observation")),{command.at(0).num(),command.at(1).num()});}
    else{c=historical::FeedbackController(matrix(input.at("jacobian")),input.at("policy").str());c.start(motion::vector_from_json(input.at("observation")),motion::vector_from_json(input.at("goal")));}
    auto action=c.next_action();auto result=c.state();result["action"]=Json();if(action){Json a=Json::list();a.push((*action)[0]);a.push((*action)[1]);result["action"]=a;}result["hardware_command_sent"]=false;write(out/"state.json",result.dump());std::cout<<result.dump()<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
