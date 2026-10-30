#include "observation_model.hpp"
#include "image_observer.hpp"
#include "workspace.hpp"
#include <chrono>
#include <iostream>
using namespace semes;
int wmain(int argc,wchar_t** wide){Utf8Arguments args(argc,wide);auto argv=args.data();try{
    if(argc!=4)throw std::runtime_error("predict_observation model.json current-image new-output");auto start=std::chrono::steady_clock::now();fs::path out=fs::u8path(argv[3]);if(fs::exists(out))throw std::runtime_error("New output required");auto root=workspace_root();auto bundle=Json::parse(read(fs::u8path(argv[1])));auto model=ObservationModel::from_json(bundle);ImageObserver extractor(root/"data/raw/hw/bmp/20261014/center.bmp",root/"data/reference/joint_center_630_20261022/assets/reference.npz",root/"data/reference/codex_experiments_20261014/joint_yt_visual_feedback_20261023/lines/protocol.json");auto observation=extractor.extract(bytes(fs::u8path(argv[2])),bundle.at("condition").str());if(!observation.at("success").number)throw std::runtime_error("Image measurement failed");auto q=model.predict(matrix(observation.at("X")));if(!cv::checkRange(q))throw std::runtime_error("Nonfinite prediction");Json result=Json::dict();result["condition"]=bundle.at("condition");result["Y"]=q.at<double>(0,0);result["T"]=q.at<double>(0,1);result["image_sha256"]=observation.at("image_sha256");result["seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();result["status"]="Development model; no independent new-photo validation or hardware command";write(out/"prediction.json",result.dump());std::cout<<result.dump()<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
