#include "image_observer.hpp"
#include "workspace.hpp"
#include <iostream>
using namespace semes;
int wmain(int argc,wchar_t** wide){
    Utf8Arguments args(argc,wide);auto argv=args.data();
    try{
        if(argc!=4)throw std::runtime_error("image_extract qr|upper|lower|both current-image new-output");
        auto root=workspace_root();fs::path out=fs::u8path(argv[3]);if(fs::exists(out))throw std::runtime_error("New output required");
        ImageObserver observer(root/"data/raw/hw/bmp/20261014/center.bmp",root/"data/reference/joint_center_630_20261022/assets/reference.npz",root/"data/reference/codex_experiments_20261014/joint_yt_visual_feedback_20261023/lines/protocol.json");
        auto j=observer.extract(bytes(fs::u8path(argv[2])),argv[1]);write(out/"observation.json",j.dump());std::cout<<j.dump()<<'\n';return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
