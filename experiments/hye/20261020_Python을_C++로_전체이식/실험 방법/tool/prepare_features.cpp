#include "direct_observer.hpp"
#include "npy_writer.hpp"
#include <iostream>
using namespace semes;
int wmain(int argc,wchar_t** wide){Utf8Arguments args(argc,wide);auto argv=args.data();try{
    if(argc!=5)throw std::runtime_error("prepare_features native-direct-directory native-line-directory image-count new-output");fs::path qr=fs::u8path(argv[1]),lines=fs::u8path(argv[2]),out=fs::u8path(argv[4]);int count=std::stoi(argv[3]);if(count<1||fs::exists(out))throw std::runtime_error("Positive count and new output required");
    std::map<std::string,std::vector<double>> values;std::map<std::string,std::vector<unsigned char>> success;
    for(int i=0;i<count;++i){auto filename=fs::u8path(std::to_string(i)+".json");auto q=Json::parse(read(qr/"images"/filename));auto line=Json::parse(read(lines/"images"/filename));auto grid=DirectObserver::qr_grid(q.at("candidates").at("homography_g5"));
        for(std::string condition:{"qr","upper","lower","both"}){auto& x=values[condition];for(int k=0;k<grid.cols;++k)x.push_back(grid.at<double>(k));bool good=q.at("candidates").at("homography_g5").at("success").number!=0;
            for(std::string part:{"upper","lower"})if(condition==part||condition=="both"){for(const auto& v:line.at("X_"+part).array)x.push_back(v.type==Json::Number?v.number:std::numeric_limits<double>::quiet_NaN());good=good&&line.at("success_"+part).number;}
            success[condition].push_back(good);
        }
    }
    std::map<std::string,Npy> arrays;Json check=Json::dict();check["images"]=count;check["successes"]=Json::dict();for(const auto& pair:values){arrays["X_"+pair.first]=numeric_npy(pair.second,{size_t(count),pair.second.size()/count});arrays["success_"+pair.first]=boolean_npy(success[pair.first],{size_t(count)});check["successes"][pair.first]=int(std::count(success[pair.first].begin(),success[pair.first].end(),1));}
    fs::create_directories(out);write_npz(out/"features.npz",arrays);check["scope"]="Four permitted observation conditions from newly computed native measurements; no targets or filenames enter feature computation";check["python_invocations"]=0;write(out/"checks.json",check.dump());std::cout<<check.dump()<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
