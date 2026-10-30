#include "direct_observer.hpp"
#include "table.hpp"
#include <iostream>
using namespace semes;
int wmain(int argc,wchar_t** wide){
    Utf8Arguments args(argc,wide);char** argv=args.data();try{if(argc!=5)throw std::runtime_error("direct_extract fixed-center geometry-reference.npz image-or-manifest.csv new-output");fs::path out=fs::u8path(argv[4]);if(fs::exists(out))throw std::runtime_error("New output required");DirectObserver observer(fs::u8path(argv[1]),fs::u8path(argv[2]));fs::path input=fs::u8path(argv[3]);
    if(input.extension()==".csv"){
        CsvTable manifest(input);fs::create_directories(out);std::ofstream failures(out/"failures.csv");failures<<"relative,error\n";int failed=0;
        for(size_t i=0;i<manifest.rows.size();++i){try{auto path=fs::u8path(manifest.text(i,"path"));if(!path.is_absolute())path=input.parent_path()/path;auto j=observer.extract(bytes(path));j["relative"]=manifest.text(i,"relative");write(out/"images"/fs::u8path(std::to_string(i)+".json"),j.dump());}catch(const std::exception& e){++failed;failures<<csv(manifest.text(i,"relative"))<<','<<csv(e.what())<<'\n';}if(i%40==0)std::cout<<i<<'/'<<manifest.rows.size()<<std::endl;}
        Json check=Json::dict();check["attempted"]=int(manifest.rows.size());check["failed"]=failed;check["native_compute"]=true;check["python_invocations"]=0;write(out/"checks.json",check.dump());return failed?2:0;
    }
    auto j=observer.extract(bytes(input));write(out/"observation.json",j.dump());std::cout<<"Native direct QR observation saved; homography_g5 success="<<j.at("candidates").at("homography_g5").at("success").dump()<<'\n';return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
