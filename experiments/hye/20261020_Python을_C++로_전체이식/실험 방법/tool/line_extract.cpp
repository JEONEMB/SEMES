#include "line_observer.hpp"
#include "table.hpp"
#include <iostream>
using namespace semes;
int wmain(int argc,wchar_t** wide){
    Utf8Arguments args(argc,wide);char** argv=args.data();
    try{
        if(argc!=5)throw std::runtime_error("line_extract fixed-center protocol-json image-or-manifest new-output");fs::path out=fs::u8path(argv[4]);if(fs::exists(out))throw std::runtime_error("New output required");LineObserver observer(fs::u8path(argv[1]),fs::u8path(argv[2]));fs::path input=fs::u8path(argv[3]);
        if(input.extension()==".csv"){
            CsvTable manifest(input);fs::create_directories(out);std::ofstream table(out/"measurements.csv"),failures(out/"failures.csv");table<<"relative,block,shift,valid,rmse,correlation,gain,offset,contrast\n";table<<std::setprecision(17);failures<<"relative,error\n";int failed=0;
            for(size_t i=0;i<manifest.rows.size();++i){std::string relative=manifest.text(i,"relative");auto file=fs::u8path(manifest.text(i,"path"));if(!file.is_absolute())file=input.parent_path()/file;try{auto j=observer.extract(bytes(file));for(size_t k=0;k<j.at("measured_shifts").size();++k){table<<csv(relative)<<','<<k<<','<<j.at("measured_shifts").at(k).dump()<<','<<j.at("valid_blocks").at(k).dump();for(auto key:{"rmse","correlation","gain","offset","contrast"})table<<','<<j.at(key).at(k).dump();table<<'\n';}write(out/"images"/fs::u8path(std::to_string(i)+".json"),j.dump());}catch(const std::exception& e){++failed;failures<<csv(relative)<<','<<csv(e.what())<<'\n';}if(i%40==0)std::cout<<i<<'/'<<manifest.rows.size()<<std::endl;}
            Json check=Json::dict();check["attempted"]=int(manifest.rows.size());check["failed"]=failed;check["native_compute"]=true;check["python_invocations"]=0;write(out/"checks.json",check.dump());return failed?2:0;
        }
        auto j=observer.extract(bytes(input));j["image_sha256"]=sha256(input);write(out/"observation.json",j.dump());std::cout<<j.dump()<<'\n';return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
