#include "io.hpp"
#include <fstream>
#include <iostream>
using namespace semes;
int wmain(int argc,wchar_t** wide){
    Utf8Arguments args(argc,wide);char** argv=args.data();
    try{
        if(argc!=4)throw std::runtime_error("prepare_vendor final-root plan.json new-manifest-directory");auto root=fs::weakly_canonical(fs::u8path(argv[1]));auto plan=Json::parse(read(fs::u8path(argv[2])));fs::path out=fs::absolute(fs::u8path(argv[3]));if(fs::exists(out))throw std::runtime_error("New manifest directory required");fs::create_directories(out);std::ofstream files(out/"files.csv");files<<"source,destination,bytes,source_sha256,copy_sha256\n";int count=0;uintmax_t total=0;
        for(auto& item:plan.array){auto source=fs::weakly_canonical(fs::u8path(item.at("source").str()));fs::path relative=fs::u8path(item.at("destination").str());if(relative.is_absolute()||relative.empty()||*relative.begin()=="..")throw std::runtime_error("Vendor destination must be relative to final");auto destination=(root/relative).lexically_normal();auto within=destination.lexically_relative(root);if(within.empty()||*within.begin()=="..")throw std::runtime_error("Vendor destination escapes final");if(!fs::exists(source))throw std::runtime_error("Missing vendor source "+source.u8string());
            auto copy=[&](const fs::path& from,const fs::path& to){auto before=sha256(from);fs::create_directories(to.parent_path());if(fs::exists(to)){if(sha256(to)!=before)throw std::runtime_error("Existing vendor copy differs; refusing overwrite");}else fs::copy_file(from,to,fs::copy_options::none);auto after=sha256(to);if(before!=after)throw std::runtime_error("Vendor copy hash mismatch");fs::last_write_time(to,fs::last_write_time(from));++count;total+=fs::file_size(from);files<<csv(from.u8string())<<','<<csv(to.lexically_relative(root).generic_u8string())<<','<<fs::file_size(from)<<','<<before<<','<<after<<'\n';};
            if(fs::is_regular_file(source))copy(source,destination);else for(auto& e:fs::recursive_directory_iterator(source))if(e.is_regular_file())copy(e.path(),destination/e.path().lexically_relative(source));std::cout<<"Verified vendor "<<relative.generic_u8string()<<std::endl;
        }
        Json check=Json::dict();check["created_kst"]=now();check["files"]=count;check["bytes"]=double(total);check["passed"]=true;write(out/"checks.json",check.dump());std::cout<<check.dump()<<'\n';return 0;
    }catch(const std::exception& e){Json j=Json::dict();j["passed"]=false;j["error"]=e.what();std::cerr<<j.dump()<<'\n';return 1;}
}
