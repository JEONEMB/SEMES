#include "io.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <set>
using namespace semes;

static bool source(const fs::path& p){
    static const std::set<std::string> extensions={".py",".pyw",".cpp",".c",".hpp",".h",".js",".mjs",".ps1",".bat",".cmd",".cmake"};
    return extensions.count(p.extension().u8string())||p.filename()=="CMakeLists.txt";
}
int wmain(int argc,wchar_t** wide){
    Utf8Arguments args(argc,wide);char** argv=args.data();
    try{
        if(argc!=4)throw std::runtime_error("prepare_workspace original-root final-root new-manifest-directory");
        const auto original=fs::weakly_canonical(fs::u8path(argv[1]));
        const auto target=fs::weakly_canonical(fs::u8path(argv[2]));
        const auto manifest=fs::absolute(fs::u8path(argv[3]));
        if(!fs::is_directory(original)||target.parent_path()!=original||target.filename()!="final")throw std::runtime_error("Expected original-root/final workspace layout");
        if(fs::exists(manifest))throw std::runtime_error("Manifest directory must be new");
        fs::create_directories(manifest);
        fs::create_directories(target/"data/raw/hw");
        fs::create_directories(target/"data/reference");
        fs::create_directories(target/"resources/web");
        fs::create_directories(target/"docs/migration");
        std::ofstream copies(manifest/"copied_files.csv"),sources(manifest/"source_inventory.csv"),errors(manifest/"errors.csv");
        copies<<"original,copy,bytes,original_sha256,copy_sha256,status\n";
        sources<<"original,bytes,created_kst,modified_kst,sha256,language\n";
        errors<<"path,error\n";
        const std::set<std::string> skipped={".git","__pycache__",".uv-cache","deps","node_modules","build",".review_ppt",".pytest_cache",".venv","venv"};
        size_t copied=0,reused=0,codes=0,failed=0;uintmax_t total=0;
        std::vector<fs::path> files;
        for(fs::recursive_directory_iterator it(original,fs::directory_options::skip_permission_denied),end;it!=end;++it){
            auto relative=it->path().lexically_relative(original);
            if(it->is_directory()){
                auto first=*relative.begin();auto name=relative.filename().u8string();
                if(first=="final"||first=="legacy"||skipped.count(name)||name.rfind(".chart-data-",0)==0)it.disable_recursion_pending();
            }else if(it->is_regular_file())files.push_back(relative);
        }
        std::sort(files.begin(),files.end());
        for(const auto& relative:files){
            try{
                auto input=original/relative;
                if(relative.filename().u8string().rfind("~$",0)==0)continue;
                auto before=timestamp(input,false,false);auto hash=sha256(input);
                if(source(input)){
                    ++codes;sources<<csv(relative.generic_u8string())<<','<<fs::file_size(input)<<','<<csv(timestamp(input,true,true))<<','<<csv(timestamp(input,false,true))<<','<<hash<<','<<csv(input.extension().u8string())<<'\n';
                    // Frontend JS is a browser resource; Python is never copied as executable runtime.
                    if(input.extension()!=".js"&&input.extension()!=".mjs")continue;
                }
                fs::path destination;
                auto generic=relative.generic_u8string();
                const std::string raw="hw-analysis/data/hw/";
                if(generic.rfind(raw,0)==0)destination=target/"data/raw/hw"/fs::u8path(generic.substr(raw.size()));
                else destination=target/"data/reference"/relative;
                fs::create_directories(destination.parent_path());
                bool existing=fs::exists(destination);
                if(existing){if(sha256(destination)!=hash)throw std::runtime_error("Existing copy differs; refusing overwrite");++reused;}
                else {fs::copy_file(input,destination,fs::copy_options::none);++copied;}
                auto after=sha256(destination);
                if(hash!=after||timestamp(input,false,false)!=before)throw std::runtime_error("Copy hash mismatch or concurrent original change");
                total+=fs::file_size(input);
                copies<<csv(generic)<<','<<csv(destination.lexically_relative(target).generic_u8string())<<','<<fs::file_size(input)<<','<<hash<<','<<after<<",verified\n";
                if((copied+reused)%250==0){copies.flush();std::cout<<"Verified "<<copied+reused<<" inputs, "<<codes<<" sources indexed"<<std::endl;}
            }catch(const std::exception& e){++failed;errors<<csv(relative.generic_u8string())<<','<<csv(e.what())<<'\n';}
        }
        Json report=Json::dict();report["created_kst"]=now();report["copied"]=int(copied);report["reused"]=int(reused);report["sources_indexed"]=int(codes);report["bytes"]=double(total);report["errors"]=int(failed);report["originals_moved"]=0;report["passed"]=failed==0;
        write(manifest/"checks.json",report.dump());std::cout<<report.dump()<<std::endl;
        return failed?2:0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
