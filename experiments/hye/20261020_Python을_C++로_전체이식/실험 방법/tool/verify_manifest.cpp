#include "table.hpp"
#include <iostream>
using namespace semes;
int wmain(int argc,wchar_t** wide_argv){
    Utf8Arguments arguments(argc,wide_argv);
    char** argv=arguments.data();
    try{
        if(argc!=5)throw std::runtime_error("verify_manifest workspace baseline-files.csv port-progress.json new-output");
        fs::path root=fs::weakly_canonical(fs::u8path(argv[1])),out=fs::u8path(argv[4]);
        if(fs::exists(out))throw std::runtime_error("New output required");
        CsvTable manifest(fs::u8path(argv[2]));
        auto progress=Json::parse(read(fs::u8path(argv[3])));
        fs::create_directories(out);
        std::ofstream report(out/"files.csv");
        report<<"path,category,baseline_sha256,current_sha256,status,error\n";
        int data_count=0,source_count=0,changed=0,errors=0;
        for(size_t i=0; i<manifest.rows.size(); ++i){
            auto name=manifest.text(i,"path"),category=manifest.text(i,"category");
            bool data=category=="data_keep_in_place",source=progress.has(name)&&manifest.text(i,"created_kst")<"2026-10-14";
            if(!data&&!source)continue;
            data_count+=data;
            source_count+=source;
            std::string current,status="unchanged",error;
            try{
                fs::path relative=fs::u8path(name);
                if(relative.is_absolute()||relative.empty()||*relative.begin()=="..")throw std::runtime_error("Unsafe manifest path");
                auto path=root/relative;
                auto before=timestamp(path,false,false);
                current=sha256(path);
                if(timestamp(path,false,false)!=before)throw std::runtime_error("File changed during verification");
                if(current!=manifest.text(i,"sha256")){
                    ++changed;
                    status="changed";
                }
            }
            catch(const std::exception& e){
                ++errors;
                status="unverified";
                error=e.what();
            }
            report<<csv(name)<<','<<category<<','<<manifest.text(i,"sha256")<<','<<current<<','<<status<<','<<csv(error)<<'\n';
        }
        Json audit=Json::dict();
        audit["created_kst"]=now();
        audit["baseline_manifest_sha256"]=sha256(fs::u8path(argv[2]));
        audit["data_files_compared"]=data_count;
        audit["ported_old_sources_compared"]=source_count;
        audit["content_changes"]=changed;
        audit["unverified"]=errors;
        audit["passed"]=changed==0&&errors==0;
        audit["scope"]="Read-only hashes of existing DATA plus progress-listed originals created before 2026-10-14. Actively changing recent experiments are deferred. No original moves or legacy writes.";
        write(out/"checks.json",audit.dump());
        std::cout<<audit.dump()<<'\n';
        return changed||errors?2:0;
    }
    catch(const std::exception& e){
        std::cerr<<e.what()<<'\n';
        return 1;
    }
}
