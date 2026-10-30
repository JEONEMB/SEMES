#include "io.hpp"
#include <windows.h>
#include <iostream>
#include <fstream>
#include <set>
using namespace semes;
int wmain(int argc,wchar_t** wide_argv){
    semes::Utf8Arguments unicode_arguments(argc,wide_argv);
    char** argv=unicode_arguments.data();
    try{
        if(argc!=4)throw std::runtime_error("archive workspace inventory-files.csv new-legacy-snapshot");
        auto root=fs::weakly_canonical(fs::u8path(argv[1]));
        auto policy_path=root/"final/config/migration_policy.json";
        if(!fs::exists(policy_path))throw std::runtime_error("Explicit migration policy required; no files were copied");
        auto policy=Json::parse(read(policy_path));
        if(!policy.has("legacy_additions_allowed")||!policy.at("legacy_additions_allowed").num()) throw std::runtime_error("Legacy additions are disabled by the user's latest migration policy");
        auto out=fs::absolute(fs::u8path(argv[3])).lexically_normal();
        auto relative=out.lexically_relative(root);
        if(relative.empty()||*relative.begin()!="legacy")throw std::runtime_error("Archive destination must be inside workspace/legacy");
        if(fs::exists(out))throw std::runtime_error("New snapshot directory required");
        fs::create_directories(out);
        std::ifstream input(fs::u8path(argv[2]));
        std::string line;
        std::getline(input,line);
        std::ofstream journal(out/"preservation.csv"),errors(out/"issues.csv");
        journal<<"original_path,archived_path,sha256,created_kst,modified_kst,archive_verified_kst,status\n";
        errors<<"path,reason\n";
        int copied=0,skipped=0,failed=0;
        while(std::getline(input,line)){
            auto cells=csvrow(line);
            if(cells.size()<10)continue;
            std::string category=cells[2];
            auto rel=fs::u8path(cells[0]);
            if(rel.is_absolute()||rel.empty()||*rel.begin()=="..")throw std::runtime_error("Unsafe manifest path");
            if(category=="dependency_or_generated"||category=="data_keep_in_place"||cells[0]=="ALGORITHM_RULES.md"||cells[0]=="AGENTS.md"){
                ++skipped;
                continue;
            }
            auto source=root/rel,destination=out/"originals"/rel;
            try{
                if(!fs::is_regular_file(source)||fs::is_symlink(source))throw std::runtime_error("Source is missing or symbolic");
                if(timestamp(source,false,false)!=cells[6]||sha256(source)!=cells[8])throw std::runtime_error("Source changed since inventory; kept in place for a separate snapshot");
                fs::create_directories(destination.parent_path());
                if(!fs::copy_file(source,destination,fs::copy_options::none))throw std::runtime_error("Copy failed");
                auto hash=sha256(destination);
                if(hash!=cells[8])throw std::runtime_error("Archive hash mismatch");
                HANDLE src=CreateFileW(source.c_str(),FILE_READ_ATTRIBUTES,FILE_SHARE_READ|FILE_SHARE_WRITE|FILE_SHARE_DELETE,nullptr,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,nullptr),dst=CreateFileW(destination.c_str(),FILE_WRITE_ATTRIBUTES,FILE_SHARE_READ,nullptr,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,nullptr);
                FILETIME created,access,modified;
                if(src!=INVALID_HANDLE_VALUE&&dst!=INVALID_HANDLE_VALUE&&GetFileTime(src,&created,&access,&modified))SetFileTime(dst,&created,&access,&modified);
                if(src!=INVALID_HANDLE_VALUE)CloseHandle(src);
                if(dst!=INVALID_HANDLE_VALUE)CloseHandle(dst);
                journal<<csv(cells[0])<<','<<csv((fs::path("originals")/rel).generic_u8string())<<','<<csv(hash)<<','<<csv(cells[5])<<','<<csv(cells[7])<<','<<csv(now())<<",verified_copy_source_retained\n";
                ++copied;
                if(copied%200==0)std::cout<<copied<<" preserved"<<std::endl;
            }
            catch(const std::exception& e){
                ++failed;
                errors<<csv(cells[0])<<','<<csv(e.what())<<'\n';
            }
        }
        Json summary=Json::dict();
        summary["created_kst"]=now();
        summary["verified_copies"]=copied;
        summary["data_generated_control_files_kept_in_place"]=skipped;
        summary["issues"]=failed;
        summary["originals_removed"]=0;
        summary["purpose"]="Verified preservation snapshot before individual C++ ports. This is not a declaration that all sources have been converted.";
        write(out/"summary.json",summary.dump());
        std::cout<<summary.dump()<<'\n';
        return failed?2:0;
    }
    catch(const std::exception& e){
        std::cerr<<e.what()<<'\n';
        return 1;
    }
}
