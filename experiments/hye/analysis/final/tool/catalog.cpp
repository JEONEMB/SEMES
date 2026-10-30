#include "io.hpp"
#include <iostream>
#include <fstream>
#include <set>
#include <algorithm>
#include <regex>
using namespace semes;
static bool component(const fs::path& p,const std::set<std::string>& names){
    for(auto& part:p)if(names.count(part.u8string()))return true;
    return false;
}
static bool source(const fs::path& p){
    return std::set<std::string>{
        ".py",".cpp",".hpp",".h",".js",".ps1"
    }
    .count(p.extension().string())!=0;
}
static std::string group(const fs::path& r){
    std::vector<std::string> parts;
    for(auto& v:r)parts.push_back(v.u8string());
    if(parts.size()==1)return "root_artifacts";
    if(parts[0]=="codex_experiments_20261014"&&parts.size()>2)return parts[0]+"/"+parts[1];
    if(parts[0]=="hw-analysis"&&parts.size()>3&&parts[1]=="tool")return "hw-analysis/tool/"+parts[2]+"/"+parts[3];
    return parts[0];
}
int wmain(int argc,wchar_t** wide_argv){
    semes::Utf8Arguments unicode_arguments(argc,wide_argv);
    char** argv=unicode_arguments.data();
    try{
        if(argc<3)throw std::runtime_error("catalog <workspace> <new-output-directory>");
        fs::path root=fs::absolute(fs::u8path(argv[1])),out=fs::absolute(fs::u8path(argv[2]));
        if(fs::exists(out))throw std::runtime_error("Catalog output must be new; previous inventory is preserved");
        fs::create_directories(out);
        std::ofstream manifest(out/"files.csv",std::ios::binary),timeline(out/"source_timeline.csv",std::ios::binary);
        manifest<<"\xef\xbb\xbfpath,experiment,category,bytes,created_utc,created_kst,modified_utc,modified_kst,sha256,stable_during_scan,author,author_evidence\n";
        timeline<<"\xef\xbb\xbfmodified_kst,created_kst,experiment,path,sha256,author,author_evidence\n";
        std::map<std::string,Json> groups;
        size_t files=0,codes=0,errors=0;
        std::ofstream err(out/"errors.txt");
        const std::set<std::string> generated={
            "__pycache__",".uv-cache","deps","node_modules","build",".git",".review_ppt"
        };
        for(fs::recursive_directory_iterator it(root,fs::directory_options::skip_permission_denied),end; it!=end; ++it){
            auto r=it->path().lexically_relative(root);
            auto first=*r.begin();
            if(it->is_directory()&&(first=="final"||first=="legacy"||r.filename()==".git")){
                it.disable_recursion_pending();
                continue;
            }
            if(!it->is_regular_file())continue;
            try{
                auto p=it->path();
                bool dep=component(r,generated);
                std::string cat=dep?"dependency_or_generated":component(r,{
                    "data","DATA","files"
                })?"data_keep_in_place":source(p)?"project_source":"result_or_document";
                std::string g=group(r),hash,mtime=timestamp(p,false,false),ctime=timestamp(p,true,false),mk=timestamp(p,false,true),ck=timestamp(p,true,true),author="unknown",evidence="no verified authorship; timestamp is not an author";
                try {
                    hash=dep?"not_hashed_generated":sha256(p);
                }
                catch(const std::exception& e) {
                    hash="unavailable_locked_or_unreadable";
                    ++errors;
                    err<<r.generic_u8string()<<": "<<e.what()<<'\n';
                }
                bool stable=hash!="unavailable_locked_or_unreadable"&&mtime==timestamp(p,false,false);
                Json item=Json::dict();
                item["path"]=r.generic_u8string();
                item["bytes"]=double(it->file_size());
                item["created_kst"]=ck;
                item["modified_kst"]=mk;
                item["sha256"]=hash;
                item["category"]=cat;
                item["stable_during_scan"]=stable;
                if(cat=="project_source"){
                    ++codes;
                    std::string text=read(p);
                    std::string hints;
                    for(auto word:{
                        "Claude","claude","Codex","codex","Anthropic"
                    }){
                        auto pos=text.find(word);
                        if(pos!=std::string::npos){
                            if(!hints.empty())hints+=" | ";
                            hints+=text.substr(pos>60?pos-60:0,180);
                        }
                    }
                    item["author_hints_unverified"]=hints;
                    item["migration_status"]="pending_individual_port_and_equivalence_check";
                    std::istringstream input(text);
                    std::string line;
                    Json symbols=Json::list(),imports=Json::list();
                    int n=0;
                    while(std::getline(input,line)){
                        ++n;
                        if(line.rfind("def ",0)==0||line.rfind("class ",0)==0)symbols.push(std::to_string(n)+": "+line);
                        if(line.rfind("import ",0)==0||line.rfind("from ",0)==0||line.find("sys.path")!=std::string::npos)imports.push(std::to_string(n)+": "+line);
                    }
                    item["symbols"]=symbols;
                    item["imports"]=imports;
                    item["opening_evidence"]=text.substr(0,1800);
                    timeline<<csv(mk)<<','<<csv(ck)<<','<<csv(g)<<','<<csv(r.generic_u8string())<<','<<csv(hash)<<','<<csv(author)<<','<<csv(evidence)<<'\n';
                }
                if(!dep){
                    if(!groups.count(g)){
                        groups[g]=Json::dict();
                        groups[g]["files"]=Json::list();
                        groups[g]["sources"]=Json::list();
                    }
                    groups[g]["files"].push(item);
                    if(cat=="project_source")groups[g]["sources"].push(item);
                }
                manifest<<csv(r.generic_u8string())<<','<<csv(g)<<','<<csv(cat)<<','<<it->file_size()<<','<<csv(ctime)<<','<<csv(ck)<<','<<csv(mtime)<<','<<csv(mk)<<','<<csv(hash)<<','<<(stable?"true":"false")<<','<<csv(author)<<','<<csv(evidence)<<'\n';
                ++files;
                if(files%1000==0)std::cout<<files<<" catalogued\n";
            }
            catch(const std::exception& e){
                ++errors;
                err<<r.generic_u8string()<<": "<<e.what()<<'\n';
            }
        }
        Json index=Json::list();
        for(auto& kv:groups){
            auto& g=kv.second;
            g["experiment"]=kv.first;
            g["author"]="unknown";
            g["time_basis"]="NTFS creation/last-write observation; neither proves experiment execution time or authorship";
            g["migration_status"]="pending_source_review";
            g["rules"]="../../../ALGORITHM_RULES.md";
            fs::path gp=out/"experiments"/fs::u8path(kv.first);
            write(gp/"inventory.json",g.dump());
            Json entry=Json::dict();
            entry["experiment"]=kv.first;
            entry["files"]=int(g.at("files").size());
            entry["sources"]=int(g.at("sources").size());
            index.push(entry);
        }
        Json summary=Json::dict();
        summary["captured_kst"]=now();
        summary["root"]=root.u8string();
        summary["files"]=int(files);
        summary["project_source_files"]=int(codes);
        summary["errors"]=int(errors);
        summary["groups"]=index;
        write(out/"index.json",summary.dump());
        std::cout<<summary.dump()<<'\n';
        return errors?2:0;
    }
    catch(const std::exception&e){
        std::cerr<<e.what()<<'\n';
        return 1;
    }
}
