#include "io.hpp"
#include <iostream>
#include <fstream>
#include <regex>
#include <set>
#include <windows.h>
using namespace semes;
static std::string kst(const std::string& s){
    if(s.size()<19)return s;
    SYSTEMTIME st{
    };
    try{
        st.wYear=WORD(std::stoi(s.substr(0,4)));
        st.wMonth=WORD(std::stoi(s.substr(5,2)));
        st.wDay=WORD(std::stoi(s.substr(8,2)));
        st.wHour=WORD(std::stoi(s.substr(11,2)));
        st.wMinute=WORD(std::stoi(s.substr(14,2)));
        st.wSecond=WORD(std::stoi(s.substr(17,2)));
    }
    catch(...){
        return s;
    }
    FILETIME ft;
    if(!SystemTimeToFileTime(&st,&ft))return s;
    ULARGE_INTEGER n;
    n.LowPart=ft.dwLowDateTime;
    n.HighPart=ft.dwHighDateTime;
    n.QuadPart+=9ULL*3600*10000000;
    ft.dwLowDateTime=n.LowPart;
    ft.dwHighDateTime=n.HighPart;
    FileTimeToSystemTime(&ft,&st);
    char b[50];
    sprintf_s(b,"%04d-%02d-%02dT%02d:%02d:%02d+09:00",st.wYear,st.wMonth,st.wDay,st.wHour,st.wMinute,st.wSecond);
    return b;
}
int wmain(int argc,wchar_t** wide_argv){
    semes::Utf8Arguments unicode_arguments(argc,wide_argv);
    char** argv=unicode_arguments.data();
    try{
        if(argc<5)throw std::runtime_error("provenance workspace claude-project-log-dir codex-sessions-dir new-output");
        fs::path root=fs::weakly_canonical(fs::u8path(argv[1])),out=fs::u8path(argv[4]);
        if(fs::exists(out))throw std::runtime_error("New output required");
        fs::create_directories(out);
        std::ofstream events(out/"write_events.csv");
        events<<"file,actor,session_log,line,event_timestamp,event_kst,operation,evidence_status\n";
        std::map<std::string,Json> files;
        size_t total=0,parsed=0;
        auto add=[&](std::string filename,const fs::path& cwd,const std::string& actor,const fs::path& log,size_t line,const std::string& timestamp,const std::string& operation){
            std::replace(filename.begin(),filename.end(),'\\','/');
            fs::path p=fs::u8path(filename);
            if(p.is_relative())p=cwd/p;
            p=p.lexically_normal();
            auto rel=p.lexically_relative(root);
            if(rel.empty()||*rel.begin()==".."||*rel.begin()=="final"||*rel.begin()=="legacy")return;
            auto key=rel.generic_u8string();
            Json j=Json::dict();
            j["actor"]=actor;
            j["session_log"]=log.u8string();
            j["line"]=double(line);
            j["event_timestamp"]=timestamp;
            j["event_kst"]=kst(timestamp);
            j["operation"]=operation;
            j["evidence_status"]="observed_tool_invocation; current bytes and successful completion require separate verification";
            files[key].push(j);
            events<<csv(key)<<','<<csv(actor)<<','<<csv(log.u8string())<<','<<line<<','<<csv(timestamp)<<','<<csv(kst(timestamp))<<','<<csv(operation)<<",tool_invocation\n";
            ++total;
        };
        for(int source=2; source<=3; source++){
            fs::path dir=fs::u8path(argv[source]);
            if(!fs::exists(dir))continue;
            for(fs::recursive_directory_iterator it(dir,fs::directory_options::skip_permission_denied),end; it!=end; ++it){
                if(!it->is_regular_file()||it->path().extension()!=".jsonl")continue;
                std::ifstream input(it->path());
                std::string line;
                size_t n=0;
                fs::path cwd=root;
                while(std::getline(input,line)){
                    ++n;
                    try{
                        auto j=Json::parse(line);
                        ++parsed;
                        std::string stamp=j.has("timestamp")&&j.at("timestamp").type==Json::String?j.at("timestamp").str():"unknown";
                        if(j.has("cwd")&&j.at("cwd").type==Json::String)cwd=fs::u8path(j.at("cwd").str());
                        if(source==2){
                            if(!j.has("message"))continue;
                            auto& message=j.at("message");
                            if(!message.has("content")||message.at("content").type!=Json::Array)continue;
                            for(auto& block:message.at("content").array){
                                if(!block.has("type")||block.at("type").str()!="tool_use"||!block.has("input"))continue;
                                std::string name=block.at("name").str();
                                auto& a=block.at("input");
                                if((name=="Write"||name=="Edit"||name=="MultiEdit")&&a.has("file_path"))add(a.at("file_path").str(),cwd,"Claude",it->path(),n,stamp,name);
                            }
                        }
                        else {
                            if(!j.has("payload"))continue;
                            auto& p=j.at("payload");
                            if(p.has("cwd")&&p.at("cwd").type==Json::String)cwd=fs::u8path(p.at("cwd").str());
                            if(!p.has("type")||(p.at("type").str()!="function_call"&&p.at("type").str()!="custom_tool_call")||(!p.has("arguments")&&!p.has("input")))continue;
                            std::string name=p.at("name").str(),args=p.has("arguments")?p.at("arguments").str():p.at("input").str(),patch=args;
                            if(name.find("apply_patch")==std::string::npos){
                                size_t pos=args.find("tools.apply_patch(");
                                if(pos==std::string::npos)continue;
                                pos=args.find('"',pos);
                                if(pos==std::string::npos)continue;
                                Json::Parser parser{
                                    args
                                };
                                parser.p=pos;
                                patch=parser.text();
                            }
                            std::regex re("\\*\\*\\* (Add|Update|Delete) File: ([^\\r\\n]+)");
                            for(std::sregex_iterator m(patch.begin(),patch.end(),re),end; m!=end; ++m)add((*m)[2],cwd,"Codex",it->path(),n,stamp,(*m)[1]);
                        }
                    }
                    catch(const std::exception&){
                        /* Truncated final log lines and non-JSON records are not authorship evidence. */
                    }
                }
            }
        }
        Json map=Json::dict();
        for(auto& kv:files)map[kv.first]=kv.second;
        write(out/"file_write_evidence.json",map.dump());
        Json summary=Json::dict();
        summary["observed_file_paths"]=int(files.size());
        summary["write_invocations"]=double(total);
        summary["parsed_records"]=double(parsed);
        summary["created_kst"]=now();
        summary["coverage"]="Direct Claude Write/Edit and Codex apply_patch invocations only. Shell writes and missing/deleted session history are not inferred. Multiple actors are retained per path.";
        write(out/"summary.json",summary.dump());
        std::cout<<summary.dump()<<'\n';
        return 0;
    }
    catch(const std::exception& e){
        std::cerr<<e.what()<<'\n';
        return 1;
    }
}
