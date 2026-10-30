#include "io.hpp"
#include "workspace.hpp"
#include <windows.h>
#include <algorithm>
#include <iostream>
#include <set>
using namespace semes;
static fs::path workspace(){
    return workspace_root();
}
static std::wstring quoted(const std::wstring& s){
    std::wstring result=L"\"";size_t slashes=0;
    for(wchar_t c:s){if(c==L'\\'){++slashes;continue;}if(c==L'\"'){result.append(slashes*2+1,L'\\');result+=c;}else {result.append(slashes,L'\\');result+=c;}slashes=0;}
    result.append(slashes*2,L'\\');result+=L'\"';return result;
}
static fs::path executable(const fs::path& root,const std::string& name){
    wchar_t buffer[32768];auto n=GetModuleFileNameW(nullptr,buffer,32768);if(!n||n==32768)throw std::runtime_error("Cannot locate executable");auto launcher=fs::path(std::wstring(buffer,n)).parent_path();auto local=launcher/fs::u8path(name+".exe");if(fs::is_regular_file(local))return local;auto build=launcher.filename()=="Release"?launcher.parent_path():root/"build";
    std::vector<fs::path> choices;for(fs::recursive_directory_iterator it(build),end;it!=end;++it){if(it->is_directory()&&(it->path().filename()=="dependencies"||fs::is_regular_file(it->path()/"CMakeCache.txt"))){it.disable_recursion_pending();continue;}if(it->is_regular_file()&&it->path().filename()==fs::u8path(name+".exe")&&it->path().parent_path().filename()=="Release")choices.push_back(it->path());}
    if(choices.size()!=1)throw std::runtime_error("Build target first: "+name);return choices.front();
}
static int launch(const fs::path& binary,const std::vector<fs::path>& arguments,const fs::path& cwd,const fs::path& log){
    std::wstring command=quoted(binary.wstring());for(auto& arg:arguments)command+=L" "+quoted(arg.wstring());
    SECURITY_ATTRIBUTES sa{sizeof(sa),nullptr,TRUE};HANDLE file=CreateFileW(log.c_str(),GENERIC_WRITE,FILE_SHARE_READ,&sa,CREATE_NEW,FILE_ATTRIBUTE_NORMAL,nullptr);if(file==INVALID_HANDLE_VALUE)throw std::runtime_error("Cannot create new run log");
    STARTUPINFOW startup{};startup.cb=sizeof(startup);startup.dwFlags=STARTF_USESTDHANDLES;startup.hStdOutput=file;startup.hStdError=file;startup.hStdInput=GetStdHandle(STD_INPUT_HANDLE);PROCESS_INFORMATION process{};
    BOOL created=CreateProcessW(binary.c_str(),command.data(),nullptr,nullptr,TRUE,CREATE_NO_WINDOW,nullptr,cwd.c_str(),&startup,&process);CloseHandle(file);if(!created)throw std::runtime_error("Cannot execute native target");
    WaitForSingleObject(process.hProcess,INFINITE);DWORD code=1;GetExitCodeProcess(process.hProcess,&code);CloseHandle(process.hThread);CloseHandle(process.hProcess);return int(code);
}
static std::string replace(std::string text,const std::string& from,const std::string& to){for(size_t pos=0;(pos=text.find(from,pos))!=std::string::npos;pos+=to.size())text.replace(pos,from.size(),to);return text;}
int wmain(int argc,wchar_t** wide){
    Utf8Arguments utf(argc,wide);char** argv=utf.data();
    try{
        auto root=workspace();auto programs=Json::parse(read(root/"config/native_programs.json"));
        if(argc==1||std::string(argv[1])=="list"){
            for(auto& p:programs.array)std::cout<<p.at("id").str()<<"\t"<<p.at("description").str()<<'\n';return 0;
        }
        if(std::string(argv[1])!="run"||argc!=4)throw std::runtime_error("semes list | semes run program-id new-output-directory");
        std::string requested=argv[2];fs::path output=fs::absolute(fs::u8path(argv[3]));if(fs::exists(output))throw std::runtime_error("Output must be new");
        const Json* selected=nullptr;for(auto& p:programs.array)if(p.at("id").str()==requested)selected=&p;if(!selected)throw std::runtime_error("Unknown native program: "+requested);
        std::set<std::string> active;Json executions=Json::list();
        auto execute=[&](auto&& self,const Json& p,const fs::path& folder)->int{
            auto id=p.at("id").str();if(!active.insert(id).second)throw std::runtime_error("Cyclic run dependency");fs::create_directories(folder);std::map<std::string,fs::path> dependencies;
            if(p.has("dependencies"))for(auto& dep:p.at("dependencies").array){std::string key=dep.str();const Json* definition=nullptr;for(auto& q:programs.array)if(q.at("id").str()==key)definition=&q;if(!definition)throw std::runtime_error("Unknown native dependency");auto target=folder/"dependencies"/fs::u8path(key);int exit=self(self,*definition,target);if(exit)return exit;dependencies[key]=target/"result";}
            std::vector<fs::path> args;Json command=Json::list();fs::path binary;
            if(p.has("runtime")){auto runtime=p.at("runtime").str();if(runtime!="opencv5"&&runtime!="opencv5-ipp")throw std::runtime_error("Unknown native runtime: "+runtime);binary=root/(runtime=="opencv5"?"build/modern/bin/Release":"build/modern-ipp/bin/Release")/fs::u8path(p.at("binary").str()+".exe");}else binary=executable(root,p.at("binary").str());
            if(!fs::is_regular_file(binary))throw std::runtime_error("Build the requested native runtime first: "+binary.u8string());command.push(binary.u8string());
            for(auto& a:p.at("arguments").array){auto value=a.str();value=replace(value,"{root}",root.generic_u8string());value=replace(value,"{reference}",(root/"data/reference").generic_u8string());value=replace(value,"{raw}",(root/"data/raw/hw/bmp").generic_u8string());value=replace(value,"{out}",(folder/"result").generic_u8string());for(auto& d:dependencies)value=replace(value,"{dependency:"+d.first+"}",d.second.generic_u8string());args.push_back(fs::u8path(value));command.push(value);}
            std::cout<<"Running "<<id<<std::endl;int code=launch(binary,args,root,folder/"console.log");Json record=Json::dict();record["id"]=id;record["command"]=command;record["exit_code"]=code;record["created_kst"]=now();record["scope"]=p.at("scope");record["source"]=p.at("source");record["python_invocations"]=0;write(folder/"run.json",record.dump());executions.push(record);active.erase(id);return code;
        };
        int exit=execute(execute,*selected,output);Json report=Json::dict();report["created_kst"]=now();report["executions"]=executions;report["exit_code"]=exit;report["python_invocations"]=0;write(output/"run_manifest.json",report.dump());return exit;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
