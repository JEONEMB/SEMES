#include "table.hpp"
#include <regex>
#include <set>
#include <algorithm>
#include <iostream>
#include <tuple>
using namespace semes;
namespace {
    struct File {
        std::string path,group,category,created,modified,hash;
        double size=0;
    };
    struct Relation {
        std::string source,target,kind,evidence;
    };
    std::string normalize(std::string path){
        std::replace(path.begin(),path.end(),'\\','/');
        while(path.rfind("./",0)==0)path=path.substr(2);
        return path;
    }
    std::string safe(std::string value){
        for(char& c:value)if(c=='\r'||c=='\n'||c=='|')c=' ';
        return value;
    }
    std::string link(const fs::path& root,const std::string& path,const std::string& label=""){
        return "["+safe(label.empty()?path:label)+"](<"+(root/fs::u8path(path)).generic_u8string()+">)";
    }
    bool starts(const std::string& text,const std::string& prefix){
        return text.rfind(prefix,0)==0;
    }
}
int wmain(int argc,wchar_t** wide_argv){
    Utf8Arguments args(argc,wide_argv);
    char** argv=args.data();
    try {
        if(argc!=7&&argc!=8)throw std::runtime_error("file_list workspace inventory-dir previous-ledger.json port-progress.json manual-relations.json new-output-dir [authorship.json]");
        fs::path root=fs::weakly_canonical(fs::u8path(argv[1])),inventory=fs::u8path(argv[2]),out=fs::absolute(fs::u8path(argv[6])),final=root/"final";
        if(fs::exists(out))throw std::runtime_error("New output directory required");
        CsvTable manifest(inventory/"files.csv");
        auto snapshot=Json::parse(read(inventory/"index.json")),ledger=Json::parse(read(fs::u8path(argv[3]))),progress=Json::parse(read(fs::u8path(argv[4]))),manual=Json::parse(read(fs::u8path(argv[5])));
        Json authors=argc==8?Json::parse(read(fs::u8path(argv[7]))):Json::dict();
        std::map<std::string,File> files;
        std::map<std::string,std::vector<std::string>> basenames;
        std::vector<std::string> sources;
        for(size_t i=0; i<manifest.rows.size(); ++i){
            File f;
            f.path=manifest.text(i,"path");
            f.group=manifest.text(i,"experiment");
            f.category=manifest.text(i,"category");
            f.created=manifest.text(i,"created_kst");
            f.modified=manifest.text(i,"modified_kst");
            f.hash=manifest.text(i,"sha256");
            f.size=manifest.number(i,"bytes");
            files.emplace(f.path,f);
            basenames[fs::u8path(f.path).filename().u8string()].push_back(f.path);
            if(f.category=="project_source")sources.push_back(f.path);
        }
        std::sort(sources.begin(),sources.end(),[&](const auto& a,const auto& b){
            return std::tie(files.at(a).created,a)<std::tie(files.at(b).created,b);
        });
        std::map<std::string,Json> previous;
        for(auto& item:ledger.array)previous[item.at("path").str()]=item;
        std::vector<Relation> relations;
        std::set<std::tuple<std::string,std::string,std::string>> seen;
        std::set<std::string> changed_since_snapshot;
        auto add=[&](const std::string& source,const std::string& target,const std::string& kind,const std::string& evidence){
            if(source!=target&&seen.emplace(source,target,kind).second)relations.push_back({
                source,target,kind,evidence
            });
        };
        auto resolve=[&](const std::string& source,std::string literal,const std::string& kind,const std::string& evidence){
            literal=normalize(literal);
            auto local=normalize((fs::u8path(source).parent_path()/fs::u8path(literal)).lexically_normal().generic_u8string());
            if(files.count(local)){
                add(source,local,kind,evidence);
                return;
            }
            if(files.count(literal)){
                add(source,literal,kind,evidence);
                return;
            }
            auto name=fs::u8path(literal).filename().u8string();
            if(basenames.count(name)&&basenames.at(name).size()==1){
                add(source,basenames.at(name)[0],"candidate_"+kind,evidence);
                return;
            }
            if(!name.empty())add(source,literal,"unresolved_"+kind,evidence);
        };
        std::regex literal(R"rx(["']([^"'\r\n]{1,240}\.(?:py|cpp|hpp|csv|npz|npy|json|pkl|png|jpg|jpeg|bmp|gif|mp4|html|md|xlsx|pptx|svg|txt|yml|yaml))["'])rx",std::regex::icase);
        std::regex module(R"rx((?:^|[;\s])from\s+([A-Za-z_][A-Za-z0-9_\.]*)\s+import|(?:^|[;\s])import\s+([A-Za-z_][A-Za-z0-9_\.]*))rx");
        for(auto& source:sources){
            auto& record=files.at(source);
            if(record.hash=="unavailable_locked_or_unreadable")continue;
            auto path=root/fs::u8path(source);
            std::string text=read(path);
            if(sha256(path)!=record.hash){
                changed_since_snapshot.insert(source);
                continue;
            }
            std::istringstream input(text);
            std::string line;
            int number=0;
            while(std::getline(input,line)){
                ++number;
                std::string at="L"+std::to_string(number)+": "+line.substr(0,220);
                std::string kind=(line.find("save")!=std::string::npos||line.find("to_csv")!=std::string::npos||line.find("to_pickle")!=std::string::npos||line.find("write")!=std::string::npos)?"literal_output_or_reference":"literal_input_or_reference";
                for(std::sregex_iterator i(line.begin(),line.end(),literal),end; i!=end; ++i)resolve(source,(*i)[1],kind,at);
                for(std::sregex_iterator i(line.begin(),line.end(),module),end; i!=end; ++i){
                    std::string name=(*i)[1].matched?(*i)[1].str():(*i)[2].str();
                    std::replace(name.begin(),name.end(),'.','/');
                    std::string candidate=normalize((fs::u8path(source).parent_path()/fs::u8path(name+".py")).generic_u8string());
                    if(files.count(candidate))add(source,candidate,"local_import",at);
                    else if(files.count(name+".py"))add(source,name+".py","root_import_candidate",at);
                }
            }
        }
        for(auto& item:manual.object){
            auto source=item.first;
            for(auto field:{
                "inputs","outputs","shared_code"
            })if(item.second.has(field))for(auto& target:item.second.at(field).array)add(source,target.str(),std::string("reviewed_")+field,"Manually reviewed source relationships");
            if(item.second.has("data_roots"))for(auto& prefix:item.second.at("data_roots").array){
                add(source,prefix.str(),"reviewed_dataset_root","Dataset relationship, not proof that every member is read on every run");
                for(auto& f:files)if(starts(f.first,prefix.str()+"/"))add(source,f.first,"dataset_member_candidate","Member of the reviewed input dataset; actual per-run subset may differ");
            }
        }
        std::map<std::string,std::vector<const Relation*>> outgoing,incoming;
        for(auto& relation:relations){
            outgoing[relation.source].push_back(&relation);
            incoming[relation.target].push_back(&relation);
        }
        fs::create_directories(out);
        Json relation_json=Json::list();
        for(auto& r:relations){
            Json j=Json::dict();
            j["source"]=r.source;
            j["target"]=r.target;
            j["kind"]=r.kind;
            j["evidence"]=r.evidence;
            relation_json.push(j);
        }
        write(out/"relations.json",relation_json.dump());
        std::ostringstream checklist;
        checklist<<"# 전체 파일 목록과 C++ 변환 체크리스트\n\n목록 기준: "<<snapshot.at("captured_kst").str()<<". 업데이트: "<<now()<<".\n\n원본 영역 "<<files.size()<<"개 파일 중 소스 "<<sources.size()<<"개를 오래된 생성 관측 시각순으로 정렬했다. 생성/수정 관측값은 실험 실행 시각·작성자와 다르다. 같은 시각의 Claude/Codex 작업은 원본 경로와 해시·별도 쓰기 호출 근거로 구별한다.\n\n- `[x]`: 개별 파일의 실제 C++ 구현·빌드·실행 검증을 확인한 경우. 렌더러·허용 수치 차이는 검증 설명에 명시한다.\n- `[ ]`: 대기, 일부 기능만 구현, 원본 변경 후 재검증 필요, 또는 동등성 미확인. 저장 결과 재생만으로 완료 표시하지 않는다.\n- 데이터·모델·문서·이미지·빌드 산출물에는 변환 체크를 붙이지 않는다. 아래 전체 파일표에 포함한다.\n- 전체 이식·경로·재실행·재현 검증 완료 후 기존 코드를 해시 검증해 legacy로 이동한다. 그 전까지 원본을 보존한다. Git 커밋·푸시는 보류한다.\n\n현재 작업 폴더 이름: `analysis_vX_YYYYMMDD_유지한것_바뀐것`. 이전 C++ 경로 변경표는 "<<link(final,"config/experiment_layout.json")<<"에 있다.\n\n이 문서는 재생성되는 보기다. 상태 원본은 "<<link(final,"config/port_progress.json")<<", 확인한 연관 관계는 "<<link(final,"config/file_relations.json")<<"에서 갱신한다.\n\n";
        int complete=0,partial=0,changed=0,existing_cpp=0;
        for(size_t index=0; index<sources.size(); ++index){
            auto& source=sources[index];
            auto& f=files.at(source);
            Json state=progress.has(source)?progress.at(source):Json::dict();
            bool done=state.has("cpp_complete")&&state.at("cpp_complete").num();
            std::string status=state.has("status")?state.at("status").str():previous.count(source)?previous.at(source).at("migration_status").str():"pending_individual_cpp_port";
            std::string destination=state.has("destination")?state.at("destination").str():previous.count(source)?previous.at(source).at("cpp_destination").str():"";
            bool stale=changed_since_snapshot.count(source)!=0;
            if(state.has("source_sha256"))stale|=state.at("source_sha256").str()!=f.hash;
            else if(previous.count(source)&&done)stale|=previous.at(source).at("sha256").str()!=f.hash;
            if(stale){
                done=false;
                ++changed;
                status="source_changed_revalidation_required";
            }
            if(done&&!destination.empty()&&!fs::exists(final/fs::u8path(destination))){
                done=false;
                status="cpp_destination_missing";
            }
            complete+=done;
            if(done){
                auto ext=fs::u8path(source).extension().u8string();
                existing_cpp+=ext==".cpp"||ext==".hpp"||ext==".h"||ext==".cc"||ext==".cxx";
            }
            partial+=!done&&!destination.empty();
            checklist<<"## "<<index+1<<". "<<safe(source)<<"\n\n- ["<<(done?'x':' ')<<"] C++ 변환 완료 — "<<link(root,source)<<"\n- 생성 관측 KST: "<<f.created<<" / 수정 관측 KST: "<<f.modified<<"\n- 상태: `"<<status<<"`\n- 원본 SHA256: `"<<f.hash<<"`\n";
            if(!destination.empty())checklist<<"- C++: "<<link(final,destination)<<"\n";
            if(state.has("validation"))checklist<<"- 검증·차이: "<<safe(state.at("validation").str())<<"\n";
            if(state.has("report"))checklist<<"- 설명: "<<link(final,state.at("report").str())<<"\n";
            if(state.has("evidence"))for(auto& evidence:state.at("evidence").array)checklist<<"- 실행 근거: "<<link(final,evidence.str())<<"\n";
            {
                Json events=authors.has(source)?authors.at(source):previous.count(source)&&previous.at(source).has("write_evidence")?previous.at(source).at("write_evidence"):Json::list();
                std::set<std::string> actors;
                for(auto& event:events.array)actors.insert(event.at("actor").str());
                checklist<<"- 관측된 원본 작업자: ";
                if(actors.empty())checklist<<"unknown";
                for(auto& actor:actors)checklist<<actor<<' ';
                checklist<<" (쓰기 도구 호출 흔적이며 현재 모든 줄의 저자를 확정하지 않음)\n";
                for(auto& event:events.array)checklist<<"- 쓰기 호출: "<<event.at("event_kst").str()<<" / "<<event.at("actor").str()<<" / "<<event.at("operation").str()<<" / 로그 "<<link(fs::path(),event.at("session_log").str(),"기록")<<":"<<int(event.at("line").num())<<"\n";
            }
            checklist<<"\n연관 파일:\n\n| 종류 | 경로 | 근거 |\n|---|---|---|\n";
            if(outgoing[source].empty())checklist<<"| 미확인 | 동적 경로·실행형 import 또는 외부 입력 추가 확인 필요 | 자동 검출 없음 |\n";
            for(auto* relation:outgoing[source]){
                if(relation->kind=="dataset_member_candidate")continue;
                std::string target=files.count(relation->target)?link(root,relation->target):"`"+safe(relation->target)+"`";
                checklist<<'|'<<relation->kind<<'|'<<target<<'|'<<safe(relation->evidence)<<"|\n";
            }
            checklist<<'\n';
        }
        std::string header="확인한 변환 완료 **"+std::to_string(complete)+" / "+std::to_string(sources.size())+"개 소스**. 미완료 "+std::to_string(sources.size()-complete)+"개(그중 C++ 경로가 있는 부분/재검증 "+std::to_string(partial)+"개). 기준 변경 "+std::to_string(changed)+"개. 기존 C++의 정리·빌드 확인 "+std::to_string(existing_cpp)+"개와 다른 언어에서 이식한 "+std::to_string(complete-existing_cpp)+"개를 포함한다.\n\n";
        std::string main=checklist.str();
        main.insert(main.find("## 1."),header);
        std::ostringstream listing;
        listing<<"# 전체 비소스 파일 목록\n\n기준 "<<snapshot.at("captured_kst").str()<<". 소스 "<<sources.size()<<"개는 상위 체크리스트로 표시한다. ‘연관 후보’는 정적 문자열/검토한 데이터셋 연결이며 실제 실행에서 읽혔다는 증거와 구분한다.\n\n";
        for(auto category:{
            "data_keep_in_place","result_or_document","dependency_or_generated"
        }){
            listing<<"## "<<category<<"\n\n| 파일 | 생성 관측 KST | 수정 관측 KST | 바이트 | 연관 소스/후보 |\n|---|---|---|---:|---|\n";
            for(auto& entry:files){
                auto& f=entry.second;
                if(f.category!=category)continue;
                listing<<'|'<<link(root,f.path)<<'|'<<f.created<<'|'<<f.modified<<'|'<<std::fixed<<std::setprecision(0)<<f.size<<'|';
                std::set<std::string> neighbors;
                for(auto* relation:incoming[f.path])neighbors.insert(relation->source);
                int count=0;
                for(auto& source:neighbors){
                    if(count++==6){
                        listing<<" 외 "<<neighbors.size()-6<<"개";
                        break;
                    }
                    listing<<link(root,source)<<" ";
                }
                if(neighbors.empty())listing<<"미확인/생성물";
                listing<<"|\n";
            }
        }
        write(out/fs::u8path("전체_비소스_파일.md"),listing.str());
        main+="## 전체 비소스 파일과 범위\n\n"+link(root,(out/fs::u8path("전체_비소스_파일.md")).lexically_relative(root).generic_u8string())+"에 DATA·결과·문서·의존성 파일을 빠짐없이 나열했다. 원본 영역 파일 수는 "+std::to_string(files.size())+"개다. .git 내부, 중복 legacy 보존 사본, final의 빌드/결과는 원본 변환 분모에서 제외한다.\n\n";
        main+="### final의 현재 C++·설정·문서 파일\n\n";
        int final_count=0;
        for(fs::recursive_directory_iterator it(final,fs::directory_options::skip_permission_denied),end; it!=end; ++it){
            auto rel=it->path().lexically_relative(final);
            auto name=rel.begin()->u8string();
            if(it->is_directory()&&(name=="build"||name=="results"||name=="third_party"||name=="data"||(name=="experiments"&&(starts(it->path().filename().u8string(),"inventory_")||starts(it->path().filename().u8string(),"catalogue_")||starts(it->path().filename().u8string(),"authorship_")||starts(it->path().filename().u8string(),"file_list_"))))){
                it.disable_recursion_pending();
                continue;
            }
            if(it->is_regular_file()){
                main+="- "+link(final,rel.generic_u8string())+"\n";
                ++final_count;
            }
        }
        main+="\n이전 동결 목록에만 남고 현재 경로에 없는 소스:\n\n";
        for(auto& old:previous)if(!files.count(old.first))main+="- `"+old.first+"` — 삭제·이동 원인 확인 전까지 완료로 세지 않음. 원본 SHA256 `"+old.second.at("sha256").str()+"`\n";
        write(out/fs::u8path("전체_파일_목록.md"),main);
        write(final/fs::u8path("전체_파일_목록.md"),main);
        Json summary=Json::dict();
        summary["updated_kst"]=now();
        summary["snapshot_kst"]=snapshot.at("captured_kst");
        summary["original_files"]=int(files.size());
        summary["original_sources"]=int(sources.size());
        summary["cpp_complete"]=complete;
        summary["existing_cpp_organized_and_verified"]=existing_cpp;
        summary["converted_to_cpp_and_verified"]=complete-existing_cpp;
        summary["not_complete"]=int(sources.size())-complete;
        summary["partial_or_unverified_with_cpp_path"]=partial;
        summary["source_changed"]=changed;
        summary["relations"]=int(relations.size());
        summary["final_authored_files"]=final_count;
        summary["inventory_manifest_sha256"]=sha256(inventory/"files.csv");
        write(out/"checks.json",summary.dump());
        std::cout<<summary.dump()<<'\n';
        return 0;
    }
    catch(const std::exception& e){
        std::cerr<<e.what()<<'\n';
        return 1;
    }
}
