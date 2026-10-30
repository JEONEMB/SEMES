#include "io.hpp"
#include <iostream>
#include <set>
#include <algorithm>
using namespace semes;
static std::string cell(std::string s){
    for(char& c:s)if(c=='\n'||c=='\r'||c=='|')c=' ';
    return s;
}
int wmain(int argc,wchar_t** wide_argv){
    semes::Utf8Arguments unicode_arguments(argc,wide_argv);
    char** argv=unicode_arguments.data();
    try{
        if(argc!=6&&argc!=7)throw std::runtime_error("organize inventory-dir authorship-json experiment-notes.json new-output workspace [port-progress.json]");
        fs::path inv=fs::u8path(argv[1]),out=fs::u8path(argv[4]),root=fs::u8path(argv[5]);
        if(fs::exists(out))throw std::runtime_error("New output directory required");
        auto index=Json::parse(read(inv/"index.json")),authors=Json::parse(read(fs::u8path(argv[2]))),notes=Json::parse(read(fs::u8path(argv[3])));
        Json progress=argc==7?Json::parse(read(fs::u8path(argv[6]))):Json::dict();
        std::ostringstream summary;
        summary<<"# 실험별 출처와 C++ 이식 현황\n\n전체 대상의 경로와 파일 해시를 보존한다. 파일 생성·수정 시각은 관측된 파일 메타데이터이며 실제 실험 실행 시각과 구별한다. 작성자는 도구 호출 기록이 있는 경우에만 관측된 작업자로 표시한다. 같은 경로에 여러 작업자가 있으면 기록을 모두 유지한다.\n\n| 실험 경로 | 소스 수 | 관측된 작업자 | 최초 생성 KST | 최종 수정 KST |\n|---|---:|---|---|---|\n";
        Json ledger=Json::list();
        std::ostringstream timeline;
        timeline<<"created_kst,modified_kst,experiment,source,sha256,observed_actors,migration_status,destination\n";
        for(auto& group:index.at("groups").array){
            auto key=group.at("experiment").str();
            auto g=Json::parse(read(inv/"experiments"/fs::u8path(key)/"inventory.json"));
            std::string earliest="9999",latest="0000";
            std::set<std::string> actors;
            std::ostringstream report;
            report<<"# "<<key<<" 실험 출처\n\n";
            if(notes.has(key)){
                auto& n=notes.at(key);
                for(auto pair:{
                    std::pair<const char*,const char*>{
                        "problem","문제"
                    },{
                        "investigation","조사"
                    },{
                        "method","해결 방법"
                    },{
                        "unchanged","유지한 부분"
                    },{
                        "changed","바꾼 부분"
                    },{
                        "difference","다른 실험과의 구분"
                    }
                })report<<"## "<<pair.second<<"\n\n"<<n.at(pair.first).str()<<"\n\n";
            }
            else report<<"## 내용 확인 범위\n\n이 폴더의 목적과 변경점은 아래 원본 코드의 시작 설명·함수·참조와 보존된 보고서에서 확인한다. 폴더명이나 시각만으로 다른 실험과의 선후 관계를 확정하지 않는다. 개별 C++ 이식과 동등성 검증은 아직 완료 표시하지 않았다.\n\n";
            report<<"## 시간과 작성 흔적\n\n";
            for(auto& item:g.at("files").array){
                earliest=std::min(earliest,item.at("created_kst").str());
                latest=std::max(latest,item.at("modified_kst").str());
            }
            report<<"관측된 최초 파일 생성: "<<earliest<<". 관측된 최종 파일 수정: "<<latest<<". 파일 복사·재생성·덮어쓰기로 실제 작업 시간과 다를 수 있다.\n\n";
            for(auto item:g.at("sources").array){
                std::string path=item.at("path").str(),status="pending_individual_cpp_port",destination="";
                if(path=="motion_speed_630_20261022/compiled.py"){
                    status="ported_compiled_lm_validated_sample_and_280_image_predictions";
                    destination="src/alignment.cpp";
                }
                else if(path=="motion_speed_630_20261022/engine.py"){
                    status="ported_feature_and_inference_engine";
                    destination="src/alignment.cpp";
                }
                else if(path=="joint_center_630_20261022/features.py"){
                    status="ported_feature_maps_validated_QR36_341_replay";
                    destination="src/alignment.cpp";
                }
                else if(path=="joint_center_630_20261022/model.py"||path=="joint_center_630_20261022/local_model.py"){
                    status="frozen_model_inference_ported_training_pending";
                    destination="src/joint.cpp";
                }
                else if(path.rfind("hw-analysis/tool/hw/eda/src/",0)==0){
                    status="cpp_preserved_output_paths_adapted_build_passed";
                    destination="tool/hw/eda/src/"+fs::u8path(path).filename().u8string();
                }
                else if(path=="codex_experiments_20261014/zero_experiment.py"){
                    status="axis_calibration_modes_ported_other_audit_paths_pending";
                    destination="src/training.cpp";
                }
                if(progress.has(path)){
                    auto& port=progress.at(path);
                    status=port.at("status").str();
                    destination=port.at("destination").str();
                    item["port_progress"]=port;
                }
                std::string actor;
                Json evidence=Json::list();
                if(authors.has(path)){
                    evidence=authors.at(path);
                    std::set<std::string> distinct;
                    for(auto& e:evidence.array){
                        distinct.insert(e.at("actor").str());
                        actors.insert(e.at("actor").str());
                    }
                    for(auto& a:distinct){
                        if(!actor.empty())actor+=" + ";
                        actor+=a;
                    }
                }
                else actor="unknown";
                item["migration_status"]=status;
                item["cpp_destination"]=destination;
                item["write_evidence"]=evidence;
                ledger.push(item);
                timeline<<csv(item.at("created_kst").str())<<','<<csv(item.at("modified_kst").str())<<','<<csv(key)<<','<<csv(path)<<','<<csv(item.at("sha256").str())<<','<<csv(actor)<<','<<csv(status)<<','<<csv(destination)<<'\n';
                report<<"### "<<fs::u8path(path).filename().u8string()<<"\n\n- 원본: `"<<path<<"`\n- 생성 KST: "<<item.at("created_kst").str()<<"\n- 수정 KST: "<<item.at("modified_kst").str()<<"\n- SHA256: `"<<item.at("sha256").str()<<"`\n- 관측된 작업자: "<<actor<<"\n- 이식 상태: `"<<status<<"`\n";
                if(!destination.empty())report<<"- C++ 위치: `"<<destination<<"`\n";
                if(progress.has(path)){
                    auto& port=progress.at(path);
                    report<<"- 검증·차이: "<<port.at("validation").str()<<"\n";
                    report<<"- 이식 기록: `"<<port.at("report").str()<<"`\n";
                    if(port.has("evidence"))for(auto& evidence:port.at("evidence").array)report<<"- 새 실행 근거: `"<<evidence.str()<<"`\n";
                }
                report<<"\n원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.\n\n";
                for(auto& e:evidence.array)report<<"- "<<e.at("event_kst").str()<<" "<<e.at("actor").str()<<" "<<e.at("operation").str()<<" 호출. 로그 `"<<e.at("session_log").str()<<"`의 "<<int(e.at("line").num())<<"행.\n";
                if(evidence.size())report<<"\n위 기록은 쓰기 도구 호출의 흔적이다. 현재 파일의 모든 줄을 해당 작업자가 썼거나 호출이 성공했다고 단정하지 않는다.\n\n";
            }
            report<<"## 보존된 결과와 문서\n\n";
            for(auto& item:g.at("files").array){
                auto p=fs::u8path(item.at("path").str());
                if(p.extension()==".md"||p.extension()==".html"||p.extension()==".json"||p.extension()==".csv")report<<"- `"<<item.at("path").str()<<"` ("<<item.at("modified_kst").str()<<")\n";
            }
            fs::path dir=out/fs::u8path(key);
            write(dir/"README.md",report.str());
            write(dir/"source_evidence.json",g.dump());
            std::string actor;
            for(auto& a:actors){
                if(!actor.empty())actor+=" + ";
                actor+=a;
            }
            if(actor.empty())actor="unknown";
            summary<<'|'<<cell(key)<<'|'<<g.at("sources").size()<<'|'<<actor<<'|'<<earliest<<'|'<<latest<<"|\n";
        }
        write(out/"README.md",summary.str());
        write(out/"migration_ledger.json",ledger.dump());
        write(out/"migration_timeline.csv",timeline.str());
        std::cout<<"Registered "<<ledger.size()<<" sources with separate file timestamps and write events\n";
        return 0;
    }
    catch(const std::exception& e){
        std::cerr<<e.what()<<'\n';
        return 1;
    }
}
