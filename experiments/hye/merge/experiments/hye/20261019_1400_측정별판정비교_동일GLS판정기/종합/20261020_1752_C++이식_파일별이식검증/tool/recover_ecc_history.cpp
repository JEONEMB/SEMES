#include "io.hpp"
#include <regex>
#include <iostream>
#include <fstream>
#include <algorithm>
using namespace semes;
std::string normalized(std::string text){
    text.erase(std::remove(text.begin(),text.end(),'\r'),text.end());
    while(!text.empty()&&text.back()=='\n')text.pop_back();
    return text;
}
int wmain(int argc,wchar_t** wide_argv){
    Utf8Arguments args(argc,wide_argv);
    char** argv=args.data();
    try{
        if(argc!=5)throw std::runtime_error("recover_ecc_history original-workspace claude-session.jsonl existing-authorship.json new-output");
        fs::path root=fs::u8path(argv[1]),log=fs::u8path(argv[2]),out=fs::u8path(argv[4]);
        if(fs::exists(out))throw std::runtime_error("New output required");
        std::ifstream stream(log);
        if(!stream)throw std::runtime_error("Cannot read historical log");
        std::string line;
        Json command,result;
        for(int number=1; std::getline(stream,line); ++number){
            if(number==544)command=Json::parse(line);
            if(number==593){
                result=Json::parse(line);
                break;
            }
        }
        std::string shell;
        for(auto& block:command.at("message").at("content").array)if(block.at("type").str()=="tool_use"&&block.has("input")&&block.at("input").has("command")){
            auto value=block.at("input").at("command").str();
            if(value.find("ecc_tune.py")!=std::string::npos)shell=value;
        }
        const std::string marker="cat > \"$S/ecc_tune.py\" <<'EOF'\n";
        size_t begin=shell.find(marker);
        if(begin==std::string::npos)throw std::runtime_error("Expected historical heredoc missing");
        begin+=marker.size();
        size_t end=shell.find("\nEOF\n",begin);
        if(end==std::string::npos)throw std::runtime_error("Heredoc terminator missing");
        fs::path source=root/"analysis_v3/v1_data/ecc_tune.py";
        bool matches=normalized(shell.substr(begin,end-begin))==normalized(read(source));
        if(!matches)throw std::runtime_error("Logged script and archived source differ; cannot assign this output to source");
        auto stdout_text=result.at("toolUseResult").at("file").at("content").str();
        std::regex row(u8"blur\\s+([0-9.]+)\\s+마스크확장\\s+([0-9]+)px\\s+gauss\\s+([0-9]+).*?잡음\\s+([0-9.]+)\\s+\\(([0-9]+)s\\)");
        std::ostringstream csv;
        csv<<"variant,blur,mask_expansion_px,ecc_filter,T_noise,seconds\n";
        int count=0;
        Json variants=Json::list();
        for(std::sregex_iterator it(stdout_text.begin(),stdout_text.end(),row),stop; it!=stop; ++it){
            Json v=Json::dict();
            v["variant"]=count;
            v["blur"]=std::stod((*it)[1]);
            v["mask_expansion_px"]=std::stoi((*it)[2]);
            v["ecc_filter"]=std::stoi((*it)[3]);
            v["T_noise"]=std::stod((*it)[4]);
            v["seconds"]=std::stoi((*it)[5]);
            variants.push(v);
            csv<<count++<<','<<(*it)[1]<<','<<(*it)[2]<<','<<(*it)[3]<<','<<(*it)[4]<<','<<(*it)[5]<<'\n';
        }
        if(count!=7)throw std::runtime_error("Expected7 original printed configurations");
        fs::create_directories(out);
        write(out/"original_variants.csv",csv.str());
        Json report=Json::dict();
        report["recovered_kst"]=now();
        report["source"]= "analysis_v3/v1_data/ecc_tune.py";
        report["source_sha256"]=sha256(source);
        report["log"]=log.generic_u8string();
        report["log_sha256"]=sha256(log);
        report["logged_code_line"]=544;
        report["logged_code_timestamp_utc"]=command.at("timestamp");
        report["logged_code_timestamp_kst"]="2026-10-12T15:37:47.651+09:00";
        report["logged_result_line"]=593;
        report["logged_result_timestamp_utc"]=result.at("timestamp");
        report["logged_result_timestamp_kst"]="2026-10-12T15:49:01.235+09:00";
        report["logged_code_matches_archived_source_normalized_line_endings"]=matches;
        report["archived_source_created_kst"]=timestamp(source,true,true);
        report["archived_source_modified_kst"]=timestamp(source,false,true);
        report["variants"]=variants;
        report["limitations"]="Historical tool output prints noise rounded to4decimals. Script contents match, but the exact then-current tracks.npz and Python/OpenCV build are not independently pinned by this log. Source-copy time and earlier logged code-write time are distinct. No code from the log is executed.";
        write(out/"checks.json",report.dump());
        auto evidence=Json::parse(read(fs::u8path(argv[3])));
        std::string key="analysis_v3/v1_data/ecc_tune.py";
        Json event=Json::dict();
        event["actor"]="Claude";
        event["session_log"]=log.generic_u8string();
        event["line"]=544;
        event["event_timestamp"]=command.at("timestamp");
        event["event_kst"]="2026-10-12T15:37:47.651+09:00";
        event["operation"]="shell heredoc in original temporary scratchpad; normalized whole script matches later archived source";
        event["evidence_status"]="content-matched historical shell invocation, not a direct write to the present archived path; exact old input NPZ not pinned";
        if(!evidence.has(key))evidence[key]=Json::list();
        evidence[key].push(event);
        write(out/"file_write_evidence.json",evidence.dump());
        std::cout<<"Recovered7 printed values; source text matches. Historical input/runtime identity remains unverified.\n";
        return 0;
    }
    catch(const std::exception& e){
        std::cerr<<e.what()<<'\n';
        return 1;
    }
}
