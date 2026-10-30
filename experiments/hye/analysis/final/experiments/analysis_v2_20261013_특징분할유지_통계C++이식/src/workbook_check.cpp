#include "features.hpp"
#include "ooxml.hpp"
#include <iostream>
using namespace semes;
using namespace v2_history;
int wmain(int argc,wchar_t** wide_argv){
    Utf8Arguments arguments(argc,wide_argv);
    char** argv=arguments.data();
    try{
        if(argc!=5)throw std::runtime_error("historical_v2_workbook_check original-workspace cpp-judge-output reference.xlsx new-output");
        fs::path root=fs::u8path(argv[1]),judge=fs::u8path(argv[2]),source=fs::u8path(argv[3]),out=fs::u8path(argv[4]);
        if(fs::exists(out))throw std::runtime_error("New output required");
        Features data(root);
        CsvTable decisions(judge/"decisions.csv"),summary(judge/"judge_summary.csv");
        ooxml::Reader workbook(source);
        fs::create_directories(out);
        std::ofstream differences(out/"differences.csv");
        differences<<"sheet,cell,expected,actual\n";
        int labels=0,mismatches=0,numeric_count=0;
        double max_distance_difference=0;
        for(int axis=0; axis<2; ++axis){
            auto sheet=workbook.sheet(axis+2);
            std::map<std::string,int> index;
            for(auto& cell:sheet)if(cell.first[0]=='B'&&cell.first.size()>1&&std::isdigit(static_cast<unsigned char>(cell.first[1]))&&cell.second.value.size()>4&&cell.second.value.substr(cell.second.value.size()-4)==".bmp")index.emplace(cell.second.value,std::stoi(cell.first.substr(1)));
            auto& rows=axis==0?data.ys:data.ts;
            if(index.size()!=rows.size())throw std::runtime_error("Workbook source row coverage differs");
            for(size_t i=0; i<decisions.rows.size(); ++i){
                if(decisions.text(i,"axis")!=(axis==0?"Y":"T"))continue;
                auto config=decisions.text(i,"configuration");
                size_t c=0;
                for(; c<data.configurations.size(); ++c)if(data.configurations[c].first==config)break;
                if(c==data.configurations.size())throw std::runtime_error("Unknown config");
                int row=index.at(decisions.text(i,"file"));
                auto compare=[&](int col,const std::string& actual){
                    std::string at=ooxml::column(col)+std::to_string(row),expected=sheet.at(at).value;
                    ++labels;
                    if(expected!=actual){
                        ++mismatches;
                        differences<<axis+2<<','<<at<<','<<csv(expected)<<','<<csv(actual)<<'\n';
                    }
                };
                if(c==0)compare(10,decisions.text(i,"rule1"));
                compare(int(11+c),decisions.text(i,"rule2"));
                if(c==3&&!decisions.text(i,"distance").empty()){
                    std::string at="R"+std::to_string(row);
                    double expected=std::stod(sheet.at(at).value),actual=decisions.number(i,"distance"),error=std::abs(expected-actual);
                    max_distance_difference=std::max(max_distance_difference,error);
                    ++numeric_count;
                    if(error>1e-6){
                        ++mismatches;
                        differences<<axis+2<<','<<at<<','<<std::setprecision(17)<<expected<<','<<actual<<'\n';
                    }
                }
            }
        }
        Json audit=Json::dict();
        audit["created_kst"]=now();
        audit["reference_sha256"]=sha256(source);
        audit["compared_decision_labels"]=labels;
        audit["compared_distances"]=numeric_count;
        audit["mismatches"]=mismatches;
        audit["max_distance_difference"]=max_distance_difference;
        audit["distance_tolerance"]=1e-6;
        audit["passed"]=mismatches==0;
        write(out/"checks.json",audit.dump());
        std::cout<<audit.dump()<<'\n';
        return mismatches?2:0;
    }
    catch(const std::exception& e){
        std::cerr<<e.what()<<'\n';
        return 1;
    }
}
