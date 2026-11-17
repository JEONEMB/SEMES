#include "alignment.hpp"
#include <iostream>
#include <fstream>
#include <set>
using namespace semes;
struct Table {
    std::map<std::string,int> columns;
    std::vector<std::vector<std::string>> rows;
    explicit Table(const fs::path& p){
        std::ifstream f(p);
        if(!f)throw std::runtime_error("Cannot read CSV "+p.u8string());
        std::string line;
        std::getline(f,line);
        auto h=csvrow(line);
        if(h[0].size()>=3&&(unsigned char)h[0][0]==239)h[0]=h[0].substr(3);
        for(int i=0; i<int(h.size()); i++)columns[h[i]]=i;
        while(std::getline(f,line)){
            auto r=csvrow(line);
            if(r.size()==h.size())rows.push_back(std::move(r));
        }
    }
    std::string get(const std::vector<std::string>& r,const std::string& c)const{
        return r.at(columns.at(c));
    }
};
int wmain(int argc,wchar_t** wide_argv){
    semes::Utf8Arguments unicode_arguments(argc,wide_argv);
    char** argv=unicode_arguments.data();
    try{
        if(argc!=10)throw std::runtime_error("audit assets center corners280.csv features.npy names.npz baseline-recomputed.csv compiled280.csv cpp-results new-output");
        Engine engine(fs::u8path(argv[1]),fs::u8path(argv[2]));
        Table source(fs::u8path(argv[3])),baseline(fs::u8path(argv[6])),compiled(fs::u8path(argv[7]));
        auto expected_features=npy(bytes(fs::u8path(argv[4])));
        auto cache=npz(fs::u8path(argv[5]));
        fs::path cpp=fs::u8path(argv[8]),out=fs::u8path(argv[9]);
        if(fs::exists(out))throw std::runtime_error("New output required");
        std::map<std::string,int> nameindex;
        for(size_t i=0; i<cache.at("names").count(); i++)nameindex[cache.at("names").text(i)]=int(i);
        std::map<int,int> ids;
        for(int i=0; i<630; i++)ids[engine.corners()[i].id]=i;
        std::map<std::string,cv::Mat> old,oldvalid;
        std::map<std::string,std::set<int>> seen;
        for(auto& r:source.rows){
            int id=std::stoi(source.get(r,"id"));
            if(!ids.count(id))continue;
            std::string name=source.get(r,"file");
            if(!old.count(name)){
                old[name]=cv::Mat(630,2,CV_64F);
                oldvalid[name]=cv::Mat(630,1,CV_32S);
            }
            int i=ids.at(id);
            if(!seen[name].insert(id).second)throw std::runtime_error("Duplicate measurement ID");
            old[name].at<double>(i,0)=std::stod(source.get(r,"x"));
            old[name].at<double>(i,1)=std::stod(source.get(r,"y"));
            oldvalid[name].at<int>(i)=source.get(r,"valid")=="True";
        }
        for(auto& kv:seen)if(kv.second.size()!=630)throw std::runtime_error("Incomplete original ID set");
        if(old.size()!=281)throw std::runtime_error("Expected 281 original measurement rows");
        double reference_max=0;
        for(int i=0; i<630; i++)reference_max=std::max({
            reference_max,std::abs(old.at("center.bmp").at<double>(i,0)-engine.corners()[i].reference.x),std::abs(old.at("center.bmp").at<double>(i,1)-engine.corners()[i].reference.y)
        });
        if(reference_max>1e-9)throw std::runtime_error("Original center reference differs");
        std::map<std::string,std::vector<std::string>> bmap,cmap;
        for(auto& r:baseline.rows)bmap[baseline.get(r,"file")]=r;
        for(auto& r:compiled.rows)cmap[compiled.get(r,"file")]=r;
        fs::create_directories(out);
        std::ofstream perfile(out/"comparison.csv");
        perfile<<"file,p12_reference_max_abs,frozen_model_prediction_diff,compiled_prediction_diff,compiled_valid_count_matches,coordinate_max_norm_to_trf,diagnostic_changes_to_trf\n";
        double p12max=0,baselinepredmax=0,compiledpredmax=0,coordmax=0;
        int axisdiff=0,compiledaxisdiff=0,validcountdiff=0,diagnosticdiff=0;
        std::vector<double> coordinate;
        for(auto& kv:old){
            std::string name=kv.first;
            if(name=="center.bmp")continue;
            cv::Mat d=kv.second-old.at("center.bmp"),f=engine.features(d);
            int index=nameindex.at(name);
            double p12diff=0;
            for(int j=0; j<12; j++)p12diff=std::max(p12diff,std::abs(f.at<double>(j)-expected_features.number(index*12+j)));
            p12max=std::max(p12max,p12diff);
            auto pred=engine.predict(f);
            auto& br=bmap.at(name);
            double bp=std::abs(pred.value-std::stod(baseline.get(br,"pred")));
            baselinepredmax=std::max(baselinepredmax,bp);
            axisdiff+=pred.axis!=baseline.get(br,"pred_axis");
            auto result=Json::parse(read(cpp/"images"/fs::u8path(name+".json")));
            auto& cr=cmap.at(name);
            double cp=std::abs(result.at("value").num()-std::stod(compiled.get(cr,"pred")));
            compiledpredmax=std::max(compiledpredmax,cp);
            compiledaxisdiff+=result.at("axis").str()!=compiled.get(cr,"pred_axis");
            bool validmatch=result.at("valid").num()==std::stod(compiled.get(cr,"valid"));
            validcountdiff+=!validmatch;
            Table fresh(cpp/"measurements"/fs::u8path(name+".csv"));
            if(fresh.rows.size()!=630)throw std::runtime_error("Missing fresh coordinate rows");
            int changes=0;
            double cmax=0;
            for(auto& r:fresh.rows){
                int i=ids.at(std::stoi(fresh.get(r,"id")));
                double x=std::stod(fresh.get(r,"x")),y=std::stod(fresh.get(r,"y")),delta=std::hypot(x-kv.second.at<double>(i,0),y-kv.second.at<double>(i,1));
                coordinate.push_back(delta);
                cmax=std::max(cmax,delta);
                changes+=std::stoi(fresh.get(r,"valid"))!=oldvalid.at(name).at<int>(i);
            }
            diagnosticdiff+=changes;
            coordmax=std::max(coordmax,cmax);
            perfile<<std::setprecision(17)<<csv(name)<<','<<p12diff<<','<<bp<<','<<cp<<','<<validmatch<<','<<cmax<<','<<changes<<'\n';
        }
        std::sort(coordinate.begin(),coordinate.end());
        Json j=Json::dict();
        j["n"]=280;
        j["original_center_coordinate_max_abs_px"]=reference_max;
        j["saved_python_p12_max_abs"]=p12max;
        j["saved_python_frozen_model_prediction_max_abs"]=baselinepredmax;
        j["saved_python_frozen_model_axis_changes"]=axisdiff;
        j["fresh_cpp_vs_python_compiled_prediction_max_abs"]=compiledpredmax;
        j["fresh_cpp_vs_python_compiled_axis_changes"]=compiledaxisdiff;
        j["fresh_cpp_vs_python_compiled_valid_count_mismatches"]=validcountdiff;
        j["cpp_lm_vs_python_trf_coordinate_max_norm_px"]=coordmax;
        j["cpp_lm_vs_python_trf_coordinate_median_norm_px"]=coordinate[coordinate.size()/2];
        j["cpp_lm_vs_python_trf_diagnostic_changes"]=diagnosticdiff;
        j["passed_p12_and_model_parity"]=p12max<1e-9&&baselinepredmax<1e-9&&!axisdiff;
        j["passed_compiled_prediction_parity"]=compiledpredmax<1e-5&&!compiledaxisdiff&&!validcountdiff;
        j["created_kst"]=now();
        write(out/"checks.json",j.dump());
        std::cout<<j.dump()<<'\n';
        return j.at("passed_p12_and_model_parity").num()&&j.at("passed_compiled_prediction_parity").num()?0:2;
    }
    catch(const std::exception&e){
        std::cerr<<e.what()<<'\n';
        return 1;
    }
}
