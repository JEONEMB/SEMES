#include "alignment.hpp"
#include <iostream>
#include <fstream>
#include <set>
using namespace semes;
static cv::Mat numbers(const Npy& a){
    if(a.shape.size()!=2)throw std::runtime_error("Expected feature matrix");
    cv::Mat X(int(a.shape[0]),int(a.shape[1]),CV_64F);
    for(size_t i=0; i<a.count(); i++)X.ptr<double>()[i]=a.number(i);
    return X;
}
int wmain(int argc,wchar_t** wide_argv){
    semes::Utf8Arguments unicode_arguments(argc,wide_argv);
    char** argv=unicode_arguments.data();
    try{
        if(argc<5)throw std::runtime_error("evaluation <features.npy OR fresh-result-directory> <names.npz> <folds.json> <new-output> [mode] [shrink] [covariance-dof]");
        fs::path feature=fs::u8path(argv[1]),out=fs::u8path(argv[4]);
        if(fs::exists(out))throw std::runtime_error("New output required");
        auto cache=npz(fs::u8path(argv[2]));
        auto folds=Json::parse(read(fs::u8path(argv[3])));
        std::vector<std::string> names;
        for(size_t i=0; i<cache.at("names").count(); i++)names.push_back(cache.at("names").text(i));
        std::map<std::string,int> lookup;
        for(int i=0; i<int(names.size()); i++)if(!lookup.emplace(names[i],i).second)throw std::runtime_error("Duplicate input name");
        cv::Mat X;
        if(fs::is_directory(feature)){
            X=cv::Mat::zeros(int(names.size()),12,CV_64F);
            for(int i=0; i<int(names.size()); i++)if(names[i]!="center.bmp"){
                auto j=Json::parse(read(feature/"images"/fs::u8path(names[i]+".json")));
                matrix(j.at("features")).copyTo(X.row(i));
            }
        }
        else X=numbers(npy(bytes(feature)));
        if(X.rows!=int(names.size()))throw std::runtime_error("Feature/name mismatch");
        std::string mode=argc>5?argv[5]:"center";
        double shrink=argc>6?std::stod(argv[6]):.001;
        int dof=argc>7?std::stoi(argv[7]):0;
        fs::create_directories(out);
        std::ofstream predictions(out/"predictions.csv"),failure(out/"failures.csv");
        std::string header="file,fold,protocol,true_Y,true_T,pred_Y,pred_T,pred_axis,axis_correct,axis_sign_correct,error_Y,error_T,both_within_009,strict_pass\n";
        predictions<<header;
        failure<<header;
        Json summaries=Json::list();
        std::set<int> alltests;
        auto run=[&](const std::vector<int>& tr,const std::vector<int>& te,const std::string& label,const std::string& protocol){
            auto model=train_axis_model(X,names,tr,mode,true,shrink,dof);
            write(out/("model_"+label+".json"),model.dump());
            int passed=0,strict=0,axisok=0,signok=0;
            double ysum=0,tsum=0,ymax=0,tmax=0;
            for(int i:te){
                auto p=predict_model(X.row(i),model,mode=="inferred_session"||mode=="separate_session");
                double v=std::stod(names[i].substr(1)),y=names[i][0]=='Y'?v:0,t=names[i][0]=='T'?v:0,ey=std::abs(p.y-y),et=std::abs(p.t-t);
                bool ac=p.axis==names[i].substr(0,1),sc=ac&&(std::signbit(p.value)==std::signbit(v)),pass=ey<=.09&&et<=.09;
                passed+=pass;
                strict+=pass&&sc;
                axisok+=ac;
                signok+=sc;
                ysum+=ey;
                tsum+=et;
                ymax=std::max(ymax,ey);
                tmax=std::max(tmax,et);
                std::ostringstream r;
                r<<std::setprecision(17)<<csv(names[i])<<','<<label<<','<<protocol<<','<<y<<','<<t<<','<<p.y<<','<<p.t<<','<<p.axis<<','<<ac<<','<<sc<<','<<ey<<','<<et<<','<<pass<<','<<(pass&&sc)<<'\n';
                predictions<<r.str();
                if(!pass||!sc)failure<<r.str();
            }
            Json j=Json::dict();
            j["fold"]=label;
            j["protocol"]=protocol;
            j["n"]=int(te.size());
            j["training_n"]=int(tr.size());
            j["Y_MAE"]=ysum/te.size();
            j["T_MAE"]=tsum/te.size();
            j["Y_max"]=ymax;
            j["T_max"]=tmax;
            j["both_within_009"]=passed;
            j["strict_pass"]=strict;
            j["axis_correct"]=axisok;
            j["axis_and_sign_correct"]=signok;
            summaries.push(j);
            std::cout<<j.dump()<<'\n';
        };
        if(folds.size()!=4)throw std::runtime_error("Exactly 4 fixed folds required");
        for(auto& f:folds.array){
            std::vector<int> tr,te;
            std::set<int> membership;
            for(auto& n:f.at("train").array){
                int i=lookup.at(n.str());
                if(!membership.insert(i).second)throw std::runtime_error("Duplicate training file");
                tr.push_back(i);
            }
            for(auto& n:f.at("test").array){
                int i=lookup.at(n.str());
                if(membership.count(i)||!alltests.insert(i).second)throw std::runtime_error("Overlapping fold membership");
                te.push_back(i);
            }
            if(tr.size()!=210||te.size()!=70)throw std::runtime_error("Expected 210/70 split");
            run(tr,te,std::to_string(int(f.at("fold").num())),"oof4");
        }
        if(alltests.size()!=280)throw std::runtime_error("OOF denominator not 280");
        std::vector<int> all;
        for(int i=0; i<int(names.size()); i++)if(names[i]!="center.bmp")all.push_back(i);
        run(all,all,"all","resub280");
        Json report=Json::dict();
        report["created_kst"]=now();
        report["feature_source"]=feature.u8string();
        report["feature_source_type"]=fs::is_directory(feature)?"fresh_cpp_lm_measurements":"saved_python_features_replay";
        report["mode"]=mode;
        report["shrink"]=shrink;
        report["covariance_dof_override"]=dof;
        report["summaries"]=summaries;
        write(out/"summary.json",report.dump());
        return 0;
    }
    catch(const std::exception& e){
        std::cerr<<e.what()<<'\n';
        return 1;
    }
}
