#include "joint.hpp"
#include <iostream>
#include <fstream>
using namespace semes;
int wmain(int argc,wchar_t** wide_argv){
    semes::Utf8Arguments unicode_arguments(argc,wide_argv);
    char** argv=unicode_arguments.data();
    try{
        if(argc!=8)throw std::runtime_error("joint_check assets center model.npz measurements.npz manifest.csv reference-predictions.csv new-output");
        Engine engine(fs::u8path(argv[1]),fs::u8path(argv[2]));
        JointOrigins model(fs::u8path(argv[3]));
        auto obs=npz(fs::u8path(argv[4]));
        fs::path out=fs::u8path(argv[7]);
        if(fs::exists(out))throw std::runtime_error("New output required");
        auto& D=obs.at("D");
        if(D.shape.size()!=3||D.shape[1]!=630||D.shape[2]!=2)throw std::runtime_error("Expected Nx630x2 measurement archive");
        std::ifstream manifest(fs::u8path(argv[5]));
        std::string line;
        std::getline(manifest,line);
        auto h=csvrow(line);
        std::map<std::string,int> col;
        for(int i=0; i<int(h.size()); i++)col[h[i]]=i;
        std::vector<std::vector<std::string>> rows;
        while(std::getline(manifest,line))rows.push_back(csvrow(line));
        if(rows.size()!=D.shape[0])throw std::runtime_error("Manifest measurement mismatch");
        std::ifstream reference(fs::u8path(argv[6]));
        std::getline(reference,line);
        h=csvrow(line);
        std::map<std::string,int> rc;
        for(int i=0; i<int(h.size()); i++)rc[h[i]]=i;
        std::map<std::string,std::vector<std::string>> expected;
        while(std::getline(reference,line)){
            auto r=csvrow(line);
            if(r.size()==h.size()&&r[rc.at("method")]=="multi_center_joint"&&r[rc.at("protocol")]=="resub341")expected[r[rc.at("relative")]]=r;
        }
        fs::create_directories(out);
        std::ofstream predictions(out/"predictions.csv");
        predictions<<"relative,Y,T,predY,predT,errorY,errorT,passed,center_id,prediction_difference,center_matches\n";
        double maxdiff=0,maxY=0,maxT=0,sumY=0,sumT=0;
        int passed=0,center_mismatch=0,compared=0;
        for(size_t i=0; i<rows.size(); i++){
            auto& r=rows[i];
            cv::Mat d(630,2,CV_64F);
            for(int k=0; k<1260; k++)d.ptr<double>()[k]=D.number(i*1260+k);
            auto p=model.predict(engine.features(d,model.feature_name()));
            double y=std::stod(r[col.at("Y")]),t=std::stod(r[col.at("T")]),py=p.at("Y").num(),pt=p.at("T").num(),ey=std::abs(py-y),et=std::abs(pt-t);
            bool pass=ey<=.09+1e-12&&et<=.09+1e-12;
            passed+=pass;
            maxY=std::max(maxY,ey);
            maxT=std::max(maxT,et);
            sumY+=ey;
            sumT+=et;
            std::string relative=r[col.at("relative")];
            double diff=0;
            bool cm=true;
            if(expected.count(relative)){
                auto& e=expected.at(relative);
                diff=std::max(std::abs(py-std::stod(e[rc.at("predY")])),std::abs(pt-std::stod(e[rc.at("predT")])));
                maxdiff=std::max(maxdiff,diff);
                cm=e[rc.at("center")]==p.at("center_id").str();
                center_mismatch+=!cm;
                ++compared;
            }
            predictions<<std::setprecision(17)<<csv(relative)<<','<<y<<','<<t<<','<<py<<','<<pt<<','<<ey<<','<<et<<','<<pass<<','<<csv(p.at("center_id").str())<<','<<diff<<','<<cm<<'\n';
        }
        Json report=Json::dict();
        report["n"]=int(rows.size());
        report["both_within_009"]=passed;
        report["Y_MAE"]=sumY/rows.size();
        report["T_MAE"]=sumT/rows.size();
        report["Y_max"]=maxY;
        report["T_max"]=maxT;
        report["compared_to_python"]=compared;
        report["max_prediction_difference"]=maxdiff;
        report["center_mismatches"]=center_mismatch;
        report["protocol"]="saved_measurement_and_frozen_model_replay_resub341";
        report["fresh_image_inference"]=false;
        report["created_kst"]=now();
        write(out/"summary.json",report.dump());
        std::cout<<report.dump()<<'\n';
        return 0;
    }
    catch(const std::exception&e){
        std::cerr<<e.what()<<'\n';
        return 1;
    }
}
