#include "calibration.hpp"
#include "table.hpp"
#include <iostream>
using namespace semes;
int wmain(int argc,wchar_t** wide){Utf8Arguments args(argc,wide);auto argv=args.data();try{
    if(argc!=4)throw std::runtime_error("calibrate_full features.npz aligned-manifest.csv new-output");auto features=npz(fs::u8path(argv[1]));CsvTable manifest(fs::u8path(argv[2]));fs::path out=fs::u8path(argv[3]);if(fs::exists(out))throw std::runtime_error("New output required");Json records=Json::list();
    for(std::string condition:{"qr","upper","lower","both"}){
        auto& values=features.at("X_"+condition);auto& good=features.at("success_"+condition);if(values.shape.size()!=2||values.shape[0]!=manifest.rows.size()||good.count()!=manifest.rows.size())throw std::runtime_error("Aligned training rows required");std::vector<int> selected;
        for(size_t i=0;i<manifest.rows.size();++i){bool finite=good.number(i)!=0;for(size_t k=0;k<values.shape[1];++k)finite=finite&&std::isfinite(values.number(i*values.shape[1]+k));if(finite)selected.push_back(int(i));}
        cv::Mat X(int(selected.size()),int(values.shape[1]),CV_64F),q(int(selected.size()),2,CV_64F);std::vector<std::string> dates;for(int row=0;row<X.rows;++row){auto i=selected[row];for(int k=0;k<X.cols;++k)X.at<double>(row,k)=values.number(size_t(i)*X.cols+k);q.at<double>(row,0)=manifest.number(i,"Y");q.at<double>(row,1)=manifest.number(i,"T");dates.push_back(manifest.text(i,"dataset"));}
        auto cal=Calibration::fit(X,q,dates,nullptr,selected);auto model=cal.json();model["condition"]=condition;model["scope"]="Full development training calibration only; no heldout accuracy or physical pose claim";auto path=out/fs::u8path(condition+".json");write(path,model.dump());auto reload=Json::parse(read(path));if(cv::norm(matrix(reload.at("projection"))-cal.projection,cv::NORM_INF)!=0)throw std::runtime_error("Calibration serialization differs");Json record=Json::dict();record["condition"]=condition;record["train_images"]=X.rows;record["sha256"]=sha256(path);record["diagnostics"]=cal.diagnostics;records.push(record);
    }
    Json check=Json::dict();check["models"]=records;check["features_sha256"]=sha256(fs::u8path(argv[1]));check["manifest_sha256"]=sha256(fs::u8path(argv[2]));check["python_invocations"]=0;check["serialization_exact"]=true;check["passed"]=true;write(out/"checks.json",check.dump());std::cout<<check.dump()<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
