#include "alignment.hpp"
#include <iostream>
#include <fstream>
#include <regex>
#include <chrono>
#include <set>
#include <numeric>
using namespace semes;
static void measurements(const Engine& e,const Measurement& m,const fs::path& out){
    std::ostringstream s;
    s<<std::setprecision(17)<<"id,region,x,y,dx,dy,valid,flags,rmse,stability,contrast,sigma_x,sigma_y\n";
    for(int i=0; i<630; i++){
        auto& c=e.corners()[i];
        s<<c.id<<','<<c.region<<','<<m.current.at<double>(i,0)<<','<<m.current.at<double>(i,1)<<','<<m.displacement.at<double>(i,0)<<','<<m.displacement.at<double>(i,1)<<','<<m.valid[i]<<','<<m.flags[i];
        for(int k=0; k<5; k++)s<<','<<m.quality.at<double>(i,k);
        s<<'\n';
    }
    write(out,s.str());
}
static cv::Mat arrmat(const Npy& a){
    if(a.shape.size()!=2)throw std::runtime_error("Expected rank-2 NPY");
    cv::Mat m(int(a.shape[0]),int(a.shape[1]),CV_64F);
    for(size_t i=0; i<a.count(); i++)m.ptr<double>()[i]=a.number(i);
    return m;
}
static Json parity(Engine& e,const fs::path& baseline,const fs::path& image,const fs::path& out){
    auto a=npz(baseline);
    Measurement m;
    auto j=e.infer(bytes(image),130,&m);
    std::string key=a.count("current_points")?"current_points":a.count("coords")?"coords":a.count("current")?"current":"D";
    cv::Mat expected=arrmat(a.at(key)),actual=key=="D"?m.displacement:m.current;
    if(expected.size()!=actual.size())throw std::runtime_error("Baseline shape mismatch");
    cv::Mat difference=actual-expected,absdiff;
    cv::absdiff(actual,expected,absdiff);
    Json audit=Json::dict();
    audit["cpp_prediction"]=j;
    audit["baseline_file"]=baseline.u8string();
    audit["baseline_sha256"]=sha256(baseline);
    audit["comparison_array"]=key;
    audit["coordinate_max_abs_px"]=cv::norm(difference,cv::NORM_INF);
    audit["coordinate_rmse_px"]=cv::norm(difference)/std::sqrt(double(difference.total()));
    if(a.count("valid")){
        int mismatch=0;
        for(int i=0; i<630; i++)mismatch+=bool(a.at("valid").number(i))!=bool(m.valid[i]);
        audit["diagnostic_mismatch_count"]=mismatch;
    }
    if(a.count("quality")){
        auto q=arrmat(a.at("quality"));
        if(q.size()==m.quality.size())audit["quality_max_abs"]=cv::norm(q-m.quality,cv::NORM_INF);
    }
    measurements(e,m,out/"measurement.csv");
    write(out/"parity.json",audit.dump());
    return audit;
}
static Json evaluate(Engine& e,const fs::path& data,const fs::path& splits,const fs::path& out,int limit){
    if(fs::exists(out))throw std::runtime_error("Evaluation output must be new");
    fs::create_directories(out);
    auto folds=Json::parse(read(splits));
    std::map<std::string,int> foldmap;
    for(auto& f:folds.array)for(auto& n:f.at("test").array){
        std::string name=n.str();
        if(!foldmap.emplace(name,int(f.at("fold").num())).second)throw std::runtime_error("Repeated test membership");
    }
    if(foldmap.size()!=280)throw std::runtime_error("Expected exact 280 OOF membership");
    std::vector<fs::path> images;
    for(auto sub:{
        "Y","Theta"
    })for(auto& ent:fs::directory_iterator(data/sub))if(ent.is_regular_file()&&ent.path().extension()==".bmp")images.push_back(ent.path());
    std::sort(images.begin(),images.end());
    if(images.size()!=280)throw std::runtime_error("Expected 280 single-axis input images");
    std::regex label("^([YT])([+-][0-9]+(?:\\.[0-9]+)?)\\.bmp$");
    std::ofstream predictions(out/"predictions.csv"),failures(out/"failures.csv");
    std::string header="file,fold,protocol,true_Y,true_T,pred_Y,pred_T,pred_axis,axis_correct,sign_correct,error_Y,error_T,both_within_009,valid,seconds,status\n";
    predictions<<header;
    failures<<header;
    std::vector<double> times;
    int n=0,ok=0,axisok=0,signok=0,failed=0;
    double sumY=0,sumT=0,maxY=0,maxT=0;
    for(auto& p:images){
        if(limit&&n>=limit)break;
        std::string name=p.filename().u8string();
        if(!foldmap.count(name))throw std::runtime_error("Image absent from fixed splits: "+name);
        std::smatch match;
        if(!std::regex_match(name,match,label))throw std::runtime_error("Invalid single-axis label: "+name);
        std::string axis=match[1];
        double value=std::stod(match[2]),y=axis=="Y"?value:0,t=axis=="T"?value:0;
        ++n;
        std::ostringstream row;
        row<<std::setprecision(17)<<csv(name)<<','<<foldmap.at(name)<<",full_training_resubstitution,"<<y<<','<<t<<',';
        try{
            Measurement m;
            auto j=e.infer(bytes(p),130,&m);
            double py=j.at("Y").num(),pt=j.at("T").num(),ey=std::abs(py-y),et=std::abs(pt-t),seconds=j.at("seconds").num();
            bool ac=j.at("axis").str()==axis,sc=ac&&std::signbit(j.at("value").num())==std::signbit(value),pass=ey<=.09&&et<=.09;
            axisok+=ac;
            signok+=sc;
            ok+=pass;
            sumY+=ey;
            sumT+=et;
            maxY=std::max(maxY,ey);
            maxT=std::max(maxT,et);
            times.push_back(seconds);
            row<<py<<','<<pt<<','<<j.at("axis").str()<<','<<ac<<','<<sc<<','<<ey<<','<<et<<','<<pass<<','<<j.at("valid").num()<<','<<seconds<<",ok\n";
            predictions<<row.str();
            if(!pass||!sc)failures<<row.str();
            write(out/"images"/fs::u8path(name+".json"),j.dump());
            measurements(e,m,out/"measurements"/fs::u8path(name+".csv"));
        }
        catch(const std::exception& ex){
            ++failed;
            row<<",,,,,,,,0,,,"<<csv(ex.what())<<'\n';
            predictions<<row.str();
            failures<<row.str();
        }
        predictions.flush();
        if(n%10==0)std::cout<<"Evaluated "<<n<<"/"<<(limit?limit:280)<<std::endl;
    }
    Json j=Json::dict();
    j["protocol"]="full_training_resubstitution";
    j["not_oof"]="Fold IDs are recorded for later training-only evaluation; this command uses the saved full-training model.";
    j["n"]=n;
    j["inference_failures"]=failed;
    j["axis_correct"]=axisok;
    j["axis_and_sign_correct"]=signok;
    j["both_within_009"]=ok;
    j["both_within_009_rate"]=double(ok)/n;
    j["Y_max_successful"]=maxY;
    j["T_max_successful"]=maxT;
    if(!failed){
        j["Y_MAE"]=sumY/n;
        j["T_MAE"]=sumT/n;
        j["Y_max"]=maxY;
        j["T_max"]=maxT;
    }
    else{
        j["Y_MAE"]=Json();
        j["T_MAE"]=Json();
        j["Y_max"]=Json();
        j["T_max"]=Json();
    }
    if(!times.empty()){
        auto sorted=times;
        std::sort(sorted.begin(),sorted.end());
        j["seconds_median"]=sorted[sorted.size()/2];
        j["seconds_mean"]=std::accumulate(times.begin(),times.end(),0.)/times.size();
        j["seconds_max"]=sorted.back();
    }
    j["created_kst"]=now();
    j["limit"]=limit;
    write(out/"summary.json",j.dump());
    return j;
}
int wmain(int argc,wchar_t** wide_argv){
    semes::Utf8Arguments unicode_arguments(argc,wide_argv);
    char** argv=unicode_arguments.data();
    try{
        if(argc<2)throw std::runtime_error("Commands: selftest | inspect npz | reference assets center | predict assets center image new-output | parity assets center baseline.npz image new-output | evaluate assets center data splits new-output [limit]");
        std::string command=argv[1];
        if(command=="selftest"){
            selftest();
            std::cout<<"Boundary Jacobian, noiseless bounded recovery, JSON Unicode and invalid input: PASS\n";
            return 0;
        }
        if(command=="inspect"){
            if(argc!=3)throw std::runtime_error("inspect requires exactly one NPZ path");
            auto a=npz(fs::u8path(argv[2]));
            for(auto& kv:a){
                std::cout<<kv.first<<": "<<kv.second.dtype<<" [";
                for(auto s:kv.second.shape)std::cout<<s<<',';
                std::cout<<"]\n";
            }
            return 0;
        }
        if(argc<4)throw std::runtime_error("Assets and fixed center required");
        Engine e(fs::u8path(argv[2]),fs::u8path(argv[3]));
        if(command=="reference"){
            Json j=Json::dict();
            j["points"]=int(e.corners().size());
            j["reference_id_region_coordinate_check"]=true;
            j["center_sha256"]=sha256(fs::u8path(argv[3]));
            std::cout<<j.dump()<<'\n';
            return 0;
        }
        if(command=="predict"&&argc==6){
            fs::path out=fs::u8path(argv[5]);
            if(fs::exists(out))throw std::runtime_error("Output directory must be new");
            Measurement m;
            auto j=e.infer(bytes(fs::u8path(argv[4])),130,&m);
            write(out/"prediction.json",j.dump());
            measurements(e,m,out/"measurement.csv");
            std::cout<<j.dump()<<'\n';
            return 0;
        }
        if(command=="parity"&&argc==7){
            fs::path out=fs::u8path(argv[6]);
            if(fs::exists(out))throw std::runtime_error("Output directory must be new");
            std::cout<<parity(e,fs::u8path(argv[4]),fs::u8path(argv[5]),out).dump()<<'\n';
            return 0;
        }
        if(command=="evaluate"&&(argc==7||argc==8)){
            std::cout<<evaluate(e,fs::u8path(argv[4]),fs::u8path(argv[5]),fs::u8path(argv[6]),argc==8?std::stoi(argv[7]):0).dump()<<'\n';
            return 0;
        }
        throw std::runtime_error("Unknown command or wrong arguments");
    }
    catch(const std::exception&e){
        std::cerr<<"ERROR: "<<e.what()<<'\n';
        return 1;
    }
}
