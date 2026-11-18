// 2단계 보조: results/models의 전체 학습 모델(20261014 280장)을 20261029 혼합 234장에 그대로 적용한다.
// 이 자료는 모두 Y +0.05에 T 0.04~0.82를 함께 준 혼합 자세라 control(Y-or-T)의 학습 범위 밖이다. 날짜 간 참고 평가로만 쓴다.
#include "bench.hpp"
#include "control_internal.hpp"
#include "experiment.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <regex>
#include <sstream>
namespace fs=std::filesystem;
namespace {
struct Variant {const char* name;int method;const char* model;};
const Variant kVariants[]={{"GN(채택측정기)",6,"m0_full.json"},{"ECC",7,"m1_full.json"},{"LK(1단)",4,"m2_full.json"}};
// 2단계 main.cpp의 p12()와 같은 식
cv::Mat p12(const bench::FixedPoints& pts,const bench::Estimate& e,const bench::Estimate& zero){
    std::vector<int> idx;for(size_t i=0;i<pts.p.size();++i)if(pts.selected[i]&&e.ok[i]&&zero.ok[i])idx.push_back(int(i));study::require(idx.size()>=12,"Too few valid points for P12");
    cv::Mat basis(int(idx.size()),6,CV_64F),diff(int(idx.size()),2,CV_64F),c;
    for(int r=0;r<int(idx.size());++r){int i=idx[r];double u=(pts.p[i].x-1906)/486,v=(pts.p[i].y-1070)/122,b[]={1,u,v,u*u,u*v,v*v};for(int j=0;j<6;++j)basis.at<double>(r,j)=b[j];
        diff.at<double>(r,0)=double(float(e.d[i].x))-double(float(zero.d[i].x));diff.at<double>(r,1)=double(float(e.d[i].y))-double(float(zero.d[i].y));}
    cv::solve(basis,diff,c,cv::DECOMP_SVD);return c.reshape(1,1).clone();
}
study::Model load_model(const fs::path& file){auto tmp=fs::temp_directory_path()/"yt_transfer_model.json";fs::copy_file(file,tmp,fs::copy_options::overwrite_existing);auto m=study::Model::load(tmp);fs::remove(tmp);return m;}
std::string bin(double t){t=std::abs(t);return t<=.09?"absT≤0.09":t<=.3?"0.09<absT≤0.3":"absT>0.3";}
}
int main(){
    try{
        fs::path repo=fs::u8path(REPO_DIR),experiment=fs::u8path(EXPERIMENT_DIR),results=experiment/"results_transfer20261029";
        if(fs::exists(results))throw std::runtime_error("results already exists; refusing to overwrite: "+results.u8string());
        cv::setNumThreads(1);
        auto control=repo/fs::u8path(u8"experiments/hye/20261028_1327_final_Y_T control/assets");auto pts=bench::load_points(control/"points.json");
        cv::Mat center=legacy_control::read_gray(control/"center.bmp");bench::Methods methods(center,pts,control);
        std::vector<study::Model> models;std::vector<bench::Estimate> zero;std::vector<cv::Mat> center_feature;
        for(auto& v:kVariants){models.push_back(load_model(experiment/"results/models"/v.model));zero.push_back(methods.run(v.method,center));center_feature.push_back(p12(pts,zero.back(),zero.back()));}
        std::vector<fs::path> files;for(auto& e:fs::recursive_directory_iterator(repo/"data/clean/clean_20261029"))if(e.path().extension()==".bmp")files.push_back(e.path());std::sort(files.begin(),files.end());
        study::require(files.size()==234,"Expected234 images");
        fs::create_directories(results);std::ofstream rows(results/"predictions.csv");rows<<std::setprecision(10)<<"variant,file,pose,bin,Y,T,predY,predT,Y_error,T_error,valid,both_pass,pred_axis,branch\n";
        struct S{int n=0,failed=0,pass=0,t_axis=0;std::vector<double> ey,et;};std::map<std::string,S> sum;std::map<std::string,std::vector<double>> pose_values;
        std::regex yr("Y([+-][0-9]+(?:\\.[0-9]+)?)"),tr("T([+-][0-9]+(?:\\.[0-9]+)?)");int index=0;
        for(auto& f:files){auto image=legacy_control::read_gray(f);auto pose=f.parent_path().filename().u8string();std::smatch m;double y=0,t=0;
            if(std::regex_search(pose,m,yr))y=std::stod(m[1]);if(std::regex_search(pose,m,tr))t=std::stod(m[1]);
            for(int v=0;v<3;++v){bool ok=false;study::Prediction q;double ey=0,et=0;
                try{auto e=methods.run(kVariants[v].method,image);q=models[v].predict(p12(pts,e,zero[v])-center_feature[v]);ey=std::abs(q.y-y);et=std::abs(q.t-t);ok=true;}catch(const std::exception&){}
                bool pass=ok&&ey<=.09&&et<=.09;
                rows<<kVariants[v].name<<",\""<<pose+"/"+f.filename().u8string()<<"\",\""<<pose<<"\","<<bin(t)<<','<<y<<','<<t<<',';
                if(ok)rows<<q.y<<','<<q.t<<','<<ey<<','<<et<<",1,"<<pass<<','<<q.axis<<','<<q.branch;else rows<<",,,,0,0,,";rows<<"\n";
                for(auto g:{std::string("전체"),bin(t)}){auto& s=sum[std::string(kVariants[v].name)+","+g];++s.n;if(!ok){++s.failed;continue;}s.pass+=pass;s.t_axis+=q.axis=="T";s.ey.push_back(ey);s.et.push_back(et);}
                if(ok)pose_values[std::string(kVariants[v].name)+"|"+pose].push_back(q.axis=="T"?q.t:1e9);}
            if(++index%20==0)std::cout<<"measured "<<index<<"/234\n"<<std::flush;}
        std::ofstream so(results/"summary.csv");so<<std::setprecision(6)<<"variant,group,n,failed,both_pass,pred_T_axis,Y_mae,Y_max,T_mae,T_max\n";
        auto mean=[](const std::vector<double>& v){double a=0;for(auto x:v)a+=x;return v.empty()?NAN:a/v.size();};auto mx=[](const std::vector<double>& v){return v.empty()?NAN:*std::max_element(v.begin(),v.end());};
        for(auto& e:sum)so<<e.first<<','<<e.second.n<<','<<e.second.failed<<','<<e.second.pass<<','<<e.second.t_axis<<','<<mean(e.second.ey)<<','<<mx(e.second.ey)<<','<<mean(e.second.et)<<','<<mx(e.second.et)<<"\n";
        // 같은 명령 반복 촬영(대부분 3장)끼리 T 예측값의 범위. T 축으로 판정되지 않은 사진이 섞이면 비움.
        std::ofstream sp(results/"repeat_spread.csv");sp<<std::setprecision(6)<<"variant,pose,n,T_pred_range\n";
        for(auto& e:pose_values){auto bar=e.first.find('|');double lo=*std::min_element(e.second.begin(),e.second.end()),hi=*std::max_element(e.second.begin(),e.second.end());
            sp<<e.first.substr(0,bar)<<",\""<<e.first.substr(bar+1)<<"\","<<e.second.size()<<','<<(hi>1e8?std::string(""):std::to_string(hi-lo))<<"\n";}
        std::cout<<"done: "<<results.u8string()<<"\n";return 0;
    }catch(const std::exception& e){std::cerr<<"error: "<<e.what()<<"\n";return 1;}
}
