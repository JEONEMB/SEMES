// 2단계: 측정 단계만 GN(채택)·ECC·LK로 바꾸고, 채택 control과 같은 P12 → 6후보·부호 게이트 학습을 같은 고정 4-fold로 다시 해서 Y/T 결과를 비교한다.
//  - 측정: 1단계 Methods (같은 630점, 같은 설정). 특징: 고정 536점 변위 − 같은 방법의 center 측정 → P12 (채택 Tracker::features와 같은 식)
//  - 학습: P12/J204 비교 실험의 train_yort (원본 fold 모델 1e-6 재현 확인된 C++ 학습기)
//  - 보조: 채택 원본 모델(GN 특징으로 학습)에 다른 측정을 그대로 넣은 '모델 호환성' 결과
#include "bench.hpp"
#include "control_internal.hpp"
#include "experiment.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <regex>
#include <sstream>
namespace fs=std::filesystem;
using Clock=std::chrono::steady_clock;
namespace {
const double kNan=std::numeric_limits<double>::quiet_NaN();
const double kTol=.09;
struct Variant {const char* name;int method;};
// 1단계 Methods 번호: 6=채택 GN, 7=ECC, 4=LK(1단)
const Variant kVariants[]={{"GN(채택측정기)",6},{"ECC",7},{"LK(1단)",4}};
const int V=3;
double ms_since(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
double quantile(std::vector<double> v,double q){if(v.empty())return kNan;std::sort(v.begin(),v.end());return v[std::min(v.size()-1,size_t(q*(v.size()-1)+.5))];}
std::vector<std::string> split(const std::string& text){std::vector<std::string> out;std::stringstream s(text);std::string x;while(std::getline(s,x,',')){if(!x.empty()&&x.back()=='\r')x.pop_back();out.push_back(x);}return out;}

struct Sample {std::string file,date,axis,session,group;fs::path path;double y=0,t=0,value=0;int fold=-1;
    cv::Mat feature[V];bool valid[V]={};std::string error[V];double ms[V]={};int used_points[V]={};};
// 20261014 280장은 채택 control의 고정 fold(검증 기록)를 그대로 쓴다. 20261020 60장은 날짜 간 참고 평가다.
std::vector<Sample> samples(const fs::path& root,const fs::path& control){
    std::map<std::string,Sample> labels;std::ifstream f(control/"validation/python_predictions.csv");study::require(bool(f),"Missing frozen labels/folds");std::string line;std::getline(f,line);
    while(std::getline(f,line)){auto v=split(line);study::require(v.size()==14,"Invalid frozen label row");if(v[12]!="raw_oof4")continue;
        Sample s;s.file=v[0];s.axis=v[1];s.value=std::stod(v[2]);s.session=v[3];s.fold=std::stoi(v[13]);s.date="20261014";s.group=std::string("1014_")+s.axis;
        if(s.axis=="Y")s.y=s.value;else s.t=s.value;s.path=root/"clean_20261014"/(s.axis=="Y"?"Y":"Theta")/fs::u8path(s.file);labels[s.file]=s;}
    study::require(labels.size()==280,"Expected frozen280 split");std::vector<Sample> data;for(auto& e:labels)data.push_back(e.second);
    std::vector<fs::path> paths;for(auto& e:fs::recursive_directory_iterator(root/"clean_20261020"))if(e.path().extension()==L".bmp")paths.push_back(e.path());std::sort(paths.begin(),paths.end());
    study::require(paths.size()==60,"Expected60 cross-date images");std::regex yr("Y([+-][0-9]+(?:\\.[0-9]+)?)"),tr("T([+-][0-9]+(?:\\.[0-9]+)?)");
    for(auto& p:paths){Sample s;s.path=p;s.file=p.parent_path().filename().u8string()+"/"+p.filename().u8string();s.date="20261020";s.axis="mixed";auto name=p.stem().u8string();std::smatch m;
        if(std::regex_search(name,m,yr))s.y=std::stod(m[1]);if(std::regex_search(name,m,tr))s.t=std::stod(m[1]);
        s.group=s.y==0&&s.t==0?"1020_센터":std::abs(s.t)>=2?"1020_혼합_T2.00":std::abs(s.t)<=kTol?"1020_혼합_absT≤0.09":"1020_혼합_0.09<absT≤1";data.push_back(s);}
    Sample c;c.file="center.bmp";c.date="20261014";c.axis="center";c.group="center";c.path=root/"clean_20261014/center.bmp";data.push_back(c);return data;
}
// 고정 536점의 (변위 − center 변위)에 판 전체 2차식을 맞춘 12개 계수. Tracker::features와 같은 순서(행 우선 6×2).
cv::Mat p12(const bench::FixedPoints& pts,const bench::Estimate& e,const bench::Estimate& zero,int& used){
    std::vector<int> idx;for(size_t i=0;i<pts.p.size();++i)if(pts.selected[i]&&e.ok[i]&&zero.ok[i])idx.push_back(int(i));used=int(idx.size());study::require(idx.size()>=12,"Too few valid points for P12");
    cv::Mat basis(int(idx.size()),6,CV_64F),diff(int(idx.size()),2,CV_64F),c;
    for(int r=0;r<int(idx.size());++r){int i=idx[r];double u=(pts.p[i].x-1906)/486,v=(pts.p[i].y-1070)/122,b[]={1,u,v,u*u,u*v,v*v};for(int j=0;j<6;++j)basis.at<double>(r,j)=b[j];
        diff.at<double>(r,0)=double(float(e.d[i].x))-double(float(zero.d[i].x));diff.at<double>(r,1)=double(float(e.d[i].y))-double(float(zero.d[i].y));}
    study::require(cv::solve(basis,diff,c,cv::DECOMP_SVD),"P12 solve failed");cv::Mat f=c.reshape(1,1).clone();study::require(cv::checkRange(f),"Nonfinite P12");return f;
}
// cv::FileStorage는 한글 경로를 열지 못하므로 ASCII 임시 경로에 쓰고 복사한다.
void save_model(const study::Model& m,const fs::path& target){auto tmp=fs::temp_directory_path()/"yt_impact_model.json";m.save(tmp);fs::copy_file(tmp,target,fs::copy_options::overwrite_existing);fs::remove(tmp);}
struct Stats{int n=0,failed=0,pass=0,axis_ok=0,sign_ok=0,axis_n=0;std::vector<double> ey,et;};
std::string stats_row(const Stats& s){auto mean=[](const std::vector<double>& v){double a=0;for(auto x:v)a+=x;return v.empty()?kNan:a/v.size();};
    auto mx=[](const std::vector<double>& v){return v.empty()?kNan:*std::max_element(v.begin(),v.end());};std::ostringstream o;o<<std::setprecision(6);
    o<<s.n<<','<<s.failed<<','<<mean(s.ey)<<','<<mx(s.ey)<<','<<mean(s.et)<<','<<mx(s.et)<<','<<s.pass<<','<<(s.axis_n?std::to_string(s.axis_ok)+"/"+std::to_string(s.axis_n):"")<<','<<(s.axis_n?std::to_string(s.sign_ok)+"/"+std::to_string(s.axis_n):"");return o.str();}
}
int main(int argc,char** argv){
    try{
        fs::path repo=fs::u8path(REPO_DIR),experiment=fs::u8path(EXPERIMENT_DIR),results=argc>1?fs::u8path(argv[1]):experiment/"results";
        if(fs::exists(results))throw std::runtime_error("results already exists; refusing to overwrite: "+results.u8string());
        cv::setNumThreads(1);
        auto root=repo/"data/clean",control=repo/fs::u8path(u8"experiments/hye/00000_final_Y_T control");
        auto data=samples(root,control);cv::Mat center=legacy_control::read_gray(control/"assets/center.bmp");auto pts=bench::load_points(control/"assets/points.json");
        bench::Methods methods(center,pts,control/"assets");std::vector<bench::Estimate> zero;for(auto& v:kVariants)zero.push_back(methods.run(v.method,center));
        fs::create_directories(results/"models");
        // 측정 (사진마다 방법 순서를 돌려가며 측정해 시간 순서 편향을 줄인다)
        for(size_t i=0;i<data.size();++i){auto& s=data[i];auto image=legacy_control::read_gray(s.path);
            for(int k=0;k<V;++k){int v=int((i+k)%V);auto t0=Clock::now();
                try{auto e=methods.run(kVariants[v].method,image);s.feature[v]=p12(pts,e,zero[v],s.used_points[v]);s.valid[v]=true;}catch(const std::exception& ex){s.error[v]=ex.what();}
                s.ms[v]=ms_since(t0);}
            if((i+1)%20==0)std::cout<<"measured "<<i+1<<"/"<<data.size()<<"\n"<<std::flush;}
        {auto tmp=fs::temp_directory_path()/"yt_impact_measurements.yml";{cv::FileStorage f(tmp.string(),cv::FileStorage::WRITE);study::require(f.isOpened(),"Cannot write measurements");f<<"samples"<<"[";
            for(auto& s:data){f<<"{"<<"file"<<s.file<<"date"<<s.date<<"image_sha256"<<legacy_control::sha256(s.path);for(int v=0;v<V;++v){std::string k="m"+std::to_string(v);f<<(k+"_valid")<<int(s.valid[v])<<(k+"_p12")<<s.feature[v]<<(k+"_points")<<s.used_points[v]<<(k+"_ms")<<s.ms[v]<<(k+"_error")<<s.error[v];}f<<"}";}f<<"]";}fs::copy_file(tmp,results/"measurements.yml");fs::remove(tmp);}
        // 고정 center 특징 차감 (P12/J204 비교 실험과 같은 처리. center 자기 측정이라 0에 가깝다)
        for(int v=0;v<V;++v){study::require(data.back().valid[v],"Center measurement failed");auto z=data.back().feature[v].clone();for(auto& s:data)if(s.valid[v])s.feature[v]-=z;}

        std::vector<int> all,transfer;for(int i=0;i<280;++i)all.push_back(i);for(int i=280;i<340;++i)transfer.push_back(i);
        legacy_control::Model frozen(control/"assets/model.json");
        std::ofstream rows(results/"predictions.csv");rows<<std::setprecision(10)<<"variant,protocol,fold,date,file,group,Y,T,predY,predT,Y_error,T_error,valid,both_pass,pred_axis,branch,used_points\n";
        std::map<std::string,Stats> summary;std::map<std::string,std::vector<double>> both_err;// variant|file -> max(eY,eT) for paired diffs
        auto evaluate=[&](const std::string& variant,const std::string& protocol,int fold,int v,const std::vector<int>& ids,const std::function<study::Prediction(const cv::Mat&)>& predict){
            for(int i:ids){const auto& s=data[i];std::vector<std::string> groups{protocol,protocol+"_"+s.group};if(fold>=0)groups.push_back(protocol+"_fold"+std::to_string(fold));
                bool ok=false;study::Prediction q;double ey=0,et=0;
                try{study::require(s.valid[v],"measurement failed: "+s.error[v]);q=predict(s.feature[v]);ey=std::abs(q.y-s.y);et=std::abs(q.t-s.t);ok=std::isfinite(ey)&&std::isfinite(et);}catch(const std::exception&){}
                bool pass=ok&&ey<=kTol&&et<=kTol;
                rows<<variant<<','<<protocol<<','<<fold<<','<<s.date<<",\""<<s.file<<"\","<<s.group<<','<<s.y<<','<<s.t<<',';
                if(ok)rows<<q.y<<','<<q.t<<','<<ey<<','<<et<<",1,"<<pass<<','<<q.axis<<','<<q.branch;else rows<<",,,,0,0,,";rows<<','<<s.used_points[v]<<"\n";
                if(protocol=="oof4")both_err[variant+"|"+s.file].push_back(ok?std::max(ey,et):kNan);
                for(auto& g:groups){auto& a=summary[variant+","+g];++a.n;if(!ok){++a.failed;continue;}a.ey.push_back(ey);a.et.push_back(et);a.pass+=pass;
                    if(s.axis=="Y"||s.axis=="T"){++a.axis_n;a.axis_ok+=q.axis==s.axis;double pv=q.axis=="Y"?q.y:q.t;a.sign_ok+=q.axis==s.axis&&pv*s.value>0;}}}};
        std::ofstream audit(results/"audit.json");audit<<std::setprecision(12)<<"{\n";
        auto as_study=[&](int v){std::vector<study::Sample> out;for(auto& s:data){study::Sample t;t.file=s.file;t.date=s.date;t.axis=s.axis;t.session=s.session;t.y=s.y;t.t=s.t;t.value=s.value;t.fold=s.fold;t.feature[0]=s.feature[v];t.valid[0]=s.valid[v];t.error[0]=s.error[v];out.push_back(t);}return out;};
        for(int v=0;v<V;++v){std::string name=kVariants[v].name;auto sd=as_study(v);double repro=0;int repro_axis=0;std::string train_error;
            try{
                for(int fold=0;fold<4;++fold){std::vector<int> train,test;for(int i:all)(data[i].fold==fold?test:train).push_back(i);study::require(train.size()==210&&test.size()==70,"Unexpected split");
                    auto model=study::train_yort(sd,train,0);save_model(model,results/"models"/(std::string("m")+std::to_string(v)+"_fold"+std::to_string(fold)+".json"));
                    evaluate(name,"oof4",fold,v,test,[&](const cv::Mat& x){return model.predict(x);});
                    // GN은 원본 fold 모델과 같은 예측이 나와야 한다 (측정·학습 재현 확인)
                    if(kVariants[v].method==6){legacy_control::Model ref(control/"validation"/("fold"+std::to_string(fold)+"_model.json"));
                        for(int i:test){if(!data[i].valid[v])continue;auto q=model.predict(data[i].feature[v]);auto r=ref.predict(data[i].feature[v]);repro=std::max(repro,std::abs((q.axis=="Y"?q.y:q.t)-r.value));repro_axis+=q.axis!=r.axis;}}}
                auto full=study::train_yort(sd,all,0);save_model(full,results/"models"/(std::string("m")+std::to_string(v)+"_full.json"));
                evaluate(name,"resub",-1,v,all,[&](const cv::Mat& x){return full.predict(x);});
                evaluate(name,"transfer20261020",-1,v,transfer,[&](const cv::Mat& x){return full.predict(x);});
            }catch(const std::exception& e){train_error=e.what();}
            // 모델 호환성: 채택 원본 모델(GN 특징으로 학습)에 이 측정의 특징을 그대로 넣는다
            evaluate(name,"frozen_model",-1,v,all,[&](const cv::Mat& x){auto c=frozen.predict(x);study::Prediction q;q.axis=c.axis;q.branch=c.branch;q.cost=c.cost;(c.axis=="Y"?q.y:q.t)=c.value;return q;});
            std::vector<double> ms;for(int i=0;i<280;++i)ms.push_back(data[i].ms[v]);
            audit<<"  \""<<name<<"\": {\"method_index\": "<<kVariants[v].method<<", \"train_error\": \""<<train_error<<"\", \"measure_ms_median_1014\": "<<quantile(ms,.5)<<", \"measure_ms_p95_1014\": "<<quantile(ms,.95);
            if(kVariants[v].method==6)audit<<", \"frozen_fold_max_difference\": "<<repro<<", \"frozen_fold_axis_mismatch\": "<<repro_axis;
            audit<<"}"<<(v+1<V?",":"")<<"\n";std::cout<<"done "<<name<<"\n";}
        audit<<"}\n";audit.close();
        std::ofstream sum(results/"summary.csv");sum<<"variant,protocol,n,failed,Y_mae,Y_max,T_mae,T_max,both_pass,axis_correct,sign_correct\n";
        for(auto& e:summary)sum<<e.first<<','<<stats_row(e.second)<<"\n";
        // 같은 사진의 OOF 두 축 오차 최댓값을 방법끼리 비교
        std::ofstream pair(results/"paired_oof.csv");pair<<std::setprecision(8)<<"file";for(auto& v:kVariants)pair<<','<<v.name;pair<<"\n";
        for(int i:all){pair<<'"'<<data[i].file<<'"';for(auto& v:kVariants){auto it=both_err.find(std::string(v.name)+"|"+data[i].file);pair<<','<<(it==both_err.end()||it->second.empty()?kNan:it->second[0]);}pair<<"\n";}
        std::cout<<"done: "<<results.u8string()<<"\n";
        return 0;
    }catch(const std::exception& e){std::cerr<<"error: "<<e.what()<<"\n";return 1;}
}
