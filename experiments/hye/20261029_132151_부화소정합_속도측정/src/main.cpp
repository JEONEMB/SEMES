// 3단계 정식 속도 측정. 같은 PC·Release·OpenCV 1스레드 기준, 기본 스레드는 따로 잰다.
//  - 초기화 5회, cold(초기화 직후 첫 호출) 5회, warm-up 10장, 20261014 280장 전체를 5회 반복(매회 사진·방법 순서를 고정 seed로 섞음)
//  - 단계: 파일 읽기(BMP→회색), 측정(방법별), P12 특징, 모델 추론. 사진별로 반복 중앙값을 낸 뒤 전체 중앙값/p95/최대를 낸다.
//  - 측정은 사진의 kCrop만 메모리에 두고 고정 캔버스에 붙여 넣어 한다. 모든 방법이 kCrop 안만 읽으므로 결과가 원본 영상과 같은지 처음에 확인한다.
#include "bench.hpp"
#include "control_internal.hpp"
#include "experiment.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <thread>
namespace fs=std::filesystem;
using Clock=std::chrono::steady_clock;
namespace {
const double kNan=std::numeric_limits<double>::quiet_NaN();
const int kInit=5,kWarm=10,kPasses=5,kPassesDefaultThreads=2;
// 2단계 모델 파일 번호(m0/m1/m2)와 1단계 Methods 번호
struct Pipeline {const char* name;int method;const char* model;};
const Pipeline kPipelines[]={{"GN(채택측정기)",6,"m0_full.json"},{"ECC",7,"m1_full.json"},{"LK(1단)",4,"m2_full.json"}};
double ms_since(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
double quantile(std::vector<double> v,double q){if(v.empty())return kNan;std::sort(v.begin(),v.end());return v[std::min(v.size()-1,size_t(q*(v.size()-1)+.5))];}
std::string num(double v,int p=3){if(!std::isfinite(v))return "";std::ostringstream o;o<<std::fixed<<std::setprecision(p)<<v;return o.str();}
// 2단계 main.cpp의 p12()와 같은 식
cv::Mat p12(const bench::FixedPoints& pts,const bench::Estimate& e,const bench::Estimate& zero){
    std::vector<int> idx;for(size_t i=0;i<pts.p.size();++i)if(pts.selected[i]&&e.ok[i]&&zero.ok[i])idx.push_back(int(i));study::require(idx.size()>=12,"Too few valid points for P12");
    cv::Mat basis(int(idx.size()),6,CV_64F),diff(int(idx.size()),2,CV_64F),c;
    for(int r=0;r<int(idx.size());++r){int i=idx[r];double u=(pts.p[i].x-1906)/486,v=(pts.p[i].y-1070)/122,b[]={1,u,v,u*u,u*v,v*v};for(int j=0;j<6;++j)basis.at<double>(r,j)=b[j];
        diff.at<double>(r,0)=double(float(e.d[i].x))-double(float(zero.d[i].x));diff.at<double>(r,1)=double(float(e.d[i].y))-double(float(zero.d[i].y));}
    cv::solve(basis,diff,c,cv::DECOMP_SVD);return c.reshape(1,1).clone();
}
study::Model load_model(const fs::path& file){auto tmp=fs::temp_directory_path()/"subpixel_speed_model.json";fs::copy_file(file,tmp,fs::copy_options::overwrite_existing);auto m=study::Model::load(tmp);fs::remove(tmp);return m;}
}
int main(int argc,char** argv){
    try{
        fs::path repo=fs::u8path(REPO_DIR),experiment=fs::u8path(EXPERIMENT_DIR),results=argc>1?fs::u8path(argv[1]):experiment/"results";
        bool quick=argc>2&&std::string(argv[2])=="quick";
        if(fs::exists(results))throw std::runtime_error("results already exists; refusing to overwrite: "+results.u8string());
        cv::setNumThreads(1);
        auto assets=repo/fs::u8path(u8"experiments/hye/00000_final_Y_T control/assets"),models=repo/fs::u8path(u8"experiments/hye/20261029_131112_부화소정합_YT결과영향/results/models");
        std::vector<fs::path> files;for(auto sub:{"Y","Theta"})for(auto& e:fs::directory_iterator(repo/"data/clean/clean_20261014"/sub))if(e.path().extension()==".bmp")files.push_back(e.path());
        std::sort(files.begin(),files.end());if(quick)files.resize(20);
        cv::Mat center=legacy_control::read_gray(assets/"center.bmp");auto pts=bench::load_points(assets/"points.json");const int M=bench::Methods::count();
        fs::create_directories(results);std::ofstream raw(results/"raw_times.csv");raw<<"threads,phase,pass,photo,method,stage,ms\n";
        auto log=[&](const std::string& threads,const std::string& phase,int pass,const std::string& photo,const std::string& method,const std::string& stage,double ms){raw<<threads<<','<<phase<<','<<pass<<",\""<<photo<<"\",\""<<method<<"\","<<stage<<','<<std::setprecision(6)<<ms<<"\n";};

        // 초기화: 채택 측정기 단독(Tracker), 비교용 9방법 전체(Methods), 2단계 모델 읽기. 각각 5회.
        for(int k=0;k<kInit;++k){auto t0=Clock::now();legacy_control::Tracker t(assets);log("1","init",k,"","GN(채택측정기)","tracker_init",ms_since(t0));}
        std::unique_ptr<bench::Methods> methods;
        for(int k=0;k<kInit;++k){auto t0=Clock::now();methods=std::make_unique<bench::Methods>(center,pts,assets);log("1","init",k,"","전체9방법","methods_init",ms_since(t0));
            // cold: 초기화 직후 첫 사진 한 장의 첫 호출
            auto first=legacy_control::read_gray(files[0]);for(int m=0;m<M;++m){auto t1=Clock::now();methods->run(m,first);log("1","cold",k,files[0].filename().u8string(),bench::Methods::name(m),"measure",ms_since(t1));}}
        std::vector<study::Model> model;std::vector<bench::Estimate> zero;
        for(auto& p:kPipelines){for(int k=0;k<kInit;++k){auto t0=Clock::now();auto mm=load_model(models/p.model);log("1","init",k,"",p.name,"model_load",ms_since(t0));}
            model.push_back(load_model(models/p.model));
            for(int k=0;k<kInit;++k){auto t0=Clock::now();auto z=methods->run(p.method,center);log("1","init",k,"",p.name,"center_measure",ms_since(t0));if(k==0)zero.push_back(z);}}

        // 사진 kCrop만 메모리에 두고 측정 때 고정 캔버스에 붙인다. 처음 3장은 원본 영상과 결과가 같은지 확인.
        std::vector<cv::Mat> crops;for(auto& f:files)crops.push_back(legacy_control::read_gray(f)(bench::kCrop).clone());
        cv::Mat canvas=center.clone();double equivalence=0;
        for(int i=0;i<3;++i){auto full=legacy_control::read_gray(files[i]);crops[i].copyTo(canvas(bench::kCrop));
            for(int m=0;m<M;++m){auto a=methods->run(m,full),b=methods->run(m,canvas);for(size_t j=0;j<a.d.size();++j)if(a.ok[j]||b.ok[j])equivalence=std::max(equivalence,a.ok[j]!=b.ok[j]?1e9:cv::norm(a.d[j]-b.d[j]));}}
        study::require(equivalence==0,"Canvas measurement differs from full image");

        auto pass_over=[&](const std::string& threads,int passes,unsigned seed_base){
            // warm-up
            for(int i=0;i<std::min<int>(kWarm,int(files.size()));++i){crops[i].copyTo(canvas(bench::kCrop));for(int m=0;m<M;++m)methods->run(m,canvas);}
            for(int pass=0;pass<passes;++pass){std::mt19937 rng(seed_base+pass);std::vector<int> order(files.size());for(size_t i=0;i<order.size();++i)order[i]=int(i);std::shuffle(order.begin(),order.end(),rng);
                for(int i:order){auto name=files[i].filename().u8string();
                    {auto t0=Clock::now();auto g=legacy_control::read_gray(files[i]);log(threads,"pass",pass,name,"공통","decode",ms_since(t0));}
                    crops[i].copyTo(canvas(bench::kCrop));std::vector<int> ms(M);for(int m=0;m<M;++m)ms[m]=m;std::shuffle(ms.begin(),ms.end(),rng);
                    for(int m:ms){auto t0=Clock::now();auto e=methods->run(m,canvas);log(threads,"pass",pass,name,bench::Methods::name(m),"measure",ms_since(t0));
                        for(size_t p=0;p<3;++p)if(kPipelines[p].method==m){auto t1=Clock::now();auto f=p12(pts,e,zero[p]);log(threads,"pass",pass,name,kPipelines[p].name,"p12",ms_since(t1));
                            auto t2=Clock::now();auto q=model[p].predict(f);log(threads,"pass",pass,name,kPipelines[p].name,"predict",ms_since(t2));(void)q;}}}
                std::cout<<"threads "<<threads<<" pass "<<pass+1<<"/"<<passes<<"\n"<<std::flush;}};
        pass_over("1",kPasses,20261029u);
        cv::setNumThreads(-1);std::string def=std::to_string(cv::getNumThreads());pass_over("기본("+def+")",kPassesDefaultThreads,20261129u);
        raw.close();

        // 요약: 사진별 반복 중앙값 → 전체 중앙값/p95/최대
        std::map<std::string,std::map<std::string,std::vector<double>>> per;// key threads|method|stage -> photo -> ms list
        std::map<std::string,std::vector<double>> once;// init / cold
        {std::ifstream in(results/"raw_times.csv");std::string line;std::getline(in,line);
            while(std::getline(in,line)){std::vector<std::string> v;std::string cur;bool q=false;for(char ch:line){if(ch=='"'){q=!q;continue;}if(ch==','&&!q){v.push_back(cur);cur.clear();}else cur+=ch;}v.push_back(cur);
                double ms=std::stod(v[6]);if(v[1]=="pass")per[v[0]+"|"+v[4]+"|"+v[5]][v[3]].push_back(ms);else once[v[0]+"|"+v[1]+"|"+v[4]+"|"+v[5]].push_back(ms);}}
        std::ofstream sum(results/"summary_speed.csv");sum<<"threads,method,stage,photos,median_ms,p95_ms,max_ms\n";
        std::map<std::string,std::map<std::string,double>> photo_median;
        for(auto& e:per){std::vector<double> med;for(auto& ph:e.second){double m=quantile(ph.second,.5);med.push_back(m);photo_median[e.first][ph.first]=m;}
            auto a=e.first.find('|'),b=e.first.rfind('|');sum<<e.first.substr(0,a)<<",\""<<e.first.substr(a+1,b-a-1)<<"\","<<e.first.substr(b+1)<<','<<med.size()<<','<<num(quantile(med,.5))<<','<<num(quantile(med,.95))<<','<<num(quantile(med,1))<<"\n";}
        // 파이프라인 합계 (읽기 + 측정 + P12 + 추론): 같은 사진·같은 반복의 단계 합을 사진별 중앙값으로
        // pipeline_in_memory: 운용 API처럼 영상이 이미 메모리에 있을 때 (측정 + P12 + 추론). pipeline_with_decode: BMP 파일 읽기까지.
        for(auto threads:{std::string("1"),"기본("+def+")"})for(auto& p:kPipelines){std::vector<double> mem,tot;
            for(auto& ph:photo_median[threads+"|공통|decode"]){double s=0;for(auto st:{std::string(p.name)+"|p12",std::string(p.name)+"|predict"}){s+=photo_median[threads+"|"+st][ph.first];}
                std::string mname;for(int m=0;m<M;++m)if(m==p.method)mname=bench::Methods::name(m);s+=photo_median[threads+"|"+mname+"|measure"][ph.first];mem.push_back(s);tot.push_back(s+ph.second);}
            sum<<threads<<",\""<<p.name<<"\",pipeline_in_memory,"<<mem.size()<<','<<num(quantile(mem,.5))<<','<<num(quantile(mem,.95))<<','<<num(quantile(mem,1))<<"\n";
            sum<<threads<<",\""<<p.name<<"\",pipeline_with_decode,"<<tot.size()<<','<<num(quantile(tot,.5))<<','<<num(quantile(tot,.95))<<','<<num(quantile(tot,1))<<"\n";}
        std::ofstream so(results/"summary_init_cold.csv");so<<"key,n,median_ms,max_ms\n";for(auto& e:once)so<<'"'<<e.first<<"\","<<e.second.size()<<','<<num(quantile(e.second,.5))<<','<<num(quantile(e.second,1))<<"\n";
        std::ofstream env(results/"environment.json");
        env<<"{\n  \"cpu\": \"Intel(R) Core(TM) Ultra 9 185H\",\n  \"logical_processors\": "<<std::thread::hardware_concurrency()<<",\n  \"opencv\": \""<<CV_VERSION<<"\",\n  \"default_threads\": "<<def<<",\n  \"build\": \"Release, MSVC v141\",\n  \"power\": \"노트북, 실행 직전 배터리 상태 1(방전 중)·충전 95%, 전원 구성표 균형 조정. 성능/효율 코어 혼합 CPU라 스레드 배치를 고정하지 않았다\",\n  \"concurrent_load\": \"다른 프로그램(편집기·브라우저·다른 에이전트)이 함께 실행 중. 실행 직전 전체 CPU 약 13%\",\n  \"photos\": "<<files.size()<<",\n  \"passes_1thread\": "<<kPasses<<",\n  \"passes_default_threads\": "<<kPassesDefaultThreads
           <<",\n  \"warmup_photos\": "<<kWarm<<",\n  \"init_repeats\": "<<kInit<<",\n  \"canvas_equivalence_max_diff\": "<<equivalence<<",\n  \"seed\": 20261029,\n  \"quick\": "<<(quick?"true":"false")<<"\n}\n";
        std::cout<<"done: "<<results.u8string()<<"\n";return 0;
    }catch(const std::exception& e){std::cerr<<"error: "<<e.what()<<"\n";return 1;}
}
