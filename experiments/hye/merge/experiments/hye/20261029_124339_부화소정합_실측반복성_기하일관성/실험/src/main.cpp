// 1B·1C: 실측 사진에는 점별 정답이 없으므로 두 가지 자기 일관성으로 측정 품질을 본다.
//  1C 기하 일관성: QR 하나는 평면 강체라서 630점 변위가 QR마다 아핀 하나로 설명되어야 한다. 적합 잔차 = 측정 잡음 + 실패.
//  1B 반복성: 20261020의 같은 명령 반복 3장끼리 점 변위 차이를 비교한다. 실제 자세 차이(재현 오차)는 QR별 아핀으로 빼고,
//     남은 잔차 RMS/√2 를 사진 한 장의 점 측정 잡음으로 추정한다.
// 정합 방법 9개는 1단계(20261029_122323)의 Methods를 그대로 쓴다.
#include "bench.hpp"
#include "control_internal.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
using namespace bench;
namespace {
const double kNan=std::numeric_limits<double>::quiet_NaN();
const std::string kCenterSha="01a69df6771b840d2fcf9cf727760592b5fb199c12277bfc173af514e26fd685";
const double kOutlier=1.0;   // 아핀 잔차가 이보다 크면 이상점으로 보고 다시 적합한다 (px)
using Clock=std::chrono::steady_clock;
double quantile(std::vector<double> v,double q){if(v.empty())return kNan;std::sort(v.begin(),v.end());return v[std::min(v.size()-1,size_t(q*(v.size()-1)+.5))];}
double rms(const std::vector<double>& v){if(v.empty())return kNan;double s=0;for(auto x:v)s+=x*x;return std::sqrt(s/v.size());}
std::string num(double v,int p=4){if(!std::isfinite(v))return "";std::ostringstream o;o<<std::fixed<<std::setprecision(p)<<v;return o.str();}
double ratio(long a,long b){return b?double(a)/b:kNan;}

struct Photo {std::string set,group,name;fs::path file;};
std::vector<Photo> photos(const fs::path& data){
    std::vector<Photo> out;auto d14=data/"clean_20261014",d20=data/"clean_20261020";
    for(auto sub:{"Y","Theta"})for(auto& e:fs::directory_iterator(d14/sub))if(e.path().extension()==".bmp")out.push_back({std::string("1014_")+(sub[0]=='Y'?"Y":"T"),"",e.path().filename().u8string(),e.path()});
    for(auto& g:fs::directory_iterator(d20)){if(!g.is_directory()||g.path().filename()=="_metadata")continue;auto group=g.path().filename().u8string();bool center=group==u8"기준 위치";
        // T+2.00은 20261014 범위(T ±1.00) 밖이라 따로 묶는다.
        std::string set=center?"1020_센터":group.find("T+2.00")!=std::string::npos?"1020_혼합_T2.00":"1020_혼합";
        for(auto& e:fs::directory_iterator(g.path()))if(e.path().extension()==".bmp")out.push_back({set,group,group+"/"+e.path().filename().u8string(),e.path()});}
    std::sort(out.begin(),out.end(),[](const Photo& a,const Photo& b){return a.name<b.name;});return out;
}
// QR마다 d = B·[1, (x−cx)/100, (y−cy)/100] 아핀을 맞추고, 1 px 넘는 점을 빼고 한 번 더 맞춘다. 반환은 점별 잔차 크기(NaN=쓰지 않음).
std::vector<double> affine_residual(const FixedPoints& pts,const std::vector<cv::Point2d>& d,const std::vector<unsigned char>& use){
    std::vector<double> out(pts.p.size(),kNan);
    for(int q=0;q<3;++q){std::vector<int> idx;for(size_t i=0;i<pts.p.size();++i)if(use[i]&&pts.region[i]==q&&std::isfinite(d[i].x)&&std::isfinite(d[i].y))idx.push_back(int(i));
        cv::Point2d c(kQr[q].x+kQr[q].width/2.,kQr[q].y+kQr[q].height/2.);
        auto fit=[&](const std::vector<int>& rows){cv::Mat X(int(rows.size()),3,CV_64F),Y(int(rows.size()),2,CV_64F),B;
            for(int r=0;r<int(rows.size());++r){auto p=pts.p[rows[r]];X.at<double>(r,0)=1;X.at<double>(r,1)=(p.x-c.x)/100;X.at<double>(r,2)=(p.y-c.y)/100;Y.at<double>(r,0)=d[rows[r]].x;Y.at<double>(r,1)=d[rows[r]].y;}
            cv::solve(X,Y,B,cv::DECOMP_SVD);return B;};
        auto resid=[&](const cv::Mat& B,int i){auto p=pts.p[i];double u=(p.x-c.x)/100,v=(p.y-c.y)/100;
            return std::hypot(d[i].x-(B.at<double>(0,0)+B.at<double>(1,0)*u+B.at<double>(2,0)*v),d[i].y-(B.at<double>(0,1)+B.at<double>(1,1)*u+B.at<double>(2,1)*v));};
        if(idx.size()<6)continue;auto B=fit(idx);std::vector<int> in;for(int i:idx)if(resid(B,i)<=kOutlier)in.push_back(i);
        if(in.size()>=6&&in.size()<idx.size())B=fit(in);for(int i:idx)out[i]=resid(B,i);}
    return out;
}
// 채택 control 특징과 같은 P12(판 전체 2차식)를 고정 536점에 맞춘 잔차 RMS. 이전 검출기 비교의 '점 잡음'과 같은 정의.
double p12_rms(const FixedPoints& pts,const std::vector<cv::Point2d>& d,const std::vector<unsigned char>& ok){
    std::vector<int> idx;for(size_t i=0;i<pts.p.size();++i)if(pts.selected[i]&&ok[i])idx.push_back(int(i));if(idx.size()<12)return kNan;
    cv::Mat X(int(idx.size()),6,CV_64F),Y(int(idx.size()),2,CV_64F),B;
    for(int r=0;r<int(idx.size());++r){auto p=pts.p[idx[r]];double u=(p.x-1906)/486,v=(p.y-1070)/122,b[]={1,u,v,u*u,u*v,v*v};for(int j=0;j<6;++j)X.at<double>(r,j)=b[j];Y.at<double>(r,0)=d[idx[r]].x;Y.at<double>(r,1)=d[idx[r]].y;}
    cv::solve(X,Y,B,cv::DECOMP_SVD);cv::Mat R=Y-X*B;double s=0;for(int r=0;r<R.rows;++r)s+=R.at<double>(r,0)*R.at<double>(r,0)+R.at<double>(r,1)*R.at<double>(r,1);return std::sqrt(s/R.rows);
}
struct Pool {long total=0,fail=0,own_invalid=0,fitted=0,outliers=0,gt05=0;std::vector<double> inlier,p12,ms;};
void add(Pool& p,const std::vector<double>& resid,const Estimate& e,double p12,double ms){
    for(size_t i=0;i<resid.size();++i){++p.total;if(!e.ok[i]){++p.fail;continue;}if(!e.own_valid[i])++p.own_invalid;
        if(!std::isfinite(resid[i]))continue;++p.fitted;if(resid[i]>kOutlier)++p.outliers;else p.inlier.push_back(resid[i]);if(resid[i]>.5)++p.gt05;}
    if(std::isfinite(p12))p.p12.push_back(p12);if(std::isfinite(ms))p.ms.push_back(ms);
}
const char* kPoolHeader="points,fail_rate,own_invalid_rate,resid_median_px,resid_p95_px,resid_rms_inlier_px,frac_resid_gt0.5,outlier_rate_gt1,p12_rms_median_px,time_median_ms,time_p95_ms";
std::string pool_row(const Pool& p){std::ostringstream o;o<<p.total<<','<<num(ratio(p.fail,p.total))<<','<<num(ratio(p.own_invalid,p.total))<<','<<num(quantile(p.inlier,.5))<<','<<num(quantile(p.inlier,.95))<<','<<num(rms(p.inlier))<<','
    <<num(ratio(p.gt05,p.fitted))<<','<<num(ratio(p.outliers,p.fitted))<<','<<num(quantile(p.p12,.5))<<','<<num(quantile(p.ms,.5),2)<<','<<num(quantile(p.ms,.95),2);return o.str();}
}
int main(int argc,char** argv){
    try{
        // 인자: [결과 폴더] [quick]. quick은 1014는 20장마다 1장, 1020은 반복 묶음 3개만 돌리는 빌드 확인용이다.
        fs::path repo=fs::u8path(REPO_DIR),experiment=fs::u8path(EXPERIMENT_DIR),results=argc>1?fs::u8path(argv[1]):experiment/"results";
        bool quick=argc>2&&std::string(argv[2])=="quick";
        if(fs::exists(results))throw std::runtime_error("results already exists; refusing to overwrite: "+results.u8string());
        cv::setNumThreads(1);
        auto data=repo/"data/clean",control=repo/fs::u8path(u8"experiments/hye/20261028_1327_final_Y_T control/assets");
        if(legacy_control::sha256(data/"clean_20261014/center.bmp")!=kCenterSha)throw std::runtime_error("center.bmp hash mismatch");
        cv::Mat center=legacy_control::read_gray(data/"clean_20261014/center.bmp");auto pts=load_points(control/"points.json");
        Methods methods(center,pts,control);const int M=Methods::count();
        auto list=photos(data);
        if(quick){std::vector<Photo> few;int k=0;std::map<std::string,int> groups;for(auto& p:list){if(p.set.rfind("1014",0)==0){if(k++%20==0)few.push_back(p);}else if(groups.size()<3||groups.count(p.group)){groups[p.group];few.push_back(p);}}list=few;}
        fs::create_directories(results);
        std::ofstream per(results/"per_image.csv"),fail(results/"outliers.csv"),errors(results/"errors.csv");
        per<<"photo,set,method,"<<kPoolHeader<<"\n";fail<<"photo,method,point_id,selected536,kind,value_px\n";errors<<"photo,method,error\n";
        std::map<std::pair<std::string,int>,Pool> pools;
        // 반복성 계산용: 1020 사진의 점 변위
        struct Kept{std::string group,name;std::vector<Estimate> e;};std::vector<Kept> kept;
        int index=0;
        for(auto& ph:list){
            cv::Mat gray=legacy_control::read_gray(ph.file);Kept k{ph.group,ph.name,{}};
            for(int m=0;m<M;++m){Estimate e;double ms=kNan;
                try{auto t0=Clock::now();e=methods.run(m,gray);ms=std::chrono::duration<double,std::milli>(Clock::now()-t0).count();}
                catch(const std::exception& ex){errors<<'"'<<ph.name<<"\","<<Methods::name(m)<<",\""<<ex.what()<<"\"\n";
                    e={std::vector<cv::Point2d>(pts.p.size(),{kNan,kNan}),std::vector<unsigned char>(pts.p.size(),0),std::vector<unsigned char>(pts.p.size(),0),std::vector<double>(pts.p.size(),kNan)};}
                auto resid=affine_residual(pts,e.d,e.ok);double p12=p12_rms(pts,e.d,e.ok);
                Pool p;add(p,resid,e,p12,ms);per<<'"'<<ph.name<<"\","<<ph.set<<','<<Methods::name(m)<<','<<pool_row(p)<<"\n";
                for(size_t i=0;i<resid.size();++i){if(!e.ok[i])fail<<'"'<<ph.name<<"\","<<Methods::name(m)<<','<<i<<','<<int(pts.selected[i])<<",fail,\n";
                    else if(resid[i]>kOutlier)fail<<'"'<<ph.name<<"\","<<Methods::name(m)<<','<<i<<','<<int(pts.selected[i])<<",affine_resid,"<<num(resid[i])<<"\n";}
                for(auto key:{std::make_pair(ph.set,m),std::make_pair(std::string("전체"),m)}){auto& g=pools[key];add(g,resid,e,p12,ms);}
                if(!ph.group.empty())k.e.push_back(e);}
            if(!ph.group.empty())kept.push_back(std::move(k));
            std::cout<<"["<<++index<<"/"<<list.size()<<"] "<<ph.name<<"\n";
        }
        std::ofstream sum(results/"summary_consistency.csv");sum<<"set,method,"<<kPoolHeader<<"\n";
        for(auto set:{"1014_Y","1014_T","1020_혼합","1020_혼합_T2.00","1020_센터","전체"})for(int m=0;m<M;++m){auto it=pools.find({set,m});if(it!=pools.end())sum<<set<<','<<Methods::name(m)<<','<<pool_row(it->second)<<"\n";}

        // 1B: 같은 묶음 안 모든 쌍
        std::ofstream rep(results/"repeat_pairs.csv");rep<<"group,photo_a,photo_b,method,common_points,raw_rms_px,raw_mean_dx_px,raw_mean_dy_px,resid_rms_inlier_px,resid_p95_px,outlier_rate_gt1,sigma_single_px\n";
        struct RP{std::vector<double> raw,inlier;long fitted=0,outliers=0,pairs=0;};std::map<std::pair<std::string,int>,RP> rp;
        for(size_t a=0;a<kept.size();++a)for(size_t b=a+1;b<kept.size();++b){if(kept[a].group!=kept[b].group)continue;
            for(int m=0;m<M;++m){auto& ea=kept[a].e[m];auto& eb=kept[b].e[m];std::vector<cv::Point2d> diff(pts.p.size());std::vector<unsigned char> both(pts.p.size());std::vector<double> raw;double sx=0,sy=0;
                for(size_t i=0;i<pts.p.size();++i){both[i]=ea.ok[i]&&eb.ok[i];diff[i]=ea.d[i]-eb.d[i];if(both[i]){raw.push_back(cv::norm(diff[i]));sx+=diff[i].x;sy+=diff[i].y;}}
                auto resid=affine_residual(pts,diff,both);std::vector<double> in;long fitted=0,out=0;for(auto r:resid)if(std::isfinite(r)){++fitted;if(r>kOutlier)++out;else in.push_back(r);}
                rep<<'"'<<kept[a].group<<"\",\""<<kept[a].name<<"\",\""<<kept[b].name<<"\","<<Methods::name(m)<<','<<raw.size()<<','<<num(rms(raw))<<','<<num(raw.empty()?kNan:sx/raw.size())<<','<<num(raw.empty()?kNan:sy/raw.size())<<','
                   <<num(rms(in))<<','<<num(quantile(in,.95))<<','<<num(ratio(out,fitted))<<','<<num(rms(in)/std::sqrt(2.))<<"\n";
                std::string cls=kept[a].group.find("T+2.00")!=std::string::npos?"T+2.00":"T+2.00 제외";
                for(const auto& c:{cls,std::string("전체")}){auto& g=rp[{c,m}];g.raw.insert(g.raw.end(),raw.begin(),raw.end());g.inlier.insert(g.inlier.end(),in.begin(),in.end());g.fitted+=fitted;g.outliers+=out;++g.pairs;}}}
        std::ofstream rs(results/"summary_repeat.csv");rs<<"groups,method,pairs,raw_rms_px,resid_rms_inlier_px,resid_p95_px,outlier_rate_gt1,sigma_single_px\n";
        for(auto c:{"T+2.00 제외","T+2.00","전체"})for(int m=0;m<M;++m){auto it=rp.find({c,m});if(it==rp.end())continue;auto& g=it->second;rs<<c<<','<<Methods::name(m)<<','<<g.pairs<<','<<num(rms(g.raw))<<','<<num(rms(g.inlier))<<','<<num(quantile(g.inlier,.95))<<','<<num(ratio(g.outliers,g.fitted))<<','<<num(rms(g.inlier)/std::sqrt(2.))<<"\n";}

        std::ofstream man(results/"manifest.json");
        man<<"{\n  \"center_sha256\": \""<<kCenterSha<<"\",\n  \"points_sha256\": \""<<legacy_control::sha256(control/"points.json")<<"\",\n  \"photos\": "<<list.size()<<",\n  \"quick\": "<<(quick?"true":"false")
           <<",\n  \"affine_outlier_px\": "<<kOutlier<<",\n  \"opencv\": \""<<CV_VERSION<<"\",\n  \"threads\": 1,\n  \"photo_list\": [\n";
        for(size_t i=0;i<list.size();++i)man<<"    {\"set\": \""<<list[i].set<<"\", \"name\": \""<<list[i].name<<"\", \"sha256\": \""<<legacy_control::sha256(list[i].file)<<"\"}"<<(i+1<list.size()?",":"")<<"\n";
        man<<"  ]\n}\n";
        std::cout<<"done: "<<results.u8string()<<"\n";return 0;
    }catch(const std::exception& e){std::cerr<<"error: "<<e.what()<<"\n";return 1;}
}
