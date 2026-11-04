// 특징점 검출기를 1단계와 같은 72개 합성 조건(정답 변위 있음)에서 세 경로로 평가한다.
//  (1) 검출 위치: center와 합성 영상에서 각각 검출해, center 점을 정답 변형으로 옮긴 위치에서 가장 가까운 검출점과의 거리
//  (2) 기술자 짝짓기: 비율 검사 0.8로 짝지은 뒤 정답과 비교한 진짜 오매칭률과, 이전 실험의 '중앙값 2 px' 대리 지표
//  (3) 검출+LK: center 검출점을 1단계의 NCC+LK(1단)로 추적한 변위 오차 (QR 구조점 630과 같은 조건)
#include "bench.hpp"
#include "control_internal.hpp"
#include "detectors.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <unordered_map>
using namespace bench;
namespace {
const double kNan=std::numeric_limits<double>::quiet_NaN();
const std::string kCenterSha="01a69df6771b840d2fcf9cf727760592b5fb199c12277bfc173af514e26fd685";
const double kRepeat=1.5;     // 검출 반복 판정 거리 (px)
const double kWrong=3.0;      // 짝이 다른 점을 가리킨 것으로 보는 오차 (px). QR 모듈 한 칸(약7 px)의 절반 아래
                              // ORB 상위 피라미드 점은 같은 점이어도 1.5 px 넘게 어긋나서, 위치 부정확과 오매칭을 나눈다.
const int kSearchX=110,kSearchY=22;  // 합성 영상 검출 영역: QR 상자를 가로 ±110, 세로 ±22 (이전 비교 실험과 같음)
const int kLk=4;              // 1단계 Methods의 NCC+LK(1단)
using Clock=std::chrono::steady_clock;
double ms_since(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
double quantile(std::vector<double> v,double q){if(v.empty())return kNan;std::sort(v.begin(),v.end());return v[std::min(v.size()-1,size_t(q*(v.size()-1)+.5))];}
std::string num(double v,int p=4){if(!std::isfinite(v))return "";std::ostringstream o;o<<std::fixed<<std::setprecision(p)<<v;return o.str();}
double ratio(long a,long b){return b?double(a)/b:kNan;}
// 1단계 main.cpp의 conditions()와 같은 72개 조건 (값을 바꾸지 않고 옮겼다)
std::vector<Condition> conditions(double sigma){
    std::vector<Condition> out;
    for(int i=0;i<=20;++i){double f=i*.05;Condition c;c.group="평행이동_x";c.tx=3+f;c.ty=-2+.35;c.name="tx+"+num(f,2);out.push_back(c);}
    for(int i=0;i<=20;++i){double f=i*.05;Condition c;c.group="평행이동_y";c.tx=3.35;c.ty=-2+f;c.name="ty+"+num(f,2);out.push_back(c);}
    for(int i=0;i<=10;++i){double f=i*.1;Condition c;c.group="평행이동+잡음";c.tx=3+f;c.ty=-1.65;c.noise=sigma;c.seed=100+i;c.name="tx+"+num(f,1)+"_noise";out.push_back(c);}
    for(double a:{-.1,-.05,-.02,.02,.05,.1}){Condition c;c.group="회전·배율";c.tx=3.4;c.ty=-1.7;c.angle_deg=a;c.noise=sigma;c.seed=200+int(out.size());c.name="rot"+num(a,2)+"deg";out.push_back(c);}
    for(double s:{.999,1.001}){Condition c;c.group="회전·배율";c.tx=3.4;c.ty=-1.7;c.scale=s;c.noise=sigma;c.seed=200+int(out.size());c.name="scale"+num(s,3);out.push_back(c);}
    struct P{const char* n;double g,o,b,k;};
    for(auto p:{P{"gain0.7_off+20",.7,20,0,1},P{"gain1.3_off-20",1.3,-20,0,1},P{"off+30",1,30,0,1},P{"blur0.7",1,0,.7,1},P{"blur1.2",1,0,1.2,1},P{"noise2x",1,0,0,2},P{"noise4x",1,0,0,4}}){
        Condition c;c.group="밝기·흐림·잡음";c.tx=3.4;c.ty=-1.7;c.gain=p.g;c.offset=p.o;c.blur=p.b;c.noise=sigma*p.k;c.seed=300+int(out.size());c.name=p.n;out.push_back(c);}
    for(int s=1;s<=4;++s){Condition c;c.group="가림";c.tx=3.4;c.ty=-1.7;c.noise=sigma;c.occlusions=3;c.seed=400+s;c.name="occlusion_seed"+std::to_string(s);out.push_back(c);}
    return out;
}
cv::Mat qr_mask(bool search){cv::Mat m=cv::Mat::zeros(kCrop.size(),CV_8U);
    for(auto r:kQr){if(search)r=cv::Rect(r.x-kSearchX,r.y-kSearchY,r.width+2*kSearchX,r.height+2*kSearchY);m(r-kCrop.tl())=255;}return m;}
// 4 px 격자로 나눈 최근접 검색
struct Grid {
    std::unordered_map<long long,std::vector<cv::Point2f>> cells;
    static long long key(int x,int y){return (long long)(x)*1000003LL+y;}
    explicit Grid(const std::vector<cv::KeyPoint>& k){for(auto& p:k)cells[key(int(std::floor(p.pt.x/4)),int(std::floor(p.pt.y/4)))].push_back(p.pt);}
    double nearest(cv::Point2d q)const{double best=1e9;int cx=int(std::floor(q.x/4)),cy=int(std::floor(q.y/4));
        for(int y=cy-1;y<=cy+1;++y)for(int x=cx-1;x<=cx+1;++x){auto it=cells.find(key(x,y));if(it==cells.end())continue;for(auto p:it->second)best=std::min(best,std::hypot(p.x-q.x,p.y-q.y));}
        return best;}
};
struct DetectPool{long ref=0,repeated=0,gt05=0;std::vector<double> loc,ms;double synth=0;long images=0;};
struct MatchPool{long ref=0,matches=0,wrong=0,proxy=0,proxy_missed=0,gt05=0;std::vector<double> err,ms;};
struct TrackPool{long total=0,fail=0,bad05=0;std::vector<double> mag,ms;};
}
int main(int argc,char** argv){
    try{
        // 인자: [결과 폴더] [quick]. quick은 조건을 10개 중 1개만 돌리는 빌드 확인용이다.
        fs::path repo=fs::u8path(REPO_DIR),experiment=fs::u8path(EXPERIMENT_DIR),results=argc>1?fs::u8path(argv[1]):experiment/"results";
        bool quick=argc>2&&std::string(argv[2])=="quick";
        if(fs::exists(results))throw std::runtime_error("results already exists; refusing to overwrite: "+results.u8string());
        cv::setNumThreads(1);
        auto data14=repo/"data/clean/clean_20261014",data20=repo/fs::u8path(u8"data/clean/clean_20261020/기준 위치");
        auto control=repo/fs::u8path(u8"experiments/hye/00000_final_Y_T control/assets");
        if(legacy_control::sha256(data14/"center.bmp")!=kCenterSha)throw std::runtime_error("center.bmp hash mismatch");
        cv::Mat center=legacy_control::read_gray(data14/"center.bmp"),ref_crop=center(kCrop);
        std::vector<cv::Mat> repeats;for(int i=1;i<=3;++i)repeats.push_back(legacy_control::read_gray(data20/("center"+std::to_string(i)+".bmp")));
        double sigma=quantile({estimate_noise(repeats[0],repeats[1]),estimate_noise(repeats[0],repeats[2]),estimate_noise(repeats[1],repeats[2])},.5);
        auto conds=conditions(sigma);if(quick){std::vector<Condition> few;for(size_t i=0;i<conds.size();i+=10)few.push_back(conds[i]);conds=few;}

        auto dets=detectors();auto ref_mask=qr_mask(false),search_mask=qr_mask(true);
        struct Ref{std::vector<cv::KeyPoint> kp;cv::Mat desc;double ms=0;std::unique_ptr<Methods> tracker;};std::vector<Ref> refs(dets.size());
        for(size_t d=0;d<dets.size();++d){auto t0=Clock::now();dets[d].run(ref_crop,ref_mask,refs[d].kp,dets[d].descriptor?&refs[d].desc:nullptr);refs[d].ms=ms_since(t0);
            if(dets[d].trackable){FixedPoints fp;for(auto& k:refs[d].kp){cv::Point2d g(k.pt.x+kCrop.x,k.pt.y+kCrop.y);int r=-1;for(int q=0;q<3;++q)if(kQr[q].contains(cv::Point(int(g.x),int(g.y))))r=q;
                    if(r<0)continue;fp.p.push_back(g);fp.region.push_back(r);fp.selected.push_back(1);}
                refs[d].tracker=std::make_unique<Methods>(center,fp,control);}
            std::cout<<dets[d].name<<": center "<<refs[d].kp.size()<<" points\n";}
        auto qr_points=load_points(control/"points.json");Methods qr_tracker(center,qr_points,control);
        if(std::string(Methods::name(kLk))!="NCC+LK(1단)")throw std::runtime_error("Stage1 method index changed");

        fs::create_directories(results);
        std::ofstream det_csv(results/"detect_per_image.csv"),match_csv(results/"match_per_image.csv"),track_csv(results/"track_per_image.csv");
        det_csv<<"condition,group,detector,center_points,synth_points,repeat_rate,loc_median_px,loc_p95_px,frac_loc_gt0.5,time_ms\n";
        match_csv<<"condition,group,detector,center_points,matches,true_mismatch_rate,proxy_outlier_rate,proxy_missed,err_median_px,err_p95_px,frac_err_gt0.5,time_ms\n";
        track_csv<<"condition,group,points_set,points,fail,median_px,p95_px,max_px,frac_fail_or_gt0.5,time_ms\n";
        std::map<std::pair<std::string,std::string>,DetectPool> dp;std::map<std::pair<std::string,std::string>,MatchPool> mp;std::map<std::pair<std::string,std::string>,TrackPool> tp;
        int index=0;
        for(const auto& c:conds){
            auto image=synthesize(center,c);cv::Mat crop=image(kCrop);
            auto truth=[&](cv::Point2f crop_point){cv::Point2d g(crop_point.x+kCrop.x,crop_point.y+kCrop.y);return warp_point(c,g)-g;};
            auto track=[&](const std::string& set,const Methods& tracker,const FixedPoints& fp){
                auto t0=Clock::now();auto e=tracker.run(kLk,image);double ms=ms_since(t0);TrackPool p;p.ms.push_back(ms);
                for(size_t i=0;i<fp.p.size();++i){++p.total;if(!e.ok[i]){++p.fail;++p.bad05;continue;}double m=cv::norm(e.d[i]-(warp_point(c,fp.p[i])-fp.p[i]));p.mag.push_back(m);if(m>.5)++p.bad05;}
                track_csv<<c.name<<','<<c.group<<','<<set<<','<<p.total<<','<<p.fail<<','<<num(quantile(p.mag,.5))<<','<<num(quantile(p.mag,.95))<<','<<num(quantile(p.mag,1))<<','<<num(ratio(p.bad05,p.total))<<','<<num(ms,2)<<"\n";
                for(auto key:{std::make_pair(c.group,set),std::make_pair(std::string("전체"),set)}){auto& g=tp[key];g.total+=p.total;g.fail+=p.fail;g.bad05+=p.bad05;g.mag.insert(g.mag.end(),p.mag.begin(),p.mag.end());g.ms.push_back(ms);}};
            track("QR구조점630",qr_tracker,qr_points);
            for(size_t d=0;d<dets.size();++d){const auto& det=dets[d];const auto& ref=refs[d];
                std::vector<cv::KeyPoint> kp;cv::Mat desc;
                auto t0=Clock::now();det.run(crop,search_mask,kp,det.descriptor?&desc:nullptr);double detect_ms=ms_since(t0);
                // (1) 검출 위치
                Grid grid(kp);DetectPool p;p.ms.push_back(detect_ms);p.synth=double(kp.size());p.images=1;
                for(auto& k:ref.kp){++p.ref;cv::Point2d target=cv::Point2d(k.pt)+truth(k.pt);double dist=grid.nearest(target);if(dist<=kRepeat){++p.repeated;p.loc.push_back(dist);if(dist>.5)++p.gt05;}}
                det_csv<<c.name<<','<<c.group<<','<<det.name<<','<<p.ref<<','<<kp.size()<<','<<num(ratio(p.repeated,p.ref))<<','<<num(quantile(p.loc,.5))<<','<<num(quantile(p.loc,.95))<<','<<num(ratio(p.gt05,p.repeated))<<','<<num(detect_ms,2)<<"\n";
                for(auto key:{std::make_pair(c.group,det.name),std::make_pair(std::string("전체"),det.name)}){auto& g=dp[key];g.ref+=p.ref;g.repeated+=p.repeated;g.gt05+=p.gt05;g.loc.insert(g.loc.end(),p.loc.begin(),p.loc.end());g.ms.push_back(detect_ms);g.synth+=p.synth;g.images+=1;}
                // (2) 기술자 짝짓기
                if(det.descriptor){auto t1=Clock::now();std::vector<std::vector<cv::DMatch>> knn;if(!desc.empty()&&!ref.desc.empty())cv::BFMatcher(det.norm).knnMatch(ref.desc,desc,knn,2);
                    std::vector<cv::Point2f> a,b;for(auto& v:knn){if(v.size()<2||v[0].distance>=.8f*v[1].distance)continue;a.push_back(ref.kp[v[0].queryIdx].pt);b.push_back(kp[v[0].trainIdx].pt);}
                    double ms=detect_ms+ms_since(t1);MatchPool m;m.ref=long(ref.kp.size());m.matches=long(a.size());m.ms.push_back(ms);
                    std::vector<double> dx,dy;for(size_t i=0;i<a.size();++i){dx.push_back(b[i].x-a[i].x);dy.push_back(b[i].y-a[i].y);}double mx=quantile(dx,.5),my=quantile(dy,.5);
                    for(size_t i=0;i<a.size();++i){cv::Point2d d(b[i].x-a[i].x,b[i].y-a[i].y);double err=cv::norm(d-truth(a[i]));bool wrong=err>kWrong,proxy=std::abs(d.x-mx)>2||std::abs(d.y-my)>2;
                        m.wrong+=wrong;m.proxy+=proxy;m.proxy_missed+=wrong&&!proxy;if(!wrong){m.err.push_back(err);if(err>.5)++m.gt05;}}
                    match_csv<<c.name<<','<<c.group<<','<<det.name<<','<<m.ref<<','<<m.matches<<','<<num(ratio(m.wrong,m.matches))<<','<<num(ratio(m.proxy,m.matches))<<','<<m.proxy_missed<<','<<num(quantile(m.err,.5))<<','<<num(quantile(m.err,.95))<<','<<num(ratio(m.gt05,long(m.err.size())))<<','<<num(ms,2)<<"\n";
                    for(auto key:{std::make_pair(c.group,det.name),std::make_pair(std::string("전체"),det.name)}){auto& g=mp[key];g.ref+=m.ref;g.matches+=m.matches;g.wrong+=m.wrong;g.proxy+=m.proxy;g.proxy_missed+=m.proxy_missed;g.gt05+=m.gt05;g.err.insert(g.err.end(),m.err.begin(),m.err.end());g.ms.push_back(ms);}}
                // (3) 검출+LK
                if(det.trackable){FixedPoints fp;for(auto& k:ref.kp){cv::Point2d g(k.pt.x+kCrop.x,k.pt.y+kCrop.y);for(int q=0;q<3;++q)if(kQr[q].contains(cv::Point(int(g.x),int(g.y)))){fp.p.push_back(g);break;}}
                    track(det.name,*ref.tracker,fp);}
            }
            std::cout<<"["<<++index<<"/"<<conds.size()<<"] "<<c.name<<"\n";
        }
        std::vector<std::string> order;for(const auto& c:conds)if(std::find(order.begin(),order.end(),c.group)==order.end())order.push_back(c.group);order.push_back("전체");
        std::ofstream sd(results/"summary_detect.csv"),sm(results/"summary_match.csv"),st(results/"summary_track.csv");
        sd<<"group,detector,center_points,synth_points_mean,repeat_rate,loc_median_px,loc_p95_px,frac_loc_gt0.5,center_detect_ms,time_median_ms,time_p95_ms\n";
        sm<<"group,detector,matches_mean,true_mismatch_rate,proxy_outlier_rate,proxy_missed,err_median_px,err_p95_px,frac_err_gt0.5,time_median_ms,time_p95_ms\n";
        st<<"group,points_set,points,fail,median_px,p95_px,max_px,frac_fail_or_gt0.5,time_median_ms,time_p95_ms\n";
        for(auto& g:order){
            for(size_t d=0;d<dets.size();++d){auto& p=dp[{g,dets[d].name}];if(!p.images)continue;
                sd<<g<<','<<dets[d].name<<','<<refs[d].kp.size()<<','<<num(p.synth/p.images,1)<<','<<num(ratio(p.repeated,p.ref))<<','<<num(quantile(p.loc,.5))<<','<<num(quantile(p.loc,.95))<<','<<num(ratio(p.gt05,p.repeated))<<','<<num(refs[d].ms,2)<<','<<num(quantile(p.ms,.5),2)<<','<<num(quantile(p.ms,.95),2)<<"\n";
                if(dets[d].descriptor){auto& m=mp[{g,dets[d].name}];sm<<g<<','<<dets[d].name<<','<<num(double(m.matches)/m.ms.size(),1)<<','<<num(ratio(m.wrong,m.matches))<<','<<num(ratio(m.proxy,m.matches))<<','<<m.proxy_missed<<','<<num(quantile(m.err,.5))<<','<<num(quantile(m.err,.95))<<','<<num(ratio(m.gt05,long(m.err.size())))<<','<<num(quantile(m.ms,.5),2)<<','<<num(quantile(m.ms,.95),2)<<"\n";}}
            std::vector<std::string> sets{"QR구조점630"};for(auto& d:dets)if(d.trackable)sets.push_back(d.name);
            for(auto& s:sets){auto& p=tp[{g,s}];if(!p.total)continue;st<<g<<','<<s<<','<<p.total<<','<<p.fail<<','<<num(quantile(p.mag,.5))<<','<<num(quantile(p.mag,.95))<<','<<num(quantile(p.mag,1))<<','<<num(ratio(p.bad05,p.total))<<','<<num(quantile(p.ms,.5),2)<<','<<num(quantile(p.ms,.95),2)<<"\n";}
        }
        std::ofstream man(results/"manifest.json");
        man<<"{\n  \"center_sha256\": \""<<kCenterSha<<"\",\n  \"points_sha256\": \""<<legacy_control::sha256(control/"points.json")<<"\",\n  \"conditions\": \"1단계(20261029_122323)와 같은 72개, quick="<<(quick?"true":"false")<<"\",\n  \"noise_sigma\": "<<sigma
           <<",\n  \"repeat_px\": "<<kRepeat<<",\n  \"wrong_match_px\": "<<kWrong<<",\n  \"ratio_test\": 0.8,\n  \"synth_search\": [\""<<kSearchX<<"\", \""<<kSearchY<<"\"],\n  \"opencv\": \""<<CV_VERSION<<"\",\n  \"threads\": 1,\n  \"not_evaluated\": {\"SIFT\": \"팀 기준 OpenCV 3.4.15 opencv_world에 xfeatures2d(contrib) 없음\", \"SURF\": \"같은 이유 + nonfree\"}\n}\n";
        std::cout<<"done: "<<results.u8string()<<"\n";return 0;
    }catch(const std::exception& e){std::cerr<<"error: "<<e.what()<<"\n";return 1;}
}
