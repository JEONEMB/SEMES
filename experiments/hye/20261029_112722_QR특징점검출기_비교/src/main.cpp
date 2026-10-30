// QR 판 특징점 검출기 비교.
// QR 구조점(채택 control 측정기, OpenCV LK)과 일반 검출기(GFTT, Harris, FAST + LK / ORB, BRISK, AKAZE, KAZE + 기술자 짝짓기)를
// 같은 1014 단일축 280장, 같은 이동장 요약(P12), 같은 단순 판정기(고정 4-fold)로 비교한다.
#include "control_internal.hpp"
#include <opencv2/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
namespace fs=std::filesystem;
using legacy_control::read_gray;

static const cv::Rect crop(1340,800,1130,530);
static const cv::Rect qr[]={cv::Rect(1552,969,184,198),cv::Rect(1809,971,184,198),cv::Rect(2067,973,185,198)};
static const int SEARCH_X=110,SEARCH_Y=22;
static const double TOL=.09,MISMATCH_PX=2.;

struct Sample{std::string name;int axis;double value;fs::path file;};   // axis 0=Y, 1=T
struct Field{std::vector<cv::Point2d> p,d;int detected=0;};                // p는 center 좌표, d는 변위
struct Summary{int n=0,inliers=0;double noise=0,shift=0;cv::Mat p12;bool ok=false;};

static std::vector<Sample> load_samples(const fs::path& data){
    std::vector<Sample> out;
    for(auto [folder,axis]:{std::pair<const char*,int>{"Y",0},{"Theta",1}})
        for(auto& e:fs::directory_iterator(data/folder))if(e.path().extension()==".bmp"){
            auto stem=e.path().stem().u8string();out.push_back({stem,axis,std::stod(stem.substr(1)),e.path()});}
    std::sort(out.begin(),out.end(),[](auto& a,auto& b){return a.axis!=b.axis?a.axis<b.axis:a.value<b.value;});
    return out;
}

static int region_of(cv::Point2d p){for(int k=0;k<3;++k)if(qr[k].contains(cv::Point(int(p.x),int(p.y))))return k;return -1;}

// QR별 NCC 정수 이동 (채택 측정기와 같은 초기값)
static std::array<cv::Point,3> qr_shifts(const cv::Mat& center,const cv::Mat& gray){
    std::array<cv::Point,3> out;
    for(int k=0;k<3;++k){cv::Rect search(qr[k].x-SEARCH_X,qr[k].y-SEARCH_Y,qr[k].width+2*SEARCH_X,qr[k].height+2*SEARCH_Y);
        cv::Mat c;cv::matchTemplate(gray(search),center(qr[k]),c,cv::TM_CCOEFF_NORMED);cv::Point loc;cv::minMaxLoc(c,nullptr,nullptr,nullptr,&loc);
        out[k]=loc-cv::Point(SEARCH_X,SEARCH_Y);}
    return out;
}

// center에서 고정한 점을 LK 정·역방향으로 추적
static Field track_lk(const cv::Mat& center,const cv::Mat& gray,const std::vector<cv::Point2d>& points){
    auto shift=qr_shifts(center,gray);Field f;f.detected=int(points.size());
    std::vector<cv::Point2f> p0,p1,back;for(auto& p:points){auto s=shift[region_of(p)];p0.emplace_back(float(p.x-crop.x),float(p.y-crop.y));p1.emplace_back(p0.back().x+s.x,p0.back().y+s.y);}
    back=p0;std::vector<unsigned char> s1,s2;std::vector<float> e1,e2;auto crit=cv::TermCriteria(cv::TermCriteria::COUNT|cv::TermCriteria::EPS,40,.001);
    cv::Mat a=center(crop),b=gray(crop);
    cv::calcOpticalFlowPyrLK(a,b,p0,p1,s1,e1,cv::Size(21,21),2,crit,cv::OPTFLOW_USE_INITIAL_FLOW);
    cv::calcOpticalFlowPyrLK(b,a,p1,back,s2,e2,cv::Size(21,21),2,crit,cv::OPTFLOW_USE_INITIAL_FLOW);
    for(size_t i=0;i<points.size();++i){if(!s1[i]||!s2[i]||cv::norm(back[i]-p0[i])>.5)continue;
        f.p.push_back(points[i]);f.d.emplace_back(double(p1[i].x-p0[i].x),double(p1[i].y-p0[i].y));}
    return f;
}

struct DescState{std::vector<cv::KeyPoint> kp;cv::Mat desc;};
static cv::Mat qr_mask(bool search){cv::Mat m=cv::Mat::zeros(crop.size(),CV_8U);
    for(auto r:qr){if(search)r=cv::Rect(r.x-SEARCH_X,r.y-SEARCH_Y,r.width+2*SEARCH_X,r.height+2*SEARCH_Y);m(r-crop.tl())=255;}return m;}

// 기술자 짝짓기: center와 현재 영상에서 각각 검출하고 비율 검사(0.8)로 짝짓는다
static Field match_desc(const cv::Ptr<cv::Feature2D>& feature,int norm,const DescState& ref,const cv::Mat& gray){
    Field f;std::vector<cv::KeyPoint> kp;cv::Mat desc;feature->detectAndCompute(gray(crop),qr_mask(true),kp,desc);f.detected=int(ref.kp.size());
    if(desc.empty()||ref.desc.empty())return f;
    std::vector<std::vector<cv::DMatch>> m;cv::BFMatcher(norm).knnMatch(ref.desc,desc,m,2);
    for(auto& v:m){if(v.size()<2||v[0].distance>=.8f*v[1].distance)continue;auto a=ref.kp[v[0].queryIdx].pt,b=kp[v[0].trainIdx].pt;
        f.p.emplace_back(a.x+crop.x,a.y+crop.y);f.d.emplace_back(b.x-a.x,b.y-a.y);}
    return f;
}

static double median(std::vector<double> v){if(v.empty())return 0;auto m=v.begin()+v.size()/2;std::nth_element(v.begin(),m,v.end());return *m;}

// 이동장 요약: 판 이동에서 2 px 넘게 벗어난 짝은 오매칭으로 세고, 나머지로 P12를 적합한다
static Summary summarize(const Field& f){
    Summary s;s.n=int(f.p.size());if(s.n<12)return s;
    std::vector<double> dx,dy;for(auto& d:f.d){dx.push_back(d.x);dy.push_back(d.y);}double mx=median(dx),my=median(dy);
    std::vector<int> keep;for(int i=0;i<s.n;++i)if(std::abs(f.d[i].x-mx)<=MISMATCH_PX&&std::abs(f.d[i].y-my)<=MISMATCH_PX)keep.push_back(i);
    s.inliers=int(keep.size());if(s.inliers<12)return s;
    cv::Mat basis(s.inliers,6,CV_64F),target(s.inliers,2,CV_64F);
    for(int r=0;r<s.inliers;++r){auto p=f.p[keep[r]];double u=(p.x-1906)/486,v=(p.y-1070)/122,b[]={1,u,v,u*u,u*v,v*v};
        for(int j=0;j<6;++j)basis.at<double>(r,j)=b[j];target.at<double>(r,0)=f.d[keep[r]].x;target.at<double>(r,1)=f.d[keep[r]].y;}
    cv::Mat coef;cv::solve(basis,target,coef,cv::DECOMP_SVD);cv::Mat res=target-basis*coef;
    s.noise=std::sqrt(res.dot(res)/(2.*s.inliers));s.p12=coef.reshape(1,1).clone();s.shift=coef.at<double>(0,0);s.ok=true;return s;
}

// 단순 판정기: 축마다 특징을 값의 2차식으로 맞추고, 잔차 공분산(축소 0.01)으로 백색화한 거리가 가장 작은 축과 값을 고른다.
// 채택 control의 영점 후보 6개, 부호 게이트는 넣지 않았다. 모든 방법에 같은 판정기를 쓰는 상대 비교용이다.
struct Judge{cv::Mat coef[2],inv;double lo[2],hi[2];};
static Judge train(const std::vector<Sample>& S,const std::vector<cv::Mat>& X,const std::vector<int>& idx){
    Judge j;cv::Mat residual;
    for(int a=0;a<2;++a){std::vector<int> rows;for(int i:idx)if(S[i].axis==a)rows.push_back(i);
        cv::Mat D(int(rows.size()),3,CV_64F),Y(int(rows.size()),12,CV_64F);j.lo[a]=1e9;j.hi[a]=-1e9;
        for(int r=0;r<int(rows.size());++r){double v=S[rows[r]].value;D.at<double>(r,0)=1;D.at<double>(r,1)=v;D.at<double>(r,2)=v*v;X[rows[r]].copyTo(Y.row(r));j.lo[a]=std::min(j.lo[a],v);j.hi[a]=std::max(j.hi[a],v);}
        cv::solve(D,Y,j.coef[a],cv::DECOMP_SVD);cv::Mat r=Y-D*j.coef[a];residual.push_back(r);}
    cv::Mat cov=residual.t()*residual/double(residual.rows-6);double m=cv::trace(cov)[0]/12;cov=.99*cov+.01*m*cv::Mat::eye(12,12,CV_64F);
    cv::invert(cov,j.inv,cv::DECOMP_SVD);return j;
}
static std::pair<int,double> predict(const Judge& j,const cv::Mat& x){
    double best=1e300;int axis=0;double value=0;
    auto cost=[&](int a,double v){cv::Mat f=(cv::Mat_<double>(1,3)<<1,v,v*v)*j.coef[a];cv::Mat r=x-f;return cv::Mat(r*j.inv*r.t()).at<double>(0);};
    for(int a=0;a<2;++a){double bv=0,bc=1e300;for(double v=j.lo[a]-.3;v<=j.hi[a]+.3;v+=.001){double c=cost(a,v);if(c<bc){bc=c;bv=v;}}
        for(double v=bv-.001;v<=bv+.001;v+=.0001){double c=cost(a,v);if(c<bc){bc=c;bv=v;}}
        if(bc<best){best=bc;axis=a;value=bv;}}
    return {axis,value};
}

struct Method{std::string name,kind;std::function<Field(const cv::Mat&)> measure;int center_points=0;};

int main(){
    try{
        auto repo=fs::u8path(REPO_DIR),exp=fs::u8path(EXPERIMENT_DIR);
        auto assets=repo/"experiments/hye/00000_final_Y_T control/assets";auto data=repo/"data/clean/clean_20261014";
        auto out=exp/"results";if(fs::exists(out))throw std::runtime_error("results 폴더가 이미 있다. 덮어쓰지 않는다");fs::create_directories(out);
        auto S=load_samples(data);if(S.size()!=280)throw std::runtime_error("1014 단일축 280장이 아니다");
        cv::Mat center=read_gray(data/"center.bmp");

        // QR 구조점 536 (채택 control과 같은 고정 점)
        std::vector<cv::Point2d> structure;{cv::FileStorage fsd((assets/"points.json").u8string(),cv::FileStorage::READ);
            for(auto it=fsd["points"].begin();it!=fsd["points"].end();++it){auto r=*it;if(int(r[3])==1)structure.emplace_back(double(r[0]),double(r[1]));}}
        legacy_control::Tracker tracker(assets);auto zero=tracker.measure(center);
        std::vector<int> selected;{cv::FileStorage fsd((assets/"points.json").u8string(),cv::FileStorage::READ);int i=0;
            for(auto it=fsd["points"].begin();it!=fsd["points"].end();++it,++i)if(int((*it)[3])==1)selected.push_back(i);}

        std::vector<Method> methods;
        methods.push_back({"QR구조점536+GN(채택측정기)","structure",[&](const cv::Mat& g){auto t=tracker.measure(g);Field f;f.detected=536;
            for(size_t k=0;k<selected.size();++k){int i=selected[k];if(!t.valid[i])continue;f.p.push_back(structure[k]);
                f.d.emplace_back(double(t.displacement.at<float>(i,0)-zero.displacement.at<float>(i,0)),double(t.displacement.at<float>(i,1)-zero.displacement.at<float>(i,1)));}return f;},536});
        methods.push_back({"QR구조점536+LK","lk",[&](const cv::Mat& g){return track_lk(center,g,structure);},536});
        auto detected_points=[&](const std::function<std::vector<cv::Point2f>(const cv::Mat&,const cv::Mat&)>& detect){
            std::vector<cv::Point2d> pts;for(auto p:detect(center(crop),qr_mask(false)))pts.emplace_back(p.x+crop.x,p.y+crop.y);return pts;};
        auto gftt=detected_points([](const cv::Mat& im,const cv::Mat& m){std::vector<cv::Point2f> p;cv::goodFeaturesToTrack(im,p,800,.01,4,m,5,false);return p;});
        auto harris=detected_points([](const cv::Mat& im,const cv::Mat& m){std::vector<cv::Point2f> p;cv::goodFeaturesToTrack(im,p,800,.01,4,m,5,true,.04);return p;});
        auto fast=detected_points([](const cv::Mat& im,const cv::Mat& m){std::vector<cv::KeyPoint> k;cv::FastFeatureDetector::create(25)->detect(im,k,m);
            cv::KeyPointsFilter::retainBest(k,800);std::vector<cv::Point2f> p;for(auto& x:k)p.push_back(x.pt);return p;});
        methods.push_back({"Shi-Tomasi(GFTT)+LK","lk",[&](const cv::Mat& g){return track_lk(center,g,gftt);},int(gftt.size())});
        methods.push_back({"Harris+LK","lk",[&](const cv::Mat& g){return track_lk(center,g,harris);},int(harris.size())});
        methods.push_back({"FAST+LK","lk",[&](const cv::Mat& g){return track_lk(center,g,fast);},int(fast.size())});
        std::map<std::string,std::pair<cv::Ptr<cv::Feature2D>,int>> descs={
            {"ORB",{cv::ORB::create(3000),cv::NORM_HAMMING}},{"BRISK",{cv::BRISK::create(20),cv::NORM_HAMMING}},
            {"AKAZE",{cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB,0,3,.0001f),cv::NORM_HAMMING}},{"KAZE",{cv::KAZE::create(false,false,.0001f),cv::NORM_L2}}};
        std::map<std::string,DescState> refs;
        for(auto name:{"ORB","BRISK","AKAZE","KAZE"}){auto& [feature,norm]=descs[name];auto& r=refs[name];feature->detectAndCompute(center(crop),qr_mask(false),r.kp,r.desc);
            methods.push_back({std::string(name)+"+기술자짝짓기","desc",[&,name](const cv::Mat& g){auto& [ft,nm]=descs[name];return match_desc(ft,nm,refs[name],g);},int(r.kp.size())});}

        std::ofstream summary(out/"summary.csv");summary<<std::setprecision(6)
            <<"method,kind,center_points,mean_matched,mismatch_pct,noise_rms_px_median,Y_slope_px_per_mm,Y_linearity_rms_px,T_slope_px_per_T,T_linearity_rms_px,"
              "pass_4fold,axis_correct,Y_max_err,T_max_err,measure_ms_median\n";
        for(auto& m:methods){
            std::cout<<"측정: "<<m.name<<std::endl;
            auto zero_summary=summarize(m.measure(center));
            std::vector<Summary> sums;std::vector<double> ms;std::ofstream per(out/("per_image_"+std::to_string(&m-&methods[0])+".csv"));
            per<<"name,axis,value,matched,inliers,noise_px,shift_px\n";
            for(auto& s:S){auto g=read_gray(s.file);auto t0=std::chrono::steady_clock::now();auto f=m.measure(g);
                ms.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count());
                auto su=summarize(f);if(su.ok&&zero_summary.ok)su.p12=su.p12-zero_summary.p12;sums.push_back(su);
                per<<s.name<<','<<s.axis<<','<<s.value<<','<<su.n<<','<<su.inliers<<','<<su.noise<<','<<su.shift<<'\n';}
            // 판 이동 직선성
            double slope[2],lin[2];
            for(int a=0;a<2;++a){cv::Mat D,Y;for(size_t i=0;i<S.size();++i)if(S[i].axis==a&&sums[i].ok){D.push_back(cv::Mat((cv::Mat_<double>(1,2)<<1,S[i].value)));Y.push_back(sums[i].shift);}
                cv::Mat c;cv::solve(D,Y,c,cv::DECOMP_SVD);cv::Mat r=Y-D*c;slope[a]=c.at<double>(1);lin[a]=std::sqrt(r.dot(r)/r.rows);}
            // 고정 4-fold: 축마다 값 순으로 정렬한 순번 % 4
            std::vector<int> fold(S.size());int counter[2]={0,0};for(size_t i=0;i<S.size();++i)fold[i]=counter[S[i].axis]++%4;
            std::vector<cv::Mat> X(S.size());bool all_ok=true;for(size_t i=0;i<S.size();++i){all_ok&=sums[i].ok;X[i]=sums[i].ok?sums[i].p12:cv::Mat::zeros(1,12,CV_64F);}
            int pass=0,axis_ok=0;double ymax=0,tmax=0;std::ofstream pred(out/("predictions_"+std::to_string(&m-&methods[0])+".csv"));pred<<"name,axis,value,pred_axis,pred_value,pass\n";
            for(int k=0;k<4;++k){std::vector<int> tr;for(size_t i=0;i<S.size();++i)if(fold[i]!=k&&sums[i].ok)tr.push_back(int(i));auto j=train(S,X,tr);
                for(size_t i=0;i<S.size();++i)if(fold[i]==k){int pa=-1;double pv=0;bool ok=false;
                    if(sums[i].ok){auto [a,v]=predict(j,X[i]);pa=a;pv=v;double ey=std::abs((a==0?v:0)-(S[i].axis==0?S[i].value:0)),et=std::abs((a==1?v:0)-(S[i].axis==1?S[i].value:0));
                        ok=ey<=TOL&&et<=TOL;axis_ok+=a==S[i].axis;ymax=std::max(ymax,ey);tmax=std::max(tmax,et);}
                    pass+=ok;pred<<S[i].name<<','<<S[i].axis<<','<<S[i].value<<','<<pa<<','<<pv<<','<<ok<<'\n';}}
            double matched=0,mis=0,tot=0;std::vector<double> noise;for(auto& s:sums){matched+=s.n;mis+=s.n-s.inliers;tot+=s.n;if(s.ok)noise.push_back(s.noise);}
            summary<<m.name<<','<<m.kind<<','<<m.center_points<<','<<matched/S.size()<<','<<(tot>0?100*mis/tot:0)<<','<<median(noise)<<','
                   <<slope[0]<<','<<lin[0]<<','<<slope[1]<<','<<lin[1]<<','<<pass<<','<<axis_ok<<','<<ymax<<','<<tmax<<','<<median(ms)<<'\n';
            summary.flush();
            std::cout<<"  통과 "<<pass<<"/280, 오매칭 "<<(tot>0?100*mis/tot:0)<<" %, 잡음 "<<median(noise)<<" px, "<<median(ms)<<" ms"<<(all_ok?"":" (요약 실패 있음)")<<std::endl;
        }
        std::ofstream methods_csv(out/"methods.csv");for(size_t i=0;i<methods.size();++i)methods_csv<<i<<','<<methods[i].name<<'\n';
        return 0;
    }catch(const std::exception& e){std::cerr<<"실패: "<<e.what()<<std::endl;return 1;}
}
