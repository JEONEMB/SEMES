// control의 측정만 바꿔 다시 학습한다: (A) 채택 측정기 LK15 + 스플라인 GN, (B) OpenCV LK만.
// 점(536), P12 정규화, 원래 4-fold와 세션, 학습식(train_yort: 6절편·2차 응답·축소 0.05)은 같게 둔다.
#include "experiment.hpp"
#include "control_internal.hpp"
#include <opencv2/video/tracking.hpp>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <regex>
#include <sstream>
using namespace study;
namespace fs=std::filesystem;
using Clock=std::chrono::steady_clock;
static double elapsed(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
static std::vector<std::string> split(const std::string& text){std::vector<std::string> out;std::stringstream ss(text);std::string s;while(std::getline(ss,s,',')){if(!s.empty()&&s.back()=='\r')s.pop_back();out.push_back(s);}return out;}

static const cv::Rect crop(1340,800,1130,530);
static const cv::Rect qr[]={cv::Rect(1552,969,184,198),cv::Rect(1809,971,184,198),cv::Rect(2067,973,185,198)};

// (B) QR별 NCC 정수 이동을 초기값으로 OpenCV LK(21×21, 피라미드 2단) 정·역방향. 역방향 오차 0.5 px 이하만 유효
struct LkTracker{
    cv::Mat center;std::vector<cv::Point2d> points;std::vector<int> region;std::vector<cv::Point2d> zero;std::vector<unsigned char> zero_ok;
    void measure(const cv::Mat& gray,std::vector<cv::Point2d>& d,std::vector<unsigned char>& ok)const{
        cv::Point shift[3];
        for(int k=0;k<3;++k){cv::Rect search(qr[k].x-110,qr[k].y-22,qr[k].width+220,qr[k].height+44);cv::Mat c;
            cv::matchTemplate(gray(search),center(qr[k]),c,cv::TM_CCOEFF_NORMED);cv::Point loc;cv::minMaxLoc(c,nullptr,nullptr,nullptr,&loc);shift[k]=loc-cv::Point(110,22);}
        std::vector<cv::Point2f> p0,p1,back;for(size_t i=0;i<points.size();++i){p0.emplace_back(float(points[i].x-crop.x),float(points[i].y-crop.y));p1.push_back(p0.back()+cv::Point2f(shift[region[i]]));}
        back=p0;std::vector<unsigned char> s1,s2;std::vector<float> e1,e2;auto crit=cv::TermCriteria(cv::TermCriteria::COUNT|cv::TermCriteria::EPS,40,.001);
        cv::calcOpticalFlowPyrLK(center(crop),gray(crop),p0,p1,s1,e1,cv::Size(21,21),2,crit,cv::OPTFLOW_USE_INITIAL_FLOW);
        cv::calcOpticalFlowPyrLK(gray(crop),center(crop),p1,back,s2,e2,cv::Size(21,21),2,crit,cv::OPTFLOW_USE_INITIAL_FLOW);
        d.resize(points.size());ok.resize(points.size());
        for(size_t i=0;i<points.size();++i){d[i]={double(p1[i].x-p0[i].x),double(p1[i].y-p0[i].y)};ok[i]=s1[i]&&s2[i]&&cv::norm(back[i]-p0[i])<=.5;}
    }
    // 채택 측정기의 Tracker::features()와 같은 정규화로 P12를 적합한다. 유효점만 쓴다
    cv::Mat features(const cv::Mat& gray,int& valid)const{
        std::vector<cv::Point2d> d;std::vector<unsigned char> ok;measure(gray,d,ok);
        std::vector<int> rows;for(size_t i=0;i<points.size();++i)if(ok[i]&&zero_ok[i])rows.push_back(int(i));valid=int(rows.size());
        require(valid>=100,"LK valid points below 100");
        cv::Mat basis(valid,6,CV_64F),diff(valid,2,CV_64F);
        for(int r=0;r<valid;++r){int i=rows[r];double u=(points[i].x-1906)/486,v=(points[i].y-1070)/122,b[]={1,u,v,u*u,u*v,v*v};
            for(int j=0;j<6;++j)basis.at<double>(r,j)=b[j];diff.at<double>(r,0)=d[i].x-zero[i].x;diff.at<double>(r,1)=d[i].y-zero[i].y;}
        cv::Mat c;require(cv::solve(basis,diff,c,cv::DECOMP_SVD),"LK P12 solve failed");return c.reshape(1,1).clone();
    }
};

static std::vector<Sample> load_samples(const fs::path& root,const fs::path& control){
    std::map<std::string,Sample> labels;std::ifstream f(control/"validation/python_predictions.csv");require(bool(f),"Missing frozen labels/folds");std::string line;std::getline(f,line);
    while(std::getline(f,line)){auto v=split(line);require(v.size()==14,"Invalid frozen label row");if(v[12]!="raw_oof4")continue;Sample s;s.file=v[0];s.axis=v[1];s.value=std::stod(v[2]);s.session=v[3];s.fold=std::stoi(v[13]);
        s.date="20261014";if(s.axis=="Y")s.y=s.value;else s.t=s.value;s.path=root/"clean_20261014"/(s.axis=="Y"?"Y":"Theta")/fs::u8path(s.file);labels[s.file]=s;}
    require(labels.size()==280,"Expected frozen280 split");std::vector<Sample> data;for(auto& e:labels)data.push_back(e.second);
    std::regex yr("Y([+-][0-9]+(?:\\.[0-9]+)?)"),tr("T([+-][0-9]+(?:\\.[0-9]+)?)");
    for(auto day:{"clean_20261020","clean_20261029"}){std::vector<fs::path> paths;for(auto& e:fs::recursive_directory_iterator(root/day))if(e.path().extension()==L".bmp")paths.push_back(e.path());std::sort(paths.begin(),paths.end());
        for(auto& p:paths){Sample s;s.path=p;s.file=fs::relative(p,root/day).generic_u8string();s.date=std::string(day).substr(6);s.axis="mixed";auto name=s.file;std::smatch m;
            if(std::regex_search(name,m,yr))s.y=std::stod(m[1]);if(std::regex_search(name,m,tr))s.t=std::stod(m[1]);data.push_back(s);}}
    Sample center;center.file="center.bmp";center.date="20261014";center.axis="center";center.path=root/"clean_20261014/center.bmp";data.push_back(center);return data;
}

struct Stats{int n=0,failed=0,pass=0,axis_ok=0;double y_sum=0,y_max=0,t_sum=0,t_max=0;};
static void evaluate(const Model& model,int k,const std::string& cond,const std::string& protocol,int fold,const std::vector<Sample>& data,const std::vector<int>& ids,std::ostream& out,std::map<std::string,Stats>& sums){
    for(int i:ids){const auto& s=data[i];std::vector<std::string> groups{protocol};
        if(s.date=="20261014")groups.push_back(protocol+(s.axis=="Y"?"_Y80":"_T200"));
        if(s.axis=="mixed")groups.push_back(protocol+(s.y==0&&s.t==0?"_centers":"_mixed"));
        bool valid=false;Prediction q;double ey=0,et=0;
        try{require(s.valid[k],"Measurement failed");q=model.predict(s.feature[k]);ey=std::abs(q.y-s.y);et=std::abs(q.t-s.t);valid=true;}catch(const std::exception&){}
        out<<cond<<','<<protocol<<','<<fold<<','<<s.date<<','<<s.file<<','<<s.y<<','<<s.t<<',';
        if(valid)out<<q.y<<','<<q.t<<','<<ey<<','<<et<<",1,"<<(ey<=.09&&et<=.09)<<','<<q.branch<<'\n';else out<<",,,,0,0,\n";
        for(auto& g:groups){auto& a=sums[cond+","+g];++a.n;if(!valid){++a.failed;continue;}a.y_sum+=ey;a.t_sum+=et;a.y_max=std::max(a.y_max,ey);a.t_max=std::max(a.t_max,et);
            a.pass+=ey<=.09&&et<=.09;a.axis_ok+=(s.axis=="Y"||s.axis=="T")?q.axis==s.axis:0;}
    }
}
static double percentile(std::vector<double> v,double p){std::sort(v.begin(),v.end());double x=(v.size()-1)*p;size_t lo=size_t(x);return v[lo]+(v[std::min(lo+1,v.size()-1)]-v[lo])*(x-lo);}

// (C) 채택 측정기의 LK 단계만: 같은 NCC, QR 영역 σ0.8 흐림, LK 15×15 피라미드 없음, 60회/1e-4, GN 없음. P12는 536점 전부(채택과 같음)
struct LkNoGn{
    cv::Mat reference;std::vector<cv::Point2f> panel[3];std::vector<int> ids[3];cv::Mat blurred[3];int n=0;
    void init(const cv::Mat& center,const std::vector<cv::Point2d>& points,const std::vector<int>& region){
        reference=center;n=int(points.size());
        for(size_t i=0;i<points.size();++i){int k=region[i];cv::Point2f c(float(points[i].x-crop.x),float(points[i].y-crop.y));auto local=qr[k].tl()-crop.tl();
            panel[k].emplace_back(c.x-float(local.x),c.y-float(local.y));ids[k].push_back(int(i));}
        for(int k=0;k<3;++k)cv::GaussianBlur(center(qr[k]).clone(),blurred[k],cv::Size(),.8);
    }
    cv::Mat displacement(const cv::Mat& gray)const{
        cv::Mat d(n,2,CV_64F);auto ref=reference(crop),cur=gray(crop);
        for(int k=0;k<3;++k){auto b=qr[k];b.x-=crop.x;b.y-=crop.y;cv::Rect search(b.x-110,b.y-22,b.width+220,b.height+44);
            cv::Mat c;cv::matchTemplate(cur(search),ref(b),c,cv::TM_CCOEFF_NORMED);cv::Point loc;cv::minMaxLoc(c,nullptr,nullptr,nullptr,&loc);cv::Point shift=loc-cv::Point(110,22);
            cv::Mat target=cur(b+shift).clone(),bl;cv::GaussianBlur(target,bl,cv::Size(),.8);
            std::vector<cv::Point2f> next=panel[k];std::vector<unsigned char> st;std::vector<float> er;
            cv::calcOpticalFlowPyrLK(blurred[k],bl,panel[k],next,st,er,cv::Size(15,15),0,cv::TermCriteria(cv::TermCriteria::COUNT|cv::TermCriteria::EPS,60,.0001),cv::OPTFLOW_USE_INITIAL_FLOW,1e-5);
            for(size_t j=0;j<next.size();++j){d.at<double>(ids[k][j],0)=next[j].x-panel[k][j].x+shift.x;d.at<double>(ids[k][j],1)=next[j].y-panel[k][j].y+shift.y;}}
        return d;
    }
};

int main(){try{
    cv::setNumThreads(1);
    auto repo=fs::u8path(REPO_DIR),exp=fs::u8path(EXPERIMENT_DIR),root=repo/"data/clean",control=repo/"experiments/hye/20261028_1327_final_Y_T control";
    auto out=exp/"results_v2";require(!fs::exists(out),"results_v2 폴더가 이미 있다. 덮어쓰지 않는다");fs::create_directories(out/"models");
    // OpenCV FileStorage는 한글 경로에 쓰지 못하므로 모델은 영문 임시 폴더에 쓰고 마지막에 복사한다
    auto model_dir=fs::temp_directory_path()/"control_lk_models_20261029_v2";fs::remove_all(model_dir);fs::create_directories(model_dir);
    auto data=load_samples(root,control);const int N=int(data.size());std::cout<<"samples "<<N<<std::endl;

    legacy_control::Tracker tracker(control/"assets");
    LkTracker lk;lk.center=legacy_control::read_gray(control/"assets/center.bmp");
    {cv::FileStorage f((control/"assets/points.json").u8string(),cv::FileStorage::READ);
        for(auto it=f["points"].begin();it!=f["points"].end();++it){auto r=*it;if(int(r[3])!=1)continue;lk.points.emplace_back(double(r[0]),double(r[1]));lk.region.push_back(int(r[2]));}}
    require(lk.points.size()==536,"Expected536 points");
    lk.measure(lk.center,lk.zero,lk.zero_ok);
    LkNoGn nogn;nogn.init(lk.center,lk.points,lk.region);cv::Mat zeroC=nogn.displacement(lk.center);
    auto p12_all=[&](const cv::Mat& disp){cv::Mat basis(536,6,CV_64F),diff(536,2,CV_64F);
        for(int i=0;i<536;++i){double u=(lk.points[i].x-1906)/486,v=(lk.points[i].y-1070)/122,b[]={1,u,v,u*u,u*v,v*v};for(int j=0;j<6;++j)basis.at<double>(i,j)=b[j];
            diff.at<double>(i,0)=disp.at<double>(i,0)-zeroC.at<double>(i,0);diff.at<double>(i,1)=disp.at<double>(i,1)-zeroC.at<double>(i,1);}
        cv::Mat c;require(cv::solve(basis,diff,c,cv::DECOMP_SVD),"P12 solve failed");return cv::Mat(c.reshape(1,1).clone());};

    // F[k][i]: 조건 k의 특징, V: 유효, T: 측정 시간
    std::vector<cv::Mat> F[3];std::vector<char> V[3];std::vector<double> T[3];for(int k=0;k<3;++k){F[k].resize(N);V[k].assign(N,0);T[k].assign(N,0);}
    for(int i=0;i<N;++i){auto img=legacy_control::read_gray(data[i].path);
        auto t0=Clock::now();try{auto t=tracker.measure(img);F[0][i]=tracker.features(t);V[0][i]=1;}catch(const std::exception&){}T[0][i]=elapsed(t0);
        t0=Clock::now();int nv=0;try{F[1][i]=lk.features(img,nv);V[1][i]=1;}catch(const std::exception&){}T[1][i]=elapsed(t0);
        t0=Clock::now();try{F[2][i]=p12_all(nogn.displacement(img));V[2][i]=1;}catch(const std::exception&){}T[2][i]=elapsed(t0);
        if((i+1)%50==0)std::cout<<"measured "<<i+1<<"/"<<N<<std::endl;}
    for(int k=0;k<3;++k){require(V[k][N-1],"center measurement failed");auto z=F[k][N-1].clone();for(int i=0;i<N;++i)if(V[k][i])F[k][i]=F[k][i]-z;}

    std::vector<int> all,m1020,m1029;for(int i=0;i<N-1;++i){if(i<280)all.push_back(i);else (data[i].date=="20261020"?m1020:m1029).push_back(i);}
    std::ofstream rows(out/"predictions.csv");rows<<std::setprecision(17)<<"condition,protocol,fold,date,file,Y,T,predY,predT,Y_error,T_error,valid,both_pass,branch\n";
    std::map<std::string,Stats> sums;double baseline=0;int mismatch=0;legacy_control::Model frozen(control/"assets/model.json");
    const char* names[]={"A_NCC_LK15_GN17(채택)","B_NCC_LK21피라미드_유효점","C_NCC_LK15(채택LK단계)_GN없음"};const char* tags[]={"A","B","C"};
    for(int k=0;k<3;++k){
        for(int i=0;i<N;++i){data[i].feature[0]=F[k][i];data[i].valid[0]=V[k][i]!=0;}
        for(int fold=0;fold<4;++fold){std::vector<int> tr,te;for(int i:all)(data[i].fold==fold?te:tr).push_back(i);require(tr.size()==210&&te.size()==70,"Unexpected split");
            auto model=train_yort(data,tr,0);model.save(model_dir/(std::string(tags[k])+"_fold"+std::to_string(fold)+".json"));evaluate(model,0,names[k],"oof4",fold,data,te,rows,sums);
            if(k==0){legacy_control::Model ref(control/"validation"/("fold"+std::to_string(fold)+"_model.json"));
                for(int i:te){auto q=model.predict(data[i].feature[0]);auto r=ref.predict(data[i].feature[0]);baseline=std::max(baseline,std::abs((q.axis=="Y"?q.y:q.t)-r.value));mismatch+=q.axis!=r.axis;}}}
        auto model=train_yort(data,all,0);model.save(model_dir/(std::string(tags[k])+"_full.json"));
        evaluate(model,0,names[k],"resub",-1,data,all,rows,sums);evaluate(model,0,names[k],"transfer20261020",-1,data,m1020,rows,sums);evaluate(model,0,names[k],"transfer20261029",-1,data,m1029,rows,sums);
        if(k==0)for(int i:all){auto q=model.predict(data[i].feature[0]);auto r=frozen.predict(data[i].feature[0]);baseline=std::max(baseline,std::abs((q.axis=="Y"?q.y:q.t)-r.value));mismatch+=q.axis!=r.axis;}
    }
    std::ofstream sum(out/"summary.csv");sum<<std::setprecision(10)<<"condition,protocol,n,failed,both_pass,axis_correct,Y_mae,Y_max,T_mae,T_max\n";
    for(auto& [key,a]:sums)sum<<key<<','<<a.n<<','<<a.failed<<','<<a.pass<<','<<a.axis_ok<<','<<a.y_sum/std::max(1,a.n-a.failed)<<','<<a.y_max<<','<<a.t_sum/std::max(1,a.n-a.failed)<<','<<a.t_max<<'\n';
    std::ofstream timing(out/"timing.csv");timing<<"condition,n,measure_median_ms,measure_p95_ms\n";
    for(int k=0;k<3;++k){std::vector<double> t;for(int i:all)t.push_back(T[k][i]);timing<<names[k]<<','<<t.size()<<','<<percentile(t,.5)<<','<<percentile(t,.95)<<'\n';}
    std::ofstream audit(out/"audit.json");audit<<std::setprecision(17)<<"{\"A_reproduces_frozen_max_difference\":"<<baseline<<",\"A_axis_mismatch\":"<<mismatch<<",\"folds\":\"validation/python_predictions.csv raw_oof4\",\"shrink\":0.05,\"final_module_changed\":false}\n";
    for(auto& e:fs::directory_iterator(model_dir))fs::copy_file(e.path(),out/"models"/e.path().filename());
    require(baseline<1e-6&&mismatch==0,"A가 기존 control을 재현하지 못했다");
    std::cout<<"done, A reproduces frozen within "<<baseline<<std::endl;return 0;
}catch(const std::exception& e){std::cerr<<"실패: "<<e.what()<<std::endl;return 1;}}
