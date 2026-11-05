#include "final_pin.hpp"
#include "observer_internal.hpp"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <map>
#include <set>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
namespace fs=std::filesystem;
using Clock=std::chrono::steady_clock;
static double ms(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
const std::vector<std::string> names={"v1_joint_GLS","v2_joint","pin_only_linear","pin_only_quadratic","qr_pin_linear","qr_pin_quadratic","qr_pin_bars_linear","qr_pin_bars_quadratic"};
static std::vector<std::string> split(std::string s,char delim){if(!s.empty()&&s.back()=='\r')s.pop_back();std::vector<std::string> v;std::string x;bool q=false;for(size_t i=0;i<s.size();++i){if(s[i]=='"'){if(q&&i+1<s.size()&&s[i+1]=='"'){x+='"';++i;}else q=!q;}else if(s[i]==delim&&!q){v.push_back(x);x.clear();}else x+=s[i];}v.push_back(x);return v;}
static cv::Mat numbers(const std::string&s){if(s.empty())return {};auto v=split(s,';');cv::Mat x(1,int(v.size()),CV_64F);for(int j=0;j<x.cols;++j)x.at<double>(j)=std::stod(v[j]);return x;}
static std::vector<int> shuffled(std::vector<int> ids,unsigned seed){std::mt19937 g(seed);for(unsigned i=unsigned(ids.size()-1);i>0;--i){unsigned n=i+1,t=unsigned(-n)%n,v;do{v=g();}while(v<t);std::swap(ids[i],ids[v%n]);}return ids;}
static cv::Mat basis(double y,double t,int degree){cv::Mat b=(cv::Mat_<double>(1,6)<<1.,y,t,y*y,y*t,t*t);return degree==1?b.colRange(0,3).clone():b;}
struct Row{int id;std::string date,path,hash,group;double y,t;cv::Mat x[8];};
static std::vector<Row> load(const fs::path&root,std::string&hash){
    const std::string rel="experiments/hye/20261104_173000_pin_random5fold/results_with_bars/features.tsv";
    hash=alignment_detail::sha256(root/fs::u8path(rel));
    cv::FileStorage provenance((root/"experiments/hye/20261104_173000_pin_random5fold/artifact_hashes_with_bars.json").u8string(),cv::FileStorage::READ);std::string expected;
    for(auto n:provenance["files"])if(std::string(n["path"])==rel)expected=std::string(n["sha256"]);
    if(expected.empty()||hash!=expected)throw std::runtime_error("Frozen feature SHA256 mismatch");
    std::map<std::string,std::vector<std::string>> truth;std::ifstream labels(root/"data/fivefold_seed20261104/_metadata/images.csv");std::string l;std::getline(labels,l);while(std::getline(labels,l)){auto v=split(l,',');if(v.size()!=11)throw std::runtime_error("Bad original metadata");truth["data/clean/clean_"+v[0]+"/"+v[2]]=v;}
    cv::FileStorage protocol((root/"experiments/hye/v2_Y_T_joint/assets/line_protocol.json").u8string(),cv::FileStorage::READ);std::vector<double> bx;
    for(auto n:protocol["blocks"]){auto b=n["read_box"];bx.push_back((double(b[0])+double(b[2]))/2.);}
    std::ifstream in(root/fs::u8path(rel));std::getline(in,l);std::vector<Row> rows;std::set<std::string> hashes;
    while(std::getline(in,l)){auto v=split(l,'\t');if(v.size()!=16)throw std::runtime_error("Bad feature row");Row r;r.id=std::stoi(v[0]);r.date=v[1];r.path=v[2];r.hash=v[3];r.group=v[4];r.y=std::stod(v[5]);r.t=std::stod(v[6]);
        if(r.id!=int(rows.size())||!hashes.insert(r.hash).second||r.path.find("center")!=std::string::npos)throw std::runtime_error("Bad image identity");auto it=truth.find(r.path);if(it==truth.end()||it->second[3]!=r.hash||std::abs(std::stod(it->second[4])-r.y)>1e-12||std::abs(std::stod(it->second[5])-r.t)>1e-12)throw std::runtime_error("Feature/label mismatch");
        auto qr=numbers(v[8]),pins=numbers(v[9]),v2=numbers(v[11]);
        if(!qr.empty()){if(qr.cols!=102||!cv::checkRange(qr))throw std::runtime_error("Invalid QR features");r.x[0]=cv::Mat(1,204,CV_64F);qr.copyTo(r.x[0].colRange(0,102));double h=0;for(int k=0;k<54;k+=2)h+=qr.at<double>(k)/2700.;for(int k=0;k<102;++k)r.x[0].at<double>(k+102)=qr.at<double>(k)*h;r.x[1]=v2;}
        if(!pins.empty()){if(pins.cols!=8||!cv::checkRange(pins))throw std::runtime_error("Invalid pin features");r.x[2]=r.x[3]=pins;}
        if(!qr.empty()&&!pins.empty()){r.x[4]=r.x[5]=final_pin::relative_features(qr,pins);r.x[6]=r.x[7]=final_pin::with_bars_features(qr,pins,bx);}rows.push_back(r);
    }if(rows.size()!=571||truth.size()!=571)throw std::runtime_error("Expected 571 original non-center images");return rows;
}
struct GLS{cv::Mat origin,P;
    void fit(const cv::Mat& X,const cv::Mat& Y){cv::Mat A=cv::Mat::ones(X.rows,3,CV_64F);Y.copyTo(A.colRange(1,3));cv::Mat co;cv::solve(A,X,co,cv::DECOMP_SVD);cv::Mat R=X-A*co,sd(1,X.cols,CV_64F),Z=R.clone();for(int k=0;k<X.cols;++k){sd.at<double>(k)=std::max(std::sqrt(R.col(k).dot(R.col(k))/R.rows),1e-6);Z.col(k)/=sd.at<double>(k);}cv::Mat cov=.99*(Z.t()*Z/double(Z.rows))+.01*cv::Mat::eye(X.cols,X.cols,CV_64F),B=co.rowRange(1,3).clone();for(int k=0;k<X.cols;++k)B.col(k)/=sd.at<double>(k);cv::Mat Q;cv::solve(cov,B.t(),Q,cv::DECOMP_SVD);cv::Mat inv;cv::invert(B*Q,inv,cv::DECOMP_SVD);P=Q*inv;for(int k=0;k<X.cols;++k)P.row(k)/=sd.at<double>(k);origin=co.row(0).clone();}
    cv::Vec2d predict(const cv::Mat&x)const{cv::Mat y=(x-origin)*P;return {y.at<double>(0),y.at<double>(1)};}
};
struct Model{int kind;GLS gls;final_pin::Regressor reg;cv::Mat world[2];
    double fit(const cv::Mat&X,const cv::Mat&Y,const cv::Mat&A){auto t=Clock::now();if(kind==0)gls.fit(X,Y);else reg.fit(X,Y,kind==3||kind==5||kind==7?2:1,kind==1?0:1e-4);double elapsed=ms(t);cv::solve(A.colRange(0,3),X,world[0],cv::DECOMP_SVD);cv::solve(A,X,world[1],cv::DECOMP_SVD);return elapsed;}
    cv::Vec2d predict(const cv::Mat&x)const{return kind==0?gls.predict(x):reg.predict(x);}
};
struct Sim{int turns=0;bool stop=false,pass=false;double y,t;};
static Sim simulate(const Model&m,const cv::Mat&anchor,const Row&r,int degree){Sim s;s.y=r.y;s.t=r.t;for(;;){cv::Mat x=anchor+(basis(s.y,s.t,degree)-basis(r.y,r.t,degree))*m.world[degree-1];auto p=m.predict(x);if(!std::isfinite(p[0]+p[1]))break;if(std::abs(p[0])<=.045&&std::abs(p[1])<=.01){s.stop=true;break;}if(s.turns==20)break;s.y-=std::max(-.5,std::min(.5,p[0]));s.t-=std::max(-.25,std::min(.25,p[1]));++s.turns;if(std::abs(s.y)>2.1||std::abs(s.t)>2.1)break;}s.pass=s.stop&&std::abs(s.y)<=.045+1e-12&&std::abs(s.t)<=.01+1e-12;return s;}
struct Stats{long n=0,valid=0,py=0,pt=0,pb=0;double ey=0,et=0,my=0,mt=0;long simpass[2]={0,0},false_stop[2]={0,0};double turns[2]={0,0};int maxturns[2]={0,0},fitfail=0;std::vector<double> rates,fitms,predms;};
static double quantile(std::vector<double> v,double q){if(v.empty())return std::numeric_limits<double>::quiet_NaN();std::sort(v.begin(),v.end());return v[std::max(0,std::min(int(v.size())-1,int(std::ceil(q*v.size()))-1))];}
static void evaluate(const fs::path&root,const fs::path&out,const std::vector<Row>&rows,const std::string&hash){
    fs::create_directories(out/"predictions");fs::create_directories(out/"full_models");
    std::ofstream members(out/"train_membership.tsv"),foldfile(out/"test_folds.tsv"),data(out/"dataset.tsv"),runs(out/"runs.tsv");
    members<<"seed\tfold\tsubset\tbudget\trank\tindex\n";foldfile<<"seed\tfold\tindex\n";data<<"index\tdate\tpath\tsha256\tgroup\tY\tT\n";data<<std::setprecision(17);for(auto&r:rows)data<<r.id<<'\t'<<r.date<<'\t'<<r.path<<'\t'<<r.hash<<'\t'<<r.group<<'\t'<<r.y<<'\t'<<r.t<<'\n';
    runs<<std::setprecision(17)<<"model\tseed\tfold\tsubset\tbudget\tn_train\tn_valid_train\tn_test\tn_valid_test\tpass_Y\tpass_T\tpass_both\tMAE_Y\tMAE_T\tmax_eY\tmax_eT\tsim1_pass\tsim2_pass\tsim1_false_stop\tsim2_false_stop\tfit_ms\tpredict_ms_per_image\tn_1014\tn_1020\tn_1029\ttrain_min_Y\ttrain_max_Y\ttrain_min_T\ttrain_max_T\terror\n";
    std::map<std::pair<int,int>,Stats> total;int fits=0;long predictions=0;
    std::vector<int> canonical;for(int i=0;i<int(rows.size());++i)canonical.push_back(i);
    for(unsigned seed:{20261105u,20261106u,20261107u}){
        auto order=shuffled(canonical,seed);std::vector<int> fold(rows.size());for(int i=0;i<int(order.size());++i)fold[order[i]]=i%5+1;
        for(int id:canonical)foldfile<<seed<<'\t'<<fold[id]<<'\t'<<id<<'\n';
        std::vector<std::ofstream> pr(8);for(int m=0;m<8;++m){pr[m].open(out/"predictions"/(names[m]+"_"+std::to_string(seed)+".tsv"));pr[m]<<std::setprecision(17)<<"fold\tsubset\tbudget\tindex\tY\tT\tpred_Y\tpred_T\tabs_eY\tabs_eT\tpass_both\tsim1_pass\tsim2_pass\tsim1_turns\tsim2_turns\terror\n";}
        for(int f=1;f<=5;++f){std::vector<int> train,test;for(int id:canonical)(fold[id]==f?test:train).push_back(id);
            if(train.size()+test.size()!=rows.size())throw std::runtime_error("Fold partition failure");
            for(int subset=0;subset<2;++subset){auto selected=shuffled(train,seed^(0x9e3779b9u*unsigned(f))^(0x85ebca6bu*unsigned(subset+1)));
                for(int budget:{10,20,40,80,160,320,0}){if(budget==0&&subset==1)continue;int count=budget==0?int(train.size()):budget;
                    std::vector<int> ids=budget==0?train:std::vector<int>(selected.begin(),selected.begin()+count);int dates[3]={0,0,0};double ymin=1e9,ymax=-1e9,tmin=1e9,tmax=-1e9;
                    for(int k=0;k<count;++k){int id=ids[k];if(fold[id]==f)throw std::runtime_error("Train/test leakage");members<<seed<<'\t'<<f<<'\t'<<subset<<'\t'<<(budget==0?"all":std::to_string(budget))<<'\t'<<k<<'\t'<<id<<'\n';auto&r=rows[id];++dates[r.date=="20261014"?0:r.date=="20261020"?1:2];ymin=std::min(ymin,r.y);ymax=std::max(ymax,r.y);tmin=std::min(tmin,r.t);tmax=std::max(tmax,r.t);}
                    for(int m=0;m<8;++m){Model model;model.kind=m;cv::Mat X,Y,A;Stats s;std::string error;double trainms=0,predictionms=0;
                        for(int id:ids){auto&r=rows[id];if(r.x[m].empty())continue;X.push_back(r.x[m]);Y.push_back(cv::Mat((cv::Mat_<double>(1,2)<<r.y,r.t)));A.push_back(basis(r.y,r.t,2));}
                        try{if(X.rows<10)throw std::runtime_error("Fewer than 10 valid training images");trainms=model.fit(X,Y,A);
                            if(budget==0){auto path=out/"full_models"/(names[m]+"_"+std::to_string(seed)+"_fold"+std::to_string(f)+".yml");if(m==0){cv::FileStorage save(path.u8string(),cv::FileStorage::WRITE);save<<"origin"<<model.gls.origin<<"projector"<<model.gls.P;}else model.reg.save(path.u8string());}
                        }catch(const std::exception&e){error=e.what();++s.fitfail;}
                        for(int id:test){auto&r=rows[id];++s.n;++predictions;cv::Vec2d p{std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::quiet_NaN()};std::string failure=error;Sim sims[2];sims[0].y=sims[1].y=r.y;sims[0].t=sims[1].t=r.t;
                            if(failure.empty())try{if(r.x[m].empty())throw std::runtime_error("Measurement failure");auto t=Clock::now();p=model.predict(r.x[m]);predictionms+=ms(t);if(!std::isfinite(p[0]+p[1]))throw std::runtime_error("Nonfinite prediction");for(int world=1;world<=2;++world)sims[world-1]=simulate(model,r.x[m],r,world);}catch(const std::exception&e){failure=e.what();}
                            double ey=std::abs(p[0]-r.y),et=std::abs(p[1]-r.t);bool py=failure.empty()&&ey<=.045+1e-12,pt=failure.empty()&&et<=.01+1e-12;
                            if(failure.empty()){++s.valid;s.ey+=ey;s.et+=et;s.my=std::max(s.my,ey);s.mt=std::max(s.mt,et);}s.py+=py;s.pt+=pt;s.pb+=py&&pt;
                            for(int w=0;w<2;++w){s.simpass[w]+=sims[w].pass;s.false_stop[w]+=sims[w].stop&&!sims[w].pass;s.turns[w]+=sims[w].turns;s.maxturns[w]=std::max(s.maxturns[w],sims[w].turns);}
                            pr[m]<<f<<'\t'<<subset<<'\t'<<(budget==0?"all":std::to_string(budget))<<'\t'<<id<<'\t'<<r.y<<'\t'<<r.t<<'\t'<<p[0]<<'\t'<<p[1]<<'\t'<<ey<<'\t'<<et<<'\t'<<(py&&pt)<<'\t'<<sims[0].pass<<'\t'<<sims[1].pass<<'\t'<<sims[0].turns<<'\t'<<sims[1].turns<<'\t'<<failure<<'\n';
                        }
                        runs<<names[m]<<'\t'<<seed<<'\t'<<f<<'\t'<<subset<<'\t'<<(budget==0?"all":std::to_string(budget))<<'\t'<<count<<'\t'<<X.rows<<'\t'<<s.n<<'\t'<<s.valid<<'\t'<<s.py<<'\t'<<s.pt<<'\t'<<s.pb<<'\t'<<s.ey/std::max(1L,s.valid)<<'\t'<<s.et/std::max(1L,s.valid)<<'\t'<<s.my<<'\t'<<s.mt<<'\t'<<s.simpass[0]<<'\t'<<s.simpass[1]<<'\t'<<s.false_stop[0]<<'\t'<<s.false_stop[1]<<'\t'<<trainms<<'\t'<<predictionms/std::max(1L,s.valid)<<'\t'<<dates[0]<<'\t'<<dates[1]<<'\t'<<dates[2]<<'\t'<<ymin<<'\t'<<ymax<<'\t'<<tmin<<'\t'<<tmax<<'\t'<<error<<'\n';
                        auto&g=total[{m,budget}];g.n+=s.n;g.valid+=s.valid;g.py+=s.py;g.pt+=s.pt;g.pb+=s.pb;g.ey+=s.ey;g.et+=s.et;g.my=std::max(g.my,s.my);g.mt=std::max(g.mt,s.mt);g.fitfail+=s.fitfail;g.rates.push_back(100.*s.pb/s.n);g.fitms.push_back(trainms);g.predms.push_back(predictionms/std::max(1L,s.valid));for(int w=0;w<2;++w){g.simpass[w]+=s.simpass[w];g.false_stop[w]+=s.false_stop[w];g.turns[w]+=s.turns[w];g.maxturns[w]=std::max(g.maxturns[w],s.maxturns[w]);}++fits;
                    }
                }
            }std::cout<<"seed "<<seed<<" fold "<<f<<" fits "<<fits<<std::endl;
        }
    }
    if(fits!=1560||predictions!=178152)throw std::runtime_error("Unexpected experiment coverage");
    std::ofstream agg(out/"aggregate.tsv");agg<<std::setprecision(17)<<"model\tbudget\tfits\ttest_evaluations\tvalid\tpass_Y_pct\tpass_T_pct\tpass_both_pct\tworst_fit_pct\tp05_fit_pct\tMAE_Y\tMAE_T\tmax_eY\tmax_eT\tsim1_pass_pct\tsim2_pass_pct\tsim1_false_stop\tsim2_false_stop\tsim1_mean_turns\tsim2_mean_turns\tsim1_max_turns\tsim2_max_turns\tmedian_fit_ms\tmedian_predict_ms\tfit_failures\n";
    for(auto&item:total){int m=item.first.first,b=item.first.second;auto&s=item.second;agg<<names[m]<<'\t'<<(b==0?"all":std::to_string(b))<<'\t'<<s.rates.size()<<'\t'<<s.n<<'\t'<<s.valid<<'\t'<<100.*s.py/s.n<<'\t'<<100.*s.pt/s.n<<'\t'<<100.*s.pb/s.n<<'\t'<<quantile(s.rates,0)<<'\t'<<quantile(s.rates,.05)<<'\t'<<s.ey/std::max(1L,s.valid)<<'\t'<<s.et/std::max(1L,s.valid)<<'\t'<<s.my<<'\t'<<s.mt<<'\t'<<100.*s.simpass[0]/s.n<<'\t'<<100.*s.simpass[1]/s.n<<'\t'<<s.false_stop[0]<<'\t'<<s.false_stop[1]<<'\t'<<s.turns[0]/s.n<<'\t'<<s.turns[1]/s.n<<'\t'<<s.maxturns[0]<<'\t'<<s.maxturns[1]<<'\t'<<quantile(s.fitms,.5)<<'\t'<<quantile(s.predms,.5)<<'\t'<<s.fitfail<<'\n';}
    std::ofstream integrity(out/"integrity.txt");integrity<<"feature_sha256 "<<hash<<"\nimages 571\nsource_dates_mixed true\nfold_seeds 20261105 20261106 20261107\nfolds 5\nsubset_repeats 2\ntrain_budgets 10 20 40 80 160 320 all\nfits "<<fits<<"\ntest_evaluations "<<predictions<<"\ntrain_test_intersection 0\ntrain_only_standardization true\ntrain_only_world_model true\nreal_robot_trials 0\n";
}
static void benchmark(const fs::path&root,const fs::path&out,const std::vector<Row>&rows){
    auto assets=root/"experiments/hye/v2_Y_T_joint/assets";auto t=Clock::now();alignment_detail::Observer qr(assets);auto center=alignment_detail::read_gray(assets/"center.bmp");final_pin::Observer pin(center);double init=ms(t);qr.measure(center);pin.measure(center);
    std::vector<int> ids;for(int i=0;i<int(rows.size());++i)ids.push_back(i);ids=shuffled(ids,20261105u^0x12345678u);
    std::ofstream raw(out/"observation_timing.tsv"),summary(out/"observation_timing_summary.tsv");raw<<std::setprecision(17)<<"index\tdecode_ms\tqr_bars_ms\tpin_ms\n";std::vector<double> costs[3];
    for(int j=0;j<30;++j){auto&r=rows[ids[j]];auto path=root/fs::u8path(r.path);if(alignment_detail::sha256(path)!=r.hash)throw std::runtime_error("Timing image hash mismatch");t=Clock::now();auto image=alignment_detail::read_gray(path);double decode=ms(t);t=Clock::now();qr.measure(image);double q=ms(t);t=Clock::now();pin.measure(image);double p=ms(t);raw<<r.id<<'\t'<<decode<<'\t'<<q<<'\t'<<p<<'\n';costs[0].push_back(decode+q);costs[1].push_back(decode+p);costs[2].push_back(decode+q+p);}
    summary<<"runtime_path\tn_images\tmedian_observation_decode_ms\tp95_observation_decode_ms\n";for(int i=0;i<3;++i)summary<<(i==0?"joint_v1_v2":i==1?"pin_only":"qr_pin_with_or_without_bars")<<"\t30\t"<<quantile(costs[i],.5)<<'\t'<<quantile(costs[i],.95)<<'\n';std::ofstream env(out/"environment.txt");env<<"OpenCV "<<CV_VERSION<<"\nC++17\nOpenCV threads 1; sequential image worker 1\ninitialization_ms "<<init<<"\n";
#ifdef _MSC_VER
    env<<"MSVC "<<_MSC_VER<<"\n";
#endif
}
int main(int argc,char**argv){try{if(argc!=4)throw std::runtime_error("learning_curve evaluate|benchmark REPO OUTPUT");cv::setNumThreads(1);fs::path root=fs::u8path(argv[2]),out=fs::u8path(argv[3]);fs::create_directories(out);std::string hash;auto rows=load(root,hash);if(std::string(argv[1])=="evaluate")evaluate(root,out,rows,hash);else if(std::string(argv[1])=="benchmark")benchmark(root,out,rows);else throw std::runtime_error("Unknown mode");return 0;}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
