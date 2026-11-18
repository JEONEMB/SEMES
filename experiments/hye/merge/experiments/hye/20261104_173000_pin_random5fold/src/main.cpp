#include "final_pin.hpp"
#include "observer_internal.hpp"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <random>
#include <algorithm>
#include <thread>
#include <atomic>
#include <map>
#include <set>
#include <limits>
#include <chrono>
#include <cmath>
namespace fs=std::filesystem;
using Clock=std::chrono::steady_clock;
static double elapsed(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
static std::vector<std::string> fields(std::string s,char delim){
    if(!s.empty()&&s.back()=='\r')s.pop_back();std::vector<std::string> v;std::string x;bool quote=false;
    for(size_t i=0;i<s.size();++i){char c=s[i];if(c=='"'){if(quote&&i+1<s.size()&&s[i+1]=='"'){x+='"';++i;}else quote=!quote;}else if(c==delim&&!quote){v.push_back(x);x.clear();}else x+=c;}v.push_back(x);return v;
}
static cv::Mat numbers(const std::string&s){if(s.empty())return {};auto v=fields(s,';');cv::Mat x(1,int(v.size()),CV_64F);for(int j=0;j<x.cols;++j)x.at<double>(j)=std::stod(v[j]);return x;}
static void write_mat(std::ostream&o,const cv::Mat&m){for(int j=0;j<m.cols;++j){if(j)o<<';';o<<m.at<double>(j);}}
struct Row{std::string date,rel,hash,group,jerror,perror;double y,t,jms=0,pms=0;int fold=0;cv::Mat qr,pins,quality,v2,bars;};
static std::vector<Row> dataset(const fs::path& root){
    std::ifstream in(root/"data/fivefold_seed20261104/_metadata/images.csv");if(!in)throw std::runtime_error("Missing original image metadata");
    std::string l;std::getline(in,l);std::vector<Row> rows;std::set<std::string> hashes,paths;
    while(std::getline(in,l)){auto v=fields(l,',');if(v.size()!=11)throw std::runtime_error("Invalid image manifest");
        Row r;r.date=v[0];r.rel="data/clean/clean_"+v[0]+"/"+v[2];r.hash=v[3];r.y=std::stod(v[4]);r.t=std::stod(v[5]);r.group=v[6];
        if(v[8].find("center")!=std::string::npos)throw std::runtime_error("Center appeared in evaluation manifest");
        if(!hashes.insert(r.hash).second||!paths.insert(r.rel).second)throw std::runtime_error("Duplicate hash/path");rows.push_back(r);
    }
    if(rows.size()!=571)throw std::runtime_error("Expected exactly 571 non-center images");
    std::sort(rows.begin(),rows.end(),[](const Row&a,const Row&b){return a.rel<b.rel;});
    // Portable Fisher-Yates: rejection-sampled mt19937, no library-dependent shuffle.
    std::vector<int> order(rows.size());for(int i=0;i<int(order.size());++i)order[i]=i;std::mt19937 gen(20261104);
    for(unsigned i=unsigned(order.size()-1);i>0;--i){unsigned n=i+1,threshold=unsigned(-n)%n,v;do{v=gen();}while(v<threshold);std::swap(order[i],order[v%n]);}
    for(int i=0;i<int(order.size());++i)rows[order[i]].fold=i%5+1;return rows;
}
static cv::Mat fmap(const cv::Mat&raw){cv::Mat z(1,204,CV_64F);raw.copyTo(z.colRange(0,102));double h=0;for(int k=0;k<54;k+=2)h+=raw.at<double>(k);h/=2700;for(int k=0;k<102;++k)z.at<double>(k+102)=raw.at<double>(k)*h;return z;}
struct GLS{cv::Mat origin,P;void fit(const cv::Mat& X,const cv::Mat& Y){
    cv::Mat A=cv::Mat::ones(X.rows,3,CV_64F);Y.copyTo(A.colRange(1,3));cv::Mat co;cv::solve(A,X,co,cv::DECOMP_SVD);
    cv::Mat R=X-A*co,sd(1,X.cols,CV_64F),Z=R.clone();for(int k=0;k<X.cols;++k){sd.at<double>(k)=std::max(std::sqrt(R.col(k).dot(R.col(k))/R.rows),1e-6);Z.col(k)/=sd.at<double>(k);}
    cv::Mat cov=.99*(Z.t()*Z/double(Z.rows))+.01*cv::Mat::eye(X.cols,X.cols,CV_64F),B=co.rowRange(1,3).clone();for(int k=0;k<X.cols;++k)B.col(k)/=sd.at<double>(k);
    cv::Mat Q;cv::solve(cov,B.t(),Q,cv::DECOMP_SVD);cv::Mat inv;cv::invert(B*Q,inv,cv::DECOMP_SVD);P=Q*inv;for(int k=0;k<X.cols;++k)P.row(k)/=sd.at<double>(k);origin=co.row(0).clone();
}cv::Vec2d predict(const cv::Mat&x)const{cv::Mat q=(x-origin)*P;return {q.at<double>(0),q.at<double>(1)};}};
static cv::Mat target(const Row&r){return (cv::Mat_<double>(1,2)<<r.y,r.t);}
static cv::Mat basis(double y,double t,int degree){cv::Mat b=(cv::Mat_<double>(1,6)<<1.,y,t,y*y,y*t,t*t);return degree==1?b.colRange(0,3).clone():b;}
static cv::Mat features(const Row&r,const std::string&m){if(m=="v1_joint_GLS")return r.qr.empty()?cv::Mat():fmap(r.qr);if(m=="v2_joint")return r.v2;if(m.find("pin_only")==0)return r.pins;if(m.find("qr_pin_bars_")==0)return r.bars;if(r.qr.empty()||r.pins.empty())return {};return final_pin::relative_features(r.qr,r.pins);}
static void measure(const fs::path&root,const fs::path&out,std::vector<Row>&rows){
    auto assets=root/std::filesystem::u8path(u8"experiments/hye/20261104_1435_v2_joint_강등/assets");alignment_detail::Observer qr(assets);final_pin::Observer pin(alignment_detail::read_gray(assets/"center.bmp"));auto bx=qr.block_centers();
    fs::create_directories(out/"measure_cache");std::atomic<int> next{0},done{0};
    auto work=[&](){for(int i;(i=next++)<int(rows.size());){auto&r=rows[i];auto cache=out/"measure_cache"/(std::to_string(i)+".yml");
        // Cache is reusable only for identical source bytes and tracker version.
        std::string actual;try{actual=alignment_detail::sha256(root/fs::u8path(r.rel));if(actual!=r.hash)throw std::runtime_error("Source SHA256 mismatch");}catch(const std::exception&e){r.jerror=r.perror=e.what();continue;}
        if(fs::exists(cache)){cv::FileStorage f(cache.u8string(),cv::FileStorage::READ);std::string hash,version;f["hash"]>>hash;f["version"]>>version;if(hash==actual&&version=="pin_cubic_v1"){f["qr"]>>r.qr;f["pins"]>>r.pins;f["quality"]>>r.quality;f["v2"]>>r.v2;f["jms"]>>r.jms;f["pms"]>>r.pms;f["jerror"]>>r.jerror;f["perror"]>>r.perror;++done;continue;}}
        try{auto image=alignment_detail::read_gray(root/fs::u8path(r.rel));try{auto t=Clock::now();auto o=qr.measure(image);r.jms=elapsed(t);r.qr=o.values;auto v=alignment_detail::v2_features(o.values,bx);r.v2=(cv::Mat_<double>(1,2)<<v[0],v[1]);}catch(const std::exception&e){r.jerror=e.what();}try{auto t=Clock::now();auto m=pin.measure(image);r.pms=elapsed(t);r.pins=m.displacement;r.quality=m.quality;}catch(const std::exception&e){r.perror=e.what();}}catch(const std::exception&e){r.jerror=r.perror=e.what();}
        cv::FileStorage f(cache.u8string(),cv::FileStorage::WRITE);f<<"hash"<<actual<<"version"<<"pin_cubic_v1"<<"qr"<<r.qr<<"pins"<<r.pins<<"quality"<<r.quality<<"v2"<<r.v2<<"jms"<<r.jms<<"pms"<<r.pms<<"jerror"<<r.jerror<<"perror"<<r.perror;
        int n=++done;if(n%25==0)std::cout<<"measured "<<n<<"/"<<rows.size()<<std::endl;
    }};std::vector<std::thread> workers;for(int k=0;k<4;++k)workers.emplace_back(work);for(auto&t:workers)t.join();
    std::ofstream f(out/"features.tsv");f<<std::setprecision(17)<<"index\tdate\trelative_path\tsha256\tgroup\tY\tT\tfold\tQR102\tpins8\tquality8\tv2\tqr_ms\tpin_ms\tqr_error\tpin_error\n";
    for(int i=0;i<int(rows.size());++i){auto&r=rows[i];f<<i<<'\t'<<r.date<<'\t'<<r.rel<<'\t'<<r.hash<<'\t'<<r.group<<'\t'<<r.y<<'\t'<<r.t<<'\t'<<r.fold<<'\t';write_mat(f,r.qr);f<<'\t';write_mat(f,r.pins);f<<'\t';write_mat(f,r.quality);f<<'\t';write_mat(f,r.v2);f<<'\t'<<r.jms<<'\t'<<r.pms<<'\t'<<r.jerror<<'\t'<<r.perror<<'\n';}
}
static void load_measurements(const fs::path&out,std::vector<Row>&rows){std::ifstream f(out/"features.tsv");if(!f)throw std::runtime_error("Run measure first");std::string l;std::getline(f,l);std::set<int> seen;while(std::getline(f,l)){auto v=fields(l,'\t');if(v.size()!=16)throw std::runtime_error("Bad measurement row");int i=std::stoi(v[0]);auto&r=rows.at(i);if(r.rel!=v[2]||r.hash!=v[3]||r.fold!=std::stoi(v[7])||!seen.insert(i).second)throw std::runtime_error("Measurement manifest mismatch");r.qr=numbers(v[8]);r.pins=numbers(v[9]);r.quality=numbers(v[10]);r.v2=numbers(v[11]);r.jms=std::stod(v[12]);r.pms=std::stod(v[13]);r.jerror=v[14];r.perror=v[15];}if(seen.size()!=rows.size())throw std::runtime_error("Incomplete measurement table");}
static void populate_bars(const fs::path&root,std::vector<Row>&rows){
    cv::FileStorage protocol((root/std::filesystem::u8path(u8"experiments/hye/20261104_1435_v2_joint_강등/assets/line_protocol.json")).u8string(),cv::FileStorage::READ);
    if(!protocol.isOpened())throw std::runtime_error("Missing line protocol");
    std::vector<double> bx;for(auto it=protocol["blocks"].begin();it!=protocol["blocks"].end();++it){auto b=(*it)["read_box"];bx.push_back((double(b[0])+double(b[2]))/2.);}
    for(auto&r:rows)if(!r.qr.empty()&&!r.pins.empty())r.bars=final_pin::with_bars_features(r.qr,r.pins,bx);
}
static void evaluate(const fs::path&out,const std::vector<Row>&rows,bool include_bars=false){
    std::vector<std::string> models={"v1_joint_GLS","v2_joint","pin_only_linear","pin_only_quadratic","qr_pin_linear","qr_pin_quadratic"};
    if(include_bars){models.push_back("qr_pin_bars_linear");models.push_back("qr_pin_bars_quadratic");}
    std::ofstream pred(out/"predictions.tsv"),summary(out/"summary.tsv"),sim(out/"simulation.tsv"),integrity(out/"integrity.txt");
    pred<<std::setprecision(17)<<"model\tfold\tindex\tdate\tY\tT\tpred_Y\tpred_T\tabs_eY\tabs_eT\tpass_Y\tpass_T\tpass_both\terror\n";
    summary<<std::setprecision(17)<<"model\tfold\tn_test\tn_train_valid\tn_valid\tpass_Y\tpass_T\tpass_both\tMAE_Y_valid\tMAE_T_valid\tmax_eY_valid\tmax_eT_valid\n";
    sim<<std::setprecision(17)<<"model\tfold\tindex\tworld_degree\tturns\tstopped_by_estimate\ttrue_pass\tfinal_Y\tfinal_T\tstatus\n";
    std::set<std::string> joint_keys;int crossed=0;std::map<std::string,std::set<int>> group_folds;for(const auto&r:rows)group_folds[r.group].insert(r.fold);for(const auto&g:group_folds)crossed+=g.second.size()>1;
    integrity<<"seed 20261104\nnoncenter_images "<<rows.size()<<"\nsource_hashes_unique true\nimage_test_once true\ngroups_crossing_folds "<<crossed<<"\nrandom_image_split_intentionally_allows_repeated_pose_overlap true\n";
    for(const auto&name:models)for(int fold=1;fold<=5;++fold){
        cv::Mat X,Y,A,forwardX;int ntest=0;for(const auto&r:rows){auto x=features(r,name);if(r.fold==fold){++ntest;continue;}if(x.empty())continue;X.push_back(x);Y.push_back(target(r));A.push_back(basis(r.y,r.t,2));forwardX.push_back(x);}
        if(X.rows<10)throw std::runtime_error("Too few valid training images");GLS gls;final_pin::Regressor regression;int degree=name.find("quadratic")!=std::string::npos?2:1;
        if(name=="v1_joint_GLS")gls.fit(X,Y);else regression.fit(X,Y,degree,name=="v2_joint"?0:1e-4);
        auto predict=[&](const cv::Mat&x){return name=="v1_joint_GLS"?gls.predict(x):regression.predict(x);};
        if(name=="v1_joint_GLS"){cv::FileStorage f((out/(name+"_fold"+std::to_string(fold)+".yml")).u8string(),cv::FileStorage::WRITE);f<<"origin"<<gls.origin<<"projector"<<gls.P;}else{
            auto path=(out/(name+"_fold"+std::to_string(fold)+".yml")).u8string();regression.save(path);final_pin::Regressor loaded;loaded.load(path);auto p=predict(X.row(0)),q=loaded.predict(X.row(0));if(cv::norm(p-q)>1e-10)throw std::runtime_error("Model serialization parity failed");
        }
        int valid=0,py=0,pt=0,pb=0;double ey=0,et=0,my=0,mt=0;
        cv::Mat forward[2];cv::solve(A.colRange(0,3),forwardX,forward[0],cv::DECOMP_SVD);cv::solve(A,forwardX,forward[1],cv::DECOMP_SVD);
        for(int i=0;i<int(rows.size());++i){const auto&r=rows[i];if(r.fold!=fold)continue;auto x=features(r,name);cv::Vec2d q{std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::quiet_NaN()};std::string error;
            if(x.empty())error="measurement_failed";else{q=predict(x);if(!std::isfinite(q[0]+q[1]))error="nonfinite_prediction";}
            double a=std::abs(q[0]-r.y),b=std::abs(q[1]-r.t);bool y=error.empty()&&a<=.045+1e-12,t=error.empty()&&b<=.01+1e-12;
            if(error.empty()){++valid;ey+=a;et+=b;my=std::max(my,a);mt=std::max(mt,b);}py+=y;pt+=t;pb+=y&&t;
            pred<<name<<'\t'<<fold<<'\t'<<i<<'\t'<<r.date<<'\t'<<r.y<<'\t'<<r.t<<'\t'<<q[0]<<'\t'<<q[1]<<'\t'<<a<<'\t'<<b<<'\t'<<y<<'\t'<<t<<'\t'<<(y&&t)<<'\t'<<error<<'\n';
            if(!joint_keys.insert(name+":"+std::to_string(i)).second)throw std::runtime_error("Duplicate OOF prediction");
            for(int world=1;world<=2;++world){double sy=r.y,st=r.t;int turns=0;bool stopped=false;std::string status=x.empty()?"measurement_failed":"turn_limit";
                // Simulator owns true pose. Controller receives only generated features.
                // Keep the held-out anchor residual constant; forward coefficients are train-only.
                if(!x.empty())for(;;){cv::Mat observed=x+(basis(sy,st,world)-basis(r.y,r.t,world))*forward[world-1];auto e=predict(observed);if(!std::isfinite(e[0]+e[1])){status="nonfinite_prediction";break;}
                    if(std::abs(e[0])<=.045&&std::abs(e[1])<=.01){stopped=true;status="estimate_stop";break;}if(turns==20)break;
                    // Joint simultaneous correction, constrained to current calibration extent.
                    sy-=std::max(-.5,std::min(.5,e[0]));st-=std::max(-.25,std::min(.25,e[1]));++turns;
                    if(std::abs(sy)>2.1||std::abs(st)>2.1){status="outside_calibrated_extent";break;}
                }
                bool pass=stopped&&std::abs(sy)<=.045+1e-12&&std::abs(st)<=.01+1e-12;sim<<name<<'\t'<<fold<<'\t'<<i<<'\t'<<world<<'\t'<<turns<<'\t'<<stopped<<'\t'<<pass<<'\t'<<sy<<'\t'<<st<<'\t'<<status<<'\n';
            }
        }
        summary<<name<<'\t'<<fold<<'\t'<<ntest<<'\t'<<X.rows<<'\t'<<valid<<'\t'<<py<<'\t'<<pt<<'\t'<<pb<<'\t'<<ey/std::max(1,valid)<<'\t'<<et/std::max(1,valid)<<'\t'<<my<<'\t'<<mt<<'\n';
        std::cout<<"evaluated "<<name<<" fold "<<fold<<std::endl;
    }
    if(joint_keys.size()!=models.size()*rows.size())throw std::runtime_error("OOF prediction count mismatch");
    integrity<<"OOF_predictions "<<joint_keys.size()<<"\nmodel_reload_parity_pass true\nscaling_train_only true\nforward_surrogate_train_only true\nreal_robot_trials 0\n";
    for(const auto&name:models)if(name.find("qr_pin_")==0){cv::Mat X,Y;for(const auto&r:rows){auto x=features(r,name);if(!x.empty()){X.push_back(x);Y.push_back(target(r));}}final_pin::Regressor model;model.fit(X,Y,name.find("quadratic")!=std::string::npos?2:1);model.save((out/("development_all_"+name+".yml")).u8string());}
}
static void summarize(const fs::path&out,const std::vector<Row>&rows){
    struct Stats{int n=0,valid=0,py=0,pt=0,pb=0;double ey=0,et=0,my=0,mt=0;std::map<int,int> fn,fp;std::map<std::string,std::vector<cv::Vec2d>> repeated;};
    std::map<std::string,Stats> stats;std::ifstream in(out/"predictions.tsv");std::string l;std::getline(in,l);
    while(std::getline(in,l)){auto v=fields(l,'\t');if(v.size()!=14)throw std::runtime_error("Bad prediction record");auto&s=stats[v[0]];int fold=std::stoi(v[1]),i=std::stoi(v[2]);++s.n;++s.fn[fold];s.py+=std::stoi(v[10]);s.pt+=std::stoi(v[11]);int pass=std::stoi(v[12]);s.pb+=pass;s.fp[fold]+=pass;if(v[13].empty()){++s.valid;double ey=std::stod(v[8]),et=std::stod(v[9]);s.ey+=ey;s.et+=et;s.my=std::max(s.my,ey);s.mt=std::max(s.mt,et);s.repeated[rows.at(i).group].push_back({std::stod(v[6]),std::stod(v[7])});}}
    std::ofstream table(out/"aggregate.tsv"),repeat(out/"repeat_spread.tsv"),md(out/"measured_results.md");
    table<<std::setprecision(17)<<"model\tn\tvalid\tpass_Y\tpass_T\tpass_both\tMAE_Y\tMAE_T\tmax_eY\tmax_eT\tfold_both_min_pct\tfold_both_max_pct\tfold_both_sd_pp\n";
    repeat<<std::setprecision(17)<<"model\tgroup\tn_valid\tY_range\tT_range\tY_max_deviation_from_mean\tT_max_deviation_from_mean\n";
    md<<"# Measured random-image 5-fold development results\n\n571 non-center images. Y tolerance 0.045, T tolerance 0.01. Failures remain in denominators.\n\n| Model | Valid | Y pass | T pass | Both pass | MAE Y | MAE T | Max Y | Max T | Fold both min-max |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---|\n";
    for(auto&item:stats){auto&s=item.second;std::vector<double> rates;for(auto&p:s.fn)rates.push_back(100.*s.fp[p.first]/p.second);double mean=0,var=0;for(double x:rates)mean+=x/rates.size();for(double x:rates)var+=(x-mean)*(x-mean)/rates.size();double lo=*std::min_element(rates.begin(),rates.end()),hi=*std::max_element(rates.begin(),rates.end());
        table<<item.first<<'\t'<<s.n<<'\t'<<s.valid<<'\t'<<s.py<<'\t'<<s.pt<<'\t'<<s.pb<<'\t'<<s.ey/std::max(1,s.valid)<<'\t'<<s.et/std::max(1,s.valid)<<'\t'<<s.my<<'\t'<<s.mt<<'\t'<<lo<<'\t'<<hi<<'\t'<<std::sqrt(var)<<'\n';
        md<<std::fixed<<std::setprecision(5)<<"| "<<item.first<<" | "<<s.valid<<"/"<<s.n<<" | "<<s.py<<" | "<<s.pt<<" | "<<s.pb<<" ("<<100.*s.pb/s.n<<"%) | "<<s.ey/std::max(1,s.valid)<<" | "<<s.et/std::max(1,s.valid)<<" | "<<s.my<<" | "<<s.mt<<" | "<<lo<<"-"<<hi<<"% |\n";
        for(const auto&g:s.repeated)if(g.second.size()>=2){cv::Vec2d mean(0.,0.),lo=g.second.front(),hi=lo,dev(0.,0.);for(auto x:g.second){mean+=x*(1./g.second.size());for(int k=0;k<2;++k){lo[k]=std::min(lo[k],x[k]);hi[k]=std::max(hi[k],x[k]);}}for(auto x:g.second)for(int k=0;k<2;++k)dev[k]=std::max(dev[k],std::abs(x[k]-mean[k]));repeat<<item.first<<'\t'<<g.first<<'\t'<<g.second.size()<<'\t'<<hi[0]-lo[0]<<'\t'<<hi[1]-lo[1]<<'\t'<<dev[0]<<'\t'<<dev[1]<<'\n';}
        std::cout<<item.first<<" both "<<s.pb<<"/"<<s.n<<" max Y "<<s.my<<" T "<<s.mt<<std::endl;
    }
    struct Sim{int n=0,stop=0,pass=0,false_stop=0,maxturn=0;double turns=0;};std::map<std::pair<std::string,int>,Sim> ss;std::ifstream sf(out/"simulation.tsv");std::getline(sf,l);while(std::getline(sf,l)){auto v=fields(l,'\t');if(v.size()!=10)throw std::runtime_error("Bad simulation record");auto&s=ss[{v[0],std::stoi(v[3])}];++s.n;int t=std::stoi(v[4]);bool stop=std::stoi(v[5]),pass=std::stoi(v[6]);s.stop+=stop;s.pass+=pass;s.false_stop+=stop&&!pass;s.maxturn=std::max(s.maxturn,t);s.turns+=t;}
    std::ofstream sm(out/"simulation_summary.tsv");sm<<"model\tworld_degree\tn\tstopped\ttrue_pass\tfalse_stop\tmean_turns\tmax_turns\n";md<<"\n## Surrogate simulation, not physical robot trials\n\nTraining-only forward response, fixed held-out residual, noiseless ideal motion, simultaneous correction.\n\n| Model | World degree | True pass | False stop | Mean turns | Max turns |\n|---|---:|---:|---:|---:|---:|\n";
    for(auto&p:ss){auto&s=p.second;sm<<p.first.first<<'\t'<<p.first.second<<'\t'<<s.n<<'\t'<<s.stop<<'\t'<<s.pass<<'\t'<<s.false_stop<<'\t'<<s.turns/s.n<<'\t'<<s.maxturn<<'\n';md<<"| "<<p.first.first<<" | "<<p.first.second<<" | "<<s.pass<<"/"<<s.n<<" | "<<s.false_stop<<" | "<<s.turns/s.n<<" | "<<s.maxturn<<" |\n";}
}
static void selftest(const fs::path&root,const fs::path&out){
    auto center=alignment_detail::read_gray(root/std::filesystem::u8path(u8"experiments/hye/20261104_1435_v2_joint_강등/assets/center.bmp"));final_pin::Observer pin(center);auto zero=pin.measure(center);double ident=cv::norm(zero.displacement,cv::NORM_INF);if(ident>1e-4)throw std::runtime_error("Pin identity failed");
    cv::Mat shifted;cv::Mat warp=(cv::Mat_<double>(2,3)<<1,0,3.25,0,1,-2.375);cv::warpAffine(center,shifted,warp,center.size(),cv::INTER_CUBIC,cv::BORDER_REFLECT_101);auto m=pin.measure(shifted);double worst=0;for(int k=0;k<4;++k)worst=std::max(worst,std::max(std::abs(m.displacement.at<double>(2*k)-3.25),std::abs(m.displacement.at<double>(2*k+1)+2.375)));if(worst>.04)throw std::runtime_error("Pin synthetic translation error exceeds .04px");
    std::ofstream f(out/"selftest.txt");f<<std::setprecision(17)<<"identity_max_px "<<ident<<"\nsynthetic_translation_max_error_px "<<worst<<"\n";std::cout<<"identity "<<ident<<" translation "<<worst<<std::endl;
    for(const std::string kind:{"linear","quadratic","bars_linear","bars_quadratic"}){auto model=out/("development_all_qr_pin_"+kind+".yml");
    if(fs::exists(model)){auto rows=dataset(root);load_measurements(out,rows);populate_bars(root,rows);const auto&r=rows.front();auto image=alignment_detail::read_gray(root/fs::u8path(r.rel)),before=image.clone();
        final_pin::Estimator pipeline((root/std::filesystem::u8path(u8"experiments/hye/20261104_1435_v2_joint_강등/assets")).u8string(),model.u8string());auto actual=pipeline.predict(image);final_pin::Regressor reg;reg.load(model.u8string());auto expected=reg.predict(features(r,"qr_pin_"+kind));double parity=cv::norm(actual-expected),mutation=cv::norm(image,before,cv::NORM_INF);if(parity>1e-9||mutation!=0)throw std::runtime_error("Public inference parity or input preservation failed");f<<"public_image_api_parity_"<<kind<<' '<<parity<<"\ninput_mutation_max_"<<kind<<' '<<mutation<<"\n";std::cout<<"public API "<<kind<<" parity "<<parity<<std::endl;
    }}
}
int main(int argc,char**argv){try{
    if(argc!=4)throw std::runtime_error("pin_eval measure|evaluate|evaluate_bars|summarize|selftest REPO OUTPUT");cv::setNumThreads(1);fs::path root=fs::u8path(argv[2]),out=fs::u8path(argv[3]);fs::create_directories(out);std::string mode=argv[1];
    if(mode=="selftest"){selftest(root,out);return 0;}auto rows=dataset(root);
    if(mode=="measure")measure(root,out,rows);else if(mode=="evaluate"){load_measurements(out,rows);evaluate(out,rows);}else if(mode=="evaluate_bars"){load_measurements(out,rows);populate_bars(root,rows);evaluate(out,rows,true);}else if(mode=="summarize")summarize(out,rows);else throw std::runtime_error("Unknown mode");
    std::ofstream env(out/"environment.txt");env<<"OpenCV "<<CV_VERSION<<"\nC++17\nOpenCV threads 1; image workers 4\n";
#ifdef _MSC_VER
    env<<"MSVC "<<_MSC_VER<<"\n";
#endif
    return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
