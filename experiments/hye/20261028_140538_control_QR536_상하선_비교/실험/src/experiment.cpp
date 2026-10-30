#include "control_internal.hpp"
#include "observer_internal.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <regex>
#include <sstream>
#include <stdexcept>
namespace fs=std::filesystem;
using Clock=std::chrono::steady_clock;
static double ms(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
static std::vector<std::string> split(const std::string& s){std::vector<std::string> v;std::stringstream ss(s);std::string x;while(std::getline(ss,x,',')){if(!x.empty()&&x.back()=='\r')x.pop_back();v.push_back(x);}return v;}
static void check(bool ok,const std::string& message){if(!ok)throw std::runtime_error(message);}
struct Sample{fs::path path;std::string name,date,axis,session;double y=0,t=0,value=0;int fold=-1;cv::Mat q,upper,lower;bool qvalid=false,uvalid=false,lvalid=false;double decode_ms=0,q_ms=0,u_ms=0,l_ms=0;};
struct LineResult{cv::Mat x;bool valid=true;double min_correlation=1;};
class Lines{
 struct Block{cv::Rect box;std::vector<double> reference;double zero;};
 std::vector<Block> groups_[2];
public:
 Lines(const cv::Mat& center,const fs::path& protocol){cv::FileStorage f(protocol.u8string(),cv::FileStorage::READ);check(f.isOpened(),"Missing line protocol");
  for(auto it=f["blocks"].begin();it!=f["blocks"].end();++it){auto n=*it;std::string group=n["line"];check(group=="upper"||group=="lower","Unexpected line group");auto b=n["read_box"];cv::Rect r(int(b[0]),int(b[1]),int(b[2])-int(b[0]),int(b[3])-int(b[1]));check(r.width==64&&r.height==49&&(r&cv::Rect(0,0,3840,2160))==r,"Invalid permitted line mask");
   auto ref=alignment_detail::line_profile(center,r);auto z=alignment_detail::match_line(ref,ref);check(z.valid,"Invalid center line");groups_[group=="upper"?0:1].push_back({r,ref,z.shift});}
  check(groups_[0].size()==24&&groups_[1].size()==24,"Expected upper24/lower24 blocks");}
 LineResult measure(const cv::Mat& image,int group)const{LineResult out{cv::Mat(1,24,CV_64F),true,1};int i=0;for(const auto& b:groups_[group]){auto fit=alignment_detail::match_line(b.reference,alignment_detail::line_profile(image,b.box));out.x.at<double>(i++)=fit.shift-b.zero;out.valid&=fit.valid;out.min_correlation=std::min(out.min_correlation,fit.correlation);}return out;}
};
static bool valid(const Sample& s,int mode){return s.qvalid&&(!(mode&1)||s.uvalid)&&(!(mode&2)||s.lvalid);}
static cv::Mat feature(const Sample& s,int mode){std::vector<cv::Mat> parts{s.q};if(mode&1)parts.push_back(s.upper);if(mode&2)parts.push_back(s.lower);cv::Mat x;cv::hconcat(parts,x);return x;}
static const char* names[]={"QR_only","QR_upper","QR_lower","QR_both"};
static const char* branches[]={"Y1+","Y1-","Y2+","Y2-","T+","T-"};
struct Prediction{std::string axis,branch;double value=0,cost=0;};
struct Branch{std::string axis,name;cv::Mat origin,slope,curve;};
struct Fit{
 cv::Mat W;std::vector<Branch> b;
 Prediction predict(const cv::Mat& x)const{Prediction best;best.cost=std::numeric_limits<double>::infinity();for(const auto& q:b){cv::Mat z=W*(x-q.origin).t(),s=W*q.slope.t(),c=W*q.curve.t();double v=z.dot(s)/s.dot(s);for(int k=0;k<8;++k){cv::Mat j=s+2*v*c,r=z-v*s-v*v*c;double den=j.dot(j);check(den>0&&std::isfinite(den),"Degenerate solve");v+=j.dot(r)/den;}cv::Mat r=z-v*s-v*v*c;double cost=r.dot(r);if(v*(q.name.back()=='+'?1:-1)<0)cost=std::numeric_limits<double>::infinity();if(cost<best.cost)best={q.axis,q.name,v,cost};}check(std::isfinite(best.cost),"No finite branch");return best;}
};
static Fit train(const std::vector<Sample>& data,const std::vector<int>& train_ids,int mode){
 check(!train_ids.empty(),"Empty training split");for(int i:train_ids)check(valid(data[i],mode),"Training measurement failed: no samples silently dropped");int d=feature(data[train_ids[0]],mode).cols;Fit fit;std::map<std::string,Branch> fitted;
 for(const auto& axis:{std::string("Y"),std::string("T")}){std::vector<int> ids;std::vector<std::string> sessions;for(const auto* name:branches)if(name[0]==axis[0])sessions.push_back(name);for(int i:train_ids)if(data[i].axis==axis)ids.push_back(i);
  cv::Mat A=cv::Mat::zeros(int(ids.size()),int(sessions.size())+2,CV_64F),X(int(ids.size()),d,CV_64F),coeff;
  for(int r=0;r<int(ids.size());++r){const auto& sample=data[ids[r]];A.at<double>(r,0)=sample.value;A.at<double>(r,1)=sample.value*sample.value;auto pos=std::find(sessions.begin(),sessions.end(),sample.session);check(pos!=sessions.end(),"Unknown training session");A.at<double>(r,2+int(pos-sessions.begin()))=1;feature(sample,mode).copyTo(X.row(r));}
  check(cv::solve(A,X,coeff,cv::DECOMP_SVD),"Branch fit failed");for(int j=0;j<int(sessions.size());++j)fitted[sessions[j]]={axis,sessions[j],coeff.row(j+2).clone(),coeff.row(0).clone(),coeff.row(1).clone()};
 }
 for(const auto* name:branches)fit.b.push_back(fitted.at(name));
 cv::Mat R(int(train_ids.size()),d,CV_64F);for(int r=0;r<int(train_ids.size());++r){auto& s=data[train_ids[r]];auto& b=fitted.at(s.session);cv::Mat residual=feature(s,mode)-b.origin-s.value*b.slope-s.value*s.value*b.curve;residual.copyTo(R.row(r));}
 cv::Mat S=R.t()*R/double(std::max(int(train_ids.size())-16,1)),C(d,d,CV_64F);std::vector<double> sd(d);for(int i=0;i<d;++i)sd[i]=std::sqrt(std::max(S.at<double>(i,i),1e-14));
 for(int i=0;i<d;++i)for(int j=0;j<d;++j)C.at<double>(i,j)=.95*S.at<double>(i,j)/sd[i]/sd[j]+(i==j?.05:0);
 cv::Mat values,vectors;check(cv::eigen(C,values,vectors),"Covariance eigensolve failed");fit.W=cv::Mat(d,d,CV_64F);for(int i=0;i<d;++i){check(values.at<double>(i)>0,"Nonpositive regularized covariance");for(int j=0;j<d;++j)fit.W.at<double>(i,j)=vectors.at<double>(i,j)/std::sqrt(values.at<double>(i))/sd[j];}return fit;
}
static void matrix_json(std::ostream& out,const cv::Mat& x){out<<'[';for(int i=0;i<x.rows;++i){if(i)out<<',';if(x.rows!=1)out<<'[';for(int j=0;j<x.cols;++j){if(j)out<<',';out<<x.at<double>(i,j);}if(x.rows!=1)out<<']';}out<<']';}
static void save_model(const Fit& model,const fs::path& file){std::ofstream out(file);out<<std::setprecision(17)<<"{\"shrinkage\":0.05,\"W\":";matrix_json(out,model.W);out<<",\"branches\":{";bool first=true;for(const auto& b:model.b){if(!first)out<<',';first=false;out<<'"'<<b.name<<"\":[\""<<b.axis<<"\",";matrix_json(out,b.origin);out<<',';matrix_json(out,b.slope);out<<',';matrix_json(out,b.curve);out<<']';}out<<"}}\n";}
struct Stats{int n=0,failed=0,passed=0,axis_correct=0,sign_correct=0;double sum_y=0,sum_t=0,max_y=0,max_t=0;};
static void evaluate(const std::vector<Sample>& data,const std::vector<int>& ids,int mode,const Fit& model,const std::string& protocol,int fold,std::ostream& rows,std::map<std::string,Stats>& stats){
 for(int i:ids){auto& s=data[i];std::vector<std::string> groups{protocol};if(protocol=="oof4"||protocol=="resub280")groups.push_back(protocol+(s.axis=="Y"?"_Y80":"_T200"));if(protocol=="transfer20261020")groups.push_back(protocol+(s.y==0&&s.t==0?"_centers3":"_mixed57"));
  try{check(valid(s,mode),"Invalid measurement");auto q=model.predict(feature(s,mode));double y=q.axis=="Y"?q.value:0,t=q.axis=="T"?q.value:0,ey=std::abs(y-s.y),et=std::abs(t-s.t);bool pass=ey<=.09&&et<=.09;bool single=s.axis=="Y"||s.axis=="T";bool correct=single&&q.axis==s.axis,sign=correct&&std::signbit(q.value)==std::signbit(s.value);
   rows<<names[mode]<<','<<protocol<<','<<fold<<','<<s.date<<','<<s.name<<','<<s.y<<','<<s.t<<','<<q.axis<<','<<q.value<<','<<q.branch<<','<<ey<<','<<et<<",1,"<<pass<<'\n';
   for(const auto& group:groups){auto& a=stats[std::string(names[mode])+","+group];++a.n;a.passed+=pass;a.axis_correct+=correct;a.sign_correct+=sign;a.sum_y+=ey;a.sum_t+=et;a.max_y=std::max(a.max_y,ey);a.max_t=std::max(a.max_t,et);}
  }catch(const std::exception&){rows<<names[mode]<<','<<protocol<<','<<fold<<','<<s.date<<','<<s.name<<','<<s.y<<','<<s.t<<",,,,,,0,0\n";for(const auto& group:groups){auto& a=stats[std::string(names[mode])+","+group];++a.n;++a.failed;}}
 }
}
static double percentile(std::vector<double> v,double p){std::sort(v.begin(),v.end());double x=(v.size()-1)*p;size_t i=size_t(x);return v[i]+(v[std::min(i+1,v.size()-1)]-v[i])*(x-i);}
int wmain(int argc,wchar_t** argv){try{
 check(argc==5,"control_line_ablation CLEAN_ROOT CONTROL_MODULE LINE_PROTOCOL NEW_OUTPUT");cv::setNumThreads(1);fs::path root(argv[1]),module(argv[2]),protocol(argv[3]),output(argv[4]);check(!fs::exists(output),"Use NEW output directory");fs::create_directories(output/"models");
 auto start=Clock::now();legacy_control::Tracker tracker(module/"assets");legacy_control::Model frozen(module/"assets/model.json");double qr_init=ms(start);auto center=legacy_control::read_gray(module/"assets/center.bmp");start=Clock::now();Lines lines(center,protocol);double line_init=ms(start);
 std::map<std::string,Sample> labels;std::ifstream ref(module/"validation/python_predictions.csv");std::string text;std::getline(ref,text);while(std::getline(ref,text)){auto v=split(text);if(v[12]!="raw_oof4")continue;Sample s;s.name=v[0];s.axis=v[1];s.value=std::stod(v[2]);s.session=v[3];s.fold=std::stoi(v[13]);if(s.axis=="Y")s.y=s.value;else s.t=s.value;labels[s.name]=s;}
 check(labels.size()==280,"Expected280 development labels");std::vector<Sample> data;for(auto item:labels){auto s=item.second;s.date="20261014";s.path=root/"clean_20261014"/(s.axis=="Y"?"Y":"Theta")/fs::u8path(s.name);data.push_back(s);}
 std::vector<fs::path> paths;for(const auto& e:fs::recursive_directory_iterator(root/"clean_20261020"))if(e.path().extension()==L".bmp")paths.push_back(e.path());std::sort(paths.begin(),paths.end());check(paths.size()==60,"Expected60 transfer photos");std::regex yr("Y([+-][0-9]+(?:\\.[0-9]+)?)"),tr("T([+-][0-9]+(?:\\.[0-9]+)?)");for(auto path:paths){Sample s;s.path=path;s.name=path.filename().u8string();s.date="20261020";s.axis="M";std::smatch match;auto name=path.stem().u8string();if(std::regex_search(name,match,yr))s.y=std::stod(match[1]);if(std::regex_search(name,match,tr))s.t=std::stod(match[1]);data.push_back(s);}
 std::ofstream measurements(output/"measurements.csv");measurements<<std::setprecision(17)<<"date,file,qr_valid,upper_valid,lower_valid,decode_ms,qr_ms,upper_ms,lower_ms\n";
 for(int i=0;i<int(data.size());++i){auto& s=data[i];start=Clock::now();auto image=legacy_control::read_gray(s.path);s.decode_ms=ms(start);try{start=Clock::now();auto tracks=tracker.measure(image);s.q=tracker.features(tracks);s.q_ms=ms(start);s.qvalid=true;}catch(const std::exception& e){std::cerr<<s.name<<" QR: "<<e.what()<<'\n';}
  for(int k=0;k<2;++k){start=Clock::now();auto result=lines.measure(image,k);double elapsed=ms(start);if(k==0){s.upper=result.x;s.uvalid=result.valid;s.u_ms=elapsed;}else{s.lower=result.x;s.lvalid=result.valid;s.l_ms=elapsed;}}
  measurements<<s.date<<','<<s.name<<','<<s.qvalid<<','<<s.uvalid<<','<<s.lvalid<<','<<s.decode_ms<<','<<s.q_ms<<','<<s.u_ms<<','<<s.l_ms<<'\n';if((i+1)%40==0)std::cout<<"Measured "<<i+1<<"/340\n"<<std::flush;
 }
 std::vector<int> all,transfer;for(int i=0;i<int(data.size());++i)(i<280?all:transfer).push_back(i);
 std::ofstream predictions(output/"predictions.csv");predictions<<std::setprecision(17)<<"condition,protocol,fold,date,file,Y,T,pred_axis,pred_value,branch,Y_error,T_error,valid,pass\n";std::map<std::string,Stats> stats;std::vector<Fit> full;std::vector<double> train_times;
 for(int mode=0;mode<4;++mode){for(int f=0;f<4;++f){std::vector<int> train_ids,test;for(int i:all)(data[i].fold==f?test:train_ids).push_back(i);check(train_ids.size()==210&&test.size()==70,"Fold membership mismatch");auto model=train(data,train_ids,mode);save_model(model,output/"models"/(std::string(names[mode])+"_fold"+std::to_string(f)+".json"));evaluate(data,test,mode,model,"oof4",f,predictions,stats);}
  start=Clock::now();full.push_back(train(data,all,mode));train_times.push_back(ms(start));save_model(full.back(),output/"models"/(std::string(names[mode])+"_full.json"));evaluate(data,all,mode,full.back(),"resub280",-1,predictions,stats);evaluate(data,transfer,mode,full.back(),"transfer20261020",-1,predictions,stats);std::cout<<"Fitted "<<names[mode]<<"\n"<<std::flush;
 }
 double baseline_delta=0;int baseline_axis_mismatch=0;for(int i:all){auto a=frozen.predict(data[i].q);auto b=full[0].predict(data[i].q);baseline_delta=std::max(baseline_delta,std::abs(a.value-b.value));baseline_axis_mismatch+=(a.axis!=b.axis);}check(baseline_delta<1e-6&&baseline_axis_mismatch==0,"Refitted QR baseline does not reproduce original family");
 std::ofstream summary(output/"summary.csv");summary<<std::setprecision(17)<<"condition,protocol,n,measurement_failures,pass,pass_percent,axis_correct,sign_correct,Y_mae,Y_max,T_mae,T_max\n";for(const auto& entry:stats){auto s=entry.second;summary<<entry.first<<','<<s.n<<','<<s.failed<<','<<s.passed<<','<<100.*s.passed/s.n<<','<<s.axis_correct<<','<<s.sign_correct<<','<<s.sum_y/std::max(s.n-s.failed,1)<<','<<s.max_y<<','<<s.sum_t/std::max(s.n-s.failed,1)<<','<<s.max_t<<'\n';}
 std::vector<int> bench;for(const auto& axis:{std::string("Y"),std::string("T")}){std::vector<int> group;for(int i:all)if(data[i].axis==axis)group.push_back(i);for(int k=0;k<16;++k)bench.push_back(group[k*group.size()/16]);}for(int k=0;k<16;++k)bench.push_back(transfer[k*transfer.size()/16]);check(bench.size()==48,"Benchmark sample count");
 // Warm up all measurement paths before timing. Interleave upper/lower order.
 for(int i:{bench.front(),bench.back()}){auto image=legacy_control::read_gray(data[i].path);auto t=tracker.measure(image);tracker.features(t);lines.measure(image,0);lines.measure(image,1);}
 std::ofstream timing(output/"timing.csv");timing<<std::setprecision(17)<<"repeat,date,file,condition,decode_ms,qr_ms,line_ms,predict_ms,compute_ms,with_decode_ms\n";std::vector<double> compute[4],with_decode[4],line_elapsed[4];double sink=0;
 for(int repeat=0;repeat<3;++repeat){auto order=bench;if(repeat%2)std::reverse(order.begin(),order.end());for(int i:order){const auto& s=data[i];Sample now=s;start=Clock::now();auto image=legacy_control::read_gray(s.path);double decode=ms(start);start=Clock::now();auto t=tracker.measure(image);now.q=tracker.features(t);double qtime=ms(start),lt[2];
   for(int kk=0;kk<2;++kk){int k=(kk+repeat)%2;start=Clock::now();auto result=lines.measure(image,k);lt[k]=ms(start);check(result.valid,"Benchmark line failure");if(k==0)now.upper=result.x;else now.lower=result.x;}
   for(int k=0;k<4;++k){int mode=(k+repeat)%4;start=Clock::now();auto p=full[mode].predict(feature(now,mode));double ptime=ms(start);sink+=p.value;double ltime=((mode&1)?lt[0]:0)+((mode&2)?lt[1]:0),total=qtime+ltime+ptime;compute[mode].push_back(total);with_decode[mode].push_back(total+decode);line_elapsed[mode].push_back(ltime);timing<<repeat<<','<<s.date<<','<<s.name<<','<<names[mode]<<','<<decode<<','<<qtime<<','<<ltime<<','<<ptime<<','<<total<<','<<total+decode<<'\n';}
  }std::cout<<"Benchmark repeat "<<repeat+1<<"/3\n"<<std::flush;
 }
 std::ofstream times(output/"timing_summary.csv");times<<std::setprecision(17)<<"condition,n,compute_median_ms,compute_p95_ms,with_decode_median_ms,with_decode_p95_ms,line_median_ms,full_train_ms\n";for(int m=0;m<4;++m)times<<names[m]<<','<<compute[m].size()<<','<<percentile(compute[m],.5)<<','<<percentile(compute[m],.95)<<','<<percentile(with_decode[m],.5)<<','<<percentile(with_decode[m],.95)<<','<<percentile(line_elapsed[m],.5)<<','<<train_times[m]<<'\n';
 std::ofstream audit(output/"audit.json");audit<<std::setprecision(17)<<"{\"qr_init_ms\":"<<qr_init<<",\"both_lines_init_ms\":"<<line_init<<",\"baseline_refit_max_difference\":"<<baseline_delta<<",\"baseline_axis_mismatch\":"<<baseline_axis_mismatch<<",\"measured_photos\":340,\"oof_photos\":280,\"transfer_photos\":60,\"benchmark_photos\":48,\"benchmark_repeats\":3,\"threads\":1,\"shrinkage\":0.05,\"benchmark_sink\":"<<sink<<",\"scope\":\"Previously exposed development data; transfer mixed poses are outside Y-or-T assumption; paired component timing, not robot cycle timing\"}\n";
 std::cout<<"Completed. QR refit max delta="<<baseline_delta<<"\n";return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
