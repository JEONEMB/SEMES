#include "experiment.hpp"
#include "control_internal.hpp"
#include "observer_internal.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <regex>
#include <sstream>
using namespace study;
namespace fs=std::filesystem;
using Clock=std::chrono::steady_clock;
static double elapsed(Clock::time_point start){return std::chrono::duration<double,std::milli>(Clock::now()-start).count();}
static std::vector<std::string> split(const std::string& text){std::vector<std::string> out;std::stringstream stream(text);std::string s;while(std::getline(stream,s,',')){if(!s.empty()&&s.back()=='\r')s.pop_back();out.push_back(s);}return out;}
static std::vector<Sample> samples(const fs::path& root,const fs::path& control){
 std::map<std::string,Sample> labels;std::ifstream f(control/"validation/python_predictions.csv");require(bool(f),"Missing frozen labels/folds");std::string line;std::getline(f,line);
 while(std::getline(f,line)){auto v=split(line);require(v.size()==14,"Invalid frozen label row");if(v[12]!="raw_oof4")continue;Sample s;s.file=v[0];s.axis=v[1];s.value=std::stod(v[2]);s.session=v[3];s.fold=std::stoi(v[13]);s.date="20261014";if(s.axis=="Y")s.y=s.value;else s.t=s.value;s.path=root/"clean_20261014"/(s.axis=="Y"?"Y":"Theta")/fs::u8path(s.file);labels[s.file]=s;}
 require(labels.size()==280,"Expected frozen280 split");std::vector<Sample> data;for(const auto& entry:labels)data.push_back(entry.second);
 std::vector<fs::path> paths;for(const auto& entry:fs::recursive_directory_iterator(root/"clean_20261020"))if(entry.path().extension()==L".bmp")paths.push_back(entry.path());std::sort(paths.begin(),paths.end());require(paths.size()==60,"Expected60 cross-date images");
 std::regex yr("Y([+-][0-9]+(?:\\.[0-9]+)?)"),tr("T([+-][0-9]+(?:\\.[0-9]+)?)");
 for(const auto& path:paths){Sample s;s.path=path;s.file=path.filename().u8string();s.date="20261020";s.axis="mixed";auto name=path.stem().u8string();std::smatch m;if(std::regex_search(name,m,yr))s.y=std::stod(m[1]);if(std::regex_search(name,m,tr))s.t=std::stod(m[1]);data.push_back(s);}
 Sample center;center.file="center.bmp";center.date="20261014";center.axis="center";center.path=root/"clean_20261014/center.bmp";data.push_back(center);return data;
}
static void save_cache(const std::vector<Sample>& data,const fs::path& path,const fs::path& control,const fs::path& joint){
 cv::FileStorage f(path.u8string(),cv::FileStorage::WRITE);require(f.isOpened(),"Cannot save measurements");f<<"control_model_sha256"<<legacy_control::sha256(control/"assets/model.json")<<"joint_geometry_sha256"<<legacy_control::sha256(joint/"assets/geometry.json")<<"line_protocol_sha256"<<legacy_control::sha256(joint/"assets/line_protocol.json")<<"samples"<<"[";
 for(const auto& s:data){f<<"{"<<"date"<<s.date<<"file"<<s.file<<"image_sha256"<<legacy_control::sha256(s.path)<<"p12_valid"<<int(s.valid[0])<<"j204_valid"<<int(s.valid[1])<<"p12"<<s.feature[0]<<"j204"<<s.feature[1]<<"p12_ms"<<s.measurement_ms[0]<<"j204_ms"<<s.measurement_ms[1]<<"decode_ms"<<s.decode_ms<<"p12_error"<<s.error[0]<<"j204_error"<<s.error[1]<<"}";}f<<"]";
}
static void load_cache(std::vector<Sample>& data,const fs::path& path,const fs::path& control,const fs::path& joint){
 cv::FileStorage f(path.u8string(),cv::FileStorage::READ);require(f.isOpened(),"Cannot open measurement cache");
 require(std::string(f["control_model_sha256"])==legacy_control::sha256(control/"assets/model.json")&&std::string(f["joint_geometry_sha256"])==legacy_control::sha256(joint/"assets/geometry.json")&&std::string(f["line_protocol_sha256"])==legacy_control::sha256(joint/"assets/line_protocol.json"),"Measurement cache assets mismatch");
 auto rows=f["samples"];require(rows.size()==data.size(),"Cache row count mismatch");int i=0;
 for(auto it=rows.begin();it!=rows.end();++it,++i){auto n=*it;auto& s=data[i];require(std::string(n["file"])==s.file&&std::string(n["date"])==s.date,"Cache ordering mismatch");require(std::string(n["image_sha256"])==legacy_control::sha256(s.path),"Cache source image mismatch");s.valid[0]=int(n["p12_valid"])!=0;s.valid[1]=int(n["j204_valid"])!=0;n["p12"]>>s.feature[0];n["j204"]>>s.feature[1];s.measurement_ms[0]=double(n["p12_ms"]);s.measurement_ms[1]=double(n["j204_ms"]);s.decode_ms=double(n["decode_ms"]);s.error[0]=std::string(n["p12_error"]);s.error[1]=std::string(n["j204_error"]);}
}
struct Stats{int n=0,failed=0,y_pass=0,both_pass=0;double y_sum=0,y_max=0,t_sum=0,t_max=0;};
static void evaluate(const Model& model,int feature,const std::string& condition,const std::string& protocol,int fold,const std::vector<Sample>& data,const std::vector<int>& ids,std::ostream& out,std::map<std::string,Stats>& summaries){
 for(int i:ids){const auto& s=data[i];std::vector<std::string> groups{protocol};
  if(s.date=="20261014"&&(s.axis=="Y"||s.axis=="T"))groups.push_back(protocol+(s.axis=="Y"?"_Y80":"_T200"));
  if(s.date=="20261020"){groups.push_back(protocol+(s.y==0&&s.t==0?"_centers3":"_mixed57"));if(s.y!=0||s.t!=0)groups.push_back(protocol+(std::abs(s.t)<=.09?"_mixed_absT_le_0p09":"_mixed_absT_gt_0p09"));}
  bool valid=false;Prediction q;double ey=0,et=0;std::string error;
  try{require(s.valid[feature],"Measurement failed: "+s.error[feature]);q=model.predict(s.feature[feature]);ey=std::abs(q.y-s.y);et=std::abs(q.t-s.t);valid=true;}catch(const std::exception& e){error=e.what();}
  out<<condition<<','<<protocol<<','<<fold<<','<<s.date<<','<<s.file<<','<<s.y<<','<<s.t<<',';
  if(valid)out<<q.y<<','<<q.t<<','<<ey<<','<<et<<",1,"<<(ey<=.09)<<','<<(ey<=.09&&et<=.09)<<','<<q.branch;
  else out<<",,,,0,0,0,";out<<'\n';
  for(const auto& group:groups){auto& a=summaries[condition+","+group];++a.n;if(!valid){++a.failed;continue;}a.y_sum+=ey;a.t_sum+=et;a.y_max=std::max(a.y_max,ey);a.t_max=std::max(a.t_max,et);a.y_pass+=ey<=.09;a.both_pass+=ey<=.09&&et<=.09;}
 }
}
static double percentile(std::vector<double> values,double p){std::sort(values.begin(),values.end());double index=(values.size()-1)*p;size_t lower=size_t(index);return values[lower]+(values[std::min(lower+1,values.size()-1)]-values[lower])*(index-lower);}
static void save_checked(const Model& model,const fs::path& file,const cv::Mat& feature){model.save(file);auto loaded=Model::load(file);auto a=model.predict(feature),b=loaded.predict(feature);require(std::abs(a.y-b.y)+std::abs(a.t-b.t)<1e-12,"Saved-model round trip changed prediction");}
int wmain(int argc,wchar_t** argv){try{
 require(argc==5||argc==6,"compare_control CLEAN_ROOT CONTROL_MODULE JOINT_MODULE NEW_OUTPUT [MEASUREMENTS.yml]");cv::setNumThreads(1);
 fs::path root(argv[1]),control(argv[2]),joint(argv[3]),output(argv[4]);require(!fs::exists(output),"Use a NEW output directory");fs::create_directories(output/"models");
 auto data=samples(root,control);double init_p12=0,init_j204=0;
 if(argc==6)load_cache(data,fs::path(argv[5]),control,joint);
 else {
  auto start=Clock::now();legacy_control::Tracker tracker(control/"assets");init_p12=elapsed(start);start=Clock::now();alignment_detail::Observer observer(joint/"assets");init_j204=elapsed(start);
  auto center=legacy_control::read_gray(control/"assets/center.bmp");auto tracks=tracker.measure(center);tracker.features(tracks);observer.measure(center);
  for(int i=0;i<int(data.size());++i){auto& s=data[i];start=Clock::now();auto image=legacy_control::read_gray(s.path);s.decode_ms=elapsed(start);
   for(int k=0;k<2;++k){int feature=(i+k)%2;start=Clock::now();try{
     if(feature==0){auto t=tracker.measure(image);s.feature[0]=tracker.features(t);}else s.feature[1]=alignment_detail::feature_map(observer.measure(image).values);
     s.valid[feature]=true;
    }catch(const std::exception& e){s.error[feature]=e.what();}s.measurement_ms[feature]=elapsed(start);}
   if((i+1)%20==0)std::cout<<"Measured "<<i+1<<"/341\n"<<std::flush;
  }
  save_cache(data,output/"measurements.yml",control,joint);
 }
 // Center correction is a fixed reference transformation, with no held-out pose labels.
 for(int feature=0;feature<2;++feature){require(data.back().valid[feature],"Fixed center measurement failed");auto zero=data.back().feature[feature].clone();for(auto& s:data)if(s.valid[feature])s.feature[feature]-=zero;}
 std::vector<int> all,y_all,transfer;for(int i=0;i<280;++i){all.push_back(i);if(data[i].axis=="Y")y_all.push_back(i);}for(int i=280;i<340;++i)transfer.push_back(i);require(y_all.size()==80,"Expected Y80");
 std::ofstream rows(output/"predictions.csv");rows<<std::setprecision(17)<<"condition,protocol,fold,date,file,Y,T,predY,predT,Y_error,T_error,valid,Y_pass,both_pass,branch\n";
 std::ofstream timing(output/"timing.csv");timing<<std::setprecision(17)<<"condition,measurement_n,measurement_median_ms,measurement_p95_ms,initialization_ms,predict_mean_ms,training_full_ms\n";
 std::map<std::string,Stats> summaries;double baseline_delta=0;int baseline_axis_mismatch=0;legacy_control::Model frozen(control/"assets/model.json");
 for(int family=0;family<2;++family)for(int feature=0;feature<2;++feature){std::string condition=std::string(family?"YorT_":"Y_GLS_")+(feature?"J204":"P12");
  for(int fold=0;fold<4;++fold){std::vector<int> train,test;for(int i:all)if(family||data[i].axis=="Y")(data[i].fold==fold?test:train).push_back(i);if(!family)train.push_back(340);
   require(train.size()==size_t(family?210:61)&&test.size()==size_t(family?70:20),"Unexpected train/test counts");
   auto model=family?train_yort(data,train,feature):train_y(data,train,feature);save_checked(model,output/"models"/(condition+"_fold"+std::to_string(fold)+".json"),data[test.front()].feature[feature]);
   evaluate(model,feature,condition,"oof4",fold,data,test,rows,summaries);
   if(family&&feature==0){legacy_control::Model reference(control/"validation"/("fold"+std::to_string(fold)+"_model.json"));for(int i:test){auto q=model.predict(data[i].feature[0]);auto r=reference.predict(data[i].feature[0]);baseline_delta=std::max(baseline_delta,std::abs((q.axis=="Y"?q.y:q.t)-r.value));baseline_axis_mismatch+=q.axis!=r.axis;}}
  }
  auto train=family?all:y_all;if(!family)train.push_back(340);auto start=Clock::now();auto model=family?train_yort(data,train,feature):train_y(data,train,feature);double train_ms=elapsed(start);
  save_checked(model,output/"models"/(condition+"_full.json"),data[y_all.front()].feature[feature]);evaluate(model,feature,condition,"resub",-1,data,family?all:y_all,rows,summaries);evaluate(model,feature,condition,"transfer20261020",-1,data,transfer,rows,summaries);
  if(family&&feature==0)for(int i:all){auto q=model.predict(data[i].feature[0]);auto r=frozen.predict(data[i].feature[0]);baseline_delta=std::max(baseline_delta,std::abs((q.axis=="Y"?q.y:q.t)-r.value));baseline_axis_mismatch+=q.axis!=r.axis;}
  std::vector<double> times;for(int i:all)if(data[i].valid[feature])times.push_back(data[i].measurement_ms[feature]);require(!times.empty(),"No valid timing observations");
  double sink=0;start=Clock::now();for(int repeat=0;repeat<10;++repeat)for(int i:y_all){auto q=model.predict(data[i].feature[feature]);sink+=q.y+q.t;}double predict_ms=elapsed(start)/(10*y_all.size());require(std::isfinite(sink),"Invalid prediction timing");
  timing<<condition<<','<<times.size()<<','<<percentile(times,.5)<<','<<percentile(times,.95)<<','<<(feature?init_j204:init_p12)<<','<<predict_ms<<','<<train_ms<<'\n';std::cout<<"Completed "<<condition<<"\n"<<std::flush;
 }
 std::ofstream summary(output/"summary.csv");summary<<std::setprecision(17)<<"condition,protocol,n,failed,Y_mae_valid,Y_max_valid,T_mae_valid,T_max_valid,Y_pass,both_pass\n";
 for(const auto& entry:summaries){const auto& s=entry.second;summary<<entry.first<<','<<s.n<<','<<s.failed<<','<<s.y_sum/std::max(1,s.n-s.failed)<<','<<s.y_max<<','<<s.t_sum/std::max(1,s.n-s.failed)<<','<<s.t_max<<','<<s.y_pass<<','<<s.both_pass<<'\n';}
 std::ofstream audit(output/"audit.json");audit<<std::setprecision(17)<<"{\"measured_photos\":341,\"Y_oof_photos\":80,\"YorT_oof_photos\":280,\"Y_train_per_fold\":61,\"YorT_train_per_fold\":210,\"Y_shrink\":0.01,\"YorT_shrink\":0.05,\"frozen_baseline_max_difference\":"<<baseline_delta<<",\"frozen_baseline_axis_mismatch\":"<<baseline_axis_mismatch<<",\"cache_used\":"<<(argc==6?"true":"false")<<",\"scope\":\"Existing exposed development photos; original splits; transfer mixed poses are not post-T-alignment captures; final modules unchanged\"}\n";audit.close();
 require(baseline_delta<1e-6&&baseline_axis_mismatch==0,"Original P12 fitting-family reproduction failed");
 std::cout<<"Completed comparison; original baseline max difference="<<baseline_delta<<"\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
