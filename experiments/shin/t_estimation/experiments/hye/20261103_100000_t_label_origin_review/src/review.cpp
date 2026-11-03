#include "six.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <map>
#include <set>
#include <functional>
#include <numeric>
#include <algorithm>
#include <cmath>
namespace fs=std::filesystem;
using Table=std::map<std::string,std::map<std::string,std::string>>;
static std::vector<std::string> split(const std::string& s){std::stringstream f(s);std::vector<std::string> r;std::string v;while(std::getline(f,v,',')){if(!v.empty()&&v.back()=='\r')v.pop_back();r.push_back(v);}return r;}
static Table table(const fs::path& path){std::ifstream f(path);if(!f)throw std::runtime_error("Missing CSV "+path.u8string());std::string line;std::getline(f,line);auto h=split(line);Table r;while(std::getline(f,line)){auto v=split(line);if(v.size()!=h.size())throw std::runtime_error("CSV column mismatch");std::map<std::string,std::string> row;for(size_t j=0;j<h.size();j++)row[h[j]]=v[j];if(!r.emplace(row.at("path"),row).second)throw std::runtime_error("Duplicate path");}return r;}
static std::vector<six::Row> targets(std::vector<six::Row> rows,const fs::path& file){auto t=table(file);for(auto& r:rows){auto v=t.at(r.path);r.y=std::stod(v.at("Y_standard"));r.t=std::stod(v.at("T_standard"));}return rows;}
static six::Model ensemble(const std::vector<six::Row>& rows,const std::vector<int>& ids,const std::vector<six::Config>& cfg){six::Model m;m.config.method=7;m.min_t=-1;m.max_t=2;for(auto c:cfg)m.members.push_back(std::make_shared<six::Model>(six::fit(rows,ids,c)));m.weights=cv::Mat(1,(int)cfg.size(),CV_64F,cv::Scalar(1./cfg.size()));return m;}
static cv::Mat choose(const cv::Mat& oof,const std::vector<int>& train,const std::vector<double>& truth){int n=oof.cols;cv::Mat w(1,n,CV_64F,cv::Scalar(0)),best=w.clone();double minimum=1e300;std::function<void(int,int)> visit=[&](int k,int left){if(k+1==n){w.at<double>(k)=left/4.;double sse=0,maxe=0;for(int i:train){double e=(oof.row(i).dot(w)-truth[i])/.01;sse+=e*e;maxe=std::max(maxe,std::abs(e));}double score=maxe+.1*sqrt(sse/train.size());if(score<minimum){minimum=score;best=w.clone();}return;}for(int x=0;x<=left;x++){w.at<double>(k)=x/4.;visit(k+1,left-x);}};visit(0,4);return best;}
int wmain(int argc,wchar_t** argv){try{
 if(argc!=5)throw std::runtime_error("Usage: t_label_review OLD_EXPERIMENT NEW_EXPERIMENT COORDINATES raw|standard|group|all");cv::setNumThreads(1);fs::path old=argv[1],out=argv[2],coords=argv[3];bool standard=std::wstring(argv[4])==L"standard",leavegroup=std::wstring(argv[4])==L"group",auditall=std::wstring(argv[4])==L"all";if(!standard&&!leavegroup&&!auditall&&std::wstring(argv[4])!=L"raw")throw std::runtime_error("Invalid mode");
 auto original=six::read_rows(old/"observations.csv");auto metadata=table(coords);auto anomalies=table(out/"exclusions.csv");if(original.size()!=575||anomalies.size()!=8)throw std::runtime_error("Unexpected input counts");for(auto& r:original)r.session=std::stoi(metadata.at(r.path).at("group"));
 auto deployed=six::load_model(old/"assets/best_image_ensemble.json");std::vector<six::Config> configs;for(auto& m:deployed.members)configs.push_back(m->config);
 for(int policy:auditall?std::vector<int>{0}:(standard||leavegroup)?std::vector<int>{8}:std::vector<int>{8,5}){
  std::string name=(standard?"standard_crossfit":leavegroup?"nominal_group":"nominal")+std::string("_exclude")+std::to_string(policy);fs::path dir=out/"results"/name;fs::create_directories(dir/"models");
  std::vector<int> all,refs;for(int i=0;i<(int)original.size();i++){auto& r=original[i];if(r.cohort=="reference"){refs.push_back(i);continue;}if(policy>0&&anomalies.count(r.path)&&(policy==8||anomalies.at(r.path).at("sensitivity_exclude")=="1"))continue;if(r.status)throw std::runtime_error("Measurement failure");all.push_back(i);}if(all.size()!=(policy==0?571:policy==8?563:566)||refs.size()!=4)throw std::runtime_error("Unexpected evaluation count");
  std::ofstream pred(dir/"predictions.csv"),cal(dir/"calibration.csv"),audit(dir/"isolation.csv"),weights(dir/"weights.csv"),innerlog(dir/"inner_predictions.csv");innerlog<<std::setprecision(17)<<"outer,path,group,target,prediction,used_for_calibration\n";
  pred<<std::setprecision(17)<<"path,cohort,fold,group,Y,T_original,T_target,target_available,raw,global_oof,session_oof,session_shrink30,reference_zero,known_group\n";
  cal<<std::setprecision(17)<<"fold,group,n,global_shift,session_shift,shrink30_shift,reference_shift\n";audit<<"outer,inner,fit_count,validation_count,overlap\n";weights<<"fold,member,weight\n";
  std::set<int> cases;if(leavegroup){for(int i:all)cases.insert(original[i].session);}else for(int f=0;f<5;f++)cases.insert(f);
  for(int outer:cases){
   std::vector<int> train,test;for(int i:all)((leavegroup?original[i].session:original[i].fold)==outer?test:train).push_back(i);auto rows=standard?targets(original,out/("origins/outer"+std::to_string(outer)+".csv")):original;
   cv::Mat oof=cv::Mat::zeros((int)rows.size(),(int)configs.size(),CV_64F);std::vector<double> truth(rows.size());std::map<int,std::pair<double,int>> residual;
   std::vector<std::vector<six::Row>> innerdata(5);std::vector<std::vector<int>> fits(5),vals(5);std::vector<int> selection;
   for(int inner=0;inner<5;inner++){if(!leavegroup&&inner==outer)continue;
    for(int i:train)(original[i].fold==inner?vals[inner]:fits[inner]).push_back(i);
    innerdata[inner]=standard?targets(original,out/("origins/outer"+std::to_string(outer)+"_inner"+std::to_string(inner)+".csv")):original;
    Table av;if(standard)av=table(out/("origins/outer"+std::to_string(outer)+"_inner"+std::to_string(inner)+".csv"));
    for(int i:vals[inner]){truth[i]=innerdata[inner][i].t;if(!standard||av.at(original[i].path).at("available")=="1")selection.push_back(i);}
    std::set<int> fitted(fits[inner].begin(),fits[inner].end());int overlap=0;for(int i:vals[inner])overlap+=fitted.count(i);for(int i:test)overlap+=fitted.count(i);if(overlap)throw std::runtime_error("Fold leakage");audit<<outer<<','<<inner<<','<<fits[inner].size()<<','<<vals[inner].size()<<','<<overlap<<'\n';
   }
   auto chosen=configs;
   for(int k=0;k<(int)configs.size();k++){
    double best=1e300;cv::Mat winning;
    for(auto candidate:six::candidates(configs[k].method)){
     cv::Mat attempt=cv::Mat::zeros((int)rows.size(),1,CV_64F);
     for(int inner=0;inner<5;inner++){if(!leavegroup&&inner==outer)continue;auto m=six::fit(innerdata[inner],fits[inner],candidate);for(int i:vals[inner])attempt.at<double>(i)=six::predict(m,original[i].x).t;}
     double sse=0,maxe=0;for(int i:selection){double e=(attempt.at<double>(i)-truth[i])/.01;sse+=e*e;maxe=std::max(maxe,std::abs(e));}double score=maxe+.1*sqrt(sse/selection.size());
     if(score<best){best=score;chosen[k]=candidate;winning=attempt;}
    }
    winning.copyTo(oof.col(k));
   }
   auto model=ensemble(rows,train,chosen);model.weights=choose(oof,selection,truth);for(int i:train)innerlog<<outer<<','<<original[i].path<<','<<original[i].session<<','<<truth[i]<<','<<oof.row(i).dot(model.weights)<<','<<(std::find(selection.begin(),selection.end(),i)!=selection.end())<<'\n';double global=0;for(int i:selection){double e=truth[i]-oof.row(i).dot(model.weights);global+=e;auto& r=residual[original[i].session];r.first+=e;r.second++;}global/=selection.size();double reference=0;for(int i:refs)reference-=six::predict(model,original[i].x).t/refs.size();
   for(auto [group,v]:residual){double shift=v.first/v.second,shrunk=(v.first+30*global)/(v.second+30);cal<<outer<<','<<group<<','<<v.second<<','<<global<<','<<shift<<','<<shrunk<<','<<reference<<'\n';}for(int k=0;k<model.weights.cols;k++)weights<<outer<<','<<configs[k].method<<','<<model.weights.at<double>(k)<<'\n';
   six::save_model(model,dir/("models/fold"+std::to_string(outer)+".json"));auto reload=six::load_model(dir/("models/fold"+std::to_string(outer)+".json"));Table avail;if(standard)avail=table(out/("origins/outer"+std::to_string(outer)+".csv"));
   for(int i:test){double p=six::predict(model,original[i].x).t;if(!std::isfinite(p)||std::abs(p-six::predict(reload,original[i].x).t)>1e-10)throw std::runtime_error("Prediction/roundtrip error");auto it=residual.find(original[i].session);bool known=it!=residual.end();double shift=known?it->second.first/it->second.second:global,shrink=known?(it->second.first+30*global)/(it->second.second+30):global;int available=standard?std::stoi(avail.at(original[i].path).at("available")):1;
    pred<<original[i].path<<','<<original[i].cohort<<','<<outer<<','<<original[i].session<<','<<original[i].y<<','<<original[i].t<<','<<rows[i].t<<','<<available<<','<<p<<','<<p+global<<','<<p+shift<<','<<p+shrink<<','<<p+reference<<','<<known<<'\n';}
   pred.flush();cal.flush();audit.flush();std::cout<<name<<" fold "<<outer<<" done; n="<<test.size()<<'\n'<<std::flush;
  }
 }
 return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
