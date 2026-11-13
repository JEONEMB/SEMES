#include "six.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <functional>
#include <cmath>
#include <map>
#include <algorithm>
namespace fs=std::filesystem;
namespace six {void register_training_metadata(const std::vector<Row>&,const std::vector<double>&);}
struct Dataset {std::vector<six::Row> rows;std::vector<double> weight;std::vector<int> available;};
Dataset load(const fs::path& dir,const std::vector<six::Row>& base){
 Dataset d;d.rows=six::read_rows(dir/"observations.csv");
 if(d.rows.size()!=base.size())throw std::runtime_error("Dataset count mismatch");
 for(size_t i=0;i<base.size();i++){if(d.rows[i].path!=base[i].path)throw std::runtime_error("Dataset order mismatch");d.rows[i].fold=base[i].fold;}
 for(auto name:{"weights.csv","target_available.csv"}){
  std::ifstream f(dir/name);if(!f)throw std::runtime_error("Missing dataset metadata");std::string line;std::getline(f,line);int i=0;
  while(std::getline(f,line)){if(!line.empty()&&line.back()=='\r')line.pop_back();auto k=line.rfind(',');if(k==std::string::npos||line.substr(0,k)!=base.at(i).path)throw std::runtime_error("Metadata path mismatch");double v=std::stod(line.substr(k+1));if(std::string(name)=="weights.csv"){if(v<=0||v>1||!std::isfinite(v))throw std::runtime_error("Invalid row weight");d.weight.push_back(v);}else d.available.push_back((int)v);i++;}
  if(i!=(int)base.size())throw std::runtime_error("Metadata row count mismatch");
 }
 return d;
}
six::Model fit(const Dataset& d,const std::vector<six::Row>& base,const std::vector<int>& ids,six::Config cfg){six::register_training_metadata(base,d.weight);return six::fit(d.rows,ids,cfg);}
void roundtrip(const six::Model& m,const fs::path& file,const std::vector<six::Row>& base){
 six::save_model(m,file);auto r=six::load_model(file);for(auto& row:base)if(std::abs(six::predict(m,row.x).t-six::predict(r,row.x).t)>1e-10)throw std::runtime_error("Model roundtrip mismatch");
}
int wmain(int argc,wchar_t** argv){try{
 if(argc!=3)throw std::runtime_error("Usage: v3_train DATASET LEGACY_MODELS_DIRECTORY");
 cv::setNumThreads(1);fs::path root=argv[1],legacy=argv[2];auto base=six::read_rows(root/"original_observations.csv");auto full=load(root,base);
 fs::create_directories(root/"results/models");std::vector<int> all;for(int i=0;i<(int)base.size();i++)if(base[i].cohort!="reference")all.push_back(i);
 const std::vector<int> methods{0,1,2,3,6};std::vector<six::Config> configs;
 for(int k:methods)configs.push_back(six::load_model(legacy/("deployment_method"+std::to_string(k)+".json")).config);
 std::ofstream pred(root/"results/predictions.csv"),members(root/"results/member_predictions.csv"),wlog(root/"results/weights.csv");
 pred<<std::setprecision(17)<<"path,fold,cohort,T_target,T_pred,error_ticks,target_available,status\n";
 members<<std::setprecision(17)<<"path,fold,method,T_target,T_pred,target_available\n";wlog<<"fold,method,weight\n";
 std::vector<cv::Mat> weights;
 for(int outer=0;outer<5;outer++){
  Dataset data=fs::exists(root/("outer"+std::to_string(outer)) )?load(root/("outer"+std::to_string(outer)),base):full;
  std::vector<int> train,test;for(int i:all)(base[i].fold==outer?test:train).push_back(i);
  cv::Mat innerPred=cv::Mat::zeros((int)base.size(),5,CV_64F);std::vector<double> target(base.size());std::vector<int> available(base.size());
  six::Model ensemble;ensemble.config.method=7;ensemble.min_t=1e30;ensemble.max_t=-1e30;
  for(int i:train){ensemble.min_t=std::min(ensemble.min_t,data.rows[i].t);ensemble.max_t=std::max(ensemble.max_t,data.rows[i].t);}
  for(int inner=0;inner<5;inner++){
   if(inner==outer)continue;auto dir=root/("outer"+std::to_string(outer)+"_inner"+std::to_string(inner));Dataset inside=fs::exists(dir)?load(dir,base):data;
   std::vector<int> fitting,validation;for(int i:train)(base[i].fold==inner?validation:fitting).push_back(i);
   for(int i:validation){target[i]=inside.rows[i].t;available[i]=inside.available[i];}
   for(int k=0;k<5;k++){auto m=fit(inside,base,fitting,configs[k]);for(int i:validation)innerPred.at<double>(i,k)=six::predict(m,base[i].x).t;}
  }
  double best=1e300;cv::Mat selected=cv::Mat::zeros(1,5,CV_64F),current=selected.clone();
  std::function<void(int,int)> search=[&](int k,int remaining){
   if(k==4){current.at<double>(k)=remaining/4.;double sum=0,worst=0;int n=0;for(int i:train)if(available[i]){double e=(innerPred.row(i).dot(current)-target[i])*100;sum+=e*e;worst=std::max(worst,std::abs(e));n++;}double score=n?worst+.1*sqrt(sum/n):1e300;if(score<best){best=score;selected=current.clone();}return;}
   for(int v=0;v<=remaining;v++){current.at<double>(k)=v/4.;search(k+1,remaining-v);}
  };search(0,4);ensemble.weights=selected;weights.push_back(selected);
  for(int k=0;k<5;k++){
   auto m=fit(data,base,train,configs[k]);ensemble.members.push_back(std::make_shared<six::Model>(m));
   wlog<<outer<<','<<methods[k]<<','<<selected.at<double>(k)<<'\n';
   for(int i:test)members<<base[i].path<<','<<outer<<','<<methods[k]<<','<<data.rows[i].t<<','<<six::predict(m,base[i].x).t<<','<<data.available[i]<<'\n';
  }
  roundtrip(ensemble,root/("results/models/fold"+std::to_string(outer)+".json"),base);
  for(int i:test){double p=six::predict(ensemble,base[i].x).t;pred<<base[i].path<<','<<outer<<','<<base[i].cohort<<','<<data.rows[i].t<<','<<p<<','<<(p-data.rows[i].t)*100<<','<<data.available[i]<<",0\n";}
  pred.flush();members.flush();std::cout<<root.filename().u8string()<<" fold "<<outer<<" done; inner score="<<best<<'\n'<<std::flush;
 }
 six::Model deployment;deployment.config.method=7;deployment.weights=cv::Mat::zeros(1,5,CV_64F);for(auto w:weights)deployment.weights+=w/5.;deployment.min_t=1e30;deployment.max_t=-1e30;
 for(int i:all){deployment.min_t=std::min(deployment.min_t,full.rows[i].t);deployment.max_t=std::max(deployment.max_t,full.rows[i].t);}
 for(auto cfg:configs)deployment.members.push_back(std::make_shared<six::Model>(fit(full,base,all,cfg)));
 roundtrip(deployment,root/"results/models/deployment.json",base);
 std::ofstream smoke(root/"results/deployment_predictions.csv");smoke<<std::setprecision(17)<<"path,T_pred\n";for(auto& r:base)smoke<<r.path<<','<<six::predict(deployment,r.x).t<<'\n';
 std::cout<<"FINISHED "<<root.filename().u8string()<<"; "<<all.size()<<" OOF predictions; roundtrip PASS\n"<<std::flush;return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
