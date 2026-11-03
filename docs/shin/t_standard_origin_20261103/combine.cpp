#include "six.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <functional>
#include <numeric>
#include <cmath>
namespace fs=std::filesystem;
int wmain(int argc,wchar_t** argv){try{
 if(argc!=2)throw std::invalid_argument("Usage: six_combo EXPERIMENT");cv::setNumThreads(1);fs::path root=argv[1];auto rows=six::read_rows(root/"original_observations.csv");auto corrected=six::read_rows(root/"observations.csv");if(rows.size()!=corrected.size())throw std::runtime_error("Coordinate count mismatch");for(size_t i=0;i<rows.size();i++){if(rows[i].path!=corrected[i].path)throw std::runtime_error("Coordinate path mismatch");rows[i].y=corrected[i].y;rows[i].t=corrected[i].t;}std::vector<int> all;for(int i=0;i<(int)rows.size();i++)if(rows[i].cohort!="reference")all.push_back(i);
 for(int family:{7}){
 std::vector<int> methods=family==7?std::vector<int>{0,1,2,3,6}:std::vector<int>{0,1,2,3,5,6};int count=(int)methods.size();std::ofstream pred(root/("results/predictions_method"+std::to_string(family)+".csv")),metric(root/("results/metrics_method"+std::to_string(family)+".csv")),log(root/("results/configs_method"+std::to_string(family)+".csv"));pred<<std::setprecision(17)<<"path,cohort,fold,session_proxy,T_true,T_pred,error_deg,error_ticks,status,known_session,cost,inference_ms,low_quality_line_samples\n";log<<"fold,member,weight\n";double sae=0,sse=0,maxe=0;int passes=0;std::vector<cv::Mat> weights;
 for(int fold=0;fold<5;fold++){
  std::vector<int> train,test;for(int i:all)(rows[i].fold==fold?test:train).push_back(i);cv::Mat oof=cv::Mat::zeros((int)rows.size(),count,CV_64F);six::Model ensemble;ensemble.config.method=family;
  for(int k=0;k<count;k++){
   auto outer=six::load_model(root/("results/models/method"+std::to_string(methods[k])+"_fold"+std::to_string(fold)+".json"));ensemble.members.push_back(std::make_shared<six::Model>(outer));
   for(int inner=0;inner<5;inner++){if(inner==fold)continue;std::vector<int> fit,validation;for(int i:train)(rows[i].fold==inner?validation:fit).push_back(i);auto model=six::fit(rows,fit,outer.config);for(int i:validation)oof.at<double>(i,k)=six::predict(model,rows[i].x,rows[i].time).t;}
  }
  double best=1e300;cv::Mat selected(1,count,CV_64F,cv::Scalar(0)),current=selected.clone();
  std::function<void(int,int)> search=[&](int k,int remaining){if(k==count-1){current.at<double>(k)=remaining/4.;double sum=0,worst=0;for(int i:train){double p=oof.row(i).dot(current),e=(p-rows[i].t)/.01;sum+=e*e;worst=std::max(worst,std::abs(e));}double score=worst+.1*sqrt(sum/train.size());if(score<best){best=score;selected=current.clone();}return;}for(int v=0;v<=remaining;v++){current.at<double>(k)=v/4.;search(k+1,remaining-v);}};search(0,4);
  ensemble.weights=selected;weights.push_back(selected);ensemble.min_t=1e30;ensemble.max_t=-1e30;for(int i:train){ensemble.min_t=std::min(ensemble.min_t,rows[i].t);ensemble.max_t=std::max(ensemble.max_t,rows[i].t);}six::save_model(ensemble,root/("results/models/method"+std::to_string(family)+"_fold"+std::to_string(fold)+".json"));for(int k=0;k<count;k++)log<<fold<<','<<methods[k]<<','<<selected.at<double>(k)<<'\n';
  for(int i:test){auto p=six::predict(ensemble,rows[i].x,rows[i].time);double e=p.t-rows[i].t;sae+=std::abs(e);sse+=e*e;maxe=std::max(maxe,std::abs(e));passes+=std::abs(e)<=.01;pred<<rows[i].path<<','<<rows[i].cohort<<','<<fold<<','<<rows[i].session<<','<<rows[i].t<<','<<p.t<<','<<e<<','<<e/.01<<",0,"<<p.known_session<<",0,0,0\n";}std::cout<<"Combination "<<family<<" fold "<<fold<<" done\n"<<std::flush;
 }
 metric<<std::setprecision(17)<<"split,attempted,valid,MAE_deg,RMSE_deg,max_abs_deg,MAE_ticks,RMSE_ticks,max_ticks,pass_1tick\nall,571,571,"<<sae/571<<','<<sqrt(sse/571)<<','<<maxe<<','<<sae/571/.01<<','<<sqrt(sse/571)/.01<<','<<maxe/.01<<','<<passes<<'\n';
 six::Model deployment;deployment.config.method=family;deployment.weights=cv::Mat::zeros(1,count,CV_64F);for(auto& w:weights)deployment.weights+=w/5;deployment.min_t=1e30;deployment.max_t=-1e30;for(int i:all){deployment.min_t=std::min(deployment.min_t,rows[i].t);deployment.max_t=std::max(deployment.max_t,rows[i].t);}for(int method:methods)deployment.members.push_back(std::make_shared<six::Model>(six::load_model(root/("results/models/deployment_method"+std::to_string(method)+".json"))));six::save_model(deployment,root/("results/models/deployment_method"+std::to_string(family)+".json"));auto reload=six::load_model(root/("results/models/deployment_method"+std::to_string(family)+".json"));for(int i:all)if(std::abs(six::predict(deployment,rows[i].x,rows[i].time).t-six::predict(reload,rows[i].x,rows[i].time).t)>1e-10)throw std::runtime_error("Combination roundtrip failed");std::cout<<"Combination "<<family<<" max="<<maxe/.01<<" MAE="<<sae/571/.01<<" pass="<<passes<<"/571\n"<<std::flush;
 }return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
