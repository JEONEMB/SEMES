#include "t_model.hpp"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <limits>
namespace fs=std::filesystem;
struct Row {t_only::Sample s;std::string cohort;double y;int status;};
static std::vector<std::string> split(const std::string& text){std::vector<std::string> v;std::istringstream in(text);std::string s;while(std::getline(in,s,','))v.push_back(s);return v;}
static std::vector<Row> read(const fs::path& path,bool evaluation){std::ifstream f(path);if(!f)throw std::runtime_error("Missing observations");std::string text;std::getline(f,text);std::vector<Row> rows;int count=0;while(std::getline(f,text)){count++;auto v=split(text);bool isnew=v.at(1)=="new_evaluation";if(isnew!=evaluation||(!isnew&&v.at(1)=="reference"))continue;Row r;r.cohort=v[1];r.y=std::stod(v[2]);r.s.path=v[0];r.s.target=std::stod(v[3]);r.status=std::stoi(v[4]);std::string key=std::to_string(std::lround(r.y*100))+":"+std::to_string(std::lround(r.s.target*100));unsigned hash=2166136261u;for(unsigned char c:key)hash=(hash^c)*16777619u;r.s.group=hash%5;for(int j=0;j<22;j++)r.s.features[j]=std::stod(v.at(5+j));rows.push_back(r);}if(count!=575)throw std::runtime_error("Extraction incomplete: expected575 rows");return rows;}
static t_only::Config tune(const std::vector<t_only::Sample>& rows,const std::vector<int>& ids,int mode){
 t_only::Config best;double bestcost=1e300;
 for(int kind=0;kind<(mode?8:6);kind++)for(int degree=1;degree<=((mode==3&&(kind==3||kind==5))?3:((mode>=2&&kind<6)?2:1));degree++)for(double alpha:{0.,.001,.01,.1,1.,10.,100.}){
  t_only::Config cfg{kind,degree,alpha,0,0};double worst=0,sse=0;int n=0;
  for(int g=0;g<5;g++){std::vector<int> train,test;for(int i:ids)(rows[i].group==g?test:train).push_back(i);if(test.empty()||train.empty())continue;auto model=t_only::fit(rows,train,cfg);for(int i:test){double e=(t_only::predict(model,rows[i].features)-rows[i].target)/.01;worst=std::max(worst,std::abs(e));sse+=e*e;n++;}}
  double cost=worst+.1*sqrt(sse/n);if(cost<bestcost){bestcost=cost;best=cfg;}
 }return best;
}
struct Metrics {int count=0,valid=0,pass=0;double sae=0,sse=0,maximum=0;void add(double e,bool ok){count++;if(!ok)return;valid++;sae+=std::abs(e);sse+=e*e;maximum=std::max(maximum,std::abs(e));pass+=std::abs(e)<=.01;}double score()const{return maximum/.01+.1*sqrt(sse/valid)/.01;}void write(std::ostream& f,const std::string& name)const{f<<name<<','<<count<<','<<valid<<','<<sae/valid<<','<<sqrt(sse/valid)<<','<<maximum<<','<<sae/valid/.01<<','<<sqrt(sse/valid)/.01<<','<<maximum/.01<<','<<pass<<'\n';}};
int wmain(int argc,wchar_t** argv){try{
 if(argc!=4)throw std::invalid_argument("Usage: t_dataset train|evaluate|expand|axis|stress OBSERVATIONS OUTPUT_DIRECTORY");cv::setNumThreads(1);std::wstring action=argv[1];fs::path input=argv[2],out=argv[3];fs::create_directories(out);
 if(action==L"train"||action==L"expand"||action==L"axis"){ bool expanded=action==L"expand"; bool axis=action==L"axis";
 auto original=read(input,false);if(axis){original.erase(std::remove_if(original.begin(),original.end(),[](const Row& r){return r.cohort!="theta_development";}),original.end());} if(expanded){auto extra=read(input,true);for(auto& r:extra){r.cohort="additional_development";original.push_back(r);}} if(original.size()!=(expanded?571:(axis?200:337)))throw std::runtime_error("Unexpected development image count");std::vector<t_only::Sample> rows;for(auto& r:original){if(r.status)throw std::runtime_error("Development measurement failed; inspect audit before training");rows.push_back(r.s);}std::ofstream pred(out/"development_predictions.csv"),metrics(out/"development_metrics.csv"),configs(out/"development_models.csv");pred<<std::setprecision(17)<<"mode,path,cohort,fold,true_T,predicted_T,error_ticks\n";metrics<<std::setprecision(17)<<"mode,count,valid,MAE,RMSE,max,MAE_ticks,RMSE_ticks,max_ticks,pass_1tick\n";configs<<"mode,fold,representation,degree,alpha\n";double bestscore=1e300;int selected=0;
 for(int mode=0;mode<(expanded?4:3);mode++){Metrics total;for(int g=0;g<5;g++){std::vector<int> train,test;for(int i=0;i<(int)rows.size();i++)(rows[i].group==g?test:train).push_back(i);auto cfg=tune(rows,train,mode);auto model=t_only::fit(rows,train,cfg);model.requires_y_fixed_at_reference=axis;t_only::save(model,(out/("mode"+std::to_string(mode)+"_fold"+std::to_string(g)+".json")).u8string());configs<<mode<<','<<g<<','<<cfg.representation<<','<<cfg.degree<<','<<cfg.alpha<<'\n';for(int i:test){double p=t_only::predict(model,rows[i].features);total.add(p-rows[i].target,true);pred<<mode<<','<<rows[i].path<<','<<original[i].cohort<<','<<g<<','<<rows[i].target<<','<<p<<','<<(p-rows[i].target)/.01<<'\n';}}
 total.write(metrics,std::to_string(mode));std::cout<<"Development mode "<<mode<<" max="<<total.maximum/.01<<" ticks MAE="<<total.sae/total.valid/.01<<" pass="<<total.pass<<'/'<<total.count<<'\n'<<std::flush;
 std::vector<int> all;for(int i=0;i<(int)rows.size();i++)all.push_back(i);auto cfg=tune(rows,all,mode);auto model=t_only::fit(rows,all,cfg);model.requires_y_fixed_at_reference=axis;t_only::save(model,(out/("deployment_mode"+std::to_string(mode)+".json")).u8string());if(total.score()<bestscore){bestscore=total.score();selected=mode;}}
 fs::copy_file(out/("deployment_mode"+std::to_string(selected)+".json"),out/"frozen_model.json",fs::copy_options::overwrite_existing);std::ofstream freeze(out/"freeze.txt");freeze<<"selected_mode="<<selected<<"\ntraining_images="<<(expanded?571:(axis?200:337))<<"\nnew_evaluation_images_used_for_selection="<<(expanded?234:0)<<"\nindependent_test=false\n";std::cout<<"FROZEN mode="<<selected<<"; run evaluate once after freeze\n";return 0;
 }
 if(action==L"stress"){
 auto model=t_only::load((out/"frozen_model.json").u8string());auto rows=read(input,false);auto extra=read(input,true);rows.insert(rows.end(),extra.begin(),extra.end());std::ofstream summary(out/"stress_metrics.csv");summary<<std::setprecision(17)<<"cohort,count,valid,MAE,RMSE,max,MAE_ticks,RMSE_ticks,max_ticks,pass_1tick\n";
 for(const std::string cohort : {"y_development","mixed_development","new_evaluation"}){Metrics metric;for(auto& r:rows)if(r.cohort==cohort)metric.add(t_only::predict(model,r.s.features)-r.s.target,r.status==0);metric.write(summary,cohort);}return 0;
 }
 if(action==L"evaluate"){
 if(fs::exists(out/"evaluation_metrics.csv"))throw std::runtime_error("Evaluation already exists; refusing to overwrite one-shot results");auto model=t_only::load((out/"frozen_model.json").u8string());auto rows=read(input,true);if(rows.size()!=234)throw std::runtime_error("Expected234 new images");Metrics metrics;std::ofstream f(out/"evaluation_predictions.csv"),summary(out/"evaluation_metrics.csv");f<<std::setprecision(17)<<"path,Y,true_T,predicted_T,error_ticks,status,pass_1tick\n";summary<<std::setprecision(17)<<"split,count,valid,MAE,RMSE,max,MAE_ticks,RMSE_ticks,max_ticks,pass_1tick\n";
 for(auto& r:rows){double p=std::numeric_limits<double>::quiet_NaN(); if(r.status==0)p=t_only::predict(model,r.s.features); double e=p-r.s.target;metrics.add(e,r.status==0);f<<r.s.path<<','<<r.y<<','<<r.s.target<<','<<p<<','<<e/.01<<','<<r.status<<','<<(r.status==0&&std::abs(e)<=.01)<<'\n';}metrics.write(summary,"new_capture");std::cout<<"NEW234 max="<<metrics.maximum/.01<<" ticks MAE="<<metrics.sae/metrics.valid/.01<<" pass="<<metrics.pass<<'/'<<metrics.count<<" valid="<<metrics.valid<<'\n';return 0;
 }throw std::invalid_argument("Unknown action");
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
