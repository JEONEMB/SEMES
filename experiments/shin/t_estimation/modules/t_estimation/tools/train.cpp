#include "t_model.hpp"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <algorithm>
namespace fs=std::filesystem;
static std::vector<std::string> split(const std::string& s){std::vector<std::string> v;std::istringstream in(s);std::string t;while(std::getline(in,t,','))v.push_back(t);return v;}
int wmain(int argc,wchar_t** argv){try{
 cv::setNumThreads(1); if(argc!=3)throw std::invalid_argument("Usage: t_train QRWITHBAR_MODULE OUTPUT_DIRECTORY");
 fs::path module=argv[1],out=argv[2];fs::create_directories(out);std::vector<t_only::Sample> rows;std::ifstream f(module/"tests/fixtures/manifest.csv");std::string s;std::getline(f,s);
 while(std::getline(f,s)){auto v=split(s);rows.push_back({v.at(0),std::stoi(v.at(1)),std::stod(v.at(0).substr(v.at(0).find("T")+1)),{}});}
 if(rows.size()!=57)throw std::runtime_error("Expected 57 labels");f.close();f.open(module/"tests/fixtures/comparison_features.csv");std::getline(f,s);std::vector<int> seen(57,0);
 while(std::getline(f,s)){auto v=split(s);if(v.at(0)!="lanczos3")continue;if(std::stoi(v.at(2)))throw std::runtime_error("Failed observation");auto r=std::find_if(rows.begin(),rows.end(),[&](const t_only::Sample& r){return r.path==v.at(1);});if(r==rows.end())throw std::runtime_error("Unknown feature path");int index=(int)(r-rows.begin());if(seen[index]++)throw std::runtime_error("Duplicate feature path");for(int j=0;j<10;j++)r->features[j]=std::stod(v.at(4+j));}
 for(int n:seen)if(n!=1)throw std::runtime_error("Missing feature row");
 std::ofstream predictions(out/"predictions.csv"),metrics(out/"metrics.csv"),configs(out/"fold_models.csv");predictions<<std::setprecision(17)<<"phase,path,group,true_T,predicted_T,error,error_ticks,pass_1tick\n";metrics<<std::setprecision(17)<<"phase,count,MAE,RMSE,max_absolute,MAE_ticks,RMSE_ticks,max_ticks,pass_1tick\n";configs<<"phase,group,representation,degree,alpha,loss,rank\n";
 for(int phase=0;phase<3;phase++) {double sae=0,sse=0,maxe=0;int pass=0;for(int g=0;g<19;g++){
 std::vector<int> train,test;for(int i=0;i<57;i++)(rows[i].group==g?test:train).push_back(i);if(test.size()!=3)throw std::runtime_error("Expected 3 repetitions per coordinate");auto cfg=t_only::select(rows,train,phase);auto model=t_only::fit(rows,train,cfg);t_only::save(model,(out/("phase"+std::to_string(phase)+"_fold"+std::to_string(g)+".json")).u8string());configs<<phase<<','<<g<<','<<cfg.representation<<','<<cfg.degree<<','<<cfg.alpha<<','<<cfg.loss<<','<<cfg.rank<<'\n';
 for(int i:test){double p=t_only::predict(model,rows[i].features),e=p-rows[i].target,a=std::abs(e);sae+=a;sse+=e*e;maxe=std::max(maxe,a);pass+=a<=t_only::tick;predictions<<phase<<','<<rows[i].path<<','<<g<<','<<rows[i].target<<','<<p<<','<<e<<','<<e/t_only::tick<<','<<(a<=t_only::tick)<<'\n';}}
 metrics<<phase<<",57,"<<sae/57<<','<<sqrt(sse/57)<<','<<maxe<<','<<sae/57/t_only::tick<<','<<sqrt(sse/57)/t_only::tick<<','<<maxe/t_only::tick<<','<<pass<<'\n';std::cout<<"Phase "<<phase<<" MAE="<<sae/57<<" RMSE="<<sqrt(sse/57)<<" max="<<maxe<<" ("<<maxe/t_only::tick<<" ticks) pass="<<pass<<"/57\n";}
 std::vector<int> all(57);std::iota(all.begin(),all.end(),0);for(int mode=0;mode<3;mode++){auto candidate=t_only::fit(rows,all,t_only::select(rows,all,mode));t_only::save(candidate,(out/("deployment_phase"+std::to_string(mode)+".json")).u8string());} auto m=t_only::fit(rows,all,t_only::select(rows,all,0));t_only::save(m,(out/"deployment.json").u8string());auto reloaded=t_only::load((out/"deployment.json").u8string());double diff=0;for(auto& r:rows)diff=std::max(diff,std::abs(t_only::predict(m,r.features)-t_only::predict(reloaded,r.features)));if(diff>1e-12)throw std::runtime_error("Model serialization changed predictions");std::cout<<"Deployment representation="<<m.config.representation<<" degree="<<m.config.degree<<" alpha="<<m.config.alpha<<" loss="<<m.config.loss<<" rank="<<m.config.rank<<" roundtrip="<<diff<<'\n';return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
