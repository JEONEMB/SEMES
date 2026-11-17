#include "control_internal.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
namespace fs=std::filesystem;
static std::vector<std::string> split(const std::string& line){std::vector<std::string> out;std::stringstream s(line);std::string v;while(std::getline(s,v,',')){if(!v.empty()&&v.back()=='\r')v.pop_back();out.push_back(v);}return out;}
static void require(bool condition,const char* message){if(!condition)throw std::runtime_error(message);}
struct Expected{std::string axis,branch;double value;int fold;};
int wmain(int argc,wchar_t** argv){
 try{
  if(argc<4)throw std::invalid_argument("validate_control MODULE CLEAN OUTPUT [contract]");
  cv::setNumThreads(1);
  fs::path module(argv[1]),clean(argv[2]),output(argv[3]);if(argc==4&&fs::exists(output))throw std::runtime_error("Use a NEW output directory to preserve earlier results");fs::create_directories(output);
  const auto assets=module/"assets";
  initialize_control(assets.u8string());
  cv::Mat center=legacy_control::read_gray(assets/"center.bmp");
  auto initial=center.clone();auto a=control(center);auto diag=alignment_diagnostics();
  require(cv::norm(center,initial,cv::NORM_INF)==0,"Input changed");
  require(diag.tracked_points==630&&diag.valid_points==536,"Wrong point counts");
  require(!diag.pose_stability_assessed&&!diag.control_precondition_met,"False stability certification");
  cv::Mat invalid;int errors=0;
  try{control(invalid);}catch(const std::exception&){++errors;}
  invalid=cv::Mat(100,100,CV_8UC1);try{control(invalid);}catch(const std::exception&){++errors;}
  invalid=cv::Mat(2160,3840,CV_32FC1);try{control(invalid);}catch(const std::exception&){++errors;}
  invalid=cv::Mat(2160,3840,CV_8UC2);try{control(invalid);}catch(const std::exception&){++errors;}
  require(errors==4&&!alignment_diagnostics().measurement_valid,"Invalid input accepted or false diagnostic");
  try{initialize_control((output/"missing_assets").u8string());}catch(const std::exception&){++errors;}
  auto b=control(center);require(a.y==b.y&&a.t==b.t&&errors==5,"Failed initialization corrupted model");
  cv::Mat color;cv::cvtColor(center,color,cv::COLOR_GRAY2BGR);b=control(color);require(std::abs(a.y-b.y)+std::abs(a.t-b.t)<1e-12,"BGR mismatch");
  cv::cvtColor(center,color,cv::COLOR_GRAY2BGRA);b=control(color);require(std::abs(a.y-b.y)+std::abs(a.t-b.t)<1e-12,"BGRA mismatch");
  std::cout<<"Contract checks passed; control links independently of joint.\n";
  if(argc>4)return 0;
  legacy_control::Tracker tracker(assets);legacy_control::Model model(assets/"model.json");
  std::vector<legacy_control::Model> folds;for(int i=0;i<4;++i)folds.emplace_back(module/"validation"/("fold"+std::to_string(i)+"_model.json"));
  std::map<std::string,cv::Mat> expected_features;
  std::ifstream feature_file(module/"validation/python_features.csv");std::string line;std::getline(feature_file,line);
  while(std::getline(feature_file,line)){auto v=split(line);require(v.size()==13,"Invalid feature fixture");cv::Mat x(1,12,CV_64F);for(int i=0;i<12;++i)x.at<double>(i)=std::stod(v[i+1]);expected_features[v[0]]=x;}
  std::map<std::string,Expected> references[2];
  std::ifstream prediction_file(module/"validation/python_predictions.csv");std::getline(prediction_file,line);
  while(std::getline(prediction_file,line)){auto v=split(line);require(v.size()==14,"Invalid prediction fixture");int protocol=v[12]=="raw_oof4"?1:0;references[protocol][v[0]]={v[4],v[6],std::stod(v[5]),protocol?std::stoi(v[13]):-1};}
  require(expected_features.size()==281&&references[0].size()==280&&references[1].size()==280,"Missing reference rows");
  std::ofstream rows(output/"predictions.csv");rows<<std::setprecision(17)<<"file,protocol,fold,pred_axis,pred_value,branch,y_error,t_error,pass,python_prediction_difference,feature_max_difference,valid_selected\n";
  int passed[2]={},correct[2]={},total=0,valid_min=536;double max_y[2]={},max_t[2]={},sum_y[2]={},sum_t[2]={};double feature_max=0,prediction_max=0,model_only_max=0;int parity_failures=0;
  for(const auto& item:references[0]){
   const auto& name=item.first;auto image=legacy_control::read_gray(clean/(name[0]=='Y'?"Y":"Theta")/fs::u8path(name));auto tracks=tracker.measure(image);auto feature=tracker.features(tracks);int valid=tracker.valid_selected(tracks);valid_min=std::min(valid_min,valid);
   double fd=cv::norm(feature,expected_features.at(name),cv::NORM_INF);feature_max=std::max(feature_max,fd);
   const auto& expected_oof=references[1].at(name);
   for(int p=0;p<2;++p){const auto& e=references[p].at(name);const auto& m=p?folds.at(expected_oof.fold):model;
    auto q=m.predict(feature);auto original=m.predict(expected_features.at(name));
    model_only_max=std::max(model_only_max,std::abs(original.value-e.value));
    const auto& stem=fs::u8path(name).stem().u8string();size_t sign=stem.find_first_of("+-");require(sign!=std::string::npos,"Cannot parse evaluation label");
    double command=std::stod(stem.substr(sign));std::string axis=stem.substr(0,1);
    double py=q.axis=="Y"?q.value:0,pt=q.axis=="T"?q.value:0;
    double ey=std::abs(py-(axis=="Y"?command:0)),et=std::abs(pt-(axis=="T"?command:0));
    bool pass=ey<=.09&&et<=.09;passed[p]+=pass;correct[p]+=(q.axis==axis);max_y[p]=std::max(max_y[p],ey);max_t[p]=std::max(max_t[p],et);sum_y[p]+=ey;sum_t[p]+=et;
    double pd=std::abs(q.value-e.value);prediction_max=std::max(prediction_max,pd);
    if(pd>1e-4||fd>1e-4||q.axis!=e.axis||original.axis!=e.axis||std::abs(original.value-e.value)>1e-10)++parity_failures;
    rows<<name<<','<<(p?"frozen_oof4":"full_model_resub")<<','<<e.fold<<','<<q.axis<<','<<q.value<<','<<q.branch<<','<<ey<<','<<et<<','<<pass<<','<<pd<<','<<fd<<','<<valid<<'\n';
   }
   ++total;if(total%20==0)std::cout<<"Processed "<<total<<"/280\n"<<std::flush;
  }
  std::ofstream summary(output/"summary.json");summary<<std::setprecision(17)<<"{\n\"pictures\":"<<total<<",\n\"feature_max_difference\":"<<feature_max<<",\n\"prediction_max_difference\":"<<prediction_max<<",\n\"model_only_max_difference\":"<<model_only_max<<",\n\"parity_tolerance\":0.0001,\n\"parity_failures\":"<<parity_failures<<",\n\"minimum_valid_selected\":"<<valid_min<<",\n\"results\":[\n";
  for(int p=0;p<2;++p){if(p)summary<<",\n";summary<<"{\"protocol\":\""<<(p?"frozen_oof4":"full_model_resub")<<"\",\"passed\":"<<passed[p]<<",\"axis_correct\":"<<correct[p]<<",\"y_mae\":"<<sum_y[p]/total<<",\"y_max\":"<<max_y[p]<<",\"t_mae\":"<<sum_t[p]/total<<",\"t_max\":"<<max_t[p]<<"}";}
  summary<<"\n],\n\"scope\":\"Previously exposed development data; original frozen fold models, no retraining; not independent or physical robot evaluation\"\n}\n";summary.close();
  require(total==280&&passed[0]==280&&passed[1]==280,"Tolerance pass reproduction failed");require(parity_failures==0,"Python/C++ numerical correspondence failed; see results");
  std::cout<<"All280 passed in resub and frozen4-fold. Numeric parity passed.\n";return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
