#include "six.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
namespace fs=std::filesystem;
int wmain(int argc,wchar_t** argv){try{
 if(argc!=2)throw std::runtime_error("Usage: calibration_control V3_DIRECTORY");
 cv::setNumThreads(1);fs::path root=argv[1];auto nominal=six::read_rows(root/"original/original_observations.csv"),target=six::read_rows(root/"fixed_v3/observations.csv");
 std::ofstream f(root/"global_calibration_predictions.csv");f<<std::setprecision(17)<<"path,fold,T_target,uncalibrated,shift_only,affine\n";
 for(int fold=0;fold<5;fold++){
  auto model=six::load_model(root/("original/results/models/fold"+std::to_string(fold)+".json"));std::vector<int> train,test;
  for(int i=0;i<(int)nominal.size();i++)if(nominal[i].cohort!="reference")(nominal[i].fold==fold?test:train).push_back(i);
  cv::Mat a((int)train.size(),2,CV_64F),b((int)train.size(),1,CV_64F);double shift=0;
  for(int j=0;j<(int)train.size();j++){int i=train[j];double p=six::predict(model,nominal[i].x).t;a.at<double>(j,0)=1;a.at<double>(j,1)=p;b.at<double>(j)=target[i].t;shift+=target[i].t-p;}
  shift/=train.size();cv::Mat c;cv::solve(a,b,c,cv::DECOMP_SVD);
  for(int i:test){double p=six::predict(model,nominal[i].x).t;f<<nominal[i].path<<','<<fold<<','<<target[i].t<<','<<p<<','<<p+shift<<','<<c.at<double>(0)+c.at<double>(1)*p<<'\n';}
 }
 return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
