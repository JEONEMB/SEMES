#include "six.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
int wmain(int argc,wchar_t** argv){try{if(argc!=2)throw std::invalid_argument("Usage: six_diagnose EXPERIMENT");cv::setNumThreads(1);std::filesystem::path root=argv[1];auto rows=six::read_rows(root/"observations.csv");std::ofstream f(root/"results/direct_initialization.csv"),smoke(root/"results/cache_model_smoke.csv");f<<std::setprecision(17)<<"path,fold,T_true,T_initial,error_ticks\n";smoke<<std::setprecision(17)<<"method,path,T_from_cache\n";for(int fold=0;fold<5;fold++){auto model=six::load_model(root/("results/models/method4_fold"+std::to_string(fold)+".json"));for(auto& r:rows)if(r.cohort!="reference"&&r.fold==fold){auto p=six::predict(model,r.x);f<<r.path<<','<<fold<<','<<r.t<<','<<p.t<<','<<(p.t-r.t)/.01<<'\n';}}
 for(int method=0;method<=8;method++){auto model=six::load_model(root/("results/models/deployment_method"+std::to_string(method)+".json"));for(auto& r:rows)if(r.path=="Y+1.00 T+0.10/Y+1.00 T+0.10 (2).bmp"||r.path=="Theta/T+0.50.bmp")smoke<<method<<','<<r.path<<','<<six::predict(model,r.x).t<<'\n';}
 std::cout<<"All models loaded; direct initialization and cached inference checks saved\n";return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
