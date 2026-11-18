#include "six.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <set>
#include <cmath>
int wmain(int argc,wchar_t** argv){try{if(argc!=2)throw std::invalid_argument("Usage: six_sessions EXPERIMENT");cv::setNumThreads(1);std::filesystem::path root=argv[1];auto rows=six::read_rows(root/"observations.csv");std::set<int> sessions;for(auto& r:rows)if(r.cohort!="reference")sessions.insert(r.session);std::ofstream out(root/"results/session_holdout.csv");out<<std::setprecision(17)<<"method,path,held_session,T_true,T_pred,error_ticks,known_session\n";
 for(int method:{0,5})for(int held:sessions){std::vector<int> train,test;for(int i=0;i<(int)rows.size();i++)if(rows[i].cohort!="reference")(rows[i].session==held?test:train).push_back(i);six::Config cfg;cfg.method=method;cfg.alpha=.1;cfg.shrink=30;auto model=six::fit(rows,train,cfg);for(int i:test){auto p=six::predict(model,rows[i].x,rows[i].time);out<<method<<','<<rows[i].path<<','<<held<<','<<rows[i].t<<','<<p.t<<','<<(p.t-rows[i].t)/.01<<','<<p.known_session<<'\n';}std::cout<<"Session holdout method="<<method<<" session="<<held<<" n="<<test.size()<<'\n'<<std::flush;}
 return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
