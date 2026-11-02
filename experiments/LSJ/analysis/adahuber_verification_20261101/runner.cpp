#include "RcppArmadillo.h"
#include <fstream>
#include <iostream>
#include <chrono>
namespace audit { nlohmann::json events=nlohmann::json::array(); }
namespace upstream {
#include "sources/adaHuber/src/adaHuber.cpp"
}
namespace observed {
#include "generated/observed.cpp"
}
namespace diagnostic_patch {
#include "generated/diagnostic_patch.cpp"
}
using json=nlohmann::json;
int main(int argc,char**argv){
 if(argc!=3)return 2;
 std::ifstream input(argv[1]); std::ofstream output(argv[2]); std::string line;
 while(std::getline(input,line)){
  json q=json::parse(line), row; row["id"]=q["id"];
  try{
   auto yy=q.at("y").get<std::vector<double>>(); arma::vec Y(yy); const arma::vec originalY(Y);
   std::string type=q.value("type",std::string("adaptive"));
   double eps=q.value("epsilon",0.0001); int cap=q.value("iteMax",500);
   arma::mat X;
   if(q.contains("x")){
    auto xx=q.at("x").get<std::vector<std::vector<double>>>();X.set_size(xx.size(),xx.at(0).size());
    for(size_t i=0;i<xx.size();++i)for(size_t j=0;j<xx[i].size();++j)X(i,j)=xx[i][j];
   }
   auto start=std::chrono::steady_clock::now();
   if(type=="adaptive")row["result"]=upstream::adaHuberReg(X,Y,eps,cap).value;
   else if(type=="standard")row["result"]=upstream::huberReg(X,Y,eps,1.345,cap).value;
   else if(type=="mean")row["result"]=upstream::huberMeanList(Y,eps,cap).value;
   else if(type=="lasso")row["result"]=upstream::adaHuberLassoList(X,Y,q.value("lambda",0.05),0.1,1.2,eps,cap).value;
   else if(type=="cov")row["result"]=upstream::huberCov(X,eps,cap).value;
   else throw std::runtime_error("unknown test type");
   row["runtime_ms"]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
   Y=originalY; audit::events=json::array();
   if(type=="adaptive")row["observed_result"]=observed::adaHuberReg(X,Y,eps,cap).value;
   else if(type=="standard")row["observed_result"]=observed::huberReg(X,Y,eps,1.345,cap).value;
   else if(type=="mean")row["observed_result"]=observed::huberMeanList(Y,eps,cap).value;
   else if(type=="lasso")row["observed_result"]=observed::adaHuberLassoList(X,Y,q.value("lambda",0.05),0.1,1.2,eps,cap).value;
   else if(type=="cov")row["observed_result"]=observed::huberCov(X,eps,cap).value;
   row["events"]=audit::events;
   Y=originalY;
   if(q.value("diagnostic_patch",false)){
    if(type=="mean")row["patch_result"]=diagnostic_patch::huberMeanList(Y,eps,cap).value;
    else if(type=="adaptive")row["patch_result"]=diagnostic_patch::adaHuberReg(X,Y,eps,cap).value;
    else if(type=="lasso")row["patch_result"]=diagnostic_patch::adaHuberLassoList(X,Y,q.value("lambda",0.05),0.1,1.2,eps,cap).value;
   }
  }catch(const std::exception&e){row["exception"]=e.what();}
  output<<row.dump()<<'\n'; output.flush();
 }
}
