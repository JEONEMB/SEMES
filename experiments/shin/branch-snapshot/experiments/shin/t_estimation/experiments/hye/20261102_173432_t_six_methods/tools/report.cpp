#include "six.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <map>
#include <set>
#include <algorithm>
#include <cmath>
namespace fs=std::filesystem;
namespace {
std::vector<std::string> split(const std::string& line){std::istringstream s(line);std::vector<std::string> v;std::string x;while(std::getline(s,x,','))v.push_back(x);return v;}
struct Stats{int n=0,valid=0,pass=0;double sae=0,sse=0,maximum=0;void add(double e,bool ok){n++;if(!ok)return;valid++;sae+=std::abs(e);sse+=e*e;maximum=std::max(maximum,std::abs(e));pass+=std::abs(e)<=1;}double mae()const{return sae/std::max(1,valid);}double rmse()const{return sqrt(sse/std::max(1,valid));}};
struct Failure{std::string line;double error;};
}
int wmain(int argc,wchar_t** argv){try{
 if(argc!=2)throw std::invalid_argument("Usage: six_report EXPERIMENT");fs::path root=argv[1];auto source=six::read_rows(root/"observations.csv");std::map<std::string,six::Row> expected;for(auto& r:source)if(r.cohort!="reference")expected[r.path]=r;if(expected.size()!=571)throw std::runtime_error("Wrong source coverage");const char* names[]={"기준 T 회귀","잡음 공분산 결합","yaw 투영 기하","국소 RBF 회귀","공동 밝기 정합","촬영 구간 절편 보정","최악 그룹 학습","영상 전용 결합","메타데이터 포함 결합"};
 std::ofstream csv(root/"results/summary.csv"),cohort(root/"results/cohort_summary.csv"),md(root/"RESULTS.md"),audit(root/"results/validation_audit.txt");csv<<std::setprecision(17)<<"method,attempted,valid,MAE_deg,RMSE_deg,max_abs_deg,MAE_ticks,RMSE_ticks,max_ticks,pass_1tick\n";cohort<<std::setprecision(17)<<"method,cohort,attempted,valid,MAE_ticks,RMSE_ticks,max_ticks,pass_1tick\n";md<<u8"# 동일 분할 개발 평가 결과\n\n571장은 이미 개발에 노출된 자료다. Y가 미지로 변하는 조건에서 T만 평가했다. 1 tick=.01 degree.\n\n| 번호 | 방법 | MAE(tick) | RMSE(tick) | 최대(tick) | 1 tick 이내 | 추론 실패 |\n| --- | --- | ---: | ---: | ---: | ---: | ---: |\n"<<std::fixed<<std::setprecision(6);
 Stats chosen,baseline;
 for(int method=0;method<=8;method++){
  std::ifstream in(root/("results/predictions_method"+std::to_string(method)+".csv"));if(!in)throw std::runtime_error("Missing predictions");std::string line;std::getline(in,line);std::string header=line;Stats total;std::map<std::string,Stats> groups;std::set<std::string> seen;std::vector<Failure> failures;
  while(std::getline(in,line)){auto v=split(line);auto it=expected.find(v.at(0));if(it==expected.end()||!seen.insert(v[0]).second)throw std::runtime_error("Unexpected or duplicated prediction path");auto& r=it->second;if(std::stoi(v.at(2))!=r.fold||std::abs(std::stod(v.at(4))-r.t)>1e-12)throw std::runtime_error("Fold or label differs across methods");bool ok=std::stoi(v.at(8))==0;double error=ok?std::stod(v.at(7)):0;if(ok&&!std::isfinite(error))throw std::runtime_error("Nonfinite successful prediction");if(ok&&std::abs((std::stod(v.at(5))-r.t)/.01-error)>1e-9)throw std::runtime_error("Stored error differs from prediction minus label");total.add(error,ok);groups[v[1]].add(error,ok);if(!ok||std::abs(error)>1)failures.push_back({line,ok?std::abs(error):1e300});}
  if(seen.size()!=571)throw std::runtime_error("Prediction coverage incomplete");csv<<method<<','<<total.n<<','<<total.valid<<','<<total.mae()*.01<<','<<total.rmse()*.01<<','<<total.maximum*.01<<','<<total.mae()<<','<<total.rmse()<<','<<total.maximum<<','<<total.pass<<'\n';md<<'|'<<method<<'|'<<names[method]<<'|'<<total.mae()<<'|'<<total.rmse()<<'|'<<total.maximum<<'|'<<total.pass<<"/571|"<<total.n-total.valid<<"|\n";
  for(auto& g:groups){auto& s=g.second;cohort<<method<<','<<g.first<<','<<s.n<<','<<s.valid<<','<<s.mae()<<','<<s.rmse()<<','<<s.maximum<<','<<s.pass<<'\n';}
  std::sort(failures.begin(),failures.end(),[](const Failure& a,const Failure& b){return a.error>b.error;});std::ofstream f(root/("results/failures_method"+std::to_string(method)+".csv"));f<<header<<'\n';for(auto& e:failures)f<<e.line<<'\n';audit<<"method="<<method<<" unique_paths="<<seen.size()<<" matched_folds_and_labels=571 failures="<<total.n-total.valid<<'\n';if(method==0)baseline=total;if(method==7)chosen=total;
 }
 md<<u8"\n선택한 method7은 촬영 시각을 사용하지 않는다. 최대 1 tick 목표는 미달성이다.\n";md<<u8"\n기준 대비 최대오차 감소="<<(1-chosen.maximum/baseline.maximum)*100<<u8"%, MAE 감소="<<(1-chosen.mae()/baseline.mae())*100<<u8"%.\n";
 std::ifstream session(root/"results/session_holdout.csv");std::string line;std::getline(session,line);std::map<int,Stats> temporal;std::map<int,int> known;while(std::getline(session,line)){auto v=split(line);int method=std::stoi(v[0]);temporal[method].add(std::stod(v[5]),true);known[method]+=std::stoi(v[6]);}std::ofstream times(root/"results/session_summary.csv");times<<std::setprecision(17)<<"method,attempted,MAE_ticks,RMSE_ticks,max_ticks,pass_1tick,known_session\n";for(auto& pair:temporal){auto& s=pair.second;times<<pair.first<<','<<s.n<<','<<s.mae()<<','<<s.rmse()<<','<<s.maximum<<','<<s.pass<<','<<known[pair.first]<<'\n';}
 std::cout<<"Validated9 methods x571 paths, labels and folds; report regenerated\n";return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
