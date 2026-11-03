#include "qrwithbar_lanczos.hpp"
#include "observation.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <regex>
#include <map>
#include <set>
#include <chrono>
#include <cmath>
namespace fs=std::filesystem;
struct Row {std::string path,hash,group,error;double y=0,t=0,py=0,pt=0,ms=0;std::array<double,10> features{};bool ok=false,train=false,outside=false;};
static std::string escape(const std::string& s){std::string r;for(char c:s){if(c=='"'||c=='\\')r+='\\';if(c=='\n'){r+="\\n";continue;}if(c=='\r')continue;r+=c;}return r;}
static void json(std::ostream& f,const Row& r){f<<std::setprecision(17)<<"{\"path\":\""<<escape(r.path)<<"\",\"sha256\":\""<<r.hash<<"\",\"group\":\""<<r.group<<"\",\"Y\":"<<r.y<<",\"T\":"<<r.t<<",\"predY\":"<<r.py<<",\"predT\":"<<r.pt<<",\"ms\":"<<r.ms<<",\"ok\":"<<(r.ok?"true":"false")<<",\"training\":"<<(r.train?"true":"false")<<",\"outside\":"<<(r.outside?"true":"false")<<",\"error\":\""<<escape(r.error)<<"\"}";}
struct Stats {int n=0,ok=0,pass=0,tick=0;double sum[2]={},sq[2]={},mx[2]={},bias[2]={};std::string worst[2];void add(const Row&r){n++;if(!r.ok)return;ok++;double e[2]={r.py-r.y,r.pt-r.t};pass+=fabs(e[0])<=.09&&fabs(e[1])<=.09;tick+=fabs(e[0])<=.05&&fabs(e[1])<=.01;for(int k=0;k<2;k++){double a=fabs(e[k]);sum[k]+=a;sq[k]+=e[k]*e[k];bias[k]+=e[k];if(a>mx[k]){mx[k]=a;worst[k]=r.path;}}}};
static void svg(const fs::path& file,const std::vector<Row>& rows,int kind){
 double loX=0,hiX=0,loY=0,hiY=0;int count=0;
 for(auto&r:rows){if(!r.ok)continue;double a=kind==0?r.y:++count,b=kind==0?r.t:fabs(kind==1?r.py-r.y:r.pt-r.t);loX=std::min(loX,a);hiX=std::max(hiX,a);loY=std::min(loY,b);hiY=std::max(hiY,b);if(kind==0){loX=std::min(loX,r.py);hiX=std::max(hiX,r.py);loY=std::min(loY,r.pt);hiY=std::max(hiY,r.pt);}}
 if(hiX==loX)hiX=loX+1;if(hiY==loY)hiY=loY+1;double px=(hiX-loX)*.05,py=(hiY-loY)*.08;loX-=px;hiX+=px;if(kind==0)loY-=py;hiY+=py;
 auto X=[&](double x){return 90+(x-loX)/(hiX-loX)*960;};auto Y=[&](double y){return 540-(y-loY)/(hiY-loY)*460;};
 std::ofstream f(file);f<<"<svg xmlns='http://www.w3.org/2000/svg' width='1100' height='620' viewBox='0 0 1100 620'><rect width='100%' height='100%' fill='white'/><g font-family='Arial' font-size='13' fill='#333'><text x='90' y='30'>"<<(kind==0?"Fixed Lanczos-3: command circle → prediction cross":kind==1?"Absolute Y error (mm), all images":"Absolute yaw error (deg), all images")<<"</text>";
 for(int j=0;j<=5;j++){double x=loX+(hiX-loX)*j/5,y=loY+(hiY-loY)*j/5;f<<"<path d='M"<<X(x)<<",80 V540 M90,"<<Y(y)<<" H1050' stroke='#eee'/><text x='"<<X(x)<<"' y='565'>"<<std::setprecision(4)<<x<<"</text><text x='15' y='"<<Y(y)<<"'>"<<y<<"</text>";}
 f<<"<path d='M90,80 V540 H1050' fill='none' stroke='#999'/><text x='480' y='602'>"<<(kind==0?"Y (mm)":"Image index (path order; not time)")<<"</text>";if(kind==0)f<<"<text x='10' y='55'>Yaw (deg)</text>";
 count=0;for(auto&r:rows){if(!r.ok)continue;double a=kind==0?r.y:++count,b=kind==0?r.t:fabs(kind==1?r.py-r.y:r.pt-r.t);std::string color=r.train?"#777":"#0071e3";f<<"<g><title>"<<escape(r.path)<<"</title>";if(kind==0){f<<"<path d='M"<<X(a)<<','<<Y(b)<<" L"<<X(r.py)<<','<<Y(r.pt)<<"' stroke='"<<color<<"' opacity='.35'/><circle cx='"<<X(a)<<"' cy='"<<Y(b)<<"' r='2.5' fill='white' stroke='"<<color<<"'/><path d='M"<<X(r.py)-2<<','<<Y(r.pt)-2<<" l4,4 m-4,0 l4,-4' stroke='"<<color<<"'/>";}else f<<"<circle cx='"<<X(a)<<"' cy='"<<Y(b)<<"' r='2.4' fill='"<<color<<"'/>";f<<"</g>";}f<<"</g></svg>";
}
int wmain(int argc,wchar_t**argv){try{
 if(argc!=4)throw std::runtime_error("Usage: full_evaluation DATA_DIRECTORY MODULE_DIRECTORY RESULTS_DIRECTORY");
 fs::path root=argv[1],module=argv[2],out=argv[3];fs::create_directories(out);cv::setNumThreads(1);
 std::set<std::string> training;std::ifstream hf(module/"tests/fixtures/image_hash_check.csv");std::string line;std::getline(hf,line);while(std::getline(hf,line)){auto a=line.find(','),b=line.find(',',a+1);training.insert(line.substr(a+1,b-a-1));}
 if(training.size()!=57)throw std::runtime_error("Training hash inventory is not 57");
 std::vector<fs::path> files;for(auto&e:fs::recursive_directory_iterator(root))if(e.is_regular_file()&&(e.path().extension()==L".bmp"||e.path().extension()==L".png"))files.push_back(e.path());std::sort(files.begin(),files.end());
 auto start=std::chrono::steady_clock::now();qrwithbar::Estimator est((module/"assets").u8string());double init=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
 std::regex yr("Y([+-][0-9]+(?:\\.[0-9]+)?)"),tr("T([+-][0-9]+(?:\\.[0-9]+)?)");std::vector<Row> rows;std::ofstream log(out/"predictions.jsonl");std::ofstream featurecsv(out/"features.csv");featurecsv<<std::setprecision(17)<<"path,Y,T";for(int k=0;k<10;k++)featurecsv<<",f"<<k;featurecsv<<"\n";std::ofstream csv(out/"predictions.csv");csv<<"path,sha256,group,training,success,Y_command_mm,yaw_command_deg,predicted_Y_mm,predicted_yaw_deg,Y_error_mm,yaw_error_deg,outside_training_range,inference_ms,error\n"<<std::setprecision(17);
 for(auto&p:files){Row r;r.path=fs::relative(p,root).generic_u8string();std::smatch ym,tm;std::string name=p.filename().u8string();bool hasY=std::regex_search(name,ym,yr),hasT=std::regex_search(name,tm,tr);r.y=hasY?std::stod(ym[1]):0;r.t=hasT?std::stod(tm[1]):0;r.group=hasY&&hasT?"mixed":hasY?"Y-axis":hasT?"yaw-axis":"center";
  try{if(!hasY&&!hasT&&name.find("center")==std::string::npos)throw std::runtime_error("Unrecognized coordinate label");r.hash=qrwithbar::detail::sha256(p);r.train=training.count(r.hash)>0;auto image=qrwithbar::detail::read_gray(p);start=std::chrono::steady_clock::now();auto v=est.estimate(image);r.ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();r.features=v.features;r.py=v.y_mm;r.pt=v.yaw_deg;r.outside=v.outside_training_range;if(!std::isfinite(r.py)||!std::isfinite(r.pt))throw std::runtime_error("Nonfinite prediction");r.ok=true;}catch(const std::exception&e){r.error=e.what();}
  if(r.ok){featurecsv<<r.path<<','<<r.y<<','<<r.t;for(auto value:r.features)featurecsv<<','<<value;featurecsv<<'\n'<<std::flush;}json(log,r);log<<'\n'<<std::flush;csv<<'"'<<r.path<<"\","<<r.hash<<','<<r.group<<','<<r.train<<','<<r.ok<<','<<r.y<<','<<r.t<<',';if(r.ok)csv<<r.py<<','<<r.pt<<','<<r.py-r.y<<','<<r.pt-r.t;else csv<<",,,";csv<<','<<r.outside<<','<<r.ms<<",\""<<escape(r.error)<<"\"\n"<<std::flush;rows.push_back(r);
  if(rows.size()%10==0||rows.size()==files.size())std::cout<<rows.size()<<'/'<<files.size()<<" processed\n"<<std::flush;
 }
 std::map<std::string,Stats> stats;std::set<std::string> unique;for(auto&r:rows){stats["all"].add(r);stats[r.group].add(r);stats[r.train?"training57":"not_in_training"].add(r);unique.insert(r.hash);}
 std::ofstream s(out/"summary.json");s<<std::setprecision(17)<<"{\"model_commit\":\"55c5ebeda5647e3ee864758be41ba456ded0eb32\",\"model_changed\":false,\"labels\":\"filename commands\",\"images\":"<<rows.size()<<",\"unique_hashes\":"<<unique.size()<<",\"initialization_ms\":"<<init<<",\"groups\":{";bool first=true;for(auto&g:stats){if(!first)s<<',';first=false;auto&a=g.second;s<<'"'<<g.first<<"\":{\"count\":"<<a.n<<",\"success\":"<<a.ok<<",\"both009\":"<<a.pass<<",\"both_tick\":"<<a.tick;for(int k=0;k<2;k++)s<<",\""<<(k==0?"Y":"yaw")<<"_mae\":"<<(a.ok?a.sum[k]/a.ok:0)<<",\""<<(k==0?"Y":"yaw")<<"_max\":"<<a.mx[k]<<",\""<<(k==0?"Y":"yaw")<<"_rmse\":"<<(a.ok?sqrt(a.sq[k]/a.ok):0)<<",\""<<(k==0?"Y":"yaw")<<"_bias\":"<<(a.ok?a.bias[k]/a.ok:0)<<",\""<<(k==0?"Y":"yaw")<<"_worst\":\""<<escape(a.worst[k])<<'"';s<<'}';}s<<"}}";
 std::ofstream js(out/"data.js");js<<"const rows=[";for(size_t i=0;i<rows.size();i++){if(i)js<<',';json(js,rows[i]);}js<<"];";
 svg(out/"y_theta_pairs.svg",rows,0);svg(out/"Y_errors.svg",rows,1);svg(out/"yaw_errors.svg",rows,2);std::cout<<"Done. "<<rows.size()<<" images, "<<stats["all"].ok<<" successful, "<<stats["all"].pass<<" both009\n";return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
