#include "observer_internal.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <regex>
#include <cmath>
using namespace alignment_detail;
int wmain(int argc,wchar_t** argv){try{
 cv::setNumThreads(1);
 if(argc!=5)throw std::runtime_error("evaluate_alignment CLEAN_ROOT JOINT_ASSETS CONTROL_MODEL NEW_OUTPUT.csv");fs::path root(argv[1]),output(argv[4]);if(fs::exists(output))throw std::runtime_error("New output required");initialize_alignment(fs::path(argv[2]).u8string(),fs::path(argv[3]).u8string());
 fs::create_directories(output.parent_path());std::ofstream out(output);out<<"dataset,path,Y,T,jointY,jointT,joint_valid,controlY,controlT,control_valid\n"<<std::setprecision(17);
 int count=0,valid=0,passed=0,conditional=0,conditional_total=0,conditional_pass=0;double ey=0,et=0,maxy=0,maxt=0,cy=0,cmax=0;
 std::regex yr("Y([+-][0-9]+(?:\\.[0-9]+)?)"),tr("T([+-][0-9]+(?:\\.[0-9]+)?)");
 for(const auto& date:{std::string("20261014"),std::string("20261020")})for(const auto& e:fs::recursive_directory_iterator(root/("clean_"+date))){if(e.path().extension()!=L".bmp")continue;auto name=e.path().stem().u8string();std::smatch ym,tm;double y=std::regex_search(name,ym,yr)?std::stod(ym[1]):0,t=std::regex_search(name,tm,tr)?std::stod(tm[1]):0;auto src=read_gray(e.path());++count;
  out<<date<<",\""<<fs::relative(e.path(),root).u8string()<<"\","<<y<<','<<t<<',';
  try{auto q=joint(src);double dy=std::abs(q.y-y),dt=std::abs(q.t-t);ey+=dy;et+=dt;maxy=std::max(maxy,dy);maxt=std::max(maxt,dt);++valid;passed+=dy<=.09&&dt<=.09;out<<q.y<<','<<q.t<<",1,";}catch(const std::exception&){out<<",,0,";}
  if(t==0){++conditional_total;try{auto q=control(src);out<<q.y<<','<<q.t<<",1\n";++conditional;double error=std::abs(q.y-y);cy+=error;cmax=std::max(cmax,error);conditional_pass+=error<=.09;}catch(const std::exception&){out<<",,0\n";}}else out<<",,0\n";
  if(count%40==0)std::cout<<"evaluated "<<count<<std::endl;
 }
 std::cout<<std::setprecision(12)<<"joint n="<<count<<" valid="<<valid<<" pass="<<passed<<" Ymean="<<ey/std::max(1,valid)<<" Ymax="<<maxy<<" Tmean="<<et/std::max(1,valid)<<" Tmax="<<maxt<<'\n';
 std::cout<<"conditional T0 n="<<conditional_total<<" accepted="<<conditional<<" Ypass="<<conditional_pass<<" Ymean="<<cy/std::max(1,conditional)<<" Ymax="<<cmax<<" (development only)\n";return valid==count&&conditional==conditional_total?0:1;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
