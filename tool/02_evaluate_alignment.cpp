#include "alignment.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <regex>
#include <cmath>
namespace fs=std::filesystem;
static cv::Mat read_image(const fs::path& p){std::ifstream f(p,std::ios::binary);std::vector<unsigned char> b{std::istreambuf_iterator<char>(f),std::istreambuf_iterator<char>()};return cv::imdecode(b,cv::IMREAD_GRAYSCALE);}
int wmain(int argc,wchar_t** argv){try{
 cv::setNumThreads(1);if(argc!=4)throw std::runtime_error("evaluate_alignment CLEAN_ROOT JOINT_ASSETS NEW_OUTPUT.csv");fs::path root(argv[1]),output(argv[3]);if(fs::exists(output))throw std::runtime_error("New output required");initialize_joint(fs::path(argv[2]).u8string());
 fs::create_directories(output.parent_path());std::ofstream out(output);out<<"dataset,path,Y,T,jointY,jointT,valid,pass\n"<<std::setprecision(17);
 int count=0,valid=0,passed=0;double ey=0,et=0,maxy=0,maxt=0;
 std::regex yr("Y([+-][0-9]+(?:\\.[0-9]+)?)"),tr("T([+-][0-9]+(?:\\.[0-9]+)?)");
 for(const auto& date:{std::string("20261014"),std::string("20261020")})for(const auto& e:fs::recursive_directory_iterator(root/("clean_"+date))){if(e.path().extension()!=L".bmp")continue;auto name=e.path().stem().u8string();std::smatch ym,tm;double y=std::regex_search(name,ym,yr)?std::stod(ym[1]):0,t=std::regex_search(name,tm,tr)?std::stod(tm[1]):0;auto src=read_image(e.path());++count;
  out<<date<<",\""<<fs::relative(e.path(),root).u8string()<<"\","<<y<<','<<t<<',';
  try{auto q=joint(src);double dy=std::abs(q.y-y),dt=std::abs(q.t-t);ey+=dy;et+=dt;maxy=std::max(maxy,dy);maxt=std::max(maxt,dt);++valid;bool pass=dy<=.09&&dt<=.09;passed+=pass;out<<q.y<<','<<q.t<<",1,"<<pass<<'\n';}catch(const std::exception&){out<<",,0,0\n";}
  if(count%40==0)std::cout<<"evaluated "<<count<<std::endl;
 }
 std::cout<<std::setprecision(12)<<"joint n="<<count<<" valid="<<valid<<" pass="<<passed<<" Ymean="<<ey/std::max(1,valid)<<" Ymax="<<maxy<<" Tmean="<<et/std::max(1,valid)<<" Tmax="<<maxt<<" (full-model reapplication; not OOF)\n";
 return valid==count?0:1;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
