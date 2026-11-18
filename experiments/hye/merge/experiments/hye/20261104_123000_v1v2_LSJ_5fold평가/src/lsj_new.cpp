#include "theta_pipeline.hpp"
#include <opencv2/imgcodecs.hpp>
#include "utf8_io.hpp"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <limits>
using Clock=std::chrono::steady_clock;
static std::vector<std::string> split(std::string s){if(!s.empty() && s.back()=='\r')s.pop_back();std::vector<std::string> v;std::stringstream ss(s);std::string x;while(std::getline(ss,x,'\t'))v.push_back(x);return v;}
int main(int argc,char**argv){try{if(argc!=4)return 2;cv::setNumThreads(1);theta::Detector detector(theta::loadConfig(argv[1]));std::ifstream in(argv[2]);std::ofstream out(argv[3]);out<<std::setprecision(17)<<"index\tdate\tfold\tY\tT\tpred_Y_nuisance\tpred_T\tcorrection_T\tms\tupper_angle\tlower_angle\terror\n";std::string l;std::getline(in,l);int i=0;while(std::getline(in,l)){auto v=split(l);double y=std::numeric_limits<double>::quiet_NaN(),t=y,c=y,ms=y,u=y,b=y;std::string err;try{std::ifstream f(std::filesystem::u8path(v[1]),std::ios::binary);std::vector<unsigned char> bytes{std::istreambuf_iterator<char>(f),std::istreambuf_iterator<char>()};auto im=cv::imdecode(bytes,cv::IMREAD_GRAYSCALE);auto start=Clock::now();auto r=detector.estimate(im);ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count();y=r.pose.nuisance_y_mm;t=r.pose.theta_deg;c=r.pose.theta_correction_deg;u=r.upper.angle_deg;b=r.lower.angle_deg;}catch(const std::exception&e){err=e.what();}out<<i<<'\t'<<v[0]<<'\t'<<v[4]<<'\t'<<v[2]<<'\t'<<v[3]<<'\t'<<y<<'\t'<<t<<'\t'<<c<<'\t'<<ms<<'\t'<<u<<'\t'<<b<<'\t'<<err<<'\n';out.flush();if(i%25==0)std::cout<<"LSJ newer "<<i+1<<std::endl;++i;}return 0;}catch(const std::exception&e){std::cerr<<e.what();return 1;}}
