#include <limits>
#include <cmath>
#include "theta_alignment.hpp"
#include "y_alignment.hpp"
#include "sequential_alignment.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <chrono>
using Clock=std::chrono::steady_clock;
static double elapsed(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
static std::vector<std::string> split(std::string s){if(!s.empty() && s.back()=='\r')s.pop_back();std::vector<std::string> v;std::stringstream ss(s);std::string x;while(std::getline(ss,x,'\t'))v.push_back(x);return v;}
int main(int argc,char**argv){try{if(argc!=5)return 2;cv::setNumThreads(1);std::string root=argv[1];auto ref=semes::readGrayUtf8(argv[2]);auto tc=semes::ThetaConfig::load(root+"/configs/theta_alignment.json");auto yc=semes::YConfig::load(root+"/configs/y_alignment.json");semes::ThetaEstimator theta(ref,tc);semes::YEstimator y(ref,yc);semes::SequentialAligner seq(ref,tc,yc);
 std::ifstream in(argv[3]);std::ofstream out(argv[4]);out<<std::setprecision(17)<<"index\tdate\tfold\tY\tT\tpred_T\ttheta_ms\ttheta_command_available\ttheta_command\ty_command\ty_command_available\ty_ms\tsequence_y_command\tsequence_t_command\tsequence_available\tsequence_ms\tstatus\n";std::string line;std::getline(in,line);int i=0;
 while(std::getline(in,line)){auto v=split(line);double pt=std::numeric_limits<double>::quiet_NaN(),tm=std::numeric_limits<double>::quiet_NaN(),ym=std::numeric_limits<double>::quiet_NaN(),sm=std::numeric_limits<double>::quiet_NaN(),ycmd=std::numeric_limits<double>::quiet_NaN(),tcmd=std::numeric_limits<double>::quiet_NaN(),sy=std::numeric_limits<double>::quiet_NaN(),st=std::numeric_limits<double>::quiet_NaN();bool ta=false,ya=false,sa=false;std::string status;
 try{auto image=semes::readGrayUtf8(v[1]);try{auto t=Clock::now();auto r=theta.estimate(image);tm=elapsed(t);pt=r.estimatedThetaDeg;ta=r.commandAvailable;tcmd=r.commandDeg;status=r.status;}catch(const std::exception&e){status=e.what();}try{auto t=Clock::now();auto r=y.estimate(image);ym=elapsed(t);ya=r.commandAvailable;ycmd=r.commandMm;}catch(const std::exception&e){status+=";Y:"+std::string(e.what());}try{auto t=Clock::now();auto r=seq.estimate(image);sm=elapsed(t);sy=r.yMm;st=r.thetaDeg;sa=r.commandAvailable;status+=";seq:"+r.status;}catch(const std::exception&e){status+=";seq:"+std::string(e.what());}}catch(const std::exception&e){status=e.what();}
 for(auto&c:status)if(c=='\t'||c=='\n'||c=='\r')c=' ';out<<i<<'\t'<<v[0]<<'\t'<<v[4]<<'\t'<<v[2]<<'\t'<<v[3]<<'\t'<<pt<<'\t'<<tm<<'\t'<<ta<<'\t'<<tcmd<<'\t'<<ycmd<<'\t'<<ya<<'\t'<<ym<<'\t'<<sy<<'\t'<<st<<'\t'<<sa<<'\t'<<sm<<'\t'<<status<<'\n';out.flush();if(i%25==0)std::cout<<"LSJ "<<i+1<<std::endl;++i;}return 0;}catch(const std::exception&e){std::cerr<<e.what();return 1;}}
