#include "alignment.hpp"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <chrono>
#include <iostream>
#include <iomanip>
namespace fs=std::filesystem;
using Clock=std::chrono::steady_clock;
static double ms(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
int main(int argc,char**argv){try{if(argc!=6)return 2;cv::setNumThreads(1);auto start=Clock::now();initialize_joint(argv[1]);double ji=ms(start);start=Clock::now();initialize_control(argv[2]);double ci=ms(start);std::ifstream in(argv[3]);std::ofstream out(argv[4]);out<<std::setprecision(17)<<"index\tstage\tdecode_ms\tinfer_ms\tpred_Y\tpred_T\terror\n";std::string l;std::getline(in,l);int i=0;bool warmed=false;while(std::getline(in,l)){if(i!=0&&i!=70&&i!=140&&i!=210&&i!=280&&i!=294&&i!=308&&i!=322&&i!=337&&i!=395&&i!=453&&i!=511){++i;continue;}std::stringstream ss(l);std::string date,path;std::getline(ss,date,'\t');std::getline(ss,path,'\t');start=Clock::now();std::ifstream file(fs::u8path(path),std::ios::binary);std::vector<unsigned char> bytes{std::istreambuf_iterator<char>(file),std::istreambuf_iterator<char>()};auto image=cv::imdecode(bytes,cv::IMREAD_GRAYSCALE);double dm=ms(start);if(!warmed){joint(image);control(image);warmed=true;}for(auto stage:{"joint","control"}){try{start=Clock::now();auto q=std::string(stage)=="joint"?joint(image):control(image);double tm=ms(start);out<<i<<'\t'<<stage<<'\t'<<dm<<'\t'<<tm<<'\t'<<q.y<<'\t'<<q.t<<"\t\n";}catch(const std::exception&e){out<<i<<'\t'<<stage<<'\t'<<dm<<"\tnan\tnan\tnan\t"<<e.what()<<'\n';}}out.flush();++i;}std::ofstream info(argv[5]);info<<"joint_initialization_ms\t"<<ji<<"\ncontrol_initialization_ms\t"<<ci<<"\n";return 0;}catch(const std::exception&e){std::cerr<<e.what();return 1;}}
