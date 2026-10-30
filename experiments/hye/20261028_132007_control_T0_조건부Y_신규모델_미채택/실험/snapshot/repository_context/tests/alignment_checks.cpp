#include "alignment.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <cmath>
#include <stdexcept>
namespace fs=std::filesystem;
static cv::Mat image(const fs::path& p){std::ifstream f(p,std::ios::binary);std::vector<unsigned char> b{std::istreambuf_iterator<char>(f),std::istreambuf_iterator<char>()};return cv::imdecode(b,cv::IMREAD_GRAYSCALE);}
static void require(bool ok,const char* what){if(!ok)throw std::runtime_error(what);}
int wmain(int argc,wchar_t** argv){try {
 cv::setNumThreads(1);
 require(argc==4,"assets model clean required");auto src=image(fs::path(argv[3])/"center.bmp");
 bool rejected=false;try{joint(src);}catch(const std::logic_error&){rejected=true;}require(rejected,"Uninitialized call must fail");
 initialize_alignment(fs::path(argv[1]).u8string(),fs::path(argv[2]).u8string());
 auto before=src.clone();auto a=joint(src),b=alignment(src);require(std::isfinite(a.y)&&std::isfinite(a.t),"Finite joint required");require(std::abs(a.y-b.y)<1e-12&&std::abs(a.t-b.t)<1e-12,"Compatibility API differs");require(cv::norm(before,src,cv::NORM_INF)==0,"Input modified");
 auto c=control(src);require(std::isfinite(c.y)&&std::isfinite(c.t)&&alignment_diagnostics().control_precondition_met,"Conditional center failed");
 for(auto bad:{cv::Mat(),cv::Mat(300,500,CV_8UC3,cv::Scalar(255,255,255)),cv::Mat(2160,3840,CV_16U,cv::Scalar(0))}){bool threw=false;try{joint(bad);}catch(const std::exception&){threw=true;}require(threw,"Invalid image accepted");require(!alignment_diagnostics().measurement_valid,"Failure reported valid");}
 auto rotated=image(fs::path(argv[3])/"Theta/T+0.50.bmp");bool gated=false;try{control(rotated);}catch(const std::domain_error&){gated=true;}require(gated,"T nonzero must reject conditional Y");
 require(alignment_diagnostics().measurement_valid&&!alignment_diagnostics().control_precondition_met,"Gate diagnostics missing");
 std::cout<<"PASS initialization, compatibility, conditional Y, immutable input, invalid images, T gate\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
