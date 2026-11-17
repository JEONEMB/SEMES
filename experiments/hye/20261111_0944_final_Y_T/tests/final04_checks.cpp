#include "alignment.hpp"
#include "final04.hpp"
#include <opencv2/imgproc.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <cmath>
#include <chrono>
#include <limits>
#include <stdexcept>
namespace fs=std::filesystem;
namespace {
void require(bool ok,const char* text){if(!ok)throw std::runtime_error(text);}
template<class F> void rejected(F f){bool threw=false;try{f();}catch(const std::exception&){threw=true;}require(threw,"Expected rejection");}
void run(const fs::path& module) {
    cv::setNumThreads(1);
    auto center=alignment_detail::read_gray(module/"assets/center.bmp");
    rejected([&]{joint(center);});require(!alignment_diagnostics().measurement_valid,"Uninitialized reported valid");
    initialize_joint((module/"assets").u8string());
    auto before=center.clone();auto p=joint(center);
    const auto compat=alignment(center);require(std::abs(compat.y-p.y)<1e-12 && std::abs(compat.t-p.t)<1e-12,"alignment compatibility path differs");
    require(std::isfinite(p.y)&&std::isfinite(p.t),"Nonfinite output");
    require(cv::norm(center,before,cv::NORM_INF)==0,"Input was changed");
    require(alignment_diagnostics().measurement_valid && !alignment_diagnostics().pose_stability_assessed,"Bad success diagnosis");
    for(auto bad:{cv::Mat(),cv::Mat(10,10,CV_8U),cv::Mat(2160,3840,CV_16U),cv::Mat(2160,3840,CV_8UC2),cv::Mat(2160,3840,CV_8U,cv::Scalar(255))}) {
        rejected([&]{joint(bad);});require(!alignment_diagnostics().measurement_valid,"Failure reported valid");
    }
    cv::Mat bgr,bgra;cv::cvtColor(center,bgr,cv::COLOR_GRAY2BGR);cv::cvtColor(center,bgra,cv::COLOR_GRAY2BGRA);
    for(auto color:{bgr,bgra}){auto q=joint(color);require(std::abs(q.y-p.y)<1e-12 && std::abs(q.t-p.t)<1e-12,"Color conversion changed estimate");}
    rejected([&]{initialize_joint((module/"missing").u8string());});
    auto q=joint(center);require(std::abs(q.y-p.y)<1e-12 && std::abs(q.t-p.t)<1e-12,"Failed reload lost previous model");
    // Synthetic geometric oracle: absolute bar means must not depend on pins.
    cv::Mat qr=cv::Mat::zeros(1,102,CV_64F),pins(1,8,CV_64F,cv::Scalar(17));
    std::vector<double> positions;
    for(int bar=0;bar<2;++bar)for(int i=0;i<24;++i){positions.push_back(1000+10*i);qr.at<double>(54+bar*24+i)=2+bar+3*(i-11.5)*.01;}
    for(int i=0;i<54;++i)qr.at<double>(i)=i*.25;
    auto f=final04::features(qr,pins,positions);
    require(cv::norm(qr.colRange(0,54),f.colRange(0,54),cv::NORM_INF)==0,"Lost QR shape features");
    for(int bar=0;bar<2;++bar) require(std::abs(f.at<double>(54+2*bar)-(2+bar))<1e-12 && std::abs(f.at<double>(55+2*bar)-3)<1e-12,"Wrong bar feature convention");
    require(cv::norm(pins,f.colRange(58,66),cv::NORM_INF)==0,"Wrong pin order");
    final04::LinearModel model;model.load((module/"assets/huber_linear66.yml").u8string());
    rejected([&]{model.predict(cv::Mat::zeros(1,18,CV_64F));});
    f.at<double>(0)=std::numeric_limits<double>::quiet_NaN();rejected([&]{model.predict(f);});
    // Test strict loader shape/scale checks using temporary, deliberately bad models.
    const auto tmp=fs::temp_directory_path()/("semes_final04_invalid_"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".yml");
    for(int bad=0;bad<3;++bad){cv::FileStorage o(tmp.u8string(),cv::FileStorage::WRITE);o<<"degree"<<(bad==0?2:1)<<"mean"<<cv::Mat::zeros(1,66,CV_64F)<<"scale"<<(bad==1?cv::Mat::zeros(1,66,CV_64F):cv::Mat::ones(1,66,CV_64F))<<"weights"<<cv::Mat::zeros(bad==2?66:67,2,CV_64F);o.release();rejected([&]{model.load(tmp.u8string());});}
    fs::remove(tmp);
    std::cout<<"PASS initialization, input immutability, 1/3/4 channels, failed reload, invalid images, blank-image rejection, feature oracle, model schema, nonfinite input\n";
}
}
#ifdef _WIN32
int wmain(int argc,wchar_t** argv) {
#else
int main(int argc,char** argv) {
#endif
    try{require(argc==2,"Module path required");run(fs::path(argv[1]));return 0;}
    catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
