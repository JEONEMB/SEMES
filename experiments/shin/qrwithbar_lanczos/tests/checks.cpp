#include "qrwithbar_lanczos.hpp"
#include "observation.hpp"
#include <iostream>
#include <stdexcept>
template<class F> static void rejected(F action,const char* message){try{action();}catch(const std::exception&){return;}throw std::runtime_error(message);}
int main(){try{
 cv::setNumThreads(1);
 qrwithbar::detail::check_lanczos_kernel();
 rejected([]{qrwithbar::Estimator invalid("missing-calibration-directory");},"Missing assets accepted");
 qrwithbar::Estimator estimator(QC_ASSETS);
 rejected([&]{estimator.estimate(cv::Mat());},"Empty input accepted");
 rejected([&]{estimator.estimate(cv::Mat(12,12,CV_8U));},"Wrong dimensions accepted");
 rejected([&]{estimator.estimate(cv::Mat(2160,3840,CV_16U));},"Wrong depth accepted");
 rejected([&]{estimator.estimate(cv::Mat(2160,3840,CV_8UC2));},"Wrong channels accepted");
 cv::Mat blank(2160,3840,CV_8U,cv::Scalar(0));auto before=blank.clone();
 rejected([&]{estimator.estimate(blank);},"Blank image accepted as a pose");
 if(cv::norm(blank,before,cv::NORM_INF)!=0)throw std::runtime_error("Input changed");
 std::cout<<"PASS: Lanczos value/gradient, calibration/input/failure contracts, unchanged source\n";return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
