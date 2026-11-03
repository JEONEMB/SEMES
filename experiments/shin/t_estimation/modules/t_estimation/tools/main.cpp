#include "t_estimator.hpp"
#include "observation.hpp"
#include <filesystem>
#include <iostream>
#include <iomanip>
int wmain(int argc,wchar_t** argv){try {
    if(argc!=4)throw std::invalid_argument("Usage: t_estimate IMAGE QR_CALIBRATION_DIRECTORY T_MODEL_JSON");
    cv::setNumThreads(1);t_only::Estimator estimator(std::filesystem::path(argv[2]).u8string(),std::filesystem::path(argv[3]).u8string());
    auto image=qrwithbar::detail::read_gray(argv[1]);auto before=image.clone();auto e=estimator.estimate(image);
    if(cv::norm(image,before,cv::NORM_INF)!=0)throw std::runtime_error("Input image was modified");
    std::cout<<std::setprecision(17)<<"{\"t\":"<<e.t<<",\"ticks\":"<<e.ticks<<",\"tick_size\":0.01,\"requires_y_fixed_at_reference\":"<<(e.requires_y_fixed_at_reference?"true":"false")<<",\"outside_calibration_range\":"<<(e.outside_calibration_range?"true":"false")<<"}\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
