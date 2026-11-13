#pragma once
#include "qr_common.h"
#include <opencv2/opencv.hpp>
#include <array>
#include <filesystem>
#include <string>
#include <vector>
namespace qrwithbar { namespace detail {
namespace fs=std::filesystem;
using Features=std::array<double,10>;
using Warps=std::array<cv::Matx23d,3>;
struct Observation {
    std::array<qc_point,630> points;
    std::array<qc_edge_sample,120> edges;
    Features features{};
    int failures=0;
    double ms=0;
};
cv::Mat read_gray(const fs::path& path);
std::string sha256(const fs::path& path);
Warps align_qrs(const cv::Mat& image,const fs::path& calibration);
Observation analytic_observe(const cv::Mat& image,const Warps& warps);
Observation interpolated_observe(const cv::Mat& image,const cv::Mat& reference,
    const Warps& warps,const Warps& reference_warps,const Observation& initial,
    const Observation& reference_observation,int method);
double sample_lanczos_gray(const cv::Mat&,double x,double y,double& gx,double& gy);
void check_lanczos_kernel();
} }
