#pragma once
#include "alignment.hpp"
#include <array>
#include <filesystem>
#include <vector>
namespace legacy_control {
namespace fs=std::filesystem;
cv::Mat read_gray(const fs::path& file);
std::string sha256(const fs::path& file);
cv::Mat read_numbers(const cv::FileNode& node,int rows,int columns);
cv::Mat spline_coefficients(const cv::Mat& image);
double spline_sample(const cv::Mat& coefficients,double x,double y);
cv::Mat nearest_prefilter(const cv::Mat& image);
struct PointModel {cv::Point2f point;cv::Mat reference,pinverse;};
struct PanelModel {cv::Mat blurred8;std::vector<cv::Point2f> points;std::vector<int> ids;std::vector<PointModel> models;};
struct Tracks {cv::Mat displacement,quality;std::vector<unsigned char> valid;std::array<double,3> ncc;};
class Tracker {
public:
    explicit Tracker(const fs::path& assets);
    Tracks measure(const cv::Mat& gray)const;
    cv::Mat features(const Tracks& tracks)const;
    int valid_selected(const Tracks& tracks)const;
    double mean_shift(const Tracks& tracks)const;
private:
    cv::Mat reference_,zero_;std::vector<cv::Point2d> points_;std::vector<int> regions_;
    std::vector<unsigned char> selected_;PanelModel panels_[3];
};
// v2: 위·아래 바 블록(joint 와 같은 line_protocol) 정합으로 바 기울기 차 d 를 잰다.
struct LineFit {double shift,correlation;bool valid;};
std::vector<double> line_profile(const cv::Mat& image,const cv::Rect& box);
LineFit match_line(const std::vector<double>& reference,const std::vector<double>& current);
class Bars {
public:
    Bars(const fs::path& assets,const cv::Mat& center);
    // 위 바 기울기 - 아래 바 기울기 (px per 1000 px). 블록 하나라도 정합에 실패하면 예외.
    double slope_difference(const cv::Mat& gray,double* minimum_correlation=nullptr)const;
    const std::vector<double>& centers()const{return x_;}
private:
    std::vector<cv::Rect> boxes_;std::vector<double> x_;std::vector<std::vector<double>> reference_;
};
// v2 모델: y = f0 + f1*s - k*soft(t^, theta), t^ = c0 + c1*s + c2*d. soft(v,theta) 는 |v|<=theta 이면 0.
struct Estimate {double y,t,t_hat;bool outside;};
class ModelV2 {
public:
    explicit ModelV2(const fs::path& file);
    Estimate predict(double s,double d)const;
    std::vector<double> block_x;
private:
    double f0_=0,f1_=0,c0_=0,c1_=0,c2_=0,k_=0,theta_=0,s_min_=0,s_max_=0,d_min_=0,d_max_=0;
};
cv::Mat checked_gray(const cv::Mat& input);
}
