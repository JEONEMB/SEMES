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
private:
    cv::Mat reference_,zero_;std::vector<cv::Point2d> points_;std::vector<int> regions_;
    std::vector<unsigned char> selected_;PanelModel panels_[3];
};
struct Choice {std::string axis,branch;double value,cost;};
class Model {
public:
    explicit Model(const fs::path& file);
    Choice predict(const cv::Mat& feature,std::vector<Choice>* alternatives=nullptr)const;
private:
    struct Branch {std::string axis,name;cv::Mat origin,slope,curve;};
    cv::Mat whitening_,zero_;std::vector<Branch> branches_;
};
cv::Mat checked_gray(const cv::Mat& input);
}
