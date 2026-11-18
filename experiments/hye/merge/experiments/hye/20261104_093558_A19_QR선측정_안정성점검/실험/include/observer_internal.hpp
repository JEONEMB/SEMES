#pragma once
#include "alignment.hpp"
#include <filesystem>
#include <vector>
#include <memory>
#include <functional>
namespace alignment_detail {
namespace fs=std::filesystem;
cv::Mat read_gray(const fs::path& path);
std::string sha256(const fs::path& path);
cv::Mat node_matrix(const cv::FileNode& node);
cv::Mat feature_map(const cv::Mat& observation);

struct LineFit { double shift,correlation; bool valid; };
LineFit match_line(const std::vector<double>& reference,const std::vector<double>& current);
std::vector<double> line_profile(const cv::Mat& image,const cv::Rect& box);

struct Observation {cv::Mat values; double qr_correlation,line_correlation;};
struct DetailedObservation {cv::Mat values; int pad=0; bool qr_ok=false; double ncc[3]={0,0,0},ecc[3]={0,0,0},refine_px[3]={0,0,0}; cv::Point int_shift[3]; double line_corr[48]={0}; bool line_valid[48]={false}; int lines_ok=0;};
class Observer {
public:
    explicit Observer(const fs::path& assets);
    Observation measure(const cv::Mat& src) const;
    DetailedObservation measure_detailed(const cv::Mat& gray) const;
private:
    cv::Mat center_,templates_[3],masks_[3];
    std::vector<cv::Point2d> reference_points_[3];
    std::vector<cv::Rect> line_boxes_;
    std::vector<std::vector<double>> reference_profiles_;
};
struct JointModel {
    cv::Mat origin,projector,initial;
    cv::Vec2d scale,lower,upper;
    void load(const fs::path& file);
    Alignment_t predict(const cv::Mat& observation) const;
};
Observation observe(const cv::Mat& src);
Alignment_t predict_joint(const Observation& observation);

void reset_diagnostics(const std::string& stage);
void set_measurement(const Observation& observation);
void set_diagnostics(bool outside,bool precondition,const std::string& message);
}
