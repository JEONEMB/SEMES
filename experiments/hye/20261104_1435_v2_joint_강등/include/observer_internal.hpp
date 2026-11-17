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
cv::Vec2d v2_features(const cv::Mat& observation,const std::vector<double>& block_x);

struct LineFit { double shift,correlation; bool valid; };
LineFit match_line(const std::vector<double>& reference,const std::vector<double>& current);
std::vector<double> line_profile(const cv::Mat& image,const cv::Rect& box);

struct Observation {cv::Mat values; double qr_correlation,line_correlation;};
class Observer {
public:
    explicit Observer(const fs::path& assets);
    Observation measure(const cv::Mat& src) const;
private:
    cv::Mat center_,templates_[3],masks_[3];
    std::vector<cv::Point2d> reference_points_[3];
    std::vector<cv::Rect> line_boxes_;
    std::vector<std::vector<double>> reference_profiles_;
public:
    // 바 블록 48개의 중심 x (line_protocol.json 의 read_box 순서)
    std::vector<double> block_centers() const;
};
// v2 모델: (y, t) = [1, s, d] * W. W 는 3x2. 학습 범위 밖 진단용으로 학습 자료의 s, d 범위를 둔다.
struct JointModel {
    cv::Mat W;
    std::vector<double> block_x;
    double s_min=0,s_max=0,d_min=0,d_max=0;
    void load(const fs::path& file);
    Alignment_t predict(const cv::Mat& observation,bool* outside=nullptr) const;
};
Observation observe(const cv::Mat& src);
Alignment_t predict_joint(const Observation& observation);

void reset_diagnostics(const std::string& stage);
void set_measurement(const Observation& observation);
void set_diagnostics(bool outside,bool precondition,const std::string& message);
}
