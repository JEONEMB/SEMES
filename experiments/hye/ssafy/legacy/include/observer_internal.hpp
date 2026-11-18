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


struct LineFit { double shift,correlation; bool valid; };
LineFit match_line(const std::vector<double>& reference,const std::vector<double>& current);
std::vector<double> line_profile(const cv::Mat& image,const cv::Rect& box);

struct Observation {cv::Mat values; double qr_correlation,line_correlation; std::vector<cv::Mat> qr_transforms;};
class Observer {
public:
    explicit Observer(const fs::path& assets);
    Observation measure(const cv::Mat& src) const;
    // 바 48칸의 세로 이동만 잰다(measure 의 마지막 단계와 같은 계산). QR NCC, ECC 호모그래피를 건너뛰어 빠르다.
    // values: 1×48 CV_64F (read_box 순서), line_correlation: 48칸 상관의 최솟값. 칸 하나라도 맞지 않으면 예외.
    void measure_lines(const cv::Mat& gray, cv::Mat& values, double& line_correlation) const;
private:
    cv::Mat center_,templates_[3],masks_[3];
    std::vector<cv::Point2d> reference_points_[3];
    std::vector<cv::Rect> line_boxes_;
    std::vector<std::vector<double>> reference_profiles_;
public:
    // 바 블록 48개의 중심 x (line_protocol.json 의 read_box 순서)
    std::vector<double> block_centers() const;
};
}
