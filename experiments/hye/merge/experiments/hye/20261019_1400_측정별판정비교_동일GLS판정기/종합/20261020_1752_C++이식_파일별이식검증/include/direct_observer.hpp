#pragma once
#include "numeric.hpp"
#include <array>
namespace semes {
class DirectObserver {
    cv::Mat center_,points_;std::vector<int> ids_,regions_;
    std::array<cv::Mat,3> templates_,masks_;
public:
    DirectObserver(const fs::path& fixed_center,const fs::path& geometry_reference);
    Json extract(const std::vector<unsigned char>& current,bool all_candidates=true)const;
    static cv::Point2d project(const cv::Mat& homography,cv::Point2d point);
    static cv::Mat qr_grid(const Json& candidate);
};
}
