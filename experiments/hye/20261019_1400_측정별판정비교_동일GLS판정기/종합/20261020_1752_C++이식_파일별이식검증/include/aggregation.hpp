#pragma once
#include "alignment.hpp"
namespace semes {
    // The seven measurement and seven feature variants from joint_yt_improved/engine.py.
    // All observed points retain positive weights. No labels are accepted.
    cv::Mat aggregate_features(const cv::Mat& points, const cv::Mat& displacement, const cv::Mat& quality, const std::string& measurement, const std::string& feature);
}
