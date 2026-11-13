#pragma once

#include <opencv2/core.hpp>

namespace qr_pose {

struct Pose {
    double y_mm;
    double theta_deg;
};

Pose estimate(const cv::Mat &rectified_image);

}
