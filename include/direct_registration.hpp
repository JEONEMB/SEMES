#ifndef DIRECT_REGISTRATION_HPP
#define DIRECT_REGISTRATION_HPP
#include "alignment.hpp"

struct DirectRegistrationResult
{
    bool valid = false;
    double correlation = std::numeric_limits<double>::quiet_NaN();
    double initialResidual = std::numeric_limits<double>::quiet_NaN();
    cv::Mat referenceToCurrent;
    std::string status;
};

// Original-pixel reference -> current warp. K and distortion are not required.
// This measures a warp, not physical Y/theta. Low residual is not an accuracy certificate.
DirectRegistrationResult registerQrPixels(const cv::Mat &reference, const cv::Mat &current,
                                         const PlateDetectionResult &referenceDetection,
                                         const PlateDetectionResult &currentDetection,
                                         int motion = cv::MOTION_HOMOGRAPHY);
#endif
