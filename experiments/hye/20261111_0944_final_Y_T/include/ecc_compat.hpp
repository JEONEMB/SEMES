#pragma once
#include <opencv2/video/tracking.hpp>
namespace semes::ecc5 {
double findTransformECCWithMask(cv::InputArray template_image,cv::InputArray image,cv::InputArray template_mask,cv::InputArray input_mask,cv::InputOutputArray warp,int motion,cv::TermCriteria criteria,int gaussian_size);
}
