#pragma once
#include <opencv2/video/tracking.hpp>
namespace semes::ecc5p {
// semes::ecc5::findTransformECCWithMask 와 같고, 투영 경로만 정확한 bilinear 역변환을 쓴다
double findTransformECCWithMask(cv::InputArray template_image, cv::InputArray image, cv::InputArray template_mask, cv::InputArray input_mask,
                                cv::InputOutputArray warp, int motion, cv::TermCriteria criteria, int gaussian_size);
}
