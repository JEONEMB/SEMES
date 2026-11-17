// 공통 hye/ecc_compat.hpp 의 사본: homography 도 정확한 bilinear 역변환 (ecc_compat_h.cpp)
#pragma once
#include <opencv2/video/tracking.hpp>
namespace semes::ecc5h {
double findTransformECCWithMask(cv::InputArray template_image,cv::InputArray image,cv::InputArray template_mask,cv::InputArray input_mask,cv::InputOutputArray warp,int motion,cv::TermCriteria criteria,int gaussian_size);
}
