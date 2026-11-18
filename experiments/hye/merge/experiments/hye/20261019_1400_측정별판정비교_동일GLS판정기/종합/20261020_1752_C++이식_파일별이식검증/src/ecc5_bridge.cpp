#include "ecc_compat.hpp"
#include <opencv2/video/tracking.hpp>
namespace semes::ecc5 {
double findTransformECCWithMask(cv::InputArray reference,cv::InputArray current,cv::InputArray reference_mask,cv::InputArray current_mask,cv::InputOutputArray warp,int motion,cv::TermCriteria criteria,int gaussian){return cv::findTransformECCWithMask(reference,current,reference_mask,current_mask,warp,motion,criteria,gaussian);}
}
