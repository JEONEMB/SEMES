// LSJ port of opencv_contrib 3.4 modules/ximgproc FastLineDetector (3-clause BSD, see LICENSE).
// The ximgproc abstract interface/export macros are removed; the detection code is unchanged
// except that segment support pixels are returned. Only do_merge=false keeps the support list aligned.
#pragma once
#include <opencv2/core.hpp>
#include <vector>

namespace lsj_fld {
class FastLineDetectorImpl;
cv::Ptr<FastLineDetectorImpl> createFastLineDetectorImpl(int length_threshold, float distance_threshold,
        double canny_th1, double canny_th2, int canny_aperture_size, bool do_merge);
void detectWithSupport(const cv::Ptr<FastLineDetectorImpl>& fld, const cv::Mat& image,
        std::vector<cv::Vec4f>& lines, std::vector<std::vector<cv::Point2i> >& support);
}
