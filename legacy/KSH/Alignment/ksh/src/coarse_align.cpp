// coarse_align.cpp — ① Coarse Alignment 구현
#include "ksh/coarse_align.hpp"

#include "ksh/image_util.hpp"

#include <opencv2/imgproc.hpp>

namespace ksh {

cv::Point2d phase_shift(const cv::Mat &base32, const cv::Mat &moved32) {
    cv::Mat win;
    cv::createHanningWindow(win, base32.size(), CV_32F);
    // cv::phaseCorrelate 는 창 함수를 입력에 제자리로 곱하므로 복사본을 넘김
    cv::Mat a = base32.clone(), b = moved32.clone();
    return cv::phaseCorrelate(a, b, win);
}

bool coarse_alignment(const cv::Mat &ref32, const cv::Mat &cur32, const CoarseParams &params, cv::Point2d &shift_px) {
    const cv::Rect window(params.center_px.x - params.window_px.width / 2, params.center_px.y - params.window_px.height / 2,
                          params.window_px.width, params.window_px.height);
    if (window.area() == 0 || !rect_inside(window, ref32.size()) || !rect_inside(window, cur32.size())) return false;
    shift_px = phase_shift(ref32(window), cur32(window));
    return true;
}

} // namespace ksh
