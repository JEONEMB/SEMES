// coarse_align.cpp — ① Coarse Alignment 구현
#include "ksh/coarse_align.hpp"

#include "ksh/image_util.hpp"

#include <opencv2/imgproc.hpp>

#include <limits>

namespace ksh {

cv::Point2d phase_shift(const cv::Mat &base32, const cv::Mat &moved32, double *response) {
    cv::Mat win;
    cv::createHanningWindow(win, base32.size(), CV_32F);
    // cv::phaseCorrelate 는 창 함수를 입력에 제자리로 곱하므로 복사본을 넘김
    cv::Mat a = base32.clone(), b = moved32.clone();
    double r = 0.0;
    const cv::Point2d shift = cv::phaseCorrelate(a, b, win, &r);
    if (response) *response = r;
    return shift;
}

bool coarse_alignment(const cv::Mat &ref32, const cv::Mat &cur32, const CoarseParams &params, cv::Point2d &shift_px,
                      double *response) {
    if (response) *response = std::numeric_limits<double>::quiet_NaN();   // 창이 사진 밖이면 NaN 그대로
    const cv::Rect window(params.center_px.x - params.window_px.width / 2, params.center_px.y - params.window_px.height / 2,
                          params.window_px.width, params.window_px.height);
    if (window.area() == 0 || !rect_inside(window, ref32.size()) || !rect_inside(window, cur32.size())) return false;
    double r = 0.0;
    shift_px = phase_shift(ref32(window), cur32(window), &r);
    if (response) *response = r;
    return r >= params.min_response;   // 봉우리가 약함 = 가림·흔들림·범위 밖 → 이동을 믿을 수 없음
}

} // namespace ksh
