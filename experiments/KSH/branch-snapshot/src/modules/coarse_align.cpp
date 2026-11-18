// coarse_align.cpp — ① Coarse Alignment 구현
#include "ksh/coarse_align.hpp"

#include "ksh/image_util.hpp"

#include <opencv2/imgproc.hpp>

#include <limits>

namespace ksh {

// 위상 상관 (Kuglin & Hines 1975)
//  A = FFT(w·base), B = FFT(w·moved)        w = 해닝 창 (가장자리 불연속이 만드는 가짜 봉우리 억제)
//  R = A*·B / |A*·B|                         크기를 버리고 위상만 남김 → 밝기·대비 변화에 강함
//  r = IFFT(R) 의 봉우리 위치 = moved 가 base 대비 옮겨진 양 (OpenCV 가 봉우리 주변 가중 중심으로 소수 위치까지 계산)
//  response = 봉우리 세기 : 이동이 뚜렷할수록 큼 (가림·흐림·범위 밖이면 작아짐)
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

// ① Coarse Alignment : 플레이트 주변 창(설정 coarse.*)을 기준·현재 사진에서 같은 위치로 잘라 위상 상관
//  결과는 화소 수준 정확도의 큰 이동 (Θ 1° ≈ 80 px) → ② Fine Alignment 의 출발점
bool coarse_alignment(const cv::Mat &ref32, const cv::Mat &cur32, const CoarseParams &params, cv::Point2d &shift_px,
                      double *response) {
    if (response) *response = std::numeric_limits<double>::quiet_NaN();   // 창이 사진 밖이면 NaN 그대로
    // 창 : 중심 ± 크기/2 (기준 사진 좌표)
    const cv::Rect window(params.center_px.x - params.window_px.width / 2, params.center_px.y - params.window_px.height / 2,
                          params.window_px.width, params.window_px.height);
    if (window.area() == 0 || !rect_inside(window, ref32.size()) || !rect_inside(window, cur32.size())) return false;
    double r = 0.0;
    shift_px = phase_shift(ref32(window), cur32(window), &r);
    if (response) *response = r;
    return r >= params.min_response;   // 봉우리가 약함 = 가림·흔들림·범위 밖 → 이동을 믿을 수 없음
}

} // namespace ksh
