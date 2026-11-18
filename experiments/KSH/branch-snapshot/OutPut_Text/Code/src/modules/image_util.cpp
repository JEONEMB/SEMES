// image_util.cpp — 영상 공통 도구 구현
#include "ksh/image_util.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <vector>

namespace ksh {

// 컬러(BGR 3채널 · BGRA 4채널) 또는 회색 영상 → CV_32F 회색
//  정합 계산(위상 상관, IC-GN)이 실수 연산이므로 한 번만 바꿔 둠. 밝기 값 범위는 원본 그대로 (8비트면 0 ~ 255)
cv::Mat to_gray32(const cv::Mat &src) {
    cv::Mat gray, out;
    if (src.channels() == 3) cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    else if (src.channels() == 4) cv::cvtColor(src, gray, cv::COLOR_BGRA2GRAY);
    else gray = src;
    gray.convertTo(out, CV_32F);
    return out;
}

// 컬러·회색 → CV_8U 회색 (BMP 8bit 회색 사진은 복사 없이 그대로 씀)
cv::Mat to_gray8(const cv::Mat &src) {
    cv::Mat gray;
    if (src.channels() == 3) cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    else if (src.channels() == 4) cv::cvtColor(src, gray, cv::COLOR_BGRA2GRAY);
    else gray = src;
    if (gray.depth() == CV_8U) return gray;
    cv::Mat out;
    gray.convertTo(out, CV_8U, gray.depth() == CV_16U ? 1.0 / 256.0 : 1.0);
    return out;
}

// 사각형 전체가 영상 안에 있는지 (한 화소라도 밖이면 false → 잘라 내기 전에 검사해 예외 방지)
bool rect_inside(const cv::Rect &rect, const cv::Size &image_size) {
    return rect.x >= 0 && rect.y >= 0 && rect.x + rect.width <= image_size.width && rect.y + rect.height <= image_size.height;
}

// 소수 좌표 → 가장 가까운 정수 좌표 (영역을 잘라 올 때 정수 시작점이 필요)
cv::Point round_point(const cv::Point2d &p) {
    return cv::Point(int(std::lround(p.x)), int(std::lround(p.y)));
}

// 중심 ± 반폭 사각형 : 가로 = 2 × half_width_px, 세로 = 2 × half_height_px
cv::Rect rect_around(const cv::Point &center_px, int half_width_px, int half_height_px) {
    return cv::Rect(center_px.x - half_width_px, center_px.y - half_height_px, 2 * half_width_px, 2 * half_height_px);
}

// 설정의 ROI(중심, 반폭) → 기준 사진 위 사각형. 세로 반폭이 없으면(0) 정사각형
cv::Rect roi_rect(const RoiSpec &roi) {
    return rect_around(roi.center_px, roi.half_width_px, roi.half_height_px > 0 ? roi.half_height_px : roi.half_width_px);
}

// 사진 읽기 — 한글 경로 대응
//  ① cv::imread : 시스템 코드 페이지 경로(명령 인자 등)는 바로 읽힘
//  ② 실패하면 경로를 UTF-8 로 보고 파일을 바이트 배열로 읽은 뒤 cv::imdecode 로 해석 (설정 파일의 한글 경로 등)
cv::Mat read_image(const std::string &path, int flags) {
    if (path.empty()) return cv::Mat();
    cv::Mat img = cv::imread(path, flags);
    if (!img.empty()) return img;
    // UTF-8 경로 (Windows 의 cv::imread 는 한글 경로를 못 읽음)
    std::ifstream f(std::filesystem::u8path(path), std::ios::binary);
    if (!f) return cv::Mat();
    const std::vector<uchar> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (bytes.empty()) return cv::Mat();
    return cv::imdecode(bytes, flags);
}

} // namespace ksh
