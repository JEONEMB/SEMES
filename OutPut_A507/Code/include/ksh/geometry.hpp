// =====================================================================================
// ksh/geometry.hpp — 서브픽셀 측정 공통 도구 (QR 사각형 변, 나사 테두리)
//  외부 라이브러리 없이 8bit 회색 영상 버퍼만 읽는다 (cv::Mat 은 view() 로 감싸서 넘김)
// =====================================================================================
#ifndef KSH_GEOMETRY_HPP
#define KSH_GEOMETRY_HPP

#include <opencv2/core.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace ksh {
namespace geo {

constexpr double kPi = 3.14159265358979323846;

// 8bit 회색 영상 읽기 전용 보기 (cv::Mat CV_8UC1 을 복사 없이 감쌈)
struct View {
    const uint8_t *data = nullptr;
    int width = 0, height = 0, stride = 0;   // stride : 한 줄 바이트 수
    uint8_t at(int x, int y) const { return data[(size_t)y * stride + x]; }
};
inline View view(const cv::Mat &gray8) { return {gray8.ptr<uint8_t>(0), gray8.cols, gray8.rows, (int)gray8.step[0]}; }

// 최소제곱 (Householder QR). A : rows × cols (행 우선)
bool lstsq(const std::vector<double> &A, int rows, int cols, const std::vector<double> &b, std::vector<double> &x);

// 양선형 보간 (영상 밖은 가장자리 값)
double bilinear(const View &im, double x, double y);

// 직선을 따라 평균낸 단면 (vertical = 세로 모서리 → x 방향 단면)
struct Profile { std::vector<double> o, v; };
Profile line_profile(const View &im, bool vertical, double p0, double slope, double tref, double t0, double t1, double range,
                     double step = 0.25);
// 가우시안으로 다듬은 기울기의 최대점
bool grad_peak(const Profile &P, int sign, double sigma, double lim, double &pos, double &peak);
// 50% 교차 위치 (기울기 최대점 양옆 plateau 평균의 중간값)
bool half_cross(const Profile &P, int sign, double o0, double plateau, double &pos, double &contrast);

struct EdgeResult { bool ok = false; double off = 0, contrast = 0; };
EdgeResult measure_offset(const View &im, bool vertical, double p0, double slope, double tref, double t0, double t1, int sign,
                          double search, double plateau, int method = 0, double sigma = 1.0);
struct EdgeLine { bool ok = false; double pos = 0, slope = 0, rms = 0, contrast = 0; };
// 모서리 직선 : 구간 nseg 개의 평균 단면 위치 → 직선 맞춤. tref 에서의 위치와 기울기
EdgeLine measure_line(const View &im, bool vertical, double p0, double slope, double tref, double t0, double t1, int sign,
                      double search, double plateau, int nseg, bool fit_slope, int method = 0, double sigma = 1.0);

std::vector<double> box_smooth(const std::vector<double> &v, int r);
double percentile(std::vector<double> v, double q);

// 분리형 가우시안 흐림 (가장자리 복제), float 버퍼 w × h
void gauss_blur(std::vector<float> &a, int w, int h, double sigma);
double bil(const std::vector<float> &a, int w, int h, double x, double y);
double median_of(std::vector<double> v);

} // namespace geo
} // namespace ksh

#endif
