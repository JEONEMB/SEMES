// =====================================================================================
// ksh/interp_kernel.hpp — 화소 사이 밝기를 채우는 보간 커널 (실험계획 2 의 비교 대상)
//
//  linear    : 선형 (2탭)                         keys05 / keys075 : Bicubic 3차 회선 a = −0.5 / −0.75 (4탭)
//  lanczos3  : sinc × sinc(·/3) (6탭)            bspline3 / omoms3 : 3차 B-스플라인 / O-MOMS (4탭, 계수 사전 필터 필요)
//  채택 = omoms3 (B-스플라인과 실제 정확도는 동률, 합성 치우침이 가장 작음)
// =====================================================================================
#ifndef KSH_INTERP_KERNEL_HPP
#define KSH_INTERP_KERNEL_HPP

#include <opencv2/core.hpp>

#include <string>

namespace ksh {

// enum 클래스로 커널 이름을 정의
enum class Kernel { linear, keys05, keys075, lanczos3, bspline3, omoms3 };

// 설정 문자열 ↔ 커널. 모르는 이름이면 false
bool parse_kernel(const std::string &name, Kernel &kernel);
const char *kernel_name(Kernel kernel);

// 탭 반경 R : 보간에 쓰는 이웃 = 2R 칸
int kernel_radius(Kernel kernel);

// 거리 t (px) 의 가중치
double kernel_weight(Kernel kernel, double t);

// 보간점을 지나지 않는 커널(bspline3, omoms3)의 사전 필터 극점. 다른 커널은 0
double kernel_pole(Kernel kernel);

// 1차원 재귀 사전 필터 (Unser 1993, 거울 경계) — c[0..n) 를 제자리에서 계수로 바꿈
void prefilter_1d(double *c, int n, double pole);

// 2차원 영상(CV_64F)을 커널 계수로 바꿈 (가로 → 세로). 극점이 0 이면 그대로
void to_kernel_coeffs(cv::Mat &image64, Kernel kernel);

} // namespace ksh

#endif
