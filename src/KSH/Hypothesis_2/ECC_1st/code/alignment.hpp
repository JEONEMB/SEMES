#ifndef __ALIGNMENT_HPP__
#define __ALIGNMENT_HPP__

#include <opencv2/opencv.hpp>
#include <string>

// error_t : 기존 템플릿의 이름 유지 (Linux glibc 의 error_t(int) 와 겹치지 않게 MSVC 에서만 정의)
#ifdef _MSC_VER
typedef double error_t;
#endif

typedef struct _alignment_t {
    double y;    // Y 오차 (mm, 기준 대비).  로봇 보정값 = −y
    double t;    // theta 오차 (deg, 기준 대비). 로봇 보정값 = −t
} Alignment_t;

// 기준 영상(Y = 0, Θ = 0)과 ECC 모델 파일(ecc_model.yml)로 초기화. 성공하면 true
bool alignment_init(const std::string& refPath = "img/center.bmp", const std::string& modelPath = "ecc_model.yml");

// 영상 한 장 → 기준 대비 Y·Θ (가설 H2 : ECC 전역 원근 정합). 실패하면 NaN
Alignment_t alignment(cv::Mat& src);

#endif // __ALIGNMENT_HPP__
