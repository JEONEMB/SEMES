#ifndef __ALIGNMENT_HPP__
#define __ALIGNMENT_HPP__

#include <opencv2/opencv.hpp>
#include <string>
#include "phase_align.hpp"   // 위상 기반 Y·Θ 분리 알고리즘 (구조체·클래스 정의)

using namespace cv;

typedef double err_t;

// 기준(보정 완료) 상태 대비 오차.  y : mm,  t : deg
// 로봇에 줄 보정값은 부호를 반대로 (−y, −t)
typedef struct _alignment_t {
    err_t y;
    err_t t; // theta
} Alignment_t;

// 기준 영상과 보정 파일(phase_calib.yml)을 읽어 알고리즘을 준비. 처음 alignment() 호출 시 자동으로도 실행됨
bool alignment_init(const std::string& refPath = "img/center.bmp",
                    const std::string& calibPath = "phase_calib.yml");

// 내부 정렬기 접근 (시각화·디버깅용)
pa::PhaseAligner& alignment_engine();

Alignment_t alignment(Mat& src);

// === End Header Setting ===

#endif // __ALIGNMENT_HPP__
