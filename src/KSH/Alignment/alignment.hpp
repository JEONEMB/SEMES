// =====================================================================================
// alignment.hpp — 사진 한 장으로 로봇의 Y·Θ 어긋남을 추정하는 함수 (제출 형식)
//
//  입력 : 현재 자세에서 찍은 사진 (3840×2160, 컬러 또는 회색). 바꾸지 않음
//  출력 : 기준(티칭) 사진 대비 어긋남 y [mm], t [°]  — 라벨과 같은 부호
//         → 로봇 보정 이동 명령은 (−y, −t)
//  실패 : 측정이 불가능하면 y, t 에 NaN (std::isnan 으로 확인), 이유는 alignment_last_info().message
//
//  구현은 ksh 모듈(ksh/include/ksh/*.hpp)을 차례로 부르는 얇은 진입점이다
//    ① ksh::coarse_alignment → ② ksh::fine_alignment → ③ ksh::measure_features → ④ ksh::invert_pose_model
//  설정 : config/alignment.yml (기준 사진, ROI, 정렬 방법, 사용할 모델 파일). 코드에 수치 하드코딩 없음
//  OpenCV 3.4.15 API, C++17
// =====================================================================================
#ifndef __ALIGNMENT_HPP__
#define __ALIGNMENT_HPP__

#include <opencv2/opencv.hpp>

#include <string>
#include <vector>

typedef double error_t;

typedef struct _alignment_t {
    error_t y;  // Y 어긋남 [mm]
    error_t t;  // theta 어긋남 [°]
} Alignment_t;

Alignment_t alignment(cv::Mat &src);

// ---------------------------------------------------------------- 추가 함수 (선택 사용)
// 설정 파일 읽기. 호출하지 않으면 첫 alignment() 호출 때 기본 경로(config/alignment.yml)를 읽음
bool alignment_init(const std::string &config_path = "config/alignment.yml");

// 기준(티칭) 사진 교체 — 장비 기동 후 목표 자세에서 다시 찍은 사진으로 바꿀 때
bool alignment_set_reference(const cv::Mat &reference);

// 마지막 alignment() 호출의 측정 상세 (진단·시각화용)
struct AlignmentRoi {
    std::string name;
    cv::Rect ref_rect_px;   // 기준 사진에서의 위치
    cv::Point2d shift_px;   // 측정 이동 (현재 − 기준)
    double ecc_cc = 0.0;    // 정합 상관 ZNCC (1 에 가까울수록 신뢰)
    bool ok = false;
};
struct AlignmentInfo {
    bool ok = false;
    std::string message;            // 실패 이유
    cv::Point2d coarse_px;          // ① Coarse Alignment 결과
    std::vector<AlignmentRoi> rois; // ② QR 3개 + 배경 ROI
    double s_px = 0.0;              // ③ QR 평균 가로 이동
    double d_px = 0.0;              // ③ 시차 = s − 배경 평균
    double elapsed_ms = 0.0;        // 처리 시간 (회색 변환 포함)
};
const AlignmentInfo &alignment_last_info();

#endif // __ALIGNMENT_HPP__
