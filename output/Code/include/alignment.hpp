// =====================================================================================
// alignment.hpp — 사진 한 장으로 로봇의 Y·Θ 어긋남을 추정하는 함수 (제출 형식)
//
//  입력 : 현재 자세에서 찍은 사진 (3840×2160, 컬러 또는 회색). 바꾸지 않음
//  출력 : 기준(티칭) 사진 대비 어긋남 y [mm], t [°]  — 라벨과 같은 부호
//         → 로봇 보정 이동 명령은 (−y, −t)
//  실패 : 측정이 불가능하면 y, t 에 NaN (std::isnan 으로 확인), 이유는 alignment_last_info().message
//
//  구현은 ksh 모듈(include/ksh/*.hpp)을 차례로 부르는 얇은 진입점이다
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

// 기준(티칭) 사진 교체 — 장비 기동 후 목표 자세(center)에서 찍은 사진으로 바꿀 때
//  설정 relocation.enable 이면 ROI 템플릿(교보재 center)과 새 사진을 위상 상관으로 비교해
//  ROI·Coarse 창을 새 사진의 같은 물체 위치로 옮긴다 (결과는 alignment_reference_info())
//  false : 빈 사진, 크기 불일치, 재배치 실패 (이때는 이전 기준 사진과 ROI 를 그대로 유지)
bool alignment_set_reference(const cv::Mat &reference);

// 마지막 alignment() 호출의 측정 상세 (진단·시각화용)
struct AlignmentRoi {
    std::string name;
    cv::Rect ref_rect_px;   // 기준 사진에서의 위치
    cv::Point2d shift_px;   // 측정 이동 (현재 − 기준)
    double zncc = 0.0;      // 정합 상관 ZNCC (1 에 가까울수록 신뢰)
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

// 마지막 기준 사진 설정(alignment_set_reference)의 ROI 재배치 결과
struct ReferenceInfo {
    bool ok = false;                // 기준 사진 설정 성공
    bool relocated = false;         // ROI 재배치를 했는지 (relocation.enable)
    std::string message;            // 실패 이유
    cv::Point2d shift_px;           // 새 center − 템플릿 : 판 전체(Coarse 창) 이동 (px)
    std::vector<AlignmentRoi> rois; // ROI 별 이동과 옮긴 뒤의 위치 (ref_rect_px = 새 기준 사진에서의 ROI)
    double qr_spread_px = 0.0;      // QR 3개 이동 차이 (px) — 크면 회전·거리 변화 의심
};
const ReferenceInfo &alignment_reference_info();

#endif // __ALIGNMENT_HPP__
