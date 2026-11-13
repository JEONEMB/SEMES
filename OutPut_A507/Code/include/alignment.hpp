// =====================================================================================
// alignment.hpp — 사진 한 장으로 스테이지의 Y·Θ 어긋남을 추정하는 함수 (제출 형식, A507 장비용)
//
//  입력 : 현재 자세에서 찍은 사진 (3840×2160 보정 프레임, 컬러 또는 회색). 바꾸지 않음
//  출력 : 센터 대비 어긋남 y [mm], t [°]  — 라벨(스테이지 명령)과 같은 부호
//         → 스테이지 보정 이동 명령은 (−y, −t)
//         ※ A507 은 +Y, +Θ 에서 플레이트가 화면 왼쪽으로 감 (SEMES 교보재와 부호 반대). 라벨 = A507 스테이지 명령 방향
//  실패 : 측정이 불가능하면 y, t 에 NaN (std::isnan 으로 확인), 이유는 alignment_last_info().message
//
//  구현은 ksh 모듈(include/ksh/*.hpp)을 차례로 부르는 얇은 진입점이다
//    ① ksh::locate_qr → ② ksh::locate_screw → ③ ksh::measure_features → ④ ksh::invert_pose_model
//  교보재 방식과 달리 기준(center) 사진 없이 사진 한 장 안의 절대 위치로 계산한다
//  설정 : config/alignment.yml (QR 찾기, 나사 ROI, 영점, 사용할 모델 파일). 코드에 수치 하드코딩 없음
//  OpenCV 3.4.15 / 4.x API, C++17
// =====================================================================================
#ifndef __ALIGNMENT_HPP__
#define __ALIGNMENT_HPP__

#include <opencv2/opencv.hpp>

#include <string>
#include <vector>

// 제출 형식의 error_t (double). 리눅스(라즈베리파이) glibc 에는 이미 error_t(int) 가 있어 그대로 쓰면 빌드 오류
//  → 리눅스에서만 이름을 바꿔 피함 (코드에서는 그대로 error_t 로 씀. 윈도우는 교보재와 완전히 같음)
#if defined(__GLIBC__) || defined(__linux__)
#define error_t alignment_error_t
#endif
typedef double error_t;

typedef struct _alignment_t {
    error_t y;  // Y 어긋남 [mm]
    error_t t;  // theta 어긋남 [°]
} Alignment_t;

Alignment_t alignment(cv::Mat &src);

// ---------------------------------------------------------------- 추가 함수 (선택 사용)
// 설정 파일 읽기. 호출하지 않으면 첫 alignment() 호출 때 기본 경로(config/alignment.yml)를 읽음
bool alignment_init(const std::string &config_path = "config/alignment.yml");

// 기준(센터) 사진 등록 = 영점 잡기
//  목표 자세(center)에서 찍은 사진을 넘기면 그 사진의 측정값을 영점으로 삼아, 이후 alignment() 가 그 자세에서 (0, 0) 을 냄
//  사진은 저장하지 않고 영점 두 값만 기억한다. save = true 면 설정 파일의 zero: 에도 기록 (원본은 .bak)
//  주의 : 정렬할 때와 같은 방향(−쪽에서 지나쳐 접근)으로 센터에 놓고 찍을 것 (반대 방향이면 백래시만큼 Θ 가 밀림)
//  false : 빈 사진, 측정 실패
bool alignment_set_reference(const cv::Mat &reference, bool save = false);

// 영점 지우기 (설정 파일의 zero: 값으로 되돌림)
void alignment_clear_reference();

// 마지막 alignment() 호출의 측정 상세 (진단·시각화용)
struct AlignmentRoi {
    std::string name;
    cv::Rect ref_rect_px;   // 이번 사진에서 찾은 영역 (QR = 사각형, 나사 = 추정한 ROI)
    cv::Point2d shift_px;   // 교보재 형식 호환용 (항상 0 — 기준 사진 대비 이동을 쓰지 않음)
    cv::Point2d center_px;  // 측정한 중심 (서브픽셀)
    double zncc = 0.0;      // 교보재 형식 호환용 (QR = 1 − 변 맞춤 잔차, 나사 = 1 − 원 맞춤 잔차/반지름)
    bool ok = false;
};
struct AlignmentInfo {
    bool ok = false;
    std::string message;            // 실패 이유
    cv::Point2d coarse_px;          // ① QR 3개 중심 평균 (qx, qy)
    std::vector<AlignmentRoi> rois; // ①② QR 3개 + 나사 ROI
    double s_px = 0.0;              // ③ QR 중심 가로 위치
    double d_px = 0.0;              // ③ 시차 = s − 나사 가로 위치 평균
    double elapsed_ms = 0.0;        // 처리 시간 (회색 변환 포함)
};
const AlignmentInfo &alignment_last_info();

// 마지막 기준 사진 등록(alignment_set_reference) 결과
struct ReferenceInfo {
    bool ok = false;                // 등록 성공
    bool relocated = false;         // 교보재 형식 호환용 (항상 false — ROI 는 매 사진 QR 위치로 옮김)
    std::string message;            // 실패 이유
    double zero_y_mm = 0.0;         // 새 영점 (센터 사진에서 잰 값)
    double zero_t_deg = 0.0;
    std::vector<AlignmentRoi> rois; // 센터 사진에서 찾은 QR·나사
};
const ReferenceInfo &alignment_reference_info();

#endif // __ALIGNMENT_HPP__
