// =====================================================================================
// alignment.hpp — 사진 한 장으로 스테이지의 Y·Θ 어긋남을 추정하는 함수 (제출 형식, 교보재 텍스트 설정판)
//
//  입력 : 현재 자세에서 찍은 사진 (3840×2160, 컬러 또는 회색). 바꾸지 않음
//  출력 : 센터(영점) 대비 어긋남 y [mm], t [°] — 라벨(스테이지 명령)과 같은 부호 → 보정 이동 명령은 (−y, −t)
//         교보재 부호 : +Y, +Θ 에서 플레이트가 화면 오른쪽으로 감
//  실패 : 측정이 불가능하면 y, t 에 NaN (std::isnan 으로 확인), 이유는 alignment_last_info().message
//
//  기준(center) 사진 · ROI 템플릿 사진 없이 동작한다. 실행에 필요한 것은 텍스트 설정 두 개 :
//    config/alignment.yml   ROI 절대 좌표(QR 3개 · 뒷벽 홀 4곳), 측정 파라미터, 영점(zero), 보정식 파일 이름
//    config/model_s1d1.yml  보정식 계수 (align_train 이 만듦)
//  흐름 : ① ksh::locate_qr → ② ksh::locate_hole → ③ ksh::measure_features (s, d) → ④ ksh::invert_pose_model − 영점
//  OpenCV 3.4.15 / 4.x API, C++17
// =====================================================================================
#ifndef __ALIGNMENT_HPP__
#define __ALIGNMENT_HPP__

#include <opencv2/opencv.hpp>

#include <string>
#include <vector>

// 제출 형식의 error_t (double). 리눅스 glibc 에는 이미 error_t(int) 가 있어 리눅스에서만 이름을 바꿔 피함
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

// 영점 잡기 : 목표 자세(center)에서 찍은 사진의 측정값을 영점으로 삼아, 이후 alignment() 가 그 자세에서 (0, 0) 을 냄
//  사진은 저장하지 않고 영점 두 값(y_mm, t_deg)만 기억한다. save = true 면 설정 파일 zero: 에도 기록 (원본 .bak)
//  설정의 center_to_encoder 가 0 이 아니면 영점 = 센터 측정값 + center_to_encoder → 결과가 엔코더 영점 기준 (센터는 그만큼 떨어져 읽힘)
//  주의 : 정렬할 때와 같은 방향으로 센터에 접근해서 찍을 것 (반대 방향이면 백래시만큼 밀림)
bool alignment_set_reference(const cv::Mat &reference, bool save = false);

// 초기 세팅 : 센터 사진(1장 이상, 권장 3장)으로 ROI 박스 7개(QR 3 + 홀 4)의 절대 좌표를 다시 맞추고 영점을 잡는다
//  ① 지금 설정의 박스로 측정 → ② 찾은 중심으로 박스 좌표 갱신 → ③ 새 박스로 다시 재서 센터 사진들의 평균을 영점으로
//  save = true 면 설정 파일의 qr_rois · background_rois 좌표와 zero: 를 고쳐 씀 (원본 .bak). 결과는 alignment_setup_info()
bool alignment_setup(const std::vector<cv::Mat> &centers, bool save = true);

// 영점 지우기 (설정 파일의 zero: 값으로 되돌림)
void alignment_clear_reference();

// 마지막 alignment() 호출의 측정 상세 (진단·시각화용)
struct AlignmentRoi {
    std::string name;
    cv::Rect ref_rect_px;   // 이번 사진에서 쓴 박스 (설정 좌표 + 대략 이동)
    cv::Point2d shift_px;   // 형식 호환용 (항상 0 — 기준 사진 대비 이동을 쓰지 않음)
    cv::Point2d center_px;  // 측정한 중심 (서브픽셀, 사진 절대 좌표)
    double zncc = 0.0;      // 형식 호환용 (QR = 1, 홀 = 명암 차/255)
    bool ok = false;
};
struct AlignmentInfo {
    bool ok = false;
    std::string message;            // 실패 이유
    cv::Point2d coarse_px;          // ① 대략 위치 : 설정 좌표 대비 가로 이동 (x), 0 (y)
    std::vector<AlignmentRoi> rois; // ①② QR 3개 + 홀 4곳
    double s_px = 0.0;              // ③ QR 중심 가로 위치 평균
    double d_px = 0.0;              // ③ 시차 = s − 홀 가로 위치 평균
    double elapsed_ms = 0.0;
};
const AlignmentInfo &alignment_last_info();

// 마지막 영점 등록(alignment_set_reference) 결과
struct ReferenceInfo {
    bool ok = false;
    bool relocated = false;         // alignment_setup 으로 박스를 다시 맞췄으면 true
    std::string message;
    double zero_y_mm = 0.0, zero_t_deg = 0.0;
    std::vector<AlignmentRoi> rois;
};
const ReferenceInfo &alignment_reference_info();

// 마지막 초기 세팅(alignment_setup) 결과
struct SetupInfo {
    bool ok = false;
    std::string message;                     // 실패 이유
    std::vector<std::string> warnings;       // 경고 (센터 사진끼리 차이가 큼 등) — 비어 있으면 정상
    std::vector<AlignmentRoi> rois;          // 새 박스와 찾은 중심 (첫 센터 사진)
    double zero_y_mm = 0.0, zero_t_deg = 0.0;
    double s_spread_px = 0.0, d_spread_px = 0.0;   // 센터 사진들의 s·d 최대−최소
    int n_centers = 0;
};
const SetupInfo &alignment_setup_info();

#endif // __ALIGNMENT_HPP__
