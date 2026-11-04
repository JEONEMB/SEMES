// =====================================================================================
// integrated/integrated.hpp — 통합 정렬기 (김성현 ksh 추정 + 신윤석 판 검출·QR 검증)
//
//  비교 결과 (docs/integrated_comparison.html, 2026-11-01) 로 역할을 나눴다
//   · 추정 (Y, Θ)  : ksh — IC-GN QR 이동 s + 하단 홀 시차 d → 물리식 역변환
//                    같은 2차 57장·같은 좌표 보류 규칙에서 Y RMS 13 µm (QR6 역회귀 148 µm, 가장자리 추가 67 µm)
//   · 검증         : shin — 판 검출 → QR 해독(payload) → 모듈 재생성 일치 → Fourier 코너
//                    ① 판·QR 이 기준 사진과 같은가 ② 판 중심 이동 ≈ Coarse 이동 ③ 코너 QR 이동 ≈ IC-GN QR 이동
//                    두 측정이 서로 독립이라 어느 한쪽의 오정합(이웃 QR 에 붙음 등)을 잡아낸다
//  검증은 설정으로 끌 수 있다 (판 검출·해독에 사진당 약 0.5 s)
// =====================================================================================
#ifndef INTEGRATED_INTEGRATED_HPP
#define INTEGRATED_INTEGRATED_HPP

#include "alignment.hpp"          // 팀 공통 자료형 (PlateDetectionResult, AlignmentOptions)
#include "ksh/align_config.hpp"
#include "ksh/estimator.hpp"

#include <array>
#include <limits>
#include <map>
#include <string>
#include <utility>

namespace integrated {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// 통합 전용 설정 (integrated.yml 의 integrated: 묶음). 나머지 키는 ksh 형식 그대로
struct Settings {
    ksh::AlignConfig ksh;               // ROI · Coarse/Fine · 모델 · 허용 범위
    std::string config_path;
    std::string reference_path;         // 기준(center) 사진 — 측정 직전에 다시 찍은 사진 (Center 정의 2026-10-30)
    bool verify_plate = true;           // 판 검출·QR 검증 사용
    bool reject_on_guard = true;        // 검증 모순(판·QR 은 찾았는데 값이 다름) 시 Y·Θ 를 NaN 으로
    bool reject_unverified = false;     // 판을 못 찾아 검증 자체를 못 했을 때도 NaN 으로 (575장 중 4장 0.7 %, 그 4장도 추정은 허용 안)
    double plate_guard_px = 30.0;       // |판 중심 이동 − Coarse 이동| 한계 (실측 최대 10.6 px 의 약 3배)
    double corner_check_px = 0.10;      // |코너 QR 이동 − IC-GN QR 이동| 한계 (실측 최대 0.026 px 의 약 4배)
    int min_corner_points = 8;          // QR 하나당 공통 코너 최소 개수
    double center_gate_y_um = 21.0;     // 새 center 채택 검사 : 1.96·√2·로봇 반복 σ(7.5 µm)
    double center_gate_t_deg = 0.0055;  //                     1.96·√2·0.00198°
};

struct Result {
    bool ok = false;                    // 최종 판정 (추정 성공 && (검증 끔 || 검증 통과))
    std::string message;
    double y_mm = kNaN, t_deg = kNaN;   // 기준 대비 어긋남 (라벨과 같은 부호). 보정 이동 = (−Y, −Θ)
    double s_px = kNaN, d_px = kNaN;    // ksh 특징
    cv::Point2d coarse_px;
    bool estimate_ok = false;
    // 검증 (shin)
    bool plate_checked = false, plate_found = false, payload_match = false;
    std::array<std::string, 3> payloads{};
    double plate_shift_diff_px = kNaN;  // 판 중심 이동 − Coarse 이동 (가로)
    double corner_s_px = kNaN;          // Fourier 코너로 잰 QR 평균 가로 이동
    double corner_s_diff_px = kNaN;     // corner_s − s
    int corner_points = 0;
    bool verified = false;              // 검증을 끝까지 수행함 (판 찾음 + 공통 코너 충분)
    bool guard_ok = true;               // 모순 없음 (검증 못 한 경우도 true — verified 로 구분)
    double ms_estimate = 0.0, ms_verify = 0.0;
};

class Aligner {
  public:
    // 설정·모델·기준 사진 읽기. 실패 시 false 와 이유. verify_override : -1 = 설정대로, 0 = 검증 끔, 1 = 검증 켬
    bool load(const std::string &config_path, std::string &error, int verify_override = -1);
    // 기준 사진 교체 (새 center 를 찍었을 때). 판 검출도 한 번 해 둔다
    bool set_reference(const cv::Mat &reference, std::string &error);
    bool ready() const { return !ref32_.empty(); }
    Result estimate(const cv::Mat &image) const;
    // 새 center 채택 검사 : 현재 기준 대비 새 사진이 반복 정밀도 범위 안인가 (영점 상수 c 는 빼고 비교)
    bool center_gate(const cv::Mat &new_center, double &dy_um, double &dt_deg, std::string &message) const;
    const Settings &settings() const { return s_; }

  private:
    using CornerKey = std::pair<std::string, int>;   // (payload, 모듈 격자 ID)
    bool detect(const cv::Mat &image, PlateDetectionResult &det) const;
    void verify(const cv::Mat &image, const ksh::PoseEstimate &e, Result &r) const;

    Settings s_;
    cv::Mat ref32_;
    bool ref_plate_ok_ = false;
    PlateDetectionResult ref_det_;
    std::map<CornerKey, cv::Point2f> ref_corners_;
};

// 설정 파일 읽기 (ksh 키 + integrated: 묶음)
bool load_settings(const std::string &path, Settings &s, std::string &error);

// 팀 인터페이스 alignment() 가 쓰는 전역 정렬기 (team_alignment.cpp)
const Result &last_result();                                   // 마지막 alignment() 의 상세
bool configure(const std::string &path, std::string &error);   // 설정을 명시적으로 읽기

} // namespace integrated

#endif
