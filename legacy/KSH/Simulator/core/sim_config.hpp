// =====================================================================================
// sim_config.hpp — 시뮬레이터 설정 (config/simulator.yml)
//   값은 모두 설정 파일에서 읽는다. 코드 안의 숫자는 '파일에 항목이 없을 때의 기본값' 뿐이고,
//   처음 실행 때 기본값으로 설정 파일을 만들어 준다.
// =====================================================================================
#ifndef SIM_CONFIG_HPP
#define SIM_CONFIG_HPP

#include <string>

namespace sim {

struct SimConfig {
    // ---------------------------------------------------------------- 경로 (작업 폴더 기준)
    std::string image_dir = "../QR_Image/img";                      // 실제 사진 (1차·2차)
    std::string alignment_config = "../Alignment/config/alignment.yml";
    std::string world_dir = "world";                                // 파노라마·운동장 저장
    std::string result_dir = "results";

    // ---------------------------------------------------------------- 운동장(motion field) 학습
    int field_step_px = 80;            // 격자 간격 (3840, 2160 의 약수)
    int field_window_px = 128;         // 격자마다 정합하는 창 크기
    double field_theta_step_deg = 0.05;// Θ 스윕에서 이 간격의 사진만 사용 (계산 시간)
    double field_min_ok_ratio = 0.8;   // 격자 측정 성공 비율이 이보다 낮으면 '질감 없음' → 이웃 값으로 채움
    double field_max_resid_px = 0.15;  // 맞춤 잔차 RMS 가 이보다 크면 채움 대상
    double field_min_cc = 0.7;         // ECC 상관 최소
    double field_min_texture = 4.0;    // 질감 점수가 촬영 잡음의 이 배수보다 낮은 창은 재지 않음 (민무늬)

    // ---------------------------------------------------------------- 파노라마
    int canvas_margin_x_px = 160;      // 기준 사진 좌우로 넓히는 폭 (격자 간격의 배수)
    int canvas_margin_y_px = 80;       // 위아래

    // ---------------------------------------------------------------- 합성 사진
    double render_noise_sigma = 0.71;  // 촬영 잡음 σ (밝기 단위, EDA 실측)
    double render_gain = 1.0;          // 밝기 배율 (조명 변화 시험용, 1 = 변화 없음)
    bool render_reference_offset = true; // 기준 사진과 스윕 원점의 실측 차이(격자별 상수항)를 넣을지
    bool teach_center_photo = true;    // 기준(티칭) 사진 : true = 실제 center 사진(+새 잡음, 모델 학습과 같은 조건), false = 자세 (0,0) 합성 사진

    // ---------------------------------------------------------------- 로봇
    double robot_max_step_y_mm = 2.0;  // 한 번 이동 최대 (과제 조건)
    double robot_max_step_t_deg = 2.0;
    double robot_gain_error = 0.002;   // 이동 배율 오차 σ (비율)
    double robot_repeat_y_um = 2.0;    // 반복 정밀도 σ
    double robot_repeat_t_deg = 0.001;

    // ---------------------------------------------------------------- 제어
    int control_max_moves = 20;        // 이동 횟수 한도 (과제 조건)
    double tol_y_um = 90.0;            // 허용 범위 (과제 조건)
    double tol_t_deg = 0.01;
    double control_stop_ratio = 0.5;   // 추정값이 허용 범위 × 이 비율 안이면 멈춤 (추정 잡음 여유)

    // ---------------------------------------------------------------- 시험
    double start_range_y_mm = 2.0;     // 시작 자세 : 균일 분포 ±
    double start_range_t_deg = 1.0;
    int monte_carlo_trials = 50;
    unsigned int seed = 20261028;

    bool load(const std::string& path);         // 없거나 읽기 실패 → false (기본값 유지)
    bool save(const std::string& path) const;
    std::string check() const;                  // 값 검사. 문제 없으면 빈 문자열
};

} // namespace sim

#endif
