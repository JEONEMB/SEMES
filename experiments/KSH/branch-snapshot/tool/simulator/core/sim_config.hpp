// =====================================================================================
// sim_config.hpp — 시뮬레이터 설정 (config/simulator.yml)
//   값은 모두 설정 파일에서 읽는다. 코드 안의 숫자는 '파일에 항목이 없을 때의 기본값' 뿐이고,
//   처음 실행 때 기본값으로 설정 파일을 만들어 준다.
//   경로는 모두 시뮬레이터 루트(config/ 가 있는 폴더) 기준. 목록은 앞에서부터 처음 존재하는 것을 씀
// =====================================================================================
#ifndef SIM_CONFIG_HPP
#define SIM_CONFIG_HPP

#include <string>
#include <vector>

namespace sim {

struct SimConfig {
    // ---------------------------------------------------------------- 경로 (루트 기준 후보 목록)
    std::vector<std::string> data_dirs = {"data/img", "../../data/img", "../QR_Image/img"};  // 키트 → 팀 저장소(PJT/data/img) → 예전 개발 폴더
    std::vector<std::string> alignment_configs = {"config/alignment.yml", "../../config/alignment.yml", "../Alignment/config/alignment.yml"};
    std::string reference_photo = "center";   // 영점 사진 (데이터 폴더의 사진 이름) — 참 자세 (0, 0) = 이 사진. 바꾸면 가상 장면을 다시 만들어야 함
    std::string world_dir = "world";                                // 파노라마·운동장 저장
    std::string result_dir = "results";
    // 실제로 쓰는 경로 (resolvePaths 가 채움. 없으면 빈 문자열)
    std::string image_dir, alignment_config;

    // ---------------------------------------------------------------- 추정기
    std::string estimator_type = "ksh";        // ksh = 내장 (Alignment 모듈), plugin = DLL
    std::string estimator_plugin = "";         // plugin 일 때 DLL 경로 (확장자 생략 가능 : .dll / .so 를 붙여 찾음)
    std::string estimator_plugin_config = "";  // 플러그인 init 에 넘길 설정 파일 (선택)
    // 영점 맞춤 : 기준(티칭) 직후 참 자세 (0, 0) 사진을 한 번 추정해 그 값을 영점 차이로 보고, 이후 추정값에서 뺌
    //   (실제 장비에서 center 를 다시 찍어 영점을 잡는 절차와 같음)
    //   0 = 안 함, 1 = 항상, 2 = 자동 (추정기가 기준 사진 교체를 지원하지 않을 때만 — 자체 기준 사진의 영점이 다를 수 있음)
    int zero_calibration = 2;

    // ---------------------------------------------------------------- 운동장(motion field) 학습
    int field_step_px = 80;            // 격자 간격 (3840, 2160 의 약수)
    int field_window_px = 128;         // 격자마다 정합하는 창 크기
    double field_theta_step_deg = 0.05;// Θ 스윕에서 이 간격의 사진만 사용 (계산 시간)
    double field_min_ok_ratio = 0.8;   // 격자 측정 성공 비율이 이보다 낮으면 '질감 없음' → 이웃 값으로 채움
    double field_max_resid_px = 0.15;  // 맞춤 잔차 RMS 가 이보다 크면 채움 대상
    double field_min_cc = 0.7;         // ECC 상관 최소
    double field_min_texture = 4.0;    // 질감 점수가 촬영 잡음의 이 배수보다 낮은 창은 재지 않음 (민무늬)
    double field_edge_min_energy = 8.0;  // 직선 모서리 창 : 기울기 에너지가 촬영 잡음의 이 배수 이상 (후보만 넓게 — 실제 채택은 상관·잔차로 거름)
    double field_edge_min_ratio = 0.65;  //                 한 방향 기울기 비율이 이 값 이상 (가로선 = 세로 기울기, 잡음만 있으면 약 0.5)
    int field_edge_search_px = 48;       //                 선에 수직 방향 검색 범위 (Θ 1° 에서 화면 끝 이동 약 30 px)

    // ---------------------------------------------------------------- 파노라마
    int canvas_margin_x_px = 160;      // 기준 사진 좌우로 넓히는 폭 (격자 간격의 배수)
    int canvas_margin_y_px = 80;       // 위아래

    // ---------------------------------------------------------------- 합성 사진
    double render_noise_sigma = 0.71;  // 촬영 잡음 σ (밝기 단위, EDA 실측)
    double render_gain = 1.0;          // 밝기 배율 (조명 변화 시험용, 1 = 변화 없음)
    bool render_reference_offset = true; // 기준 사진과 스윕 원점의 실측 차이(격자별 상수항)를 넣을지
    bool teach_center_photo = true;    // 기준(티칭) 사진 : true = 실제 center 사진(+새 잡음, 모델 학습과 같은 조건), false = 자세 (0,0) 합성 사진

    // ---------------------------------------------------------------- 로봇
    double robot_max_step_y_mm = 2.0;  // 한 번 이동 최대 — 장비 제원 Y 2 mm
    double robot_max_step_t_deg = 1.0;  // 제원 : 한 번에 Θ 1° (Y 2 mm)
    double robot_gain_error = 0.002;   // 이동 배율 오차 σ (비율, 가정값 — 미측정)
    double robot_repeat_y_um = 7.53;   // 반복 정밀도 σ (실측 : 같은 지령 3장 97묶음, 추정 잡음 0.61 µm 제외, 2026-11-01)
    double robot_repeat_t_deg = 0.00198;

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

    // ---------------------------------------------------------------- 합성 검증
    bool validate_include_set3 = false; // 3차 사진(234장)도 검증에 넣을지 (기준 사진 = center1)

    // ---------------------------------------------------------------- 화면
    double plot_font_scale = 1.6;      // 그림 글자 배율 (1 = 이전 크기)
    int ui_font_pt = 12;               // 화면(MFC) 글자 크기 pt (이전 9)

    bool load(const std::string& path);         // 없거나 읽기 실패 → false (기본값 유지)
    bool save(const std::string& path) const;
    void resolvePaths();                        // 후보 목록 → image_dir, alignment_config
    std::string check() const;                  // 값 검사. 문제 없으면 빈 문자열
};

} // namespace sim

#endif
