// =====================================================================================
// ksh/align_config.hpp — 설정 파일(config/alignment.yml) 읽기
//  모든 수치(ROI 좌표, 창 크기, 허용 범위, 학습·평가 조건)는 설정 파일에만 둔다 (코드에 하드코딩하지 않음)
// =====================================================================================
#ifndef KSH_ALIGN_CONFIG_HPP
#define KSH_ALIGN_CONFIG_HPP

#include "ksh/features.hpp"
#include "ksh/pose_model.hpp"

#include <opencv2/core.hpp>

#include <string>

// 나의 namespace 정의 - 설정값
namespace ksh {

// 오차 임계값 구조체
struct Tolerance {
    double y_um = 45.0;
    double t_deg = 0.005;
};

// 학습 도구(align_train) 설정
struct TrainingConfig {
    std::string image_dir;          // 라벨이 붙은 사진 폴더
    std::string reference;          // 기준 사진 이름 (확장자 제외)
    int set = 1;                    // 학습에 쓸 촬영 세트
    double y_grid_mm = 0.1;         // Y 단독 사진은 이 간격의 배수만 (같은 세트의 늦게 찍은 묶음 제외)
    int theta_order_s = 1;          // s 식의 Θ 차수
    int theta_order_d = 1;          // d 식의 Θ 차수
    std::string output;             // 모델 저장 경로 (실행 폴더 기준)
};

// 평가 도구(align_eval) 설정
struct EvaluationConfig {
    std::string image_dir;
    std::string reference_set1, reference_set2;   // 세트별 기준 사진 이름
    std::string result_dir = "results";
    double near_target_t_deg = 0.2;               // 2차 '목표 근처' 묶음 : |Θ| ≤ 이 값
    double cv_near_y_mm = 0.5;                     // 교차검증 '목표 근처' : |Y| ≤ 이 값이고 |Θ| ≤ near_target_t_deg
    int cv_blocks_y = 4;                           // 블록 교차검증 : Y 단독 사진 묶음 수
    int cv_blocks_t = 10;                          //               Θ 단독 사진 묶음 수
    double score_cap = 100.0;                      // J 계산 때 정규화 오차를 자르는 값 (실패 = 이 값)
};

// 예제(main.cpp) 설정
struct DemoConfig {
    std::string image, reference;
    int display_width_px = 1600;
};

// 이미지 분석을 위한 구조체
struct AlignConfig {
    std::string path;                 // 읽은 설정 파일 경로
    std::string reference_path;       // 기준(티칭) 사진
    cv::Size image_size_px;
    FeatureParams features;           // ①②③
    std::string model_path;           // 비어 있으면 설정 파일 안의 model: 사용
    PoseModel model;                  // ④
    Tolerance tolerance;              // 프로젝트 기본 허용 범위
    Tolerance spec_tolerance;         // 과제 명세 허용 범위
    TrainingConfig training;
    EvaluationConfig evaluation;
    DemoConfig demo;
};

// 상대 경로 해석 : 실행 폴더에 있으면 그대로, 없으면 설정 파일 폴더 기준
std::string resolve_path(const std::string &config_path, const std::string &path);

// 설정과 모델을 읽음. 실패하면 false 와 이유. require_model = false 면 모델이 없어도 됨 (학습 도구)
bool load_align_config(const std::string &path, AlignConfig &cfg, std::string &error, bool require_model = true);

} // namespace ksh

#endif
