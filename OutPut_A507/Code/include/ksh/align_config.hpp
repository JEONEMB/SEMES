// =====================================================================================
// ksh/align_config.hpp — 설정 파일(config/alignment.yml) 읽기
//  모든 수치(QR 찾기, 나사 ROI, 허용 범위, 학습 조건)는 설정 파일에만 둔다 (코드에 하드코딩하지 않음)
//  사진 파일(기준 사진·템플릿)은 쓰지 않는다 — 설정은 텍스트(yml) 두 개뿐
// =====================================================================================
#ifndef KSH_ALIGN_CONFIG_HPP
#define KSH_ALIGN_CONFIG_HPP

#include "ksh/estimator.hpp"
#include "ksh/features.hpp"
#include "ksh/pose_model.hpp"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace ksh {

// 오차 임계값
struct Tolerance {
    double y_um = 90.0;    // Y 허용 오차 [µm]
    double t_deg = 0.09;   // Θ 허용 오차 [°]
};

// 학습 도구(align_train) 설정
struct TrainingSet {
    std::string name;             // 보고서에 쓸 이름 (예 grid, set1)
    std::string image_dir;        // 라벨이 붙은 사진 폴더 (하위 폴더 포함)
    std::string feature_csv;      // (선택) 이미 잰 특징 CSV — 있으면 사진을 다시 재지 않음
};
struct TrainingConfig {
    std::vector<TrainingSet> train;   // 학습에 쓸 세트
    std::vector<TrainingSet> test;    // 학습에 안 쓰고 채점만 할 세트
    int theta_order_s = 1;            // s 식의 Θ 차수
    int theta_order_d = 1;            // d 식의 Θ 차수
    int cv_blocks = 5;                // 블록 교차검증 묶음 수 (Θ 순서로 이웃끼리)
    int threads = 0;                  // 사진 측정 스레드 수 (0 = CPU 수)
    std::string output;               // 모델 저장 경로 (실행 폴더 기준)
    std::string result_dir = "results";
};

// 예제(main.cpp) 설정
struct DemoConfig {
    std::string image;                // 인자 없이 실행할 때의 사진
    int display_width_px = 1600;      // 결과 창 가로 크기
};

struct AlignConfig {
    std::string path;                 // 읽은 설정 파일 경로
    cv::Size image_size_px;           // 사진 크기 (3840 × 2160)
    FeatureParams features;           // ①②③
    std::string model_path;           // 비어 있으면 설정 파일 안의 model: 사용
    PoseModel model;                  // ④
    ZeroOffset zero;                  // 현장 영점 (zero: {y_mm, t_deg}), 기본 0
    Tolerance tolerance;              // 멈춤 판정 권장 기준
    Tolerance spec_tolerance;         // 과제 명세 기준
    TrainingConfig training;
    DemoConfig demo;
};

// 상대 경로 해석 : 실행 폴더에 있으면 그대로, 없으면 설정 파일 폴더 기준
std::string resolve_path(const std::string &config_path, const std::string &path);

// 설정과 모델을 읽음. 실패하면 false 와 이유. require_model = false 면 모델이 없어도 됨 (학습 도구)
bool load_align_config(const std::string &path, AlignConfig &cfg, std::string &error, bool require_model = true);

// 설정 파일의 zero: 블록만 바꿔 씀 (다른 줄·주석은 그대로). 원본은 <path>.bak 으로 남김
bool save_zero_offset(const std::string &path, const ZeroOffset &zero, std::string &error);

} // namespace ksh

#endif
