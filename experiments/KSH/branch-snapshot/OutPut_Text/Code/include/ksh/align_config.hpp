// =====================================================================================
// ksh/align_config.hpp — 설정 파일(config/alignment.yml) 읽기 · 쓰기
//  모든 수치(ROI 절대 좌표, 측정 파라미터, 영점, 허용 범위, 학습 조건)는 텍스트 설정 파일에만 둔다
//  기준 사진·템플릿 사진은 쓰지 않는다 — 실행에 필요한 것은 yml 두 개 (alignment.yml, 보정식 model_*.yml)
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
    double t_deg = 0.01;   // Θ 허용 오차 [°]
};

// 학습·검증 세트 (align_train)
struct TrainingSet {
    std::string name;                 // 보고서 이름 (예 1st, 4th)
    std::string image_dir;            // 라벨 사진 폴더 (하위 폴더 포함)
    std::string feature_csv;          // (선택) 이미 잰 특징 CSV — 있으면 사진을 다시 재지 않음
    std::vector<std::string> exclude; // 학습·채점에서 뺄 사진 이름 (라벨 이상 등)
    std::string zero_ref;             // 영점으로 쓸 센터 사진 이름 (모든 세트의 센터 사진에서 찾음). 비우면 영점 없이
};
struct TrainingConfig {
    std::vector<TrainingSet> train;   // 학습 세트 (세트마다 상수항을 따로 둠)
    std::vector<TrainingSet> test;    // 채점만 할 세트 (zero_ref 로 영점을 잡고 평가)
    int theta_order_s = 1;            // s 식의 Θ 차수
    int theta_order_d = 1;            // d 식의 Θ 차수
    bool balance_sets = true;         // 세트 균형 가중 (1/장수)
    int threads = 0;                  // 사진 측정 스레드 수 (0 = CPU 수)
    std::string output;               // 모델 저장 경로 (실행 폴더 기준)
    std::string result_dir = "results";
};

// 예제(main.cpp) 설정
struct DemoConfig {
    std::string image;                // 인자 없이 실행할 때의 사진
    int display_width_px = 1600;
};

struct AlignConfig {
    std::string path;                 // 읽은 설정 파일 경로
    cv::Size image_size_px;           // 사진 크기 (3840 × 2160)
    FeatureParams features;           // ①②③ (QR·배경 ROI 절대 좌표와 측정 파라미터)
    std::string model_path;           // 보정식 파일 (model_file)
    PoseModel model;                  // ④
    ZeroOffset zero;                  // 영점 (zero: {y_mm, t_deg}) — 보정식 결과에서 뺄 값. --setup / --zero 가 기록
    ZeroOffset center_to_encoder;     // 센터 사진 자세 − 엔코더 영점 (center_to_encoder: {y_mm, t_deg}). 판정 기준이 엔코더일 때
                                      //  --setup / --zero 는 zero = 센터 측정값 + 이 값 으로 기록 → 결과가 엔코더 영점 기준이 됨
    Tolerance tolerance;              // 권장 멈춤 기준
    Tolerance spec_tolerance;         // 과제 명세
    TrainingConfig training;
    DemoConfig demo;
};

// 상대 경로 해석 : 실행 폴더에 있으면 그대로, 없으면 설정 파일 폴더 기준
std::string resolve_path(const std::string &config_path, const std::string &path);

// 설정과 모델을 읽음. 실패하면 false 와 이유. require_model = false 면 모델이 없어도 됨 (학습 도구)
bool load_align_config(const std::string &path, AlignConfig &cfg, std::string &error, bool require_model = true);

// 설정 파일의 zero: 블록만 바꿔 씀 (다른 줄·주석은 그대로). 원본은 <path>.bak
bool save_zero_offset(const std::string &path, const ZeroOffset &zero, std::string &error, bool backup = true);

// 설정 파일의 ROI 줄(- { name: X, cx_px: .., cy_px: .., hw_px: .. })에서 이름이 같은 줄의 cx_px · cy_px 만 바꿔 씀
//  (초기 세팅 : 센터 사진에서 찾은 중심으로 박스를 다시 맞춤). backup = true 면 원본을 <path>.bak 으로
bool save_roi_centers(const std::string &path, const std::vector<BoxSpec> &boxes, std::string &error, bool backup = true);

} // namespace ksh

#endif
