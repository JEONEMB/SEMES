// =====================================================================================
// ksh/pose_model.hpp — ④ 특징(s, d) ↔ 자세(Y, Θ) 모델 : 학습(fit) · 추정(invert) · 저장
//
//  특징 = c + a·Y[mm] + Σ_k b_k·Θ[°]^k   (Y 는 1차, Θ 차수는 s·d 각각 설정 — 1 = 직선)
//  추정 : 두 식을 동시에 만족하는 (Y, Θ) — 1/σ² 가중 가우스–뉴턴, 출발점 = 1차 항만 쓴 두 직선의 교점
//  학습 : 최소제곱(SVD). σ = 학습 잔차 RMS (잡음이 작은 식을 더 믿도록 가중에 씀)
// =====================================================================================
#ifndef KSH_POSE_MODEL_HPP
#define KSH_POSE_MODEL_HPP

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace ksh {

struct FeatureModel {
    double c_px = 0.0;                    // 상수항 (기준 사진과 학습 스윕 원점의 차이)
    double a_px_per_mm = 0.0;             // Y 감도
    std::vector<double> b_px_per_deg_k;   // [b1, b2, ...] — 길이 = Θ 차수
    double sigma_px = 1.0;                // 학습 잔차 RMS (역산 가중치)
};

struct PoseModel {
    FeatureModel s, d;
};

// 학습 표본 한 장
struct FeatureSample {
    std::string stem;
    double y_mm = 0.0, t_deg = 0.0;   // 정답 (라벨)
    double s_px = 0.0, d_px = 0.0;    // 측정 특징
};

// 특징값과 Y·Θ 미분
double eval_feature(const FeatureModel &m, double y_mm, double t_deg, double *dv_dy = nullptr, double *dv_dt = nullptr);

// (s, d) → (Y, Θ). 두 식이 Y·Θ 를 구별하지 못하면(행렬식 0) false
bool invert_pose_model(const PoseModel &model, double s_px, double d_px, double &y_mm, double &t_deg);

// 표본으로 계수 학습. theta_order_* ≥ 1. 표본 수가 계수보다 적으면 false
bool fit_pose_model(const std::vector<FeatureSample> &samples, int theta_order_s, int theta_order_d, PoseModel &model);

// 여러 촬영 세트로 학습 : 기울기(a, b)는 공통, 상수항 c 는 세트마다 따로 (촬영 날마다 다른 영점 차이를 흡수)
//  balance = true 면 세트마다 1/장수 가중 (장수가 많은 세트가 기울기를 끌고 가지 않게)
//  model.c = 첫 세트의 상수항 (실제 사용 때는 영점(zero)을 빼므로 어느 세트를 기준으로 해도 결과는 같음)
//  set_c_s, set_c_d : 세트별 상수항 (진단용, nullptr 가능). σ = 세트 상수를 쓴 잔차 RMS
bool fit_pose_model_sets(const std::vector<std::vector<FeatureSample>> &sets, int theta_order_s, int theta_order_d, bool balance,
                         PoseModel &model, std::vector<double> *set_c_s = nullptr, std::vector<double> *set_c_d = nullptr);

// 설정 노드(model: {s: {...}, d: {...}}) 읽기 — 이전 1차식 형식(b_px_per_deg)도 허용
bool read_pose_model(const cv::FileNode &node, PoseModel &model);

// 모델 파일 저장 (학습 출처를 함께 기록). note = 자유 문장
bool save_pose_model(const std::string &path, const PoseModel &model, const std::string &note, int n_samples);

// 모델 파일 읽기 (최상위에 model: 노드)
bool load_pose_model(const std::string &path, PoseModel &model);

} // namespace ksh

#endif
