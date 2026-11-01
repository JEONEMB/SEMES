// =====================================================================================
// ksh/evaluation.hpp — 정확도 평가 : 오차, 허용 범위 통과, 점수 J, 블록 교차검증
//
//  J = √ mean[ (eY / tolY)² + (eΘ / tolΘ)² ]   (항마다 cap 에서 자름, 측정 실패 = 두 항 모두 cap)
//      1 이하면 평균 오차가 허용 범위 안. 작을수록 좋음
//  블록 교차검증 : 라벨 순으로 이웃한 사진을 통째로 묶어 빼고 예측 (이웃 사진이 학습에 섞여 점수가 부풀지 않도록)
// =====================================================================================
#ifndef KSH_EVALUATION_HPP
#define KSH_EVALUATION_HPP

#include "ksh/align_config.hpp"
#include "ksh/pose_model.hpp"

#include <string>
#include <vector>

namespace ksh {

struct PosePrediction {
    std::string stem, group;
    double y_mm = 0.0, t_deg = 0.0;           // 정답
    double y_est_mm = 0.0, t_est_deg = 0.0;   // 추정
    bool ok = false;                          // 측정·추정 성공
    double elapsed_ms = 0.0;
};

double error_y_um(const PosePrediction &p);
double error_t_deg(const PosePrediction &p);
bool within_tolerance(const PosePrediction &p, const Tolerance &tol);

// J 점수 (위 정의)
double score_j(const std::vector<PosePrediction> &preds, const Tolerance &tol, double cap);

struct GroupSummary {
    std::string group;
    int n = 0, n_ok = 0;
    int pass = 0;        // 프로젝트 허용 범위 안
    int pass_spec = 0;   // 과제 허용 범위 안
    double rms_y_um = 0, max_y_um = 0, rms_t_deg = 0, max_t_deg = 0;
    double mean_ms = 0;
    double j = 0;
};

// group 이름별 요약 (처음 나온 순서). all_label 이 비어 있지 않으면 전체 줄을 맨 끝에 추가
std::vector<GroupSummary> summarize_groups(const std::vector<PosePrediction> &preds, const Tolerance &tol, const Tolerance &spec,
                                           double cap, const std::string &all_label = "");

// 블록 번호 : Y 단독(Θ = 0) 사진은 Y 순으로 n_y 묶음, Θ 단독 사진은 Θ 순으로 n_t 묶음 (번호 n_y.. )
std::vector<int> block_folds(const std::vector<FeatureSample> &samples, int n_y_blocks, int n_t_blocks);

// 묶음마다 나머지로 학습 → 그 묶음 예측. group 은 "Y 단독" / "Θ 단독"
std::vector<PosePrediction> cross_validate(const std::vector<FeatureSample> &samples, const std::vector<int> &folds,
                                           int theta_order_s, int theta_order_d);

// 요약 표 (마크다운 — 콘솔 출력과 .md 저장에 같이 씀)
std::string summary_table(const std::vector<GroupSummary> &groups, const Tolerance &tol, const Tolerance &spec);

// 사진별 결과 CSV
bool save_predictions_csv(const std::string &path, const std::vector<PosePrediction> &preds, const Tolerance &tol, const Tolerance &spec);

} // namespace ksh

#endif
