// =====================================================================================
// yt_model.hpp — 특징 → (Y, Θ) 추정 모델과 평가 도구 (트랙 A·B 공통)
//
//  모델 : 특징 k 마다  f_k = c_k + a_k·Y + b_k·Θ + q_k·Θ²   (1차 세트로 최소제곱 학습)
//  역산 : 측정 f 에서 (Y, Θ) 를 가중 Gauss–Newton 으로 풂 (가중 = 1 / 학습 잔차 분산)
//  평가 : 한 장 빼고 학습(LOO), 0.01° 계단, 다른 날(2차) 교차, teaching 보정 이동으로 현장 보정
// =====================================================================================
#ifndef YT_MODEL_HPP
#define YT_MODEL_HPP

#include "yt_common.hpp"
#include <functional>

namespace yt {

struct Obs {              // 사진 한 장의 특징
    std::string stem;
    double y = 0, t = 0;  // 라벨 (정답)
    int set = 1, shot = 0;
    std::vector<double> f;
};

class PoseModel {
public:
    // 학습 : obs 의 특징 (NaN 이 있는 사진은 제외)
    bool fit(const std::vector<Obs>& obs);
    // 역산 : f → (Y mm, Θ °)
    bool invert(const std::vector<double>& f, double& y, double& t) const;
    std::vector<double> predict(double y, double t) const;
    int nFeat() const { return int(coef_.size()); }
    // 특징별 계수 (c, a, b, q) 와 잔차 σ
    const std::vector<std::vector<double>>& coef() const { return coef_; }
    const std::vector<double>& sigma() const { return sig_; }
    // 현장 보정 : 측정 특징을 f' = g·f + h 로 고쳐 쓰도록 특징별 (g, h) 를 known (라벨 아는) 사진으로 맞춤
    void calibrate(const std::vector<Obs>& known);
    std::vector<double> correct(const std::vector<double>& f) const;
private:
    std::vector<std::vector<double>> coef_;
    std::vector<double> sig_;
    std::vector<double> gain_, off_;
};

struct Est { std::string stem; double y = 0, t = 0, ye = 0, te = 0; int set = 1, shot = 0; bool ok = false; };

// 1차 세트 LOO 추정
std::vector<Est> leaveOneOut(const std::vector<Obs>& set1);
// 학습된 모델로 추정 (보정 포함)
std::vector<Est> estimateAll(const PoseModel& m, const std::vector<Obs>& obs);

// 요약 : Θ 오차(°), Y 오차(µm) 통계 (조건 함수로 부분 집합)
struct Summary { Stat t, y; int both = 0, n = 0; };
Summary summarize(const std::vector<Est>& e, const std::function<bool(const Est&)>& sel);
std::string summaryRow(const Summary& s);

// 0.01° 계단 : 1차 Θ 스윕 이웃 쌍 (차가 정확히 0.01°) 의 추정 ΔΘ
Stat stepTest(const std::vector<Est>& e, double step, bool theta, std::vector<double>* out = nullptr);

void writeEst(const std::string& path, const std::vector<Est>& e);

} // namespace yt

#endif
