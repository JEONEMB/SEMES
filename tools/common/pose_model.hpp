#pragma once
// Y, T 를 계산하는 선형식의 학습과 예측. data/coefficients/ 의 계수를 만든 식과 같다.
//
//   1. 값마다 평균(mean)과 표준편차(scale)로 표준화한다
//   2. Y = 절편 + Σ weight_Y × (값 − mean) / scale,  T 도 같은 식
//   3. 계수는 두 축을 함께 보는 Huber 가중 최소제곱으로 6번 고쳐 구한다 (가끔 섞인 이상한 사진에 끌려가지 않게)
//      사진마다 r = max(|Y 오차| / 0.045, |T 오차| / 0.01), 가중치 = min(1, 1 / r)
//   4. 절편을 뺀 계수에 아주 작은 ridge (1e-7 × 사진 수) 를 더해 계산을 안정시킨다
#include <array>
#include <string>
#include <vector>

namespace tools {

const int kValueCount = 5;                 // [0] QR, [1~4] 나사 1~4 가로 이동 (px)
const double kToleranceYmm = 0.045;        // 통과 기준
const double kToleranceTdeg = 0.01;
extern const char* const kValueNames[kValueCount];   // "qr", "screw1" … "screw4"

using Values = std::array<double, kValueCount>;

bool passes(double error_y_mm, double error_t_deg);

struct PoseModel {
    std::vector<int> used;                 // 쓰는 값 번호 (예: {0, 1, 3, 4})
    std::vector<double> mean, scale;       // used 순서
    double intercept_y = 0, intercept_t = 0;
    std::vector<double> weight_y, weight_t;

    // values[i] 와 정답 (y, t) 로 학습한다.
    void fit(const std::vector<int>& used_values, const std::vector<Values>& values, const std::vector<double>& truth_y, const std::vector<double>& truth_t);
    void predict(const Values& values, double& y_mm, double& t_deg) const;
};

// 나사 조합 이름 ("1234", "134", "none") ↔ 쓰는 값 번호
std::vector<int> used_values_of(const std::string& screws);
extern const char* const kScrewSetNames[16];     // 계수 파일을 만드는 16가지
extern const char* const kCheckSetNames[11];     // 점검식을 만드는 11가지 (나사 2개 이상)

}   // namespace tools
