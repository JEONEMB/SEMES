#pragma once
// 계수와 점검식의 학습. data/model/coefficients/, data/model/checks/ 의 파일을 만든 식과 같다.
// 학습한 결과는 측정 코드의 자료형(alignment::CoefficientSet, alignment::AgreementCheck) 그대로이고,
// 예측도 측정 코드의 함수(alignment::compute_yt)를 쓴다. 그래서 학습, 평가, 실사용의 계산이 같다.
//
// 계수 (Y, T 를 계산하는 선형식)
//   1. 값마다 평균(mean)과 표준편차(scale)로 표준화한다
//   2. Y = 절편 + Σ weight_Y × (값 − mean) / scale,  T 도 같은 식
//   3. 계수는 두 축을 함께 보는 Huber 가중 최소제곱으로 6번 고쳐 구한다 (가끔 섞인 이상한 사진에 끌려가지 않게)
//      사진마다 r = max(|Y 오차| / 통과 기준 Y, |T 오차| / 통과 기준 T), 가중치 = min(1, 1 / r)
//   4. 절편을 뺀 계수에 아주 작은 ridge (1e-7 × 사진 수) 를 더해 계산을 안정시킨다
//
// 점검식 (값마다 "이 Y, T 라면 이 값이어야 한다")
//   기대값 = c1 + cY·Y + cT·T + cYY·Y² + cYT·Y·T + cTT·T²   (Y, T = 그 조합의 계수로 낸 값)
//   scale  = 1.4826 × (잔차 − 잔차 가운데값)의 절댓값 가운데값   (정상 사진에서 흔들리는 정도)
//   기준값 = 1.5 × (학습 사진 점검값의 99.9% 지점)
#include "steps/03_compute_yt.hpp"
#include "steps/04_check_result.hpp"
#include <array>
#include <filesystem>
#include <string>
#include <vector>

namespace tools {

using alignment::kValueCount;              // 값 5개: [0] QR, [1~4] 나사 1~4 가로 이동 (px)
using alignment::kValueNames;
using Values = std::array<double, kValueCount>;

const double kToleranceYmm = 0.045;        // 통과 기준 (학습, 평가, 채점이 모두 이 값을 쓴다)
const double kToleranceTdeg = 0.01;
bool passes(double error_y_mm, double error_t_deg);

extern const char* const kScrewSetNames[16];     // 계수 파일을 만드는 나사 조합 16가지 ("1234", "134", …, "none")
extern const char* const kCheckSetNames[11];     // 점검식을 만드는 11가지 (나사 2개 이상)
bool has_agreement_check(const std::string& screws);

// 나사 조합 screws 의 계수를 학습한다. values[i] 와 정답 (truth_y[i], truth_t[i]) 가 사진 한 장.
alignment::CoefficientSet fit_coefficient_set(const std::string& screws, const std::vector<Values>& values,
                                              const std::vector<double>& truth_y, const std::vector<double>& truth_t);

// 계수로 Y, T 를 낸다 (측정 코드의 compute_yt 그대로).
alignment::YT predict(const alignment::CoefficientSet& set, const Values& values);

// 그 조합의 점검식을 학습한다.
alignment::AgreementCheck fit_agreement_check(const alignment::CoefficientSet& set, const std::vector<Values>& values);

// 측정 코드가 읽는 형식으로 쓴다 (photo_count 는 파일 머리말에 적는 학습 사진 수).
void write_coefficient_file(const std::filesystem::path& path, const alignment::CoefficientSet& set, int photo_count);
void write_agreement_file(const std::filesystem::path& path, const alignment::AgreementCheck& check, int photo_count);

}   // namespace tools
