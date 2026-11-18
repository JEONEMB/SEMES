#pragma once
// 사용할 나사별로 이동량 계산식과 이상 여부 확인식 생성
// 학습: 측정값과 정답으로 계산식의 숫자 결정. 큰 오차의 영향 축소
// 확인식: 계산한 자세에서 예상되는 QR과 나사 이동량
// 이상 판정 기준: 학습 점검값의 상위 0.1% 경계 × 1.5
#include "steps/03_compute_yt.hpp"
#include "steps/04_check_result.hpp"
#include <array>
#include <filesystem>
#include <string>
#include <vector>

namespace tools {

using alignment::kValueCount;              // 측정값 5개(px): QR, 나사 1~4
using alignment::kValueNames;
using Values = std::array<double, kValueCount>;

const double kToleranceYmm = 0.045;        // 학습과 평가 공통 허용오차
const double kToleranceTdeg = 0.01;

// passes: 두 방향의 오차가 허용 범위인지 확인
//   입력: error_y_mm (Y 오차), error_t_deg (T 오차)
//   출력: 두 오차 모두 허용 범위 안이면 true
bool passes(double error_y_mm, double error_t_deg);

extern const char* const kScrewSetNames[16];     // 계산식 파일을 만드는 나사 조합 16가지 ("1234", "134", ..., "none")
extern const char* const kCheckSetNames[11];     // 점검식을 만드는 11가지 (나사 2개 이상)
// has_agreement_check: 이상 여부 확인 대상인지 조회
//   입력: screws (사용할 나사 번호)
//   출력: 대상 조합이면 true
bool has_agreement_check(const std::string& screws);

// fit_coefficient_set: 사진별 정답에 맞는 이동량 계산식 생성
//   입력: screws (사용 나사), values (측정값), truth_y와 truth_t (사진별 정답)
//   출력: CoefficientSet (Y와 T 계산용 숫자)
//   조건: 사진마다 측정값과 정답 한 쌍, 빈 자료와 NaN, 무한대 제외
alignment::CoefficientSet fit_coefficient_set(const std::string& screws, const std::vector<Values>& values,
                                              const std::vector<double>& truth_y, const std::vector<double>& truth_t);

// predict: 사진 속 이동량으로 판의 이동량 계산
//   입력: set (이동량 계산식), values (QR과 나사 이동량)
//   출력: YT (Y 이동량 mm, T 회전량 도)
alignment::YT predict(const alignment::CoefficientSet& set, const Values& values);

// fit_agreement_check: QR과 나사 측정값의 이상 여부 확인식 생성
//   입력: set (이동량 계산식), values (학습 사진의 측정값)
//   출력: AgreementCheck (예상 이동량 계산식과 이상 판정 기준)
alignment::AgreementCheck fit_agreement_check(const alignment::CoefficientSet& set, const std::vector<Values>& values);

// write_coefficient_file: 이동량 계산용 숫자 저장
//   입력: path (저장 경로), set (계산식), photo_count (학습 사진 수)
//   출력: 없음 (탭으로 열을 나눈 TSV 파일 저장)
void write_coefficient_file(const std::filesystem::path& path, const alignment::CoefficientSet& set, int photo_count);

// write_agreement_file: 이상 여부 확인용 숫자 저장
//   입력: path (저장 경로), check (확인식), photo_count (학습 사진 수)
//   출력: 없음 (탭으로 열을 나눈 TSV 파일 저장)
void write_agreement_file(const std::filesystem::path& path, const alignment::AgreementCheck& check, int photo_count);

}   // tools
