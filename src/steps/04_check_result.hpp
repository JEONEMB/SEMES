#pragma once
// 4단계: 값 일치 여부와 학습 범위 점검
// 비교값: 실제 이동량 - 예상 이동량 - 영점 설정 때의 차이
// Warning 또는 Failed 결정: alignment.cpp
#include "common/reference.hpp"
#include "steps/03_compute_yt.hpp"   // kValueCount, value_index_of
#include <string>
#include <vector>

namespace alignment {

// 나사 조합별 점검식. 파일: checks/screws_XXXX.tsv
struct AgreementCheck {
    std::string screws;                 // 나사 조합 (계산식 조합과 같은 이름)
    double limit = 0;                   // 값 불일치 판정 한계
    struct Row {
        int value_index = 0;            // [0] QR, [1~4] 나사 1~4
        double c[6] = {};               // 예상값 = c0 + c1*Y + c2*T + c3*Y*Y + c4*Y*T + c5*T*T
        double usual_spread = 1;        // 정상 사진에서 (측정값 - 예상값) 이 흔들리는 정도 (px)
    };
    std::vector<Row> rows;
};

struct AgreementResult {
    bool checked = false;      // 이 조합의 점검식이 있어서 점검했는지
    double value = 0;          // 점검값 = 가장 크게 어긋난 값의 (어긋난 양 ÷ usual_spread)
    double limit = 0;
    int worst_value_index = -1;   // 가장 크게 어긋난 값 ([0] QR, [1~4] 나사)
    // agrees: 값 일치 여부 확인
    //   입력: 없음 (현재 점검 결과 사용)
    //   출력: 점검 미실시 또는 한계 이내이면 true
    bool agrees() const { return !checked || value <= limit; }
};

// load_agreement_checks: 조합별 점검식 읽기
//   입력: data_folder (모델 폴더)
//   출력: 성공 여부, checks, error. 열 수 없는 파일은 생략
bool load_agreement_checks(const std::string& data_folder, std::vector<AgreementCheck>& checks, std::string& error);

// find_agreement_check: 나사 조합의 점검식 조회
//   입력: checks (점검식 목록), screws (조합 이름)
//   출력: 해당 조합의 이상 여부 확인식. 없으면 nullptr(없음)
const AgreementCheck* find_agreement_check(const std::vector<AgreementCheck>& checks, const std::string& screws);

// expected_differences: 측정값과 예상값의 차이 계산
//   입력: check (이상 여부 확인식), values (QR과 나사 이동량), y_mm와 t_deg (영점 차감 전)
//   출력: differences (측정값 - 예상값)
void expected_differences(const AgreementCheck& check, const double values[kValueCount], double y_mm, double t_deg, double differences[kValueCount]);

// check_values_agree: 값 일치 점검
//   입력: check (이상 여부 확인식), values (QR과 나사 이동량), 영점 차감 전 Y와 T, baseline (기준에서 실제값과 예상값의 차이 5개, 없으면 모두 0)
//   출력: AgreementResult (점검값, 한계, 최대 불일치 항목)
AgreementResult check_values_agree(const AgreementCheck& check, const double values[kValueCount], double y_mm, double t_deg, const double baseline[kValueCount]);

// check_in_trained_range: 학습 범위 확인
//   입력: reference, y_mm, t_deg (영점 차감 전)
//   출력: Y, T 모두 범위 안이면 true. 양 끝 포함
bool check_in_trained_range(const Reference& reference, double y_mm, double t_deg);

}   // alignment
