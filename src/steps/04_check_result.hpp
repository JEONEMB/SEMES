#pragma once
// 4단계: 결과를 믿을 만한지 점검한다. 점검에 걸려도 값은 그대로 내고 경고만 붙인다.
//
// (가) 값 일치 점검
//   판의 자세는 Y, T 두 개뿐인데 잰 값은 여러 개(QR 1 + 나사 2~4)다.
//   정상이면 모든 값이 한 자세로 함께 설명된다. 나사 하나가 가려져 엉뚱하게 재지면 그 값만 어긋난다.
//   값마다 "이 Y, T 라면 이 값이어야 한다"(기대값)를 미리 학습한 2차식으로 구해 실제 값과 비교한다.
//
// (나) 범위 점검
//   계산한 Y, T 가 계수를 학습한 범위 밖이면 오차가 커질 수 있다고 알린다.
#include "common/reference.hpp"
#include <string>
#include <vector>

namespace alignment {

// 나사 조합 하나의 값 일치 점검식 (data/checks/screws_XXXX.tsv)
struct AgreementCheck {
    std::string screws;                 // 나사 조합 (계수 조합과 같은 이름)
    double limit = 0;                   // 점검값이 이보다 크면 값들이 서로 맞지 않는 것으로 본다
    struct Row {
        int value_index = 0;            // [0] QR, [1~4] 나사 1~4
        double c[6] = {};               // 기대값 = c0 + c1·Y + c2·T + c3·Y² + c4·Y·T + c5·T²
        double usual_spread = 1;        // 정상 사진에서 (측정값 − 기대값) 이 흔들리는 정도 (px)
    };
    std::vector<Row> rows;
};

struct AgreementResult {
    bool checked = false;      // 이 조합의 점검식이 있어서 점검했는지
    double value = 0;          // 점검값 = 가장 크게 어긋난 값의 (어긋난 양 ÷ usual_spread)
    double limit = 0;
    int worst_value_index = -1;   // 가장 크게 어긋난 값 ([0] QR, [1~4] 나사)
    bool agrees() const { return !checked || value <= limit; }
};

// data/checks/ 에 있는 점검식을 모두 읽는다 (없는 조합은 건너뛴다).
bool load_agreement_checks(const std::string& data_folder, std::vector<AgreementCheck>& checks, std::string& error);

// 조합 이름으로 점검식을 찾는다. 없으면 nullptr.
const AgreementCheck* find_agreement_check(const std::vector<AgreementCheck>& checks, const std::string& screws);

// 값마다 (측정값 − 기대값). y_mm, t_deg 는 영점을 빼기 전 계산값.
void expected_differences(const AgreementCheck& check, const double values[5], double y_mm, double t_deg, double differences[5]);

// baseline: 영점을 잡을 때의 (측정값 − 기대값). 그날 화면 상태를 0 으로 삼기 위해 뺀다. 없으면 모두 0.
AgreementResult check_values_agree(const AgreementCheck& check, const double values[5], double y_mm, double t_deg, const double baseline[5]);

// Y, T 가 학습 범위 안인지.
bool check_in_trained_range(const Reference& reference, double y_mm, double t_deg);

}   // namespace alignment
