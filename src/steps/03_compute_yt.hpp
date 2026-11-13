#pragma once
// 3단계: QR 이동과 나사 이동으로 Y(mm), T(도) 를 계산한다.
//
// 원리
//   - 판이 회전(T)하면 QR 과 나사가 화면에서 거의 같이 움직인다 (1도에 약 80 px)
//   - 판이 Y 로 움직이면 QR 과 나사가 다르게 움직인다 (1 mm 에 QR 약 14 px, 나사 약 11 px)
//   그래서 "전체가 얼마나 움직였나" 로 T 를, "QR 과 나사가 얼마나 다르게 움직였나" 로 Y 를 안다.
//
// 계산식 (계수는 사진 702장으로 미리 학습해 data/coefficients/ 에 있다)
//   Y = intercept_Y + Σ weight_Y × (값 − mean) / scale        T 도 같은 식
//
// 찾은 나사 조합마다 계수가 따로 있다 (screws_1234, screws_134, …, screws_none).
#include <array>
#include <string>
#include <vector>

namespace alignment {

struct CoefficientSet {
    std::string screws;                    // 쓰는 나사 번호 (예: "1234", "13", "none")
    std::array<bool, 4> uses_screw{};      // 나사마다 쓰는지
    bool precise = false;                  // 왼쪽·오른쪽 나사가 모두 있어 정밀한 조합인지 (아니면 대략값)
    double intercept_y = 0, intercept_t = 0;
    // 값 순서: [0] QR, [1~4] 나사 1~4. 쓰지 않는 값은 weight 가 0.
    double mean[5] = {}, scale[5] = {1, 1, 1, 1, 1}, weight_y[5] = {}, weight_t[5] = {};
};

struct YT {
    double y_mm = 0;
    double t_deg = 0;
};

// data/coefficients/ 의 모든 조합을 읽는다. 실패하면 false 와 이유.
bool load_coefficient_sets(const std::string& data_folder, std::vector<CoefficientSet>& sets, std::string& error);

// 찾은 나사(found)에 맞는 조합을 고른다. 없으면 nullptr.
const CoefficientSet* choose_coefficient_set(const std::vector<CoefficientSet>& sets, const std::array<bool, 4>& found);

// values: [0] QR 가로 이동, [1~4] 나사 1~4 가로 이동 (px)
YT compute_yt(const CoefficientSet& set, const double values[5]);

}   // namespace alignment
