#pragma once
// 3단계: QR과 나사 이동량(px) -> Y(mm), T(도)
// 계산: 각 측정값의 크기 조정 -> 항목별 곱셈 -> 합계에 기본값 추가
// 계산식: <모델 폴더>/coefficients/의 나사 조합별 파일
#include <array>
#include <string>
#include <vector>

namespace alignment {

// 가로 이동량 5개(px): [0] QR, [1~4] 나사
const int kValueCount = 5;
const char* const kValueNames[kValueCount] = {"qr", "screw1", "screw2", "screw3", "screw4"};   // 계산식, 점검식 파일에 쓰는 이름

// value_index_of: 측정값 이름 조회
//   입력: name (qr, screw1~screw4)
//   출력: 값 번호 0~4. 미등록 이름은 -1
int value_index_of(const std::string& name);

struct CoefficientSet {
    std::string screws;                    // 쓰는 나사 번호 (예: "1234", "13", "none")
    std::array<bool, 4> uses_screw{};      // 나사별 사용 여부
    bool precise = false;                  // 좌우 나사 확보 여부. false이면 대략값
    double intercept_y = 0, intercept_t = 0;   // 각 항목의 합계에 마지막으로 더할 기본값
    // 값 순서: [0] QR, [1~4] 나사 1~4. 쓰지 않는 값은 weight가 0.
    // mean: 평균, scale: 퍼진 정도, weight_y와 weight_t: Y와 T 계산에 곱할 값
    double mean[kValueCount] = {}, scale[kValueCount] = {1, 1, 1, 1, 1}, weight_y[kValueCount] = {}, weight_t[kValueCount] = {};
};

struct YT {
    double y_mm = 0;    // Y 이동량(mm)
    double t_deg = 0;   // T 회전량(도)
};

// load_coefficient_sets: 나사 16조합의 계산식 읽기
//   입력: data_folder (모델 폴더)
//   출력: 성공 여부, sets (계산식 목록), error (실패 사유)
bool load_coefficient_sets(const std::string& data_folder, std::vector<CoefficientSet>& sets, std::string& error);

// choose_coefficient_set: 검출 나사에 맞는 계산식 선택
//   입력: sets (계산식 목록), found (나사별 검출 여부)
//   출력: 해당 조합의 계산식. 없으면 nullptr(없음)
const CoefficientSet* choose_coefficient_set(const std::vector<CoefficientSet>& sets, const std::array<bool, 4>& found);

// compute_yt: 이동량에서 Y, T 계산
//   입력: set (조합 계산식), values (QR, 나사 1~4의 가로 이동 px)
//   출력: YT (영점 차감 전 mm와 도)
YT compute_yt(const CoefficientSet& set, const double values[kValueCount]);

}   // alignment
