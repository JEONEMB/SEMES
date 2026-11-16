// 3단계: 검출한 나사 조합의 계산식으로 Y, T 계산
#include "steps/03_compute_yt.hpp"
#include <fstream>
#include <filesystem>
#include <set>
#include <cmath>
#include <stdexcept>
#include <sstream>

namespace alignment {

namespace {

// 가능한 나사 조합 16가지 (없음 포함)
const char* const kSetNames[] = {"1234", "234", "134", "124", "123", "12", "13", "14", "23", "24", "34", "1", "2", "3", "4", "none"};

// load_one_set: 조합 계산식 파일 읽기
//   입력: path (파일 경로), screws (나사 조합)
//   출력: 성공 여부, set (읽은 계산식), error (실패 사유)
bool load_one_set(const std::string& path, const std::string& screws, CoefficientSet& set, std::string& error) {
    std::ifstream file(std::filesystem::u8path(path));
    if (!file) { error = "계수 파일을 읽지 못했습니다: " + path; return false; }
    set = CoefficientSet();
    set.screws = screws;
    std::set<std::string> seen;
    std::string line;
    int line_number = 0;
    while (std::getline(file, line)) {
        ++line_number;
        if (line.empty() || line[0] == '#') continue;
        std::istringstream cells(line);
        std::string name;
        double mean, scale, weight_y, weight_t;
        cells >> name;
        if (!seen.insert(name).second) { error = "중복된 모델 행: " + path + ":" + std::to_string(line_number); return false; }
        if (name == "name") continue;   // 열 이름 행
        if (!(cells >> mean >> scale >> weight_y >> weight_t)) { error = "계수 파일의 줄을 읽지 못했습니다: " + path; return false; }
        std::string extra;
        if ((cells >> extra) || !std::isfinite(mean) || !std::isfinite(scale) || !std::isfinite(weight_y) || !std::isfinite(weight_t) || scale <= 0) { error = "유효하지 않은 계수: " + path + ":" + std::to_string(line_number); return false; }
        if (name == "intercept") { set.intercept_y = weight_y; set.intercept_t = weight_t; continue; }
        const int index = value_index_of(name);
        if (index < 0 || !(scale > 0)) { error = "계수 파일에 모르는 값이 있습니다: " + name + " (" + path + ")"; return false; }
        set.mean[index] = mean;
        set.scale[index] = scale;
        set.weight_y[index] = weight_y;
        set.weight_t[index] = weight_t;
        if (index >= 1) set.uses_screw[screw_of_value(index)] = true;
    }
    std::set<std::string> expected{"name", "intercept", "qr"};
    if (screws != "none") for (char c : screws) { expected.insert("screw" + std::string(1,c)); expected.insert("screw" + std::string(1,c) + "_area"); }
    if (seen != expected) { error = "계수 필수 행 누락 또는 조합 불일치: " + path; return false; }
    const bool has_left = set.uses_screw[0] || set.uses_screw[2];    // 나사 1, 3 이 왼쪽
    const bool has_right = set.uses_screw[1] || set.uses_screw[3];   // 나사 2, 4 가 오른쪽
    set.precise = has_left && has_right;
    return true;
}

}   // 파일 내부

// value_index_of: 측정값 이름 조회
//   입력: name (qr, screw1~screw4)
//   출력: 값 번호 0~4. 미등록 이름은 -1
int value_index_of(const std::string& name) {
    for (int i = 0; i < kValueCount; ++i)
        if (name == kValueNames[i]) return i;
    return -1;
}

// load_coefficient_sets: 나사 16조합의 계산식 읽기
//   입력: data_folder (모델 폴더)
//   출력: 성공 여부, sets (계산식 목록), error (실패 사유)
bool load_coefficient_sets(const std::string& data_folder, std::vector<CoefficientSet>& sets, std::string& error) {
    sets.clear();
    for (const char* name : kSetNames) {
        CoefficientSet set;
        if (!load_one_set(data_folder + "/coefficients/screws_" + name + ".tsv", name, set, error)) return false;
        sets.push_back(set);
    }
    return true;
}

// choose_coefficient_set: 검출 나사에 맞는 계산식 선택
//   입력: sets (계산식 목록), found (나사별 검출 여부)
//   출력: 해당 조합의 계산식. 없으면 nullptr(없음)
const CoefficientSet* choose_coefficient_set(const std::vector<CoefficientSet>& sets, const std::array<bool, 4>& found) {
    for (const CoefficientSet& set : sets)
        if (set.uses_screw == found) return &set;
    return nullptr;
}

// compute_yt: 이동량에서 Y, T 계산
//   입력: set (조합 계산식), values (QR, 나사 1~4 테두리, 나사 1~4 둘레 영역의 가로 이동 px)
//   출력: YT (영점 차감 전 mm와 도)
YT compute_yt(const CoefficientSet& set, const double values[kValueCount]) {
    YT result{set.intercept_y, set.intercept_t};
    for (int i = 0; i < kValueCount; ++i) {
        if (set.weight_y[i] == 0 && set.weight_t[i] == 0) continue;   // 이 조합에서 쓰지 않는 값
        const double normalized = (values[i] - set.mean[i]) / set.scale[i];
        result.y_mm += set.weight_y[i] * normalized;
        result.t_deg += set.weight_t[i] * normalized;
    }
    return result;
}

}   // alignment
