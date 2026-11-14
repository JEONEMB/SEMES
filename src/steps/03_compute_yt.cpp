#include "steps/03_compute_yt.hpp"
#include <fstream>
#include <sstream>

namespace alignment {

namespace {

// 가능한 나사 조합 16가지 (없음 포함)
const char* const kSetNames[] = {"1234", "234", "134", "124", "123", "12", "13", "14", "23", "24", "34", "1", "2", "3", "4", "none"};

// 값 이름 → 값 번호. 모르는 이름이면 −1.
int value_index(const std::string& name) {
    if (name == "qr") return 0;
    if (name.size() == 6 && name.compare(0, 5, "screw") == 0 && name[5] >= '1' && name[5] <= '4') return name[5] - '0';
    return -1;
}

bool load_one_set(const std::string& path, const std::string& screws, CoefficientSet& set, std::string& error) {
    std::ifstream file(path);
    if (!file) { error = "계수 파일을 읽지 못했습니다: " + path; return false; }
    set = CoefficientSet();
    set.screws = screws;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream cells(line);
        std::string name;
        double mean, scale, weight_y, weight_t;
        cells >> name;
        if (name == "name") continue;   // 머리줄
        if (!(cells >> mean >> scale >> weight_y >> weight_t)) { error = "계수 파일의 줄을 읽지 못했습니다: " + path; return false; }
        if (name == "intercept") { set.intercept_y = weight_y; set.intercept_t = weight_t; continue; }
        const int index = value_index(name);
        if (index < 0 || !(scale > 0)) { error = "계수 파일에 모르는 값이 있습니다: " + name + " (" + path + ")"; return false; }
        set.mean[index] = mean;
        set.scale[index] = scale;
        set.weight_y[index] = weight_y;
        set.weight_t[index] = weight_t;
        if (index >= 1) set.uses_screw[index - 1] = true;
    }
    const bool has_left = set.uses_screw[0] || set.uses_screw[2];    // 나사 1, 3 이 왼쪽
    const bool has_right = set.uses_screw[1] || set.uses_screw[3];   // 나사 2, 4 가 오른쪽
    set.precise = has_left && has_right;
    return true;
}

}   // namespace

bool load_coefficient_sets(const std::string& data_folder, std::vector<CoefficientSet>& sets, std::string& error) {
    sets.clear();
    for (const char* name : kSetNames) {
        CoefficientSet set;
        if (!load_one_set(data_folder + "/coefficients/screws_" + name + ".tsv", name, set, error)) return false;
        sets.push_back(set);
    }
    return true;
}

const CoefficientSet* choose_coefficient_set(const std::vector<CoefficientSet>& sets, const std::array<bool, 4>& found) {
    for (const CoefficientSet& set : sets)
        if (set.uses_screw == found) return &set;
    return nullptr;
}

YT compute_yt(const CoefficientSet& set, const double values[5]) {
    YT result{set.intercept_y, set.intercept_t};
    for (int i = 0; i < 5; ++i) {
        if (set.weight_y[i] == 0 && set.weight_t[i] == 0) continue;   // 이 조합에서 쓰지 않는 값
        const double normalized = (values[i] - set.mean[i]) / set.scale[i];
        result.y_mm += set.weight_y[i] * normalized;
        result.t_deg += set.weight_t[i] * normalized;
    }
    return result;
}

}   // namespace alignment
