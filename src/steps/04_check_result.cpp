#include "steps/04_check_result.hpp"
#include <cmath>
#include <fstream>
#include <sstream>

namespace alignment {

namespace {

const char* const kCheckNames[] = {"1234", "234", "134", "124", "123", "12", "13", "14", "23", "24", "34"};

int value_index(const std::string& name) {
    if (name == "qr") return 0;
    if (name.size() == 6 && name.compare(0, 5, "screw") == 0 && name[5] >= '1' && name[5] <= '4') return name[5] - '0';
    return -1;
}

// 파일이 없으면 found = false 로 두고 true 를 돌려준다 (점검식이 없는 조합은 점검하지 않는다).
bool load_one_check(const std::string& path, const std::string& screws, AgreementCheck& check, bool& found, std::string& error) {
    std::ifstream file(path);
    found = bool(file);
    if (!found) return true;
    check = AgreementCheck();
    check.screws = screws;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream cells(line);
        std::string name;
        cells >> name;
        if (name == "name") continue;   // 머리줄
        if (name == "threshold") { cells >> check.limit; continue; }
        AgreementCheck::Row row;
        row.value_index = value_index(name);
        for (double& c : row.c) cells >> c;
        cells >> row.usual_spread;
        if (row.value_index < 0 || !cells || !(row.usual_spread > 0)) { error = "점검식 파일의 줄을 읽지 못했습니다: " + path; return false; }
        check.rows.push_back(row);
    }
    if (check.rows.empty() || !(check.limit > 0)) { error = "점검식 파일이 비었습니다: " + path; return false; }
    return true;
}

}   // namespace

bool load_agreement_checks(const std::string& data_folder, std::vector<AgreementCheck>& checks, std::string& error) {
    checks.clear();
    for (const char* name : kCheckNames) {
        AgreementCheck check;
        bool found = false;
        if (!load_one_check(data_folder + "/checks/screws_" + name + ".tsv", name, check, found, error)) return false;
        if (found) checks.push_back(check);
    }
    return true;
}

const AgreementCheck* find_agreement_check(const std::vector<AgreementCheck>& checks, const std::string& screws) {
    for (const AgreementCheck& check : checks)
        if (check.screws == screws) return &check;
    return nullptr;
}

void expected_differences(const AgreementCheck& check, const double values[5], double y_mm, double t_deg, double differences[5]) {
    for (int i = 0; i < 5; ++i) differences[i] = 0;
    for (const AgreementCheck::Row& row : check.rows) {
        const double expected = row.c[0] + row.c[1] * y_mm + row.c[2] * t_deg + row.c[3] * y_mm * y_mm + row.c[4] * y_mm * t_deg + row.c[5] * t_deg * t_deg;
        differences[row.value_index] = values[row.value_index] - expected;
    }
}

AgreementResult check_values_agree(const AgreementCheck& check, const double values[5], double y_mm, double t_deg, const double baseline[5]) {
    double differences[5];
    expected_differences(check, values, y_mm, t_deg, differences);
    AgreementResult result;
    result.checked = true;
    result.limit = check.limit;
    for (const AgreementCheck::Row& row : check.rows) {
        const double off = std::abs(differences[row.value_index] - baseline[row.value_index]) / row.usual_spread;
        if (off > result.value) { result.value = off; result.worst_value_index = row.value_index; }
    }
    return result;
}

bool check_in_trained_range(const Reference& reference, double y_mm, double t_deg) {
    return y_mm >= reference.trained_y_mm[0] && y_mm <= reference.trained_y_mm[1] && t_deg >= reference.trained_t_deg[0] && t_deg <= reference.trained_t_deg[1];
}

}   // namespace alignment
