// 4단계: 예상 이동량과 가장 크게 다른 항목으로 이상 여부 확인
#include "steps/04_check_result.hpp"
#include <cmath>
#include <fstream>
#include <sstream>

namespace alignment {

namespace {

const char* const kCheckNames[] = {"1234", "234", "134", "124", "123", "12", "13", "14", "23", "24", "34"};

// load_one_check: 조합 점검식 파일 읽기
//   입력: path (파일 경로), screws (나사 조합)
//   출력: 성공 여부, check, found, error. 파일 열기 실패는 found = false
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
        if (name == "name") continue;   // 열 이름 행
        if (name == "threshold") { cells >> check.limit; continue; }
        AgreementCheck::Row row;
        row.value_index = value_index_of(name);
        for (double& c : row.c) cells >> c;
        cells >> row.usual_spread;
        if (row.value_index < 0 || !cells || !(row.usual_spread > 0)) { error = "점검식 파일의 줄을 읽지 못했습니다: " + path; return false; }
        check.rows.push_back(row);
    }
    if (check.rows.empty() || !(check.limit > 0)) { error = "점검식 파일이 비었습니다: " + path; return false; }
    return true;
}

}   // 파일 내부

// load_agreement_checks: 조합별 점검식 읽기
//   입력: data_folder (모델 폴더)
//   출력: 성공 여부, checks, error. 열 수 없는 파일은 생략
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

// find_agreement_check: 나사 조합의 점검식 조회
//   입력: checks (점검식 목록), screws (조합 이름)
//   출력: 해당 조합의 이상 여부 확인식. 없으면 nullptr(없음)
const AgreementCheck* find_agreement_check(const std::vector<AgreementCheck>& checks, const std::string& screws) {
    for (const AgreementCheck& check : checks)
        if (check.screws == screws) return &check;
    return nullptr;
}

// expected_differences: 측정값과 예상값의 차이 계산
//   입력: check (이상 여부 확인식), values (QR과 나사 이동량), y_mm와 t_deg (영점 차감 전)
//   출력: differences (측정값 - 예상값)
void expected_differences(const AgreementCheck& check, const double values[kValueCount], double y_mm, double t_deg, double differences[kValueCount]) {
    for (int i = 0; i < kValueCount; ++i) differences[i] = 0;
    for (const AgreementCheck::Row& row : check.rows) {
        const double expected = row.c[0] + row.c[1] * y_mm + row.c[2] * t_deg + row.c[3] * y_mm * y_mm + row.c[4] * y_mm * t_deg + row.c[5] * t_deg * t_deg;
        differences[row.value_index] = values[row.value_index] - expected;
    }
}

// check_values_agree: 값 일치 점검
//   입력: check (이상 여부 확인식), values (QR과 나사 이동량), 영점 차감 전 Y와 T, baseline (기준에서 실제값과 예상값의 차이 5개, 없으면 모두 0)
//   출력: AgreementResult (점검값, 한계, 최대 불일치 항목)
AgreementResult check_values_agree(const AgreementCheck& check, const double values[kValueCount], double y_mm, double t_deg, const double baseline[kValueCount]) {
    double differences[kValueCount];
    // 1. 실제 이동량과 예상 이동량의 차이 계산
    expected_differences(check, values, y_mm, t_deg, differences);
    AgreementResult result;
    result.checked = true;
    result.limit = check.limit;
    // 2. 영점 때의 차이를 빼고 평소 흔들림과 비교
    for (const AgreementCheck::Row& row : check.rows) {
        const double off = std::abs(differences[row.value_index] - baseline[row.value_index]) / row.usual_spread;
        if (off > result.value) { result.value = off; result.worst_value_index = row.value_index; }
    }
    return result;
}

// check_in_trained_range: 학습 범위 확인
//   입력: reference, y_mm, t_deg (영점 차감 전)
//   출력: Y, T 모두 범위 안이면 true. 양 끝 포함
bool check_in_trained_range(const Reference& reference, double y_mm, double t_deg) {
    return y_mm >= reference.trained_y_mm[0] && y_mm <= reference.trained_y_mm[1] && t_deg >= reference.trained_t_deg[0] && t_deg <= reference.trained_t_deg[1];
}

}   // alignment
