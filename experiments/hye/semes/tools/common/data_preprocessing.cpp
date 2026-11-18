// 사진별 정정과 제외 규칙 읽기, 적용, 누락 확인
#include "data_preprocessing.hpp"
#include "tables.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iterator>
#include <stdexcept>

namespace tools {
namespace {

// normalized_relative_path: 회차 안 사진 경로 통일
//   입력: text (사진 경로)
//   출력: 구분자를 /로 맞춘 경로. 상위 폴더나 절대 경로는 오류
std::string normalized_relative_path(std::string text) {
    std::replace(text.begin(), text.end(), '\\', '/');
    const auto path = std::filesystem::u8path(text);
    if (text.empty() || path.is_absolute() || path.has_root_name()) throw std::runtime_error("사진 키는 회차 안의 상대 경로여야 합니다");
    for (const auto& part : path)
        if (part == "..") throw std::runtime_error("사진 키에 상위 폴더(..)를 쓸 수 없습니다");
    return path.lexically_normal().generic_u8string();
}

// number: 규칙 표의 숫자 확인
//   입력: text (숫자 문자열)
//   출력: 숫자. 뒤에 다른 글자, NaN, 무한대가 있으면 오류
double number(const std::string& text) {
    size_t end = 0;
    const double value = std::stod(text, &end);
    if (end != text.size() || !std::isfinite(value)) throw std::runtime_error("유한한 숫자가 필요합니다: " + text);
    return value;
}

// read_bytes: 파일 내용을 변환 없이 읽기
//   입력: path (파일 경로)
//   출력: 성공 여부, bytes (파일 내용)
bool read_bytes(const std::filesystem::path& path, std::string& bytes) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return false;
    bytes.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    return !file.bad();
}

}   // 파일 내부

// load_preprocessing_rules: 사진별 정정과 제외 규칙 읽기
//   입력: path (규칙 파일)
//   출력: 성공 여부, rules (규칙 목록), error (실패 사유)
bool load_preprocessing_rules(const std::filesystem::path& path, std::vector<PreprocessingRule>& rules, std::string& error) {
    rules.clear();
    Table table;
    if (!read_table(path, table)) { error = "전처리 규칙을 읽지 못했습니다: " + path.u8string(); return false; }
    const std::vector<std::string> header = {"rule_id", "session", "relative_path", "expected_raw_y_mm", "expected_raw_t_deg",
        "verified_y_mm", "verified_t_deg", "training_allowed", "reason", "provenance"};
    if (table.header != header) { error = "전처리 규칙의 열 이름 또는 순서가 잘못됐습니다: " + path.u8string(); return false; }
    std::set<std::string> ids, keys;
    try {
        for (const auto& row : table.rows) {
            if (row.size() != header.size()) throw std::runtime_error("전처리 규칙은 10개 열이어야 합니다");
            PreprocessingRule rule;
            rule.decision.rule_id = row[0]; rule.session = row[1];
            if (row[0].empty() || row[1].empty() || row[8].empty() || row[9].empty()) throw std::runtime_error("규칙 ID, 회차, 사유, 출처는 비울 수 없습니다");
            if (row[1].find_first_of("/\\") != std::string::npos || row[1] == "." || row[1] == "..") throw std::runtime_error("회차는 폴더 이름 하나여야 합니다");
            rule.relative_path = normalized_relative_path(row[2]);
            rule.expected_raw_y_mm = number(row[3]); rule.expected_raw_t_deg = number(row[4]);
            rule.decision.verified_y_mm = number(row[5]); rule.decision.verified_t_deg = number(row[6]);
            if (row[7] != "0" && row[7] != "1") throw std::runtime_error("training_allowed는 0 또는 1이어야 합니다");
            rule.decision.training_allowed = row[7] == "1";
            rule.decision.reason = row[8]; rule.decision.provenance = row[9];
            if (!ids.insert(row[0]).second || !keys.insert(rule.session + "/" + rule.relative_path).second)
                throw std::runtime_error("규칙 ID 또는 사진 키가 중복됐습니다: " + row[0]);
            rules.push_back(rule);
        }
    } catch (const std::exception& e) {
        error = path.u8string() + ": " + e.what(); rules.clear(); return false;
    }
    return true;
}

// apply_preprocessing_rule: 사진에 확인된 정정값 적용
//   입력: rules (규칙 목록), session (회차), relative_path (회차 안 경로), 원본 Y와 T
//   출력: 성공 여부, decision (적용 결과), error. 해당 규칙이 없으면 원본값 유지
bool apply_preprocessing_rule(const std::vector<PreprocessingRule>& rules, const std::string& session, const std::string& relative_path,
                              double raw_y_mm, double raw_t_deg, PreprocessingDecision& decision, std::string& error) {
    decision = PreprocessingDecision();
    if (!std::isfinite(raw_y_mm) || !std::isfinite(raw_t_deg)) { error = "원본 명령이 유한한 숫자가 아닙니다"; return false; }
    decision.verified_y_mm = raw_y_mm; decision.verified_t_deg = raw_t_deg;
    decision.provenance = "raw_filename";
    try {
        const std::string key = normalized_relative_path(relative_path);
        for (const auto& rule : rules) {
            if (rule.session != session || rule.relative_path != key) continue;
            if (std::abs(raw_y_mm - rule.expected_raw_y_mm) > 1e-9 || std::abs(raw_t_deg - rule.expected_raw_t_deg) > 1e-9) {
                error = "전처리 규칙의 예상 원본 명령과 다릅니다: " + rule.decision.rule_id + " (" + session + "/" + key + ")";
                return false;
            }
            decision = rule.decision;
            break;
        }
    } catch (const std::exception& e) { error = e.what(); return false; }
    return true;
}

// validate_preprocessing_coverage: 현재 회차의 규칙 누락 확인
//   입력: rules (규칙 목록), present_sessions (현재 회차), applied_rule_ids (적용한 규칙 번호)
//   출력: 모두 적용했으면 true. 누락 시 false와 error
bool validate_preprocessing_coverage(const std::vector<PreprocessingRule>& rules, const std::set<std::string>& present_sessions,
                                     const std::set<std::string>& applied_rule_ids, std::string& error) {
    for (const auto& rule : rules) {
        if (present_sessions.count(rule.session) && !applied_rule_ids.count(rule.decision.rule_id)) {
            error = "회차는 있지만 전처리 규칙의 사진을 찾지 못했습니다: " + rule.decision.rule_id + " (" + rule.session + "/" + rule.relative_path + ")";
            return false;
        }
    }
    return true;
}

// write_preprocessing_metadata: 표 생성에 쓴 규칙 원문 보관
//   입력: tables (표 폴더), rules_path (규칙 파일)
//   출력: 성공 여부, error. 규칙 사본과 형식 버전 저장
bool write_preprocessing_metadata(const std::filesystem::path& tables, const std::filesystem::path& rules_path, std::string& error) {
    std::string bytes;
    if (!read_bytes(rules_path, bytes)) { error = "적용 규칙 원문을 읽지 못했습니다: " + rules_path.u8string(); return false; }
    std::ofstream snapshot(tables / "preprocessing_rules.tsv", std::ios::binary);
    std::ofstream version(tables / "preprocessing_version.txt", std::ios::binary);
    snapshot << bytes; version << "1\n";
    snapshot.close(); version.close();
    if (!snapshot || !version) { error = "전처리 규칙 기록을 저장하지 못했습니다: " + tables.u8string(); return false; }
    return true;
}

// validate_preprocessing_cache: 기존 표를 다시 써도 되는지 확인
//   입력: tables (기존 표 폴더), rules_path (현재 규칙)
//   출력: 규칙과 처리 기록이 일치하면 true. 다르면 false와 error
bool validate_preprocessing_cache(const std::filesystem::path& tables, const std::filesystem::path& rules_path, std::string& error) {
    std::string current, snapshot, version;
    Table audit;
    if (!read_bytes(rules_path, current)) { error = "현재 전처리 규칙을 읽지 못했습니다: " + rules_path.u8string(); return false; }
    if (!read_bytes(tables / "preprocessing_version.txt", version) || version != "1\n" ||
        !read_bytes(tables / "preprocessing_rules.tsv", snapshot) || !read_table(tables / "preprocessing.tsv", audit) ||
        audit.rows.empty() || audit.column("verified_t_deg") < 0 || audit.column("training_used") < 0) {
        error = "현재 전처리 이력이 없는 표입니다. prepare_dataset --output <새 표 폴더>로 다시 준비하세요: " + tables.u8string();
        return false;
    }
    if (current != snapshot) {
        error = "표를 만든 뒤 전처리 규칙이 바뀌었습니다. prepare_dataset --output <새 표 폴더>로 다시 준비하세요: " + tables.u8string();
        return false;
    }
    return true;
}

}   // tools
