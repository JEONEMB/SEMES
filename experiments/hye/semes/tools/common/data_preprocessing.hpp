#pragma once
// 원본 명령 보존, 확인된 정정값과 학습 제외 규칙 적용
#include <filesystem>
#include <set>
#include <string>
#include <vector>

namespace tools {

struct PreprocessingDecision {
    double verified_y_mm = 0, verified_t_deg = 0;   // 확인된 학습용 Y(mm), T(도)
    bool training_allowed = true;   // 학습에 사용할 사진인지 여부
    std::string rule_id, reason, provenance;   // 적용 규칙 번호, 사유, 확인 근거
};

struct PreprocessingRule {
    std::string session, relative_path;   // 촬영 회차와 회차 안 사진 경로
    double expected_raw_y_mm = 0, expected_raw_t_deg = 0;   // 규칙 적용 전 확인할 원본 명령
    PreprocessingDecision decision;   // 해당 사진에 적용할 정정과 제외 내용
};

// load_preprocessing_rules: 사진별 정정과 제외 규칙 읽기
//   입력: path (규칙 파일)
//   출력: 성공 여부, rules (규칙 목록), error (실패 사유)
bool load_preprocessing_rules(const std::filesystem::path& path, std::vector<PreprocessingRule>& rules, std::string& error);
// apply_preprocessing_rule: 사진에 확인된 정정값 적용
//   입력: rules (규칙 목록), session (회차), relative_path (회차 안 경로), 원본 Y와 T
//   출력: 성공 여부, decision (적용 결과), error. 해당 규칙이 없으면 원본값 유지
bool apply_preprocessing_rule(const std::vector<PreprocessingRule>& rules, const std::string& session, const std::string& relative_path,
                              double raw_y_mm, double raw_t_deg, PreprocessingDecision& decision, std::string& error);
// validate_preprocessing_coverage: 현재 회차의 규칙 누락 확인
//   입력: rules (규칙 목록), present_sessions (현재 회차), applied_rule_ids (적용한 규칙 번호)
//   출력: 모두 적용했으면 true. 누락 시 false와 error
bool validate_preprocessing_coverage(const std::vector<PreprocessingRule>& rules, const std::set<std::string>& present_sessions,
                                     const std::set<std::string>& applied_rule_ids, std::string& error);
// write_preprocessing_metadata: 표 생성에 쓴 규칙 원문 보관
//   입력: tables (표 폴더), rules_path (규칙 파일)
//   출력: 성공 여부, error. 규칙 사본과 형식 버전 저장
bool write_preprocessing_metadata(const std::filesystem::path& tables, const std::filesystem::path& rules_path, std::string& error);
// validate_preprocessing_cache: 기존 표를 다시 써도 되는지 확인
//   입력: tables (기존 표 폴더), rules_path (현재 규칙)
//   출력: 규칙과 처리 기록이 일치하면 true. 다르면 false와 error
bool validate_preprocessing_cache(const std::filesystem::path& tables, const std::filesystem::path& rules_path, std::string& error);

}   // tools
