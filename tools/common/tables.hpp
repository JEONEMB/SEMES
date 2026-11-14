#pragma once
// 첫 줄에 열 이름을 둔 탭 구분 표(TSV) 읽고 쓰기
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace tools {

struct Table {
    std::vector<std::string> header;
    std::vector<std::vector<std::string>> rows;

    // column: 열 이름 조회
    //   입력: name (열 이름)
    //   출력: 열 번호. 없으면 -1
    int column(const std::string& name) const;
};

// read_table: TSV 표 읽기
//   입력: path (파일 경로)
//   출력: 성공 여부, table (읽은 표)
bool read_table(const std::filesystem::path& path, Table& table);

// format_number: 숫자 표시
//   입력: value (숫자), digits (표시할 숫자 자릿수, 소수 자릿수 아님)
//   출력: 숫자 문자열
std::string format_number(double value, int digits = 10);

// format_rate: 통과율 표시
//   입력: passed (통과 수), total (전체 수)
//   출력: 통과율 문자열. 예: 96.4% (270/280)
std::string format_rate(int passed, int total);

// make_empty_folder: 빈 출력 폴더 준비
//   입력: folder (출력 경로)
//   출력: 성공 여부, error. 기존 폴더가 비어 있지 않으면 false
bool make_empty_folder(const std::filesystem::path& folder, std::string& error);

}   // tools
