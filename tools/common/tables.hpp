#pragma once
// 탭으로 나눈 표(.tsv) 읽고 쓰기. 첫 줄은 머리줄이다.
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace tools {

struct Table {
    std::vector<std::string> header;
    std::vector<std::vector<std::string>> rows;
    // 열 이름 → 열 번호. 없으면 -1.
    int column(const std::string& name) const;
};

// 표를 읽는다. 못 읽으면 false.
bool read_table(const std::filesystem::path& path, Table& table);

// 숫자를 글자로 (소수 자리 digits).
std::string format_number(double value, int digits = 10);

// 백분율 글자 "96.4% (270/280)".
std::string format_rate(int passed, int total);

// 폴더를 만든다. 이미 있고 비어 있지 않으면 false (덮어쓰지 않는다).
bool make_empty_folder(const std::filesystem::path& folder, std::string& error);

}   // namespace tools
