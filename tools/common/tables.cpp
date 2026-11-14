// TSV 읽기, 숫자 표시, 빈 출력 폴더 생성
#include "tables.hpp"
#include <fstream>
#include <iomanip>
#include <sstream>

namespace fs = std::filesystem;

namespace tools {

// column: 열 이름 조회
//   입력: name (열 이름)
//   출력: 열 번호. 없으면 -1
int Table::column(const std::string& name) const {
    for (size_t i = 0; i < header.size(); ++i)
        if (header[i] == name) return int(i);
    return -1;
}

namespace {

// split_tabs: 탭으로 구분한 한 줄 나누기
//   입력: line (표의 한 줄)
//   출력: 각 칸의 문자열 목록
std::vector<std::string> split_tabs(const std::string& line) {
    std::vector<std::string> cells;
    std::string cell;
    std::istringstream stream(line);
    while (std::getline(stream, cell, '\t')) cells.push_back(cell);
    return cells;
}

}   // 파일 내부

// read_table: TSV 표 읽기
//   입력: path (파일 경로)
//   출력: 성공 여부, table (읽은 표)
bool read_table(const fs::path& path, Table& table) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return false;
    table = Table();
    std::string line;
    bool first = true;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (first && line.size() >= 3 && static_cast<unsigned char>(line[0]) == 0xEF) line = line.substr(3);   // UTF-8 BOM
        if (line.empty() || line[0] == '#') continue;
        if (first) { table.header = split_tabs(line); first = false; continue; }
        table.rows.push_back(split_tabs(line));
    }
    return !first;
}

// format_number: 숫자 표시
//   입력: value (숫자), digits (표시할 숫자 자릿수, 소수 자릿수 아님)
//   출력: 숫자 문자열
std::string format_number(double value, int digits) {
    std::ostringstream text;
    text << std::setprecision(digits) << value;
    return text.str();
}

// format_rate: 통과율 표시
//   입력: passed (통과 수), total (전체 수)
//   출력: 통과율 문자열. 예: 96.4% (270/280)
std::string format_rate(int passed, int total) {
    if (total == 0) return "-";
    std::ostringstream text;
    text << std::fixed << std::setprecision(1) << 100.0 * passed / total << "% (" << passed << "/" << total << ")";
    return text.str();
}

// make_empty_folder: 빈 출력 폴더 준비
//   입력: folder (출력 경로)
//   출력: 성공 여부, error. 기존 폴더가 비어 있지 않으면 false
bool make_empty_folder(const fs::path& folder, std::string& error) {
    if (fs::exists(folder) && !fs::is_empty(folder)) {
        error = "출력 폴더가 이미 있고 비어 있지 않습니다 (덮어쓰지 않습니다): " + folder.u8string();
        return false;
    }
    fs::create_directories(folder);
    return true;
}

}   // tools
