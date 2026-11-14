#include "tables.hpp"
#include <fstream>
#include <iomanip>
#include <sstream>

namespace fs = std::filesystem;

namespace tools {

int Table::column(const std::string& name) const {
    for (size_t i = 0; i < header.size(); ++i)
        if (header[i] == name) return int(i);
    return -1;
}

namespace {

std::vector<std::string> split_tabs(const std::string& line) {
    std::vector<std::string> cells;
    std::string cell;
    std::istringstream stream(line);
    while (std::getline(stream, cell, '\t')) cells.push_back(cell);
    return cells;
}

}   // namespace

bool read_table(const fs::path& path, Table& table) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return false;
    table = Table();
    std::string line;
    bool first = true;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (first && line.size() >= 3 && static_cast<unsigned char>(line[0]) == 0xEF) line = line.substr(3);   // UTF-8 표시
        if (line.empty() || line[0] == '#') continue;
        if (first) { table.header = split_tabs(line); first = false; continue; }
        table.rows.push_back(split_tabs(line));
    }
    return !first;
}

std::string format_number(double value, int digits) {
    std::ostringstream text;
    text << std::setprecision(digits) << value;
    return text.str();
}

std::string format_rate(int passed, int total) {
    if (total == 0) return "-";
    std::ostringstream text;
    text << std::fixed << std::setprecision(1) << 100.0 * passed / total << "% (" << passed << "/" << total << ")";
    return text.str();
}

bool make_empty_folder(const fs::path& folder, std::string& error) {
    if (fs::exists(folder) && !fs::is_empty(folder)) {
        error = "출력 폴더가 이미 있고 비어 있지 않습니다 (덮어쓰지 않습니다): " + folder.u8string();
        return false;
    }
    fs::create_directories(folder);
    return true;
}

}   // namespace tools
