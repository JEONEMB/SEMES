// 사진 ID 기준으로 학습용 표 결합
#include "dataset.hpp"
#include "tables.hpp"
#include <cmath>
#include <limits>
#include <map>

namespace fs = std::filesystem;

namespace tools {

namespace {

// to_number: 표의 숫자 읽기
//   입력: text (표의 한 칸)
//   출력: 숫자. 빈 칸과 nan은 측정값 없음
double to_number(const std::string& text) {
    if (text == "nan" || text.empty()) return std::numeric_limits<double>::quiet_NaN();
    return std::stod(text);
}

// read_required: 필수 자료 표 읽기
//   입력: path (표 경로)
//   출력: 성공 여부, table (읽은 표), error (실패 사유)
bool read_required(const fs::path& path, Table& table, std::string& error) {
    if (read_table(path, table)) return true;
    error = "읽지 못했습니다: " + path.u8string() + ". prepare_dataset 을 먼저 실행하세요";
    return false;
}

}   // 파일 내부

// load_tables: 학습용 표 읽기
//   입력: folder (tables 폴더)
//   출력: 성공 여부, photos (사진 자료), sessions (회차별 영점), error (실패 사유)
bool load_tables(const fs::path& folder, std::vector<DatasetPhoto>& photos, std::vector<SessionZero>& sessions, std::string& error) {
    Table photo_table, feature_table, label_table, split_table, session_table;
    if (!read_required(folder / "photos.tsv", photo_table, error) || !read_required(folder / "features.tsv", feature_table, error) ||
        !read_required(folder / "labels.tsv", label_table, error) || !read_required(folder / "splits.tsv", split_table, error) ||
        !read_required(folder / "sessions.tsv", session_table, error))
        return false;

    photos.clear();
    std::map<int, size_t> index_of;
    for (const auto& row : photo_table.rows) {
        DatasetPhoto photo;
        photo.id = std::stoi(row.at(0));
        photo.session = row.at(1);
        photo.path = row.at(2);
        photo.command_y_mm = std::stod(row.at(3));
        photo.command_t_deg = std::stod(row.at(4));
        index_of[photo.id] = photos.size();
        photos.push_back(photo);
    }
    auto find = [&](const std::string& id_text) -> DatasetPhoto* {
        const auto it = index_of.find(std::stoi(id_text));
        return it == index_of.end() ? nullptr : &photos[it->second];
    };
    for (const auto& row : feature_table.rows) {
        DatasetPhoto* photo = find(row.at(0));
        if (!photo) continue;
        photo->measured = row.at(1) == "1";
        for (int k = 0; k < kValueCount; ++k) photo->values[k] = to_number(row.at(2 + k));
    }
    for (const auto& row : label_table.rows) {
        DatasetPhoto* photo = find(row.at(0));
        if (!photo) continue;
        photo->label_y_mm = std::stod(row.at(1));
        photo->label_t_deg = std::stod(row.at(2));
    }
    for (const auto& row : split_table.rows) {
        DatasetPhoto* photo = find(row.at(0));
        if (!photo) continue;
        photo->usable = row.at(1) == "1";
        photo->final_check = row.at(2) == "1";
        for (int r = 0; r < kRepeatCount; ++r) photo->fold[r] = std::stoi(row.at(3 + r));
    }
    sessions.clear();
    for (const auto& row : session_table.rows)
        sessions.push_back({row.at(0), std::stod(row.at(1)), std::stod(row.at(2)), std::stoi(row.at(3))});
    return true;
}

}   // tools
