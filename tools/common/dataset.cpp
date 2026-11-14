#include "dataset.hpp"
#include "tables.hpp"
#include <cmath>
#include <limits>
#include <map>

namespace fs = std::filesystem;

namespace tools {

namespace {

double to_number(const std::string& text) {
    if (text == "nan" || text.empty()) return std::numeric_limits<double>::quiet_NaN();
    return std::stod(text);
}

bool read_required(const fs::path& path, Table& table, std::string& error) {
    if (read_table(path, table)) return true;
    error = "읽지 못했습니다: " + path.u8string() + ". prepare_dataset 을 먼저 실행하세요";
    return false;
}

}   // namespace

bool load_prepared(const fs::path& folder, std::vector<DatasetPhoto>& photos, std::vector<SessionZero>& sessions, std::string& error) {
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

}   // namespace tools
