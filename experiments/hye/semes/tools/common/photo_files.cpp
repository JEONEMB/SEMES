// 사진 경로 수집과 파일명 명령값 해석
#include "photo_files.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <regex>

namespace fs = std::filesystem;

namespace tools {

// parse_command_from_file_name: 파일명에서 명령값 해석
//   입력: file_name (사진 파일 이름)
//   출력: 성공 여부, command_y_mm, command_t_deg. Y와 T 모두 없으면 false
bool parse_command_from_file_name(const std::string& file_name, double& command_y_mm, double& command_t_deg) {
    static const std::regex y_pattern("Y([+-]?[0-9]+(\\.[0-9]+)?)");
    static const std::regex t_pattern("T([+-]?[0-9]+(\\.[0-9]+)?)");
    std::smatch match;
    bool found = false;
    command_y_mm = command_t_deg = 0;
    if (std::regex_search(file_name, match, y_pattern)) { command_y_mm = std::stod(match[1]); found = true; }
    if (std::regex_search(file_name, match, t_pattern)) { command_t_deg = std::stod(match[1]); found = true; }
    return found;
}

// find_photos: 하위 폴더 포함 사진 검색
//   입력: folder (검색 폴더)
//   출력: 이름순 .bmp, .png 사진 목록
std::vector<PhotoFile> find_photos(const fs::path& folder) {
    std::vector<PhotoFile> photos;
    if (!fs::is_directory(folder)) return photos;
    for (const auto& entry : fs::recursive_directory_iterator(folder)) {
        if (!entry.is_regular_file()) continue;
        std::string extension = entry.path().extension().u8string();
        std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) { return char(std::tolower(c)); });
        if (extension != ".bmp" && extension != ".png") continue;
        PhotoFile photo;
        photo.path = entry.path();
        photo.name = entry.path().filename().u8string();
        photo.has_command = parse_command_from_file_name(photo.name, photo.command_y_mm, photo.command_t_deg);
        photos.push_back(photo);
    }
    std::sort(photos.begin(), photos.end(), [](const PhotoFile& a, const PhotoFile& b) { return a.path < b.path; });
    return photos;
}

// list_subfolders: 바로 아래 폴더 검색
//   입력: folder (상위 폴더)
//   출력: 이름순 하위 폴더 이름 목록
std::vector<std::string> list_subfolders(const fs::path& folder) {
    std::vector<std::string> names;
    if (!fs::is_directory(folder)) return names;
    for (const auto& entry : fs::directory_iterator(folder))
        if (entry.is_directory()) names.push_back(entry.path().filename().u8string());
    std::sort(names.begin(), names.end());
    return names;
}

// is_zero_command: 영점 명령 확인
//   입력: photo (사진 정보)
//   출력: 명령값이 (0, 0)이면 true
bool is_zero_command(const PhotoFile& photo) {
    return photo.has_command && std::abs(photo.command_y_mm) < 1e-9 && std::abs(photo.command_t_deg) < 1e-9;
}

}   // tools
