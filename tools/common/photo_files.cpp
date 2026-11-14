#include "photo_files.hpp"
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <regex>

namespace fs = std::filesystem;

namespace tools {

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

std::vector<std::string> list_subfolders(const fs::path& folder) {
    std::vector<std::string> names;
    if (!fs::is_directory(folder)) return names;
    for (const auto& entry : fs::directory_iterator(folder))
        if (entry.is_directory()) names.push_back(entry.path().filename().u8string());
    std::sort(names.begin(), names.end());
    return names;
}

cv::Mat read_gray_photo(const fs::path& path) {
    std::ifstream file(path, std::ios::binary);
    const std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    return bytes.empty() ? cv::Mat() : cv::imdecode(bytes, cv::IMREAD_GRAYSCALE);
}

bool is_zero_command(const PhotoFile& photo) {
    return photo.has_command && std::abs(photo.command_y_mm) < 1e-9 && std::abs(photo.command_t_deg) < 1e-9;
}

}   // namespace tools
