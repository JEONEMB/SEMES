#pragma once
// 사진 파일 찾기, 파일 이름에서 명령값 읽기, 사진 읽기.
//
// 파일 이름 규칙 (SEMES 에서 받은 그대로)
//   "Y+0.05.bmp"              → Y 0.05, T 0
//   "T-0.30.bmp"              → Y 0, T -0.30
//   "Y+0.05 T+0.01 (2).bmp"   → Y 0.05, T 0.01 (괄호 안 반복 번호는 무시)
//   "center.bmp"              → 명령값 없음
#include <opencv2/core.hpp>
#include <filesystem>
#include <string>
#include <vector>

namespace tools {

struct PhotoFile {
    std::filesystem::path path;
    std::string name;            // 파일 이름 (UTF-8)
    bool has_command = false;    // 파일 이름에서 명령값을 읽었는지
    double command_y_mm = 0;
    double command_t_deg = 0;
};

// 파일 이름에서 명령값을 읽는다. Y, T 둘 다 없으면 false.
bool parse_command_from_file_name(const std::string& file_name, double& command_y_mm, double& command_t_deg);

// folder 아래(하위 폴더 포함)의 .bmp, .png 사진을 이름 순으로 찾는다.
std::vector<PhotoFile> find_photos(const std::filesystem::path& folder);

// folder 바로 아래 폴더들의 이름 (이름 순).
std::vector<std::string> list_subfolders(const std::filesystem::path& folder);

// 사진을 회색으로 읽는다 (한글 경로도 읽히게 바이트로 읽어 푼다). 못 읽으면 빈 Mat.
cv::Mat read_gray_photo(const std::filesystem::path& path);

// 명령값이 (0, 0) 인지 (영점 사진인지).
bool is_zero_command(const PhotoFile& photo);

}   // namespace tools
