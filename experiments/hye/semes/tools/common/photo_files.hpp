#pragma once
// 사진 검색과 파일명 명령값 해석
//   Y+0.05.bmp -> Y 0.05, T 0
//   T-0.30.bmp -> Y 0, T -0.30
//   Y+0.05 T+0.01 (2).bmp -> Y 0.05, T 0.01
//   center.bmp -> 명령값 없음
#include <opencv2/core.hpp>
#include <filesystem>
#include <string>
#include <vector>

namespace tools {

struct PhotoFile {
    std::filesystem::path path;
    std::string name;            // 파일 이름 (UTF-8)
    bool has_command = false;    // 파일 이름에서 명령값 해석 성공 여부
    double command_y_mm = 0;
    double command_t_deg = 0;
};

// parse_command_from_file_name: 파일명에서 명령값 해석
//   입력: file_name (사진 파일 이름)
//   출력: 성공 여부, command_y_mm, command_t_deg. Y와 T 모두 없으면 false
bool parse_command_from_file_name(const std::string& file_name, double& command_y_mm, double& command_t_deg);

// find_photos: 하위 폴더 포함 사진 검색
//   입력: folder (검색 폴더)
//   출력: 이름순 .bmp, .png 사진 목록
std::vector<PhotoFile> find_photos(const std::filesystem::path& folder);

// list_subfolders: 바로 아래 폴더 검색
//   입력: folder (상위 폴더)
//   출력: 이름순 하위 폴더 이름 목록
std::vector<std::string> list_subfolders(const std::filesystem::path& folder);

// is_zero_command: 영점 명령 확인
//   입력: photo (사진 정보)
//   출력: 명령값이 (0, 0)이면 true
bool is_zero_command(const PhotoFile& photo);

}   // tools
