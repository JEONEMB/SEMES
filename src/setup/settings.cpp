// 기본값 적용 후 파일 설정 반영. 외부 프로그램에서 호출할 함수: initialize
#include "setup/settings.hpp"
#include <opencv2/core.hpp>
#include <filesystem>
#include <fstream>

namespace alignment {

// load_settings: 측정 설정 읽기
//   입력: settings_file (비우면 기본값)
//   출력: 성공 여부, settings (읽은 설정), error (실패 사유)
bool load_settings(const std::string& settings_file, Settings& settings, std::string& error) {
    settings = Settings();
    if (settings_file.empty()) return true;

    const auto unreadable = [&] {
        error = "설정 파일을 읽지 못했습니다: " + settings_file;
        return false;
    };
    std::ifstream input(std::filesystem::u8path(settings_file), std::ios::binary);
    if (!input.is_open()) { input.clear(); input.open(settings_file, std::ios::binary); }
    if (!input) return unreadable();
    const std::string contents((std::istreambuf_iterator<char>(input)), {});
    cv::FileStorage file;
    try {
        if (!file.open(contents, cv::FileStorage::READ | cv::FileStorage::MEMORY)) return unreadable();
    } catch (const cv::Exception&) {
        return unreadable();
    }

    const cv::FileNode screws = file["screws"];
    if (!screws.empty()) {
        settings.use_screw = {false, false, false, false};
        for (const cv::FileNode& node : screws) {
            const int number = int(node);
            if (number < 1 || number > 4) { error = "설정의 screws 는 1~4 여야 합니다"; return false; }
            settings.use_screw[number - 1] = true;
        }
        const bool has_left = settings.use_screw[0] || settings.use_screw[2];
        const bool has_right = settings.use_screw[1] || settings.use_screw[3];
        if (!has_left || !has_right) { error = "설정의 screws 에 왼쪽(1, 3)과 오른쪽(2, 4) 나사가 하나씩은 있어야 합니다"; return false; }
    }
    if (!file["fail_on_disagreement"].empty()) settings.fail_on_disagreement = int(file["fail_on_disagreement"]) != 0;
    if (!file["log_file"].empty()) file["log_file"] >> settings.log_file;
    return true;
}

}   // alignment
