#include "setup/settings.hpp"
#include <opencv2/core.hpp>

namespace alignment {

bool load_settings(const std::string& settings_file, Settings& settings, std::string& error) {
    settings = Settings();
    if (settings_file.empty()) return true;

    cv::FileStorage file(settings_file, cv::FileStorage::READ);
    if (!file.isOpened()) { error = "설정 파일을 읽지 못했습니다: " + settings_file; return false; }

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

}   // namespace alignment
