#pragma once
// 설정: config/settings.json 에서 읽는다. 파일을 주지 않으면 아래 기본값을 쓴다.
#include <array>
#include <string>

namespace alignment {

struct Settings {
    std::array<bool, 4> use_screw = {true, true, true, true};   // 쓸 나사 (기본: 4개 모두)
    bool fail_on_disagreement = false;   // 값들이 서로 맞지 않을 때 실패로 표시할지 (기본: 경고만)
    std::string log_file;                // 경고·실패를 적을 파일 (기본: 없음, 콘솔에만)
};

// settings_file 이 비어 있으면 기본값. 읽지 못하거나 값이 잘못되면 false 와 이유.
bool load_settings(const std::string& settings_file, Settings& settings, std::string& error);

}   // namespace alignment
