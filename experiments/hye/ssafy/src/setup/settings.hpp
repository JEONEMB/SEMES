#pragma once
// 측정 설정. 파일 미지정 시 기본값 사용
#include <array>
#include <string>

namespace alignment {

struct Settings {
    std::array<bool, 4> use_screw = {true, true, true, true};   // 사용 나사. 기본: 4개 모두
    bool fail_on_disagreement = false;   // 값 불일치 시 실패 처리 여부. 기본: 경고
    std::string log_file;                // 알림 로그 경로. 기본: 콘솔만 사용
};

// load_settings: 측정 설정 읽기
//   입력: settings_file (비우면 기본값)
//   출력: 성공 여부, settings (읽은 설정), error (실패 사유)
bool load_settings(const std::string& settings_file, Settings& settings, std::string& error);

}   // alignment
