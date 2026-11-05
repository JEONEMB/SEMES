// =====================================================================================
// examples/ksh/sim_hooks.cpp — 예시 : ksh Alignment 를 플러그인으로 만들 때의 연결점
//   팀원도 같은 모양으로 쓰면 된다 (초기화 함수, 기준 사진 교체 함수가 있을 때)
//   · init           : 설정 파일을 읽음 (plugin_config 가 비어 있으면 루트의 config/alignment.yml)
//   · set_reference  : 시뮬레이터가 넘긴 기준(티칭) 사진으로 교체
//   · 사진 형식       : ksh 는 회색을 그대로 받으므로 BGR 로 바꾸지 않음 (변환 시간 절약)
// =====================================================================================
#include "sim_team_hooks.hpp"

#include "alignment.hpp"

#include <filesystem>
#include <iostream>
#include <sstream>

bool sim_team_init(const std::string& /*plugin_dir*/, const std::string& config_path, std::string& message) {
    std::error_code ec;
    std::string cfg = config_path;
    // 설정 파일 후보 : 배포 키트 → 팀 저장소(PJT/tool/simulator 에서 실행) → 예전 개발 폴더
    if (cfg.empty()) {
        for (const char* c : {"config/alignment.yml", "../../config/alignment.yml", "../Alignment/config/alignment.yml"}) {
            if (std::filesystem::exists(c, ec)) { cfg = c; break; }
        }
        if (cfg.empty()) cfg = "config/alignment.yml";
    }
    // 설정의 기준 사진 경로가 없을 때의 경고는 감춤 (시뮬레이터가 기준 사진을 따로 넘김)
    std::ostringstream sink;
    std::streambuf* old = std::cerr.rdbuf(sink.rdbuf());
    const bool ok = alignment_init(cfg);
    std::cerr.rdbuf(old);
    if (!ok) { message = "alignment_init 실패 : " + cfg; return false; }
    return true;
}

bool sim_team_set_reference(const cv::Mat& reference) { return alignment_set_reference(reference); }

bool sim_team_wants_bgr() { return false; }

void sim_team_before_estimate() {}

bool sim_team_output_is_command() { return false; }   // 어긋남 추정값을 돌려줌
