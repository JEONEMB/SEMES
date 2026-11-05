// =====================================================================================
// examples/joint_style/sim_hooks.cpp — 예시 : 초기화 함수가 따로 있는 팀원 코드 (initialize_joint 형태)
//   팀 공통 헤더에 아래처럼 선언된 경우 이 파일을 team/<이름>/sim_hooks.cpp 로 복사해 이름만 맞추면 된다
//       void initialize_joint(const std::string &joint_assets);
//   · init           : 자산 폴더 = simulator.yml 의 estimator.plugin_config → 비어 있으면 DLL 옆 assets/
//   · set_reference  : 기준 사진을 자산 폴더(center.bmp)에서 직접 읽는 코드라 교체 미지원 (false)
//   · 사진 형식       : 회색을 그대로 받는 코드면 false (1/3/4 채널 모두 받으면 false 가 빠름)
//   예외(std::exception)는 래퍼가 잡아 실패로 바꾼다
// =====================================================================================
#include "sim_team_hooks.hpp"

#include "alignment.hpp"

#include <filesystem>

bool sim_team_init(const std::string& plugin_dir, const std::string& config_path, std::string& message) {
    namespace fs = std::filesystem;
    const std::string assets = !config_path.empty() ? config_path : (fs::u8path(plugin_dir) / "assets").u8string();
    try {
        initialize_joint(assets);
    } catch (const std::exception& e) {
        message = std::string("initialize_joint 실패 : ") + e.what();
        return false;
    }
    return true;
}

bool sim_team_set_reference(const cv::Mat& /*reference*/) { return false; }

bool sim_team_wants_bgr() { return false; }

void sim_team_before_estimate() {}

bool sim_team_output_is_command() { return false; }   // 어긋남 추정값을 돌려줌
