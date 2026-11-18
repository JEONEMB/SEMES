// =====================================================================================
// sim_hooks_default.cpp — team/<이름>/sim_hooks.cpp 가 없을 때 쓰는 기본 연결점
// =====================================================================================
#include "sim_team_hooks.hpp"

bool sim_team_init(const std::string& /*plugin_dir*/, const std::string& /*config_path*/, std::string& /*message*/) { return true; }

bool sim_team_set_reference(const cv::Mat& /*reference*/) { return false; }

bool sim_team_wants_bgr() { return true; }

void sim_team_before_estimate() {}

bool sim_team_output_is_command() { return false; }
