// =====================================================================================
// examples/command_style/sim_hooks.cpp — 예시 : '보정 명령' 을 돌려주는 제어기형 팀원 코드 (이승재 LSJ 형태)
//   팀 공통 헤더에 아래가 있을 때 team/<이름>/sim_hooks.cpp 로 복사해 쓴다
//       void configureAlignment(const cv::Mat& reference, const std::string& thetaConfig, const std::string& yConfig);
//       void acknowledgeAlignmentMoveExecuted();   // 이동을 실제로 마쳤다고 알림
//       void startNewAlignmentSession();           // 새 정렬 작업 (이동 횟수 0 부터)
//       Alignment_t alignment(cv::Mat& src);       // 보정 명령 (Y mm, Θ °) — 한 번에 한 축, (0, 0) = 정렬 끝
//   · init           : 설정 파일 위치 찾기 (plugin_config 폴더 → DLL 옆 configs/ → 루트 configs/)
//   · set_reference  : 시험마다 기준(티칭) 사진으로 다시 설정 + 새 정렬 작업 시작
//   · before_estimate: 직전에 낸 이동 명령을 시뮬레이터가 실행했으므로 완료를 알림 (낸 명령이 없으면 예외 → 무시)
//   · 결과            : 보정 명령 → 래퍼가 부호를 바꿔 어긋남으로 넘김
// =====================================================================================
#include "sim_team_hooks.hpp"

#include "alignment.hpp"

#include <filesystem>
#include <vector>

namespace {
std::string g_thetaConfig, g_yConfig;

void acknowledgeIfPending() {
    try {
        acknowledgeAlignmentMoveExecuted();
    } catch (...) {   // 기다리는 이동이 없음 (첫 사진, 정렬 끝, 측정 실패 뒤)
    }
}
} // namespace

bool sim_team_init(const std::string& plugin_dir, const std::string& config_path, std::string& message) {
    namespace fs = std::filesystem;
    std::vector<fs::path> dirs;
    if (!config_path.empty()) dirs.push_back(fs::u8path(config_path));
    dirs.push_back(fs::u8path(plugin_dir) / "configs");
    dirs.push_back("configs");
    std::error_code ec;
    for (const auto& d : dirs) {
        if (fs::exists(d / "theta_alignment.json", ec) && fs::exists(d / "y_alignment.json", ec)) {
            g_thetaConfig = (d / "theta_alignment.json").u8string();
            g_yConfig = (d / "y_alignment.json").u8string();
            return true;
        }
    }
    message = "theta_alignment.json / y_alignment.json 을 찾지 못함 (plugin_config, DLL 옆 configs, 루트 configs)";
    return false;
}

bool sim_team_set_reference(const cv::Mat& reference) {
    acknowledgeIfPending();   // 이전 시험이 멈춤 판정으로 끝나 남은 명령
    configureAlignment(reference, g_thetaConfig, g_yConfig);
    startNewAlignmentSession();
    return true;
}

bool sim_team_wants_bgr() { return false; }   // 회색·컬러 모두 받음 → 변환 생략

void sim_team_before_estimate() { acknowledgeIfPending(); }

bool sim_team_output_is_command() { return true; }
