// =====================================================================================
// sim_team_hooks.hpp — 팀원 코드와 플러그인 래퍼 사이의 '선택' 연결점
//
//  래퍼(team_alignment_plugin.cpp)는 팀 공통 함수 하나만 부른다
//      Alignment_t alignment(cv::Mat &src);      (include/alignment.hpp)
//  그 밖에 초기화(모델·기준 사진 읽기)나 기준 사진 교체 함수가 있으면
//  team/<이름>/sim_hooks.cpp 를 만들어 아래 함수를 구현한다. 파일이 없으면 기본 동작(sim_hooks_default.cpp)
// =====================================================================================
#ifndef SIM_TEAM_HOOKS_HPP
#define SIM_TEAM_HOOKS_HPP

#include <opencv2/core.hpp>

#include <string>

// 플러그인을 읽을 때 한 번. plugin_dir = DLL 폴더, config_path = simulator.yml 의 estimator.plugin_config
// 작업 폴더 = 시뮬레이터 루트 (data/img, config 가 있는 곳 — 팀 저장소 루트와 같은 구조)
// 기본 : 아무것도 안 하고 성공
bool sim_team_init(const std::string& plugin_dir, const std::string& config_path, std::string& message);

// 기준(티칭) 사진 교체. 지원하지 않으면 false → 시뮬레이터는 팀원 코드의 자체 기준 사진으로 진행
// 기본 : false
bool sim_team_set_reference(const cv::Mat& reference);

// alignment() 에 넘길 사진 형식 : true = BGR 3채널 (실제 카메라 사진과 같음), false = 받은 그대로 (합성 사진은 회색)
// 기본 : true
bool sim_team_wants_bgr();

// alignment() 를 부르기 직전 (사진 한 장마다). 시뮬레이터는 이전 결과가 '이동' 이었다면 그 이동을 이미 실행한 뒤 부른다
//   → 이동 완료를 알려야 하는 제어기형 코드(예 acknowledgeAlignmentMoveExecuted)는 여기서 알림
// 기본 : 아무것도 안 함
void sim_team_before_estimate();

// alignment() 결과의 뜻 : false = 어긋남 추정값 (라벨과 같은 부호, 시뮬레이터가 −값만큼 이동)
//                         true  = 보정 명령 (그대로 이동할 양) → 래퍼가 부호를 바꿔 넘김. (0, 0) = 정렬 끝
// 기본 : false
bool sim_team_output_is_command();

#endif
