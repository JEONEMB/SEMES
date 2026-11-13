// =====================================================================================
// sim_paths.hpp — 실행 위치와 무관한 경로 처리
//
//  시뮬레이터 루트 = config/simulator.yml 이 있는 폴더. 찾은 뒤 그 폴더를 작업 폴더로 바꾼다
//   → 설정 안의 상대 경로(data/img, world, results, plugins …)가 언제나 루트 기준이 됨
//   → exe 더블 클릭, VS 디버거, 다른 폴더에서 실행 모두 같은 결과
//  찾는 순서 : 명시한 루트 → 현재 작업 폴더 → exe 폴더 → exe 의 상위 폴더 (최대 kSearchUpLevels 단계)
//             각 폴더에서 '그 폴더' 와 '그 안의 Simulator/' 를 확인
//  문자열 경로는 모두 UTF-8 로 다룬다 (Windows 한글 경로 : fs::u8path 로 변환해서 열기)
// =====================================================================================
#ifndef SIM_PATHS_HPP
#define SIM_PATHS_HPP

#include <opencv2/core.hpp>

#include <filesystem>
#include <string>
#include <vector>

namespace sim {

constexpr int kSearchUpLevels = 4;                        // exe 위로 올라가며 찾는 단계 수
constexpr const char* kConfigRelPath = "config/simulator.yml";

std::filesystem::path exeDir();                           // 실행 파일이 있는 폴더
std::filesystem::path u8p(const std::string& utf8);       // UTF-8 → path
std::string u8s(const std::filesystem::path& p);          // path → UTF-8

// 루트를 찾아 작업 폴더로 바꿈. rootArg 가 비어 있지 않으면 그것만 확인. 성공 시 root 에 절대 경로
bool enterRoot(const std::string& rootArg, std::filesystem::path& root, std::string& message);

// 후보 경로 중 처음 존재하는 것 (없으면 빈 문자열)
std::string firstExisting(const std::vector<std::string>& candidates);

// 한글 경로에서도 동작하는 사진 읽기 (파일 바이트 → cv::imdecode)
cv::Mat readImage(const std::string& utf8Path, int flags);

} // namespace sim

#endif
