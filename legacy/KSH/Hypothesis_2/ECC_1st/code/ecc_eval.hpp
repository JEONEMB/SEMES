// =====================================================================================
// ecc_eval.hpp — 가설 H2 검증용 데이터 목록 · 평가 · 저장
//
//  데이터 (img/DATA.md 참고)
//    1차 세트 : img/center.bmp, img/Y/Y±a.bmp (80), img/Theta/T±b.bmp (200)       — 한 축씩, 반복 없음
//    2차 세트 : img/일부 데이터 추출/기준 위치/center1~3.bmp,
//               img/일부 데이터 추출/Y+a T+b/… (1)(2)(3).bmp (19 조합 × 3장)       — Y·Θ 동시, 같은 자세 3장
//  한글 폴더 이름 때문에 경로는 std::filesystem::path(와이드 문자)로 다루고,
//  영상은 파일을 바이트로 읽어 cv::imdecode 로 연다 (OpenCV 3.4 imread 는 한글 경로 불가).
// =====================================================================================
#ifndef __ECC_EVAL_HPP__
#define __ECC_EVAL_HPP__

#include "ecc_align.hpp"
#include <filesystem>
#include <string>
#include <vector>

namespace ev {

namespace fs = std::filesystem;

// 영상 한 장
struct Item {
    fs::path    path;
    std::string name;            // 파일 이름 (UTF-8)
    std::string set;             // 2차 세트 : 폴더 이름 (같은 자세 묶음), 1차 세트 : "Y" / "Theta" / "center"
    int         shot = 0;        // 같은 자세에서 몇 번째 촬영인지 (2차 세트 1~3)
    bool        hasTruth = false;
    double      trueY = 0, trueT = 0;   // mm, deg
    bool        train = false;          // 학습에 사용
    ea::Measure m;
    ea::Estimate e;
};

std::string u8(const fs::path& p);                       // 경로 → UTF-8 문자열
cv::Mat     readGray(const fs::path& p);                  // 한글 경로 대응 영상 읽기
bool        parseTruth(const std::string& stem, double& y, double& t);   // "Y+0.05 T+0.01 (2)", "T-0.35", "center"

// 목록 만들기
std::vector<Item> listSet1(const fs::path& imgDir);       // 1차 세트 (center 제외) : 값 순서로 짝수 번째 = 학습
std::vector<Item> listSet2(const fs::path& set2Dir);      // 2차 세트 (기준 위치의 center 포함)

// 측정 (진행 상황 출력)
void measureAll(const ea::EccAligner& al, std::vector<Item>& items, const char* label);

// 결과 저장 : MFC 앱 [결과 그래프] 와 같은 열 (file, trueY_mm, estY_mm, errY_um, trueT_deg, estT_deg, errT_deg, …)
bool writeCsv(const fs::path& path, const std::vector<Item>& items, double tolY_mm, double tolT_deg);

// 같은 자세 3장 반복성 그림
cv::Mat drawRepeatability(const std::vector<Item>& set2, double tolT_deg);

} // namespace ev

#endif // __ECC_EVAL_HPP__
