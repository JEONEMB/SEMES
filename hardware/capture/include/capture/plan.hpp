// plan.hpp — 학습·평가 사진 촬영 계획 (찍을 자세 목록과 파일 이름)
//
//  SEMES 가 준 사진과 같은 폴더 구성·파일 이름을 만든다. 그래야 김성현(align_train, dataset.cpp)·
//  이혜원(01_prepare_clean 이후 도구) 쪽이 코드 수정 없이 읽는다.
//
//   세트 1 (단독 축, 학습)      center.bmp,  Y/Y+0.05.bmp,  Theta/T-0.35.bmp
//   세트 2 (동시 이동, 평가)    center1.bmp, Y+0.50 T+0.10/Y+0.50 T+0.10 (1).bmp
//
//  값은 모두 센터 기준 상대값. 단위는 스테이지 프로토콜과 같은 정수 (Y µm, Θ 0.001°) — 실수 반올림 오차 없음
//  카메라·스테이지와 무관한 순수 함수라 PC 에서도 시험할 수 있다 (tests/plan_test.cpp)
#ifndef CAPTURE_PLAN_HPP
#define CAPTURE_PLAN_HPP

#include <cstdint>
#include <string>
#include <vector>

namespace capture {

// 센터 기준 상대 자세
struct Rel {
    int32_t y_um = 0;    // 1 = 1 µm
    int32_t t_mdeg = 0;  // 1 = 0.001°
};

// 사진 한 장
struct ShotSpec {
    Rel rel;
    std::string dir;   // 출력 폴더 아래 하위 폴더 ("" = 바로 아래)
    std::string stem;  // 확장자 뺀 파일 이름
};

// 1/1000 단위 정수 → "+0.05", "-1.20", "+0.045" (소수 2자리, 필요하면 3자리. 0 은 "+0.00")
std::string fmtMilli(int32_t v);

// "0.05", "-1.2", "+2" → 1/1000 단위 정수. 소수 넷째 자리 이하(프로토콜 분해능 밖)나 형식 오류면 false
bool parseMilli(const std::string &s, int32_t &out);

// 세트 1: Y 만 이동 (Θ = 0). -range ~ +range, step 간격, 0 제외. 폴더 "Y", 이름 "Y+0.05"
std::vector<ShotSpec> planY(int32_t range_um, int32_t step_um);

// 세트 1: Θ 만 이동 (Y = 0). 폴더 "Theta", 이름 "T+0.01"
std::vector<ShotSpec> planT(int32_t range_mdeg, int32_t step_mdeg);

// 세트 2: 동시 이동. 자세마다 repeat 장. 폴더·이름 "Y+0.50 T+0.10" / "Y+0.50 T+0.10 (1)"
std::vector<ShotSpec> planMixed(const std::vector<Rel> &poses, int repeat);

// 격자 (학습): Y -y_range~+y_range (y_step) × Θ -t_range~+t_range (t_step), 4사분면 모든 점 (센터 제외).
//  'ㄹ'자 순서: Θ 오름차순 줄마다 Y 방향을 번갈아 (짝수 줄 Y 오름, 홀수 줄 Y 내림) — 이미 이 순서로 돌려준다.
//  폴더 "Grid", 이름 "Y+0.10 T-0.05 (1)" (김성현 dataset.cpp 의 동시 이동 라벨 형식)
std::vector<ShotSpec> planGrid(int32_t y_range_um, int32_t y_step_um, int32_t t_range_mdeg, int32_t t_step_mdeg);

// 세트 2 기본 자세 20개: Y {0.05, 0.2, 0.5, 1.0} mm × Θ {0.01, 0.05, 0.2, 0.5, 1.0}°
//  (SEMES 2차: 양의 Y 0.05~1 mm × Θ 0.01~2°. Θ 2° 는 학습 범위 ±1° 밖이라 기본에서 뺌)
std::vector<Rel> defaultMixedPoses();

// "y_mm,t_deg" 한 줄에 하나. 빈 줄, # 주석, 숫자가 아닌 첫 줄(머리글)은 건너뜀
bool parsePoseCsv(const std::string &text, std::vector<Rel> &out, std::string &err);

// 이동 순서: Θ 오름차순, 같은 Θ 안에서 Y 오름차순 (이동 거리를 줄이고 접근 방향을 한쪽으로 유지)
void sortForTravel(std::vector<ShotSpec> &shots);

}  // namespace capture

#endif  // CAPTURE_PLAN_HPP
