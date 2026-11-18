// =====================================================================================
// h1_split.hpp — 가설 H1 검증 : "공유 Θ 오차는 화소 잡음이다" (QR 행 분할 시험)
//
//  방법 : 같은 사진에서 QR 영역을 위에서부터 같은 높이의 N 조각(2 = 위·아래, 3 = 상·중·하)으로 나눠
//         (경계마다 2·halfY 행을 비워 조각끼리 화소를 공유하지 않음) 조각마다 독립적으로
//         특징(t, s)을 재고, 조각마다 보정 모델을 따로 학습해 Θ·Y 를 추정한다.
//  판정 : 조각끼리의 Θ 오차 상관 ρ
//         ρ ≈ 0  → 화소마다 독립인 잡음      → 화소를 더 쓰거나 측정기를 개선하면 줄어듦
//         ρ ≈ 1  → 사진 전체에 공통인 오차    → 실제 흔들림·영상 전체 효과, 알고리즘으로 못 줄임
//         (N = 3) 이웃 조각 ρ ≈ 먼 조각(상–하) ρ → 공통 오차가 세로로 고르다
//                 이웃 조각 ρ  > 먼 조각 ρ       → 위아래로 변하는 성분이 있다 (판의 앞뒤 숙임 등, 추정)
//  근거 : 분할 반분 신뢰도(split-half reliability)의 원리를 응용한 자체 설계 시험 (특정 논문의 절차 아님)
//  출력 : <outDir>/h1_split_results.csv, h1_split_summary.txt, h1_split.png
// =====================================================================================
#ifndef __H1_SPLIT_HPP__
#define __H1_SPLIT_HPP__

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace h1 {

// 조각 두 개 (a, b) 사이의 비교
struct PairStat {
    int    a = 0, b = 0;       // 조각 번호 (0 = 맨 위)
    double rhoT = 0, rhoY = 0; // Θ·Y 오차 상관 (검증 샘플)
    double sharedT = 0;        // 공통 성분 크기 √max(0, cov) (deg)
    double rhoCurv = 0;        // 곡률 단서 κ = tL + tR − 2·tC 잔차 상관
    double rhoDs = 0;          // 배율 차 단서 Δs = sR − sL 잔차 상관
};

// 스윕(Y 또는 Θ) 하나에 대한 결과
struct SweepStat {
    int    n = 0;                  // 검증 샘플 수
    double fullT = 0, fullY = 0;   // 분할하지 않은 전체 측정의 Θ RMS (deg), Y RMS (um)
    std::vector<double> partT;     // 조각별 Θ RMS (deg)
    std::vector<double> partY;     // 조각별 Y RMS (um)
    double avgT = 0;               // 조각 평균 추정의 Θ RMS (deg)
    double indepT = 0;             // "조각 오차가 서로 독립"이라면 기대되는 조각 평균의 Θ RMS (deg)
    std::vector<PairStat> pairs;   // 모든 조각 쌍
};

struct Result {
    bool ok = false;
    int  nParts = 2;
    std::vector<std::pair<int, int>> rows;  // 조각별 사용 행 [시작, 끝) (QR 영역 기준)
    SweepStat sweep[2];        // [0] Y 스윕, [1] Θ 스윕
    std::string verdict;       // 판정 문장 (영문, 그래프·콘솔 공용)
};

// nParts : 2 (위·아래) 또는 3 (상·중·하),  outDir : 결과 폴더 (없으면 만듦),  gui : 창 표시 여부
Result runSplitTest(const cv::Mat& ref, const std::string& imgDir, const std::string& outDir,
                    int nParts, bool gui);

} // namespace h1

#endif // __H1_SPLIT_HPP__
