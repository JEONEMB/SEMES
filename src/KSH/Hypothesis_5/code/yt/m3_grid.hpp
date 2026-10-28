// =====================================================================================
// m3_grid.hpp — M3 : 샘플링 모아레 (Ri 2010) / 격자법 적용 가능성 (Grédiac 2016)
//
//  샘플링 모아레 : 피치 p (정수 px) 인 격자 영상의 각 행을 p 간격으로 솎아(시작점 k = 0..p−1) 선형 보간하면
//                 위상이 2πk/p 씩 밀린 모아레 무늬 p 장이 생김 → 위상 이동법으로 위상 φ(x)
//                 변위 u = p · Δφ / 2π (기준 대비), 변형률 ε = (p / 2π) · ∂Δφ/∂x
//  적용 조건   : 대상이 '간격이 일정한 격자' 여야 함 → 주기성 점수(행 평균 파워 스펙트럼의 최대 봉우리 / 중앙값)로 판정
// =====================================================================================
#ifndef YT_M3_GRID_HPP
#define YT_M3_GRID_HPP

#include "yt_common.hpp"

namespace yt {
namespace m3 {

// 영상(행 방향 격자)의 화소별 모아레 위상 (rad, 감기 풀지 않음), CV_64F
cv::Mat moirePhase(const cv::Mat& img32, int pitch);
// 두 영상의 위상 차로 행 평균 가로 변위 (px) 와 변형률
struct MoireResult { double u = 0, eps = 0; bool ok = false; };
MoireResult moireDisplacement(const cv::Mat& ref32, const cv::Mat& cur32, int pitch);

// 주기성 점수 : 행마다 Hann 창 FFT 파워를 평균 → 최대 봉우리 / 중앙값, 봉우리 주기(px)
struct Periodicity { double score = 0, period = 0; };
Periodicity periodicity(const cv::Mat& img32);

bool verify(const std::vector<Sample>& ds, const std::string& outDir);

} // namespace m3
} // namespace yt

#endif
