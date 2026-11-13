// =====================================================================================
// bias_h6.hpp — 가설 H6 : 대응점의 서브픽셀 위치 의존 오차(pixel-locking) 진단과 보정
//
//  근거 : Sur 외 (2021), Bornert 외 — 보간·표본화가 서브픽셀 위치에 따라 주기적인 위치 치우침을 만든다
//  1) 영상마다 dx 를 기저(1, u, u², v, uv)로 맞춘 잔차 r 를 구함 (일반 최소제곱)
//  2) 점마다 고정된 성분(점별 평균)을 빼고, 남은 성분을 현재 영상 서브픽셀 위치 φ = frac(x) 의
//     1·2차 조화항 sin/cos(2πφ), sin/cos(4πφ) 로 설명 (전역 곡선 / 점별 곡선)
//  3) 보정 : 학습 영상으로 맞춘 점별 곡선을 모든 영상의 x 좌표에서 뺀 뒤 H3 를 다시 실행
// =====================================================================================
#ifndef __H3_BIAS_H6_HPP__
#define __H3_BIAS_H6_HPP__

#include "field_model.hpp"
#include <vector>

namespace h3 {

// dx 잔차 (일반 최소제곱, 무효 점 NaN)
std::vector<double> plainResidualX(const std::vector<cv::Point2d>& P0, const Track& t, const FieldSpec& s);

struct H6Diag {
    double residSd = 0;           // 전체 잔차 σ
    double pointConstSd = 0;      // 점마다 고정된 성분 σ
    double remainSd = 0;          // 고정 성분을 뺀 나머지 σ
    double globalExplained = 0;   // 전역 pixel-locking 곡선 설명력
    double globalAmp = 0;         // 전역 1차 조화 진폭 (px)
    double pointExplainedCV = 0;  // 점별 곡선 교차검증 설명력
};
H6Diag diagnoseH6(const std::vector<cv::Point2d>& P0, const std::vector<Track>& tracks,
                  const std::vector<int>& imgIdx, const FieldSpec& s);

// 점별 곡선 학습 (학습 영상) → 보정된 트랙
std::vector<Track> correctH6(const std::vector<cv::Point2d>& P0, const std::vector<Track>& tracks,
                             const std::vector<int>& trainIdx, const FieldSpec& s);

} // namespace h3

#endif
