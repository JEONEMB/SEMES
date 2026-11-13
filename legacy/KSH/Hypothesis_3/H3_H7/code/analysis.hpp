// =====================================================================================
// analysis.hpp — H3 · H6 · H7 오차 분석 도구
//
//  physicsCheck  : 변위장 계수를 Θ·Y 에 대해 맞춰 순수 카메라 회전 예측(f·θ, u², uv 항)과 비교
//  noiseFloor    : 영상 안 점 잔차로 u² 계수 불확도 → 점 잡음만으로 생기는 Θ 한계
//  temporalNoise : 연속 촬영 자세의 촬영 간 점 잔차 차이 → 시간 잡음
//  ParallaxTruth : 배경 시차 추정값(legacy/center_check 결과 CSV)을 정답으로 사용
//  backlash      : 배경 시차 추정과 지령값의 차이를 접근 방향(부호)별로 비교
// =====================================================================================
#ifndef __H3_ANALYSIS_HPP__
#define __H3_ANALYSIS_HPP__

#include "dataset.hpp"
#include "field_model.hpp"
#include "session_h7.hpp"
#include <string>
#include <vector>

namespace h3 {

struct TermFit { std::string name; double slope = 0, quad = 0, resid = 0; };
struct PhysicsCheck { std::vector<TermFit> theta, y; };
PhysicsCheck physicsCheck(const std::vector<Item>& items, const std::vector<cv::Point2d>& P0, const std::vector<Track>& tr);

struct NoiseFloor { double sigU2 = 0, sensU2 = 0, thetaSigma = 0; };
NoiseFloor noiseFloor(const std::vector<int>& idx, const std::vector<cv::Point2d>& P0, const std::vector<Track>& tr,
                      const FieldSpec& s, double sensU2);

struct TemporalRow { std::string pose; double pointSd = 0; double u2[3] = { 0, 0, 0 }; double thetaEq[3] = { 0, 0, 0 }; };
std::vector<TemporalRow> temporalNoise(const std::vector<Item>& items, const std::vector<cv::Point2d>& P0,
                                       const std::vector<Track>& tr, const FieldSpec& s, double sensU2);

// 배경 시차 정답 : 영상 번호별 (Y, Θ) (없으면 valid=false)
struct Truth { double y_mm = 0, t_deg = 0; bool valid = false; };
std::vector<Truth> loadParallaxTruth(const std::vector<Item>& items, const std::string& csvSet1, const std::string& csvSet2);

struct VsTruth { int n = 0, pass = 0; double rms = 0, bias = 0, truthVsLabelRms = 0; };
VsTruth compareToTruth(const std::vector<int>& idx, const std::vector<Estimate>& est,
                       const std::vector<Truth>& truth, const std::vector<Item>& items);

// 접근 방향 차 : 배경 시차 − 지령값 을 부호별 평균 → 절반 차이
struct Backlash { double halfY_um = 0, halfT_deg = 0; int nY = 0, nT = 0; };
Backlash backlashFromTruth(const std::vector<Item>& items, const std::vector<Truth>& truth);

// 결과 CSV 에서 1차 세트 Θ 스윕 검증(홀수) 행 통계 (legacy 방법 비교용)
bool set1ThetaTestStat(const std::string& csv, EvalStat& st);

} // namespace h3

#endif
