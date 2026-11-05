// =====================================================================================
// sim_draw.hpp — 시뮬레이터 화면 그림 (콘솔 창 · MFC 공용, BGR 8 bit 영상으로 돌려줌)
//   cv::putText 는 한글을 못 쓰므로 그림 안 글자는 영어. 설명은 각 앱의 한글 문구로.
// =====================================================================================
#ifndef SIM_DRAW_HPP
#define SIM_DRAW_HPP

#include "sim_runner.hpp"
#include "alignment.hpp"

#include <opencv2/core.hpp>

namespace sim {

// 그림 글자 배율 (설정 display.plot_font_scale). 그리기 전에 한 번 정함
void setPlotFontScale(double scale);
double plotFontScale();

// 현재 카메라 사진 + 모델이 본 ROI + 참값·추정값 글자 (가로 width_px 로 축소)
cv::Mat drawCamera(const cv::Mat& img8, const StepRecord* step, const AlignmentInfo* info, int width_px);
// 파노라마 + 기준 위치(초록) · 지나온 위치(회색) · 현재 위치(빨강) 사각형
cv::Mat drawPanorama(const World& w, const TrialResult* trial, int width_px);
// (Y, Θ) 평면 궤적 : 왼쪽 = 전체, 오른쪽 = 허용 범위 확대
cv::Mat drawPlane(const TrialResult& t, const SimConfig& cfg, cv::Size size);
// 반복마다 |참 오차| / 허용 범위 (로그 눈금)
cv::Mat drawConvergence(const TrialResult& t, const SimConfig& cfg, cv::Size size);
// 다회 시험 : 이동 횟수 분포 + 멈춘 뒤 오차 산점도
cv::Mat drawMonte(const MonteSummary& m, const SimConfig& cfg, cv::Size size);
// 운동장 지도 : Y 감도 (px/mm, 깊이에 반비례) 를 색으로
cv::Mat drawField(const World& w, int width_px);

} // namespace sim

#endif
