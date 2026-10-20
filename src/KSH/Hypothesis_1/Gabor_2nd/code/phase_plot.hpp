// =====================================================================================
// phase_plot.hpp — 위상 기반 Y·Θ 분리 결과 시각화 (OpenCV 그리기만 사용, 추가 라이브러리 없음)
//   ※ OpenCV putText 는 한글을 그리지 못하므로 그래프 안 글자는 영어로 표기
// =====================================================================================
#ifndef __PHASE_PLOT_HPP__
#define __PHASE_PLOT_HPP__

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include "phase_align.hpp"

namespace pp {

// 산점도·선·허용 범위 띠를 그리는 그래프 한 칸
class Panel {
public:
    Panel(cv::Mat& canvas, cv::Rect area, double x0, double x1, double y0, double y1);
    void frame(const std::string& title, const std::string& xLabel, const std::string& yLabel);
    void band(double yLo, double yHi, cv::Scalar color);                 // 가로 띠 (허용 오차 범위)
    void hline(double y, cv::Scalar color, int thick = 1);
    void line(double xa, double ya, double xb, double yb, cv::Scalar color, int thick = 2);
    void point(double x, double y, cv::Scalar color, int r = 3, bool filled = true);
    void text(const std::string& s, int row, cv::Scalar color);         // 그래프 안 왼쪽 위 글자 (row = 줄 번호)
    cv::Point px(double x, double y) const;

    // 자료 범위에 맞춰 축 범위 정하기 (여유 margin 비율, 최소 범위 minSpan)
    static void autoRange(const std::vector<double>& v, double& lo, double& hi, double margin = 0.1, double minSpan = 0);

private:
    cv::Mat& c_;
    cv::Rect a_;
    double x0_, x1_, y0_, y1_;
};

// ① 특징 스윕 : 행 = [tC, sL, sC, sR], 열 = [Y 스윕(vs Y), Θ 스윕(vs Θ)], 학습/검증 점 + 보정 직선
cv::Mat plotFeatureSweeps(const std::vector<pa::Sample>& samples, const pa::Calibration& cal);

// ② 추정 오차 : 검증 샘플의 Y 오차(µm)·Θ 오차(deg)를 Y 스윕 / Θ 스윕별로, 허용 오차 띠와 함께
cv::Mat plotEstimation(const std::vector<pa::Sample>& samples, double tolY_mm, double tolT_deg);

// ③ QR 측정 오버레이 : 기준 위치(빨간 점선)·측정 위치(색 실선)·이동 화살표·t, s 값
cv::Mat drawQROverlay(const cv::Mat& image, const pa::PhaseAligner& aligner,
                      const pa::Features& f, const pa::Estimate& est, const std::string& title);

// ④ QR 별 측정 품질 : 유효 화소 비율, 맞춤 잔차, 위상 기울기 흩어짐 (전체 영상 평균)
cv::Mat plotQRQuality(const std::vector<pa::Sample>& samples);

// ⑤ 2변수 vs 3변수 비교 : 검증 샘플의 Y 오차를 두 모델로 겹쳐 그림
cv::Mat plotModelCompare(const std::vector<pa::Sample>& twoVar, const std::vector<pa::Sample>& threeVar,
                         double tolY_mm, double tolT_deg);

// ⑥ 추정된 공통 배율 변화(ppm) : Y 스윕 / Θ 스윕 사진 순서대로
cv::Mat plotMagnification(const std::vector<pa::Sample>& samples, const std::vector<double>& magPpm);

// 폴더 만들기 (중간 폴더까지 만들고, 이미 있으면 그대로)
void makeDir(const std::string& dir);

// 저장 + 화면 표시 (창 크기는 화면에 맞게 줄임)
void saveAndShow(const std::string& window, const cv::Mat& img, const std::string& path, bool show = true);

} // namespace pp

#endif // __PHASE_PLOT_HPP__
