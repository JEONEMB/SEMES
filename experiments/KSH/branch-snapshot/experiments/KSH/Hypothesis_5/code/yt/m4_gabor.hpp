// =====================================================================================
// m4_gabor.hpp — M4 : Gabor 국소 위상으로 화소별 가로 이동 → QR 별 기울기(원근)
//
//  근거 : Miao 2023 (Vision-based displacement measurements using local image phase), 기존 H1 구현
//  필터 : 복소 Gabor (가로 파장 λ = 12 px, σx = 4, σy = 6 — H1 과 같은 값). 응답 위상 φ 는 영상이 d 밀리면 같이 밀림
//  이동 : d(x) = gi − Δφ / φ'  (gi = 미리 맞춘 정수 이동, Δφ = 현재 − 기준 위상, φ' = 기준·현재 위상 기울기의 평균)
//  신뢰 : 응답 크기 하위 40 % 제외, 위상 기울기가 2π/λ 에서 30 % 이상 벗어난 화소 제외
//  QR 별 : 열 중앙값 d(x) 를 d = t + s·(x − cx)/100 으로 맞춤 → t = 이동, s = 100 px 당 늘어남 (= 100·ux)
// =====================================================================================
#ifndef YT_M4_GABOR_HPP
#define YT_M4_GABOR_HPP

#include "yt_common.hpp"

namespace yt {
namespace m4 {

struct QrFit { double t = 0, s = 0; int n = 0; };
struct BandResult {
    QrFit q[3];            // QR_L, QR_C, QR_R
    std::vector<double> col;   // 띠의 열별 이동 (px), NaN = 무효
    int x0 = 0;
    bool ok = false;
};

// ref·cur 에서 띠 [x0, x1) × [y0, y1) 의 열별 이동과 QR 별 맞춤. gi = 정수 사전 이동 (현재 영상에서 띠를 자를 위치)
// qrCx = QR 중심 x 3개 (기준 좌표)
BandResult measureBand(const cv::Mat& ref, const cv::Mat& cur, int x0, int x1, int y0, int y1, int gi, const int qrCx[3]);

bool verify(const std::vector<Sample>& ds, const std::string& outDir, const std::string& m1Csv);

} // namespace m4
} // namespace yt

#endif
