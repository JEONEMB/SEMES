#pragma once
#include <opencv2/core.hpp>

// QR 값: QR 3개를 각각 통째로 센터 사진의 QR 조각과 NCC 로 맞춘 이동(정수 봉우리 + 3점 포물선 소수 이동)의 평균.
// 2026-11-13 사용자 지시로 536점 추적(약 29 ms + 초기화 0.4 초)을 대신한다. 실험 20261113_1700 에서 정확도가 같고(처음 보는 날 99.70% 대 99.64%)
// 출력 차이가 Y 1.5 µm, T 0.2 mdeg 흔들림이며 약 5 ms 다. 탐색 창은 536점 추적과 같다(QR 상자 ± site.json 의 search_x, search_y).
namespace final00 {
struct QrShift {
    double dx, dy;        // QR 3개 이동의 평균 (px, 새 − 센터)
    double min_ncc;       // QR 3개 NCC 봉우리의 최솟값
};
class QrNcc {
public:
    explicit QrNcc(const cv::Mat& center_gray);
    // gray: 3840×2160 8비트 회색. NCC 가 0.8 아래거나 봉우리가 탐색 창 끝이면 예외.
    QrShift measure(const cv::Mat& gray) const;
private:
    cv::Mat templates_[3];
};
}
