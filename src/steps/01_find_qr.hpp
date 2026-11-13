#pragma once
// 1단계: QR 3개가 기준 사진에서 가로로 몇 px 움직였는지 잰다.
//
// 방법 (NCC = 조각을 한 칸씩 옮겨 가며 가장 닮은 자리를 찾는 방법)
//   1. 기준 사진의 QR 조각과 새 사진을 살짝 흐린다 (선명한 사진에서 소수 위치가 치우치는 것을 막는다)
//   2. QR 주변 탐색 창에서 NCC 로 가장 닮은 자리를 찾는다 (1 px 단위)
//   3. 그 자리와 양옆의 닮은 정도로 포물선을 그려 소수 px 까지 구한다
//
// 탐색 창은 좁게 시작해서, 못 찾으면 넓힌다: ±110 px → ±250 px → 사진 전체 폭
#include "common/reference.hpp"

namespace alignment {

enum class QrSearch {
    Narrow,      // ±110 px 에서 3개 모두 찾음 (평소)
    Wide,        // ±250 px 에서 3개 모두 찾음 (큰 이동)
    FullWidth,   // 사진 전체 폭에서 3개 모두 찾음 (아주 큰 이동)
    Partial,     // 사진 전체 폭에서 1~2개만 찾음 (QR 일부가 가려졌거나 사진 밖)
    NotFound     // 하나도 못 찾음
};

struct QrShift {
    QrSearch search = QrSearch::NotFound;
    double dx = 0;         // 찾은 QR 들의 가로 이동 평균 (px, 오른쪽이 +)
    double dy = 0;         // 세로 이동 평균 (px, 아래가 +). Y·T 계산에는 쓰지 않고 나사를 찾을 때만 쓴다
    int found_count = 0;   // 찾은 QR 수
    double lowest_similarity = 0;   // 찾은 QR 들의 NCC 중 가장 낮은 값 (0~1)
};

class QrFinder {
public:
    explicit QrFinder(const Reference& reference);
    QrShift find_qr_shift(const cv::Mat& gray) const;

private:
    struct Match { bool found = false; double dx = 0, dy = 0, similarity = 0; };
    // QR 하나를 가로 ±half_width 창에서 찾는다. half_width 가 0 이면 사진 전체 폭.
    Match find_one(const cv::Mat& gray, int qr_index, int half_width) const;
    // 세 QR 을 같은 창 크기로 찾아 평균을 낸다.
    QrShift find_all(const cv::Mat& gray, int half_width) const;

    cv::Rect boxes_[3];
    cv::Mat templates_[3];   // 흐린 기준 사진에서 잘라 낸 QR 조각
};

}   // namespace alignment
