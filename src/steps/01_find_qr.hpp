#pragma once
// 1단계: QR 가로 이동량(px)
// 처리: 잔무늬 줄이기 -> 가장 닮은 위치 찾기 -> 소수 픽셀 위치 계산
// 탐색 범위: ±110 px -> ±250 px -> 사진 전체 폭
#include "common/reference.hpp"

namespace alignment {

// QR 양옆 탐색 반너비. 알림 문구와 공유
const int kNarrowHalfWidth = 110;   // px. 기본 탐색
const int kWideHalfWidth = 250;     // px. 기본 탐색 창에서 3개를 모두 검출하지 못한 경우
const int kSearchHalfHeight = 22;   // 세로 탐색 반높이(px). 모든 가로 범위에 공통 적용

enum class QrSearch {
    Narrow,      // ±110 px에서 3개 모두 찾음 (기본 탐색)
    Wide,        // ±250 px에서 3개 모두 찾음 (큰 이동)
    FullWidth,   // 사진 전체 폭에서 3개 모두 찾음 (아주 큰 이동)
    Partial,     // 사진 전체 폭에서 1~2개만 찾음 (QR 일부가 가려졌거나 사진 밖)
    NotFound     // 전체 미검출
};

struct QrShift {
    QrSearch search = QrSearch::NotFound;
    double dx = 0;         // 검출 QR의 평균 가로 이동(px, 오른쪽이 +)
    double dy = 0;         // 세로 이동 평균(px, 아래가 +). 나사 탐색용
    int found_count = 0;   // 검출 QR 수
    double lowest_similarity = 0;   // 검출 QR의 최소 닮은 정도(0~1)
};

class QrFinder {
public:
    // QrFinder: QR 기준 영상 준비
    //   입력: reference (기준 사진과 QR 위치)
    //   출력: QR 탐색기
    explicit QrFinder(const Reference& reference);

    // find_qr_shift: 범위를 넓혀 QR 탐색
    //   입력: gray (회색 사진)
    //   출력: QrShift (검출 상태와 평균 이동량)
    QrShift find_qr_shift(const cv::Mat& gray) const;

    struct Match { bool found = false; double dx = 0, dy = 0, similarity = 0; };

    // find_one: QR 하나 탐색
    //   입력: gray (회색 사진), qr_index (0~2), half_width (가로 반너비 px, 0이면 전체 폭)
    //   출력: Match (검출 여부, 이동량, 닮은 정도)
    Match find_one(const cv::Mat& gray, int qr_index, int half_width) const;

private:
    // find_all: 같은 범위에서 QR 3개 탐색
    //   입력: gray (회색 사진), half_width (가로 반너비 px, 0이면 전체 폭)
    //   출력: QrShift (검출한 QR의 평균 이동량)
    QrShift find_all(const cv::Mat& gray, int half_width) const;

    cv::Rect boxes_[3];
    cv::Mat templates_[3];   // 흐린 기준 사진에서 잘라 낸 QR 영상 영역
};

}   // alignment
