#pragma once
// 1단계: QR 가로 이동량(px)
// 처리: QR 띠에서 QR 모양 봉우리 찾기(대략) -> 개수와 간격 확인 -> QR 마다 소수 위치 계산(정밀)
// 판정: 봉우리가 정확히 3개이고 간격이 맞을 때만 왼쪽부터 QR 1, 2, 3. 그 밖은 실패
#include "common/reference.hpp"
#include <string>
#include <vector>

namespace alignment {

// 대략 단계 (QR 띠: 사진 전체 폭 × QR 상자 세로 범위 ± kBandMarginY)
const int kBandMarginY = 40;            // QR 띠 세로 여유(px). 판의 세로 이동 허용 범위
const int kCoarseScale = 4;             // 대략 탐색 축소 배율 (1/4)
const double kPeakThreshold = 0.5;      // QR 봉우리 문턱. 실측: QR 봉우리 최소 0.77, QR 이 아닌 봉우리 최대 0.25
const int kMinPeakGap = 100;            // 서로 다른 봉우리로 보는 최소 가로 거리(px)
const int kQrGapTolerance = 25;         // 이웃 QR 간격 허용 오차(px). 기준 간격은 reference.json QR 상자에서 계산

// 정밀 단계
const int kFineHalfWidth = 8;           // 대략 위치 둘레 정밀 탐색 반너비(px). 가로, 세로 공통
const double kMinSimilarity = 0.8;      // 정밀 위치를 인정하는 최소 닮은 정도

struct QrMatch {
    bool found = false;           // 정밀 위치를 얻었는가
    double dx = 0, dy = 0;        // 기준 QR 자리에서의 소수 이동(px)
    double similarity = 0;        // 닮은 정도(0~1)
};

struct QrShift {
    bool found = false;           // QR 3개 모두 정밀 위치를 얻었는가
    QrMatch qr[3];                // QR 1, 2, 3 의 정밀 결과 (대략 판정을 통과했을 때만 채움)
    double dx = 0;                // QR 3개 평균 가로 이동(px, 오른쪽이 +)
    double dy = 0;                // QR 3개 평균 세로 이동(px, 아래가 +). 나사 탐색용
    double lowest_similarity = 0; // 정밀 단계 QR 3개의 최소 닮은 정도(0~1)
    int peak_count = 0;           // 대략 단계에서 문턱 이상 봉우리 수
    std::string problem;          // 실패 사유 (found = false 일 때)
};

class QrFinder {
public:
    // QrFinder: QR 기준 영상과 QR 띠 준비
    //   입력: reference (기준 사진과 QR 위치)
    //   출력: QR 탐색기
    explicit QrFinder(const Reference& reference);

    // find_qr_shift: QR 3개의 이동량
    //   입력: gray (회색 사진)
    //   출력: QrShift (검출 여부, 평균 이동량, 실패 사유)
    QrShift find_qr_shift(const cv::Mat& gray) const;

    struct Peak { int x = 0, y = 0; double similarity = 0; };   // 대략 위치: QR 왼쪽 위 모서리의 사진 좌표(px, 정수)

    // find_peaks: QR 띠의 QR 모양 봉우리
    //   입력: gray (회색 사진)
    //   출력: 높은 순 봉우리 최대 4개. 문턱 미만도 포함 (점검, 그림용)
    std::vector<Peak> find_peaks(const cv::Mat& gray) const;

    using Match = QrMatch;

    // find_one: QR 하나의 정밀 위치
    //   입력: gray (회색 사진), qr_index (0~2), coarse_dx와 coarse_dy (대략 정수 이동 px)
    //   출력: Match (검출 여부, 소수 이동량, 닮은 정도)
    Match find_one(const cv::Mat& gray, int qr_index, int coarse_dx, int coarse_dy) const;

    // qr_band: QR 띠 영역 (사진 좌표, 그림용)
    cv::Rect qr_band() const { return band_; }

    // expected_gap: 이웃 QR 의 기준 간격(px)
    double expected_gap(int left_index) const { return boxes_[left_index + 1].x - boxes_[left_index].x; }

private:
    cv::Rect boxes_[3];
    cv::Rect band_;
    cv::Mat fine_templates_[3];     // 흐린 기준 사진에서 잘라 낸 QR 영상
    cv::Mat coarse_templates_[3];   // 1/4 로 줄인 QR 영상
};

}   // alignment
