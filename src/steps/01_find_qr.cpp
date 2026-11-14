// 1단계: QR 위치 탐색과 평균 이동량 계산
#include "steps/01_find_qr.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>

namespace alignment {

namespace {

const double kBlurSigma = 2.0;        // QR 비교 전 잔무늬를 줄이는 흐림 정도(px)
const int kBlurMargin = 8;            // 필터 경계의 영향을 줄이기 위해 탐색 영역 밖에 추가하는 여유 폭(px)
const double kMinSimilarity = 0.8;    // QR 검출을 허용하는 최소 닮은 정도

// blurred_area: 가장자리까지 고르게 흐림 처리
//   입력: gray (회색 사진), area (최종 영역)
//   출력: 흐림 처리 후 area 크기로 자른 영상
cv::Mat blurred_area(const cv::Mat& gray, const cv::Rect& area) {
    const cv::Rect wider = cv::Rect(area.x - kBlurMargin, area.y - kBlurMargin, area.width + 2 * kBlurMargin, area.height + 2 * kBlurMargin) & kPhotoRect;
    cv::Mat blurred;
    cv::GaussianBlur(gray(wider), blurred, cv::Size(0, 0), kBlurSigma);
    return blurred(area - wider.tl()).clone();
}

// parabola_offset: 가장 닮은 위치의 앞뒤 값으로 소수 이동량 계산
//   입력: before, at_peak, after (연속한 닮은 정도 3개)
//   출력: 소수 이동량(-0.5~+0.5 px)
double parabola_offset(double before, double at_peak, double after) {
    const double curvature = before - 2 * at_peak + after;
    return curvature < 0 ? 0.5 * (before - after) / curvature : 0.0;
}

}   // 파일 내부

// QrFinder: 비교할 기준 QR 준비
//   입력: reference (기준 사진과 QR 위치)
//   출력: QR별 비교 사진 저장
QrFinder::QrFinder(const Reference& reference) {
    cv::Mat blurred;
    cv::GaussianBlur(reference.photo, blurred, cv::Size(0, 0), kBlurSigma);
    for (int k = 0; k < 3; ++k) {
        boxes_[k] = reference.qr[k];
        templates_[k] = blurred(boxes_[k]).clone();
    }
}

// find_one: QR 하나 탐색
//   입력: gray (회색 사진), qr_index (0~2), half_width (가로 반너비 px, 0이면 전체 폭)
//   출력: Match (검출 여부, 이동량, 닮은 정도)
QrFinder::Match QrFinder::find_one(const cv::Mat& gray, int qr_index, int half_width) const {
    Match match;
    const cv::Rect box = boxes_[qr_index];
    const int left = half_width > 0 ? box.x - half_width : 0;
    const int right = half_width > 0 ? box.x + box.width + half_width : kPhotoWidth;
    const cv::Rect window = cv::Rect(left, box.y - kSearchHalfHeight, right - left, box.height + 2 * kSearchHalfHeight) & kPhotoRect;
    if (window.width <= box.width || window.height <= box.height) return match;

    cv::Mat similarity;
    // 탐색 영역에서 기준 QR과 가장 닮은 위치 찾기
    cv::matchTemplate(blurred_area(gray, window), templates_[qr_index], similarity, cv::TM_CCOEFF_NORMED);
    double best = 0;
    cv::Point peak;
    cv::minMaxLoc(similarity, nullptr, &best, nullptr, &peak);

    // 앞뒤 비교값이 없는 경계는 소수 위치 계산 제외
    const bool at_window_edge = peak.x <= 0 || peak.y <= 0 || peak.x >= similarity.cols - 1 || peak.y >= similarity.rows - 1;
    if (best < kMinSimilarity || at_window_edge) return match;

    const double fine_x = parabola_offset(similarity.at<float>(peak.y, peak.x - 1), similarity.at<float>(peak.y, peak.x), similarity.at<float>(peak.y, peak.x + 1));
    const double fine_y = parabola_offset(similarity.at<float>(peak.y - 1, peak.x), similarity.at<float>(peak.y, peak.x), similarity.at<float>(peak.y + 1, peak.x));
    match.found = true;
    match.dx = window.x + peak.x + fine_x - box.x;
    match.dy = window.y + peak.y + fine_y - box.y;
    match.similarity = best;
    return match;
}

// find_all: 같은 범위에서 QR 3개 탐색
//   입력: gray (회색 사진), half_width (가로 반너비 px, 0이면 전체 폭)
//   출력: QrShift (검출한 QR의 평균 이동량)
QrShift QrFinder::find_all(const cv::Mat& gray, int half_width) const {
    Match matches[3];
    // QR 3개의 위치를 각각 동시에 탐색
    cv::parallel_for_(cv::Range(0, 3), [&](const cv::Range& range) {
        for (int k = range.start; k < range.end; ++k) matches[k] = find_one(gray, k, half_width);
    });

    QrShift shift;
    shift.lowest_similarity = 1.0;
    for (const Match& match : matches) {
        if (!match.found) continue;
        shift.dx += match.dx;
        shift.dy += match.dy;
        shift.lowest_similarity = std::min(shift.lowest_similarity, match.similarity);
        ++shift.found_count;
    }
    if (shift.found_count == 0) { shift.lowest_similarity = 0; return shift; }
    shift.dx /= shift.found_count;
    shift.dy /= shift.found_count;
    return shift;
}

// find_qr_shift: 범위를 넓혀 QR 탐색
//   입력: gray (회색 사진)
//   출력: QrShift (검출 상태와 평균 이동량)
QrShift QrFinder::find_qr_shift(const cv::Mat& gray) const {
    QrShift shift = find_all(gray, kNarrowHalfWidth);
    if (shift.found_count == 3) { shift.search = QrSearch::Narrow; return shift; }

    shift = find_all(gray, kWideHalfWidth);
    if (shift.found_count == 3) { shift.search = QrSearch::Wide; return shift; }

    shift = find_all(gray, 0);   // 사진 전체 폭
    if (shift.found_count == 3) shift.search = QrSearch::FullWidth;
    else if (shift.found_count > 0) shift.search = QrSearch::Partial;
    else shift.search = QrSearch::NotFound;
    return shift;
}

}   // alignment
