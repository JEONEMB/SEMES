#include "steps/01_find_qr.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>

namespace alignment {

namespace {

const double kBlurSigma = 2.0;        // px. NCC 전에 흐리는 정도
const int kBlurMargin = 8;            // px. 흐림이 창 가장자리에서 달라지지 않게 이만큼 넓게 잘라 흐린다
const double kMinSimilarity = 0.8;    // NCC 가 이보다 낮으면 QR 이 아니라고 본다

// area 부분을 흐려서 돌려준다. 가장자리 효과가 없도록 조금 넓게 잘라 흐린 뒤 area 만 남긴다.
cv::Mat blurred_area(const cv::Mat& gray, const cv::Rect& area) {
    const cv::Rect wider = cv::Rect(area.x - kBlurMargin, area.y - kBlurMargin, area.width + 2 * kBlurMargin, area.height + 2 * kBlurMargin) & kPhotoRect;
    cv::Mat blurred;
    cv::GaussianBlur(gray(wider), blurred, cv::Size(0, 0), kBlurSigma);
    return blurred(area - wider.tl()).clone();
}

// 가장 닮은 자리(peak)와 양옆 값으로 포물선을 그려 꼭짓점이 peak 에서 얼마나 떨어졌는지 구한다 (−0.5 ~ +0.5 px).
double parabola_offset(double before, double at_peak, double after) {
    const double curvature = before - 2 * at_peak + after;
    return curvature < 0 ? 0.5 * (before - after) / curvature : 0.0;
}

}   // namespace

QrFinder::QrFinder(const Reference& reference) {
    cv::Mat blurred;
    cv::GaussianBlur(reference.photo, blurred, cv::Size(0, 0), kBlurSigma);
    for (int k = 0; k < 3; ++k) {
        boxes_[k] = reference.qr[k];
        templates_[k] = blurred(boxes_[k]).clone();
    }
}

QrFinder::Match QrFinder::find_one(const cv::Mat& gray, int qr_index, int half_width) const {
    Match match;
    const cv::Rect box = boxes_[qr_index];
    const int left = half_width > 0 ? box.x - half_width : 0;
    const int right = half_width > 0 ? box.x + box.width + half_width : kPhotoWidth;
    const cv::Rect window = cv::Rect(left, box.y - kSearchHalfHeight, right - left, box.height + 2 * kSearchHalfHeight) & kPhotoRect;
    if (window.width <= box.width || window.height <= box.height) return match;

    cv::Mat similarity;
    cv::matchTemplate(blurred_area(gray, window), templates_[qr_index], similarity, cv::TM_CCOEFF_NORMED);
    double best = 0;
    cv::Point peak;
    cv::minMaxLoc(similarity, nullptr, &best, nullptr, &peak);

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

QrShift QrFinder::find_all(const cv::Mat& gray, int half_width) const {
    Match matches[3];
    // QR 3개는 서로 독립이라 동시에 찾는다 (하나씩 찾을 때와 결과가 같다).
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

}   // namespace alignment
