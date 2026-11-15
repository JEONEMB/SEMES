// 1단계: QR 띠 봉우리로 QR 3개를 찾고 평균 이동량 계산
#include "steps/01_find_qr.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>

namespace alignment {

namespace {

const double kBlurSigma = 2.0;          // 정밀 비교 전 잔무늬를 줄이는 흐림 정도(px)
const int kBlurMargin = 8;              // 필터 경계의 영향을 줄이기 위해 탐색 영역 밖에 추가하는 여유 폭(px)
const double kCoarseBlurSigma = 0.5;    // 줄인 영상의 흐림 정도(px)

// shrink: 대략 탐색용으로 줄이기 (면적 평균 후 약한 흐림)
//   입력: image (회색 영상)
//   출력: 1/kCoarseScale 크기 영상
cv::Mat shrink(const cv::Mat& image) {
    cv::Mat small;
    cv::resize(image, small, cv::Size(), 1.0 / kCoarseScale, 1.0 / kCoarseScale, cv::INTER_AREA);
    cv::GaussianBlur(small, small, cv::Size(0, 0), kCoarseBlurSigma);
    return small;
}

// blurred_area: 가장자리까지 고르게 흐림 처리
//   입력: gray (회색 사진), area (최종 영역)
//   출력: 흐림 처리 후 area 크기로 자른 영상
cv::Mat blurred_area(const cv::Mat& gray, const cv::Rect& area) {
    const cv::Rect wider = cv::Rect(area.x - kBlurMargin, area.y - kBlurMargin, area.width + 2 * kBlurMargin, area.height + 2 * kBlurMargin) & kPhotoRect;
    cv::Mat blurred;
    cv::GaussianBlur(gray(wider), blurred, cv::Size(0, 0), kBlurSigma);
    return blurred(area - wider.tl()).clone();
}

// parabola_offset: 연속한 값 3개를 지나는 포물선의 꼭대기 위치
//   입력: before, at_peak, after (가운데가 가장 큰 연속한 값 3개)
//   출력: 가운데 칸 기준 꼭대기 위치(-0.5~+0.5 px)
double parabola_offset(double before, double at_peak, double after) {
    const double curvature = before - 2 * at_peak + after;
    return curvature < 0 ? 0.5 * (before - after) / curvature : 0.0;
}

// gaussian_offset: 가장 닮은 위치의 앞뒤 값으로 소수 이동량 계산 (3점 가우스 맞춤)
//   흐린 영상끼리의 닮은 정도 봉우리는 종 모양에 가깝다. 종 모양은 로그를 취하면 포물선이므로
//   세 값의 로그로 포물선 꼭대기를 구한다. 값이 0 이하이면 로그를 쓸 수 없어 포물선으로 계산
//   입력: before, at_peak, after (연속한 닮은 정도 3개)
//   출력: 소수 이동량(-0.5~+0.5 px)
double gaussian_offset(double before, double at_peak, double after) {
    if (before <= 0 || at_peak <= 0 || after <= 0) return parabola_offset(before, at_peak, after);
    return parabola_offset(std::log(before), std::log(at_peak), std::log(after));
}

}   // 파일 내부

// QrFinder: 비교할 기준 QR 과 QR 띠 준비
//   입력: reference (기준 사진과 QR 위치)
//   출력: QR별 정밀, 대략 비교 영상 저장
QrFinder::QrFinder(const Reference& reference) {
    cv::Mat blurred;
    cv::GaussianBlur(reference.photo, blurred, cv::Size(0, 0), kBlurSigma);
    for (int k = 0; k < 3; ++k) {
        boxes_[k] = reference.qr[k];
        fine_templates_[k] = blurred(boxes_[k]).clone();
        coarse_templates_[k] = shrink(reference.photo(boxes_[k]));
    }
    const cv::Rect all_qr = boxes_[0] | boxes_[1] | boxes_[2];
    band_ = cv::Rect(0, all_qr.y - kBandMarginY, kPhotoWidth, all_qr.height + 2 * kBandMarginY) & kPhotoRect;
}

// find_peaks: QR 띠의 QR 모양 봉우리
//   입력: gray (회색 사진)
//   출력: 높은 순 봉우리 최대 4개 (QR 왼쪽 위 모서리 사진 좌표)
std::vector<QrFinder::Peak> QrFinder::find_peaks(const cv::Mat& gray) const {
    // QR 템플릿 3개로 각각 비교하고 자리마다 가장 큰 값 사용 (QR 마다 자기 템플릿과 맞음)
    const cv::Mat small_band = shrink(gray(band_));
    cv::Mat scores[3];
    cv::parallel_for_(cv::Range(0, 3), [&](const cv::Range& range) {
        for (int k = range.start; k < range.end; ++k) cv::matchTemplate(small_band, coarse_templates_[k], scores[k], cv::TM_CCOEFF_NORMED);
    });
    // QR 상자 크기가 1 px 씩 다르면 비교 결과 크기도 다름. 왼쪽 위 기준 좌표는 같으므로 공통 크기로 잘라 비교
    const cv::Size common(std::min({scores[0].cols, scores[1].cols, scores[2].cols}), std::min({scores[0].rows, scores[1].rows, scores[2].rows}));
    const cv::Rect area(cv::Point(0, 0), common);
    const cv::Mat best = cv::max(cv::max(scores[0](area), scores[1](area)), scores[2](area));

    // 가로 곡선: 세로 방향 최댓값과 그 행
    std::vector<float> column_best(best.cols, -1.f);
    std::vector<int> column_row(best.cols, 0);
    for (int y = 0; y < best.rows; ++y)
        for (int x = 0; x < best.cols; ++x)
            if (best.at<float>(y, x) > column_best[x]) { column_best[x] = best.at<float>(y, x); column_row[x] = y; }

    // 봉우리: 양옆보다 높은 칸. 높은 순으로 kMinPeakGap 이상 떨어진 것만 남김
    std::vector<int> candidates;
    for (int x = 1; x + 1 < best.cols; ++x)
        if (column_best[x] >= column_best[x - 1] && column_best[x] > column_best[x + 1]) candidates.push_back(x);
    std::sort(candidates.begin(), candidates.end(), [&](int a, int b) { return column_best[a] > column_best[b]; });

    std::vector<Peak> peaks;
    for (int x : candidates) {
        const int photo_x = band_.x + x * kCoarseScale;
        const bool separate = std::all_of(peaks.begin(), peaks.end(), [&](const Peak& p) { return std::abs(p.x - photo_x) >= kMinPeakGap; });
        if (!separate) continue;
        peaks.push_back({photo_x, band_.y + column_row[x] * kCoarseScale, column_best[x]});
        if (peaks.size() == 4) break;
    }
    return peaks;
}

// find_one: QR 하나의 정밀 위치
//   입력: gray (회색 사진), qr_index (0~2), coarse_dx와 coarse_dy (대략 정수 이동 px)
//   출력: Match (검출 여부, 소수 이동량, 닮은 정도)
QrFinder::Match QrFinder::find_one(const cv::Mat& gray, int qr_index, int coarse_dx, int coarse_dy) const {
    Match match;
    const cv::Rect box = boxes_[qr_index];
    const cv::Rect window = cv::Rect(box.x + coarse_dx - kFineHalfWidth, box.y + coarse_dy - kFineHalfWidth,
                                     box.width + 2 * kFineHalfWidth, box.height + 2 * kFineHalfWidth) & kPhotoRect;
    if (window.width <= box.width || window.height <= box.height) return match;

    cv::Mat similarity;
    // 대략 위치 둘레에서 기준 QR 과 가장 닮은 위치 찾기
    cv::matchTemplate(blurred_area(gray, window), fine_templates_[qr_index], similarity, cv::TM_CCOEFF_NORMED);
    double best = 0;
    cv::Point peak;
    cv::minMaxLoc(similarity, nullptr, &best, nullptr, &peak);

    // 앞뒤 비교값이 없는 경계는 소수 위치 계산 제외
    const bool at_window_edge = peak.x <= 0 || peak.y <= 0 || peak.x >= similarity.cols - 1 || peak.y >= similarity.rows - 1;
    if (best < kMinSimilarity || at_window_edge) return match;

    const double fine_x = gaussian_offset(similarity.at<float>(peak.y, peak.x - 1), similarity.at<float>(peak.y, peak.x), similarity.at<float>(peak.y, peak.x + 1));
    const double fine_y = gaussian_offset(similarity.at<float>(peak.y - 1, peak.x), similarity.at<float>(peak.y, peak.x), similarity.at<float>(peak.y + 1, peak.x));
    match.found = true;
    match.dx = window.x + peak.x + fine_x - box.x;
    match.dy = window.y + peak.y + fine_y - box.y;
    match.similarity = best;
    return match;
}

// find_qr_shift: QR 3개의 이동량
//   입력: gray (회색 사진)
//   출력: QrShift (검출 여부, 평균 이동량, 실패 사유)
QrShift QrFinder::find_qr_shift(const cv::Mat& gray) const {
    QrShift shift;

    // 대략: 문턱 이상 봉우리가 정확히 3개여야 함
    std::vector<Peak> qr_peaks;
    for (const Peak& peak : find_peaks(gray))
        if (peak.similarity >= kPeakThreshold) qr_peaks.push_back(peak);
    shift.peak_count = int(qr_peaks.size());
    if (qr_peaks.size() != 3) {
        shift.problem = "QR 띠에서 QR 모양 " + std::to_string(qr_peaks.size()) + "개 검출 (3개여야 함)";
        return shift;
    }

    // 왼쪽부터 QR 1, 2, 3. 이웃 간격이 기준 간격과 맞아야 함
    std::sort(qr_peaks.begin(), qr_peaks.end(), [](const Peak& a, const Peak& b) { return a.x < b.x; });
    for (int k = 0; k < 2; ++k) {
        const double gap = qr_peaks[k + 1].x - qr_peaks[k].x;
        if (std::abs(gap - expected_gap(k)) > kQrGapTolerance) {
            shift.problem = "QR " + std::to_string(k + 1) + ", " + std::to_string(k + 2) + " 간격 " + std::to_string(int(std::lround(gap))) +
                            " px (기준 " + std::to_string(int(std::lround(expected_gap(k)))) + " px)";
            return shift;
        }
    }

    // 정밀: QR 3개를 각각 동시에. 대략 이동 = 봉우리 자리 - 기준 QR 자리
    cv::parallel_for_(cv::Range(0, 3), [&](const cv::Range& range) {
        for (int k = range.start; k < range.end; ++k) shift.qr[k] = find_one(gray, k, qr_peaks[k].x - boxes_[k].x, qr_peaks[k].y - boxes_[k].y);
    });

    shift.lowest_similarity = 1.0;
    for (int k = 0; k < 3; ++k) {
        if (!shift.qr[k].found) {
            shift.problem = "QR " + std::to_string(k + 1) + " 정밀 위치 미확보 (닮은 정도 0.8 미만)";
            return shift;
        }
        shift.dx += shift.qr[k].dx;
        shift.dy += shift.qr[k].dy;
        shift.lowest_similarity = std::min(shift.lowest_similarity, shift.qr[k].similarity);
    }
    shift.dx /= 3;
    shift.dy /= 3;
    shift.found = true;
    return shift;
}

}   // alignment
