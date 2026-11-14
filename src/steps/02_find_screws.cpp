#include "steps/02_find_screws.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <vector>

namespace alignment {

namespace {

// ---- 1. locate_screw 에서 쓰는 값 ----
const int kPatchRadius = 45;                       // px. 나사 조각의 반지름
const int kPatchSide = 2 * kPatchRadius + 1;       // 91 px
const int kNearSearch = 60;                        // px. QR 이 움직인 자리 주변을 먼저 찾는 범위 (±)
const int kFarSearch = 260;                        // px. 거기서 못 찾으면 기준 위치 주변을 넓게 찾는 범위 (±)
const double kMinSimilarity = 0.6;                 // NCC 가 이보다 낮으면 나사가 아니라고 본다
const int kMaxAwayFromQr = 100;                    // px. 나사 이동이 QR 이동과 이보다 다르면 다른 나사를 잡은 것으로 본다

// ---- 2. refine_by_brightness 에서 쓰는 값 ----
const int kBrightnessMaxSteps = 50;
const double kBrightnessDone = 1e-4;               // px. 한 번에 움직인 양이 이보다 작으면 끝낸다
const double kBrightnessMaxStep = 3.0;             // px. 한 번에 이보다 많이 움직이려 하면 잘못 맞춘 것으로 본다
const double kMaxDriftFromLocate = 8.0;            // px. 1 에서 찾은 자리에서 이보다 멀어지면 잘못 맞춘 것으로 본다

// ---- 3. refine_by_edges 에서 쓰는 값 ----
const int kEdgeHalf = 70;                          // px. 테두리 그림의 반쪽 크기
const int kEdgeSide = 2 * kEdgeHalf + 1;           // 141 px
const int kEdgeMaskRadius = 45;                    // px. 나사 중심에서 이 안쪽만 맞춘다
const double kEdgeBlurSigma = 1.5;                 // px. 테두리를 뽑기 전에 흐리는 정도
const int kEdgeSmoothing = 5;                      // px. 맞추기 전에 한 번 더 흐리는 창 크기
const int kEdgeMaxSteps = 200;
const double kEdgeDone = 1e-8;                     // 닮은 정도가 이보다 적게 변하면 끝낸다

cv::Rect square_around(cv::Point center, int half) {
    return cv::Rect(center.x - half, center.y - half, 2 * half + 1, 2 * half + 1);
}

// ------------------------------------------------------------------------------------------------
// 2 에서 쓰는 도구: 소수 px 위치에서 조각 잘라 내기 (3차 보간)
// ------------------------------------------------------------------------------------------------

double cubic_weight(double distance) {
    const double a = -0.75;
    const double x = std::abs(distance);
    if (x <= 1) return (a + 2) * x * x * x - (a + 3) * x * x + 1;
    if (x < 2) return a * x * x * x - 5 * a * x * x + 8 * a * x - 4 * a;
    return 0;
}

// gray 에서 왼쪽 위가 (left, top) 인 91×91 조각을 잘라 낸다. 사진 밖으로 나가면 false.
bool cut_patch(const cv::Mat& gray, double left, double top, cv::Mat& patch) {
    const int base_x = int(std::floor(left)), base_y = int(std::floor(top));
    if (base_x < 1 || base_y < 1 || base_x + kPatchSide + 2 >= gray.cols || base_y + kPatchSide + 2 >= gray.rows) return false;
    double weight_x[4], weight_y[4];
    for (int k = 0; k < 4; ++k) {
        weight_x[k] = cubic_weight(left - base_x - (k - 1));
        weight_y[k] = cubic_weight(top - base_y - (k - 1));
    }
    patch.create(kPatchSide, kPatchSide, CV_64F);
    for (int y = 0; y < kPatchSide; ++y) {
        for (int x = 0; x < kPatchSide; ++x) {
            double value = 0;
            for (int j = 0; j < 4; ++j) {
                const unsigned char* row = gray.ptr<unsigned char>(base_y + y + j - 1) + base_x + x - 1;
                double along_row = 0;
                for (int i = 0; i < 4; ++i) along_row += weight_x[i] * row[i];
                value += weight_y[j] * along_row;
            }
            patch.at<double>(y, x) = value;
        }
    }
    return true;
}

// ------------------------------------------------------------------------------------------------
// 3 에서 쓰는 도구: 테두리 그림과 ECC 맞춤
// ------------------------------------------------------------------------------------------------

// 밝기가 급히 바뀌는 곳(테두리)이 밝게 나오는 그림을 만든다.
cv::Mat edge_picture(const cv::Mat& patch) {
    cv::Mat as_float, blurred, slope_x, slope_y, strength;
    patch.convertTo(as_float, CV_32F);
    cv::GaussianBlur(as_float, blurred, cv::Size(0, 0), kEdgeBlurSigma);
    cv::Sobel(blurred, slope_x, CV_32F, 1, 0, 3);
    cv::Sobel(blurred, slope_y, CV_32F, 0, 1, 3);
    cv::magnitude(slope_x, slope_y, strength);
    return strength;
}

// 소수 px 위치의 값 (이웃 4점 보간). 그림 밖은 0.
double value_at(const cv::Mat& picture, double x, double y) {
    const int x0 = int(std::floor(x)), y0 = int(std::floor(y));
    const double fx = x - x0, fy = y - y0;
    auto pixel = [&](int yy, int xx) {
        return (xx < 0 || yy < 0 || xx >= picture.cols || yy >= picture.rows) ? 0.0 : double(picture.at<float>(yy, xx));
    };
    return (1 - fy) * ((1 - fx) * pixel(y0, x0) + fx * pixel(y0, x0 + 1)) + fy * ((1 - fx) * pixel(y0 + 1, x0) + fx * pixel(y0 + 1, x0 + 1));
}

// ECC 맞춤 (평행이동만): current 를 (shift_x, shift_y) 만큼 옮겨 읽으면 reference 와 가장 닮도록 shift 를 구한다.
// OpenCV findTransformECC(MOTION_TRANSLATION, 마스크) 와 같은 식이다. OpenCV 것은 옮겨 읽을 때 좌표를 1/32 px 로
// 끊어서 약 0.02 px 의 계단 오차가 생기기 때문에, 같은 식을 정확한 보간으로 직접 푼다.
bool ecc_shift(const cv::Mat& reference, const cv::Mat& current, const cv::Mat& mask, double& shift_x, double& shift_y) {
    cv::Mat smooth, slope_x, slope_y;
    cv::GaussianBlur(current, smooth, cv::Size(kEdgeSmoothing, kEdgeSmoothing), 0, 0);
    const cv::Matx13f half_difference(-0.5f, 0.0f, 0.5f);
    cv::filter2D(smooth, slope_x, -1, half_difference);
    cv::filter2D(smooth, slope_y, -1, half_difference.t());
    slope_x.setTo(0, mask == 0);
    slope_y.setTo(0, mask == 0);

    const int count = kEdgeSide * kEdgeSide;
    std::vector<double> moved(count), moved_slope_x(count), moved_slope_y(count), reference_centered(count);
    std::vector<unsigned char> in_mask(count);
    shift_x = shift_y = 0;
    double likeness = -1, previous_likeness = -kEdgeDone;

    for (int step = 1; step <= kEdgeMaxSteps && std::abs(likeness - previous_likeness) >= kEdgeDone; ++step) {
        // 원(마스크) 둘레 상자만 계산한다. 상자 밖은 모든 합에 0 만 더해서 결과가 같다.
        const int reach = kEdgeMaskRadius + 3 + int(std::ceil(std::max(std::abs(shift_x), std::abs(shift_y))));
        const int low = std::max(0, kEdgeHalf - reach), high = std::min(kEdgeSide - 1, kEdgeHalf + reach);

        // (가) 지금 shift 로 옮겨 읽는다
        double moved_sum = 0, reference_sum = 0;
        int inside = 0;
        for (int y = low; y <= high; ++y) {
            for (int x = low; x <= high; ++x) {
                const int i = y * kEdgeSide + x;
                const double source_x = x + shift_x, source_y = y + shift_y;
                moved[i] = value_at(smooth, source_x, source_y);
                moved_slope_x[i] = value_at(slope_x, source_x, source_y);
                moved_slope_y[i] = value_at(slope_y, source_x, source_y);
                const int mask_x = cvRound(source_x), mask_y = cvRound(source_y);
                in_mask[i] = (mask_x >= 0 && mask_y >= 0 && mask_x < kEdgeSide && mask_y < kEdgeSide && mask.at<uchar>(mask_y, mask_x)) ? 1 : 0;
                if (in_mask[i]) { moved_sum += moved[i]; reference_sum += reference.at<float>(y, x); ++inside; }
            }
        }
        if (inside == 0) return false;

        // (나) 평균을 빼고 닮은 정도(상관)와 맞춤에 필요한 합을 구한다
        const double moved_mean = moved_sum / inside, reference_mean = reference_sum / inside;
        double moved_power = 0, reference_power = 0, product = 0;
        double hxx = 0, hxy = 0, hyy = 0;                 // 기울기끼리의 합 (2×2 행렬)
        double moved_on_x = 0, moved_on_y = 0;            // 기울기 · 옮긴 그림
        double reference_on_x = 0, reference_on_y = 0;    // 기울기 · 기준 그림
        for (int y = low; y <= high; ++y) {
            for (int x = low; x <= high; ++x) {
                const int i = y * kEdgeSide + x;
                if (in_mask[i]) {
                    moved[i] -= moved_mean;
                    reference_centered[i] = double(reference.at<float>(y, x)) - reference_mean;
                    moved_power += moved[i] * moved[i];
                    reference_power += reference_centered[i] * reference_centered[i];
                    product += reference_centered[i] * moved[i];
                } else {
                    reference_centered[i] = 0;
                }
                hxx += moved_slope_x[i] * moved_slope_x[i];
                hxy += moved_slope_x[i] * moved_slope_y[i];
                hyy += moved_slope_y[i] * moved_slope_y[i];
                moved_on_x += moved_slope_x[i] * moved[i];
                moved_on_y += moved_slope_y[i] * moved[i];
                reference_on_x += moved_slope_x[i] * reference_centered[i];
                reference_on_y += moved_slope_y[i] * reference_centered[i];
            }
        }
        const double determinant = hxx * hyy - hxy * hxy;
        if (!(determinant > 0) || !(moved_power > 0) || !(reference_power > 0)) return false;
        previous_likeness = likeness;
        likeness = product / std::sqrt(moved_power * reference_power);
        if (std::isnan(likeness)) return false;

        // (다) 밝기 크기 차이(scale)를 맞춘 뒤, 남은 차이를 줄이는 쪽으로 shift 를 고친다
        const double solved_x = (hyy * moved_on_x - hxy * moved_on_y) / determinant;
        const double solved_y = (hxx * moved_on_y - hxy * moved_on_x) / determinant;
        const double scale_top = moved_power - (moved_on_x * solved_x + moved_on_y * solved_y);
        const double scale_bottom = product - (reference_on_x * solved_x + reference_on_y * solved_y);
        if (scale_bottom <= 0) return false;
        const double scale = scale_top / scale_bottom;
        double error_on_x = 0, error_on_y = 0;
        for (int y = low; y <= high; ++y) {
            for (int x = low; x <= high; ++x) {
                const int i = y * kEdgeSide + x;
                const double error = scale * reference_centered[i] - moved[i];
                error_on_x += moved_slope_x[i] * error;
                error_on_y += moved_slope_y[i] * error;
            }
        }
        shift_x += (hyy * error_on_x - hxy * error_on_y) / determinant;
        shift_y += (hxx * error_on_y - hxy * error_on_x) / determinant;
    }
    return true;
}

// 테두리 맞춤에 쓰는 원. OpenCV ECC 와 같은 방식으로 가장자리 한 겹을 뺀다.
cv::Mat make_edge_mask() {
    cv::Mat disk = cv::Mat::zeros(kEdgeSide, kEdgeSide, CV_32F);
    for (int y = 0; y < kEdgeSide; ++y)
        for (int x = 0; x < kEdgeSide; ++x)
            if (std::hypot(double(x - kEdgeHalf), double(y - kEdgeHalf)) <= kEdgeMaskRadius) disk.at<float>(y, x) = 1;
    cv::GaussianBlur(disk, disk, cv::Size(kEdgeSmoothing, kEdgeSmoothing), 0, 0);
    disk *= (0.5 / 0.95);
    cv::Mat mask;
    disk.convertTo(mask, CV_8U);
    return mask;
}

}   // namespace

ScrewFinder::ScrewFinder(const Reference& reference) {
    for (int k = 0; k < 4; ++k) {
        centers_[k] = reference.screw[k];
        reference.photo(square_around(centers_[k], kPatchRadius)).convertTo(brightness_templates_[k], CV_64F);
        cv::GaussianBlur(edge_picture(reference.photo(square_around(centers_[k], kEdgeHalf))), edge_templates_[k], cv::Size(kEdgeSmoothing, kEdgeSmoothing), 0, 0);
    }
    edge_mask_ = make_edge_mask();
}

std::array<ScrewShift, 4> ScrewFinder::find_screw_shifts(const cv::Mat& gray, double qr_dx, double qr_dy, const std::array<bool, 4>& wanted) const {
    std::array<ScrewShift, 4> shifts;
    const cv::Point expected_shift(int(std::lround(qr_dx)), int(std::lround(qr_dy)));
    // 나사 4개는 서로 독립이라 동시에 찾는다 (하나씩 찾을 때와 결과가 같다).
    cv::parallel_for_(cv::Range(0, 4), [&](const cv::Range& range) {
        for (int k = range.start; k < range.end; ++k) {
            if (wanted[k]) shifts[k] = find_one(gray, k, expected_shift);
            else shifts[k].problem = "설정에서 쓰지 않는 나사";
        }
    });
    return shifts;
}

ScrewShift ScrewFinder::find_one(const cv::Mat& gray, int screw_index, cv::Point expected_shift) const {
    ScrewShift result;
    cv::Point located;
    if (!locate_screw(gray, screw_index, expected_shift, located, result.similarity)) {
        result.problem = "나사를 찾지 못함 (가려졌거나 사진 밖)";
        return result;
    }
    double rough_dx = 0, rough_dy = 0;
    if (!refine_by_brightness(gray, screw_index, located, rough_dx, rough_dy)) {
        result.problem = "나사 위치를 다듬지 못함 (가림·반사)";
        return result;
    }
    const cv::Point start(cvRound(rough_dx), cvRound(rough_dy));
    if (!refine_by_edges(gray, screw_index, start, result.dx, result.dy)) {
        result.problem = "나사 테두리를 맞추지 못함 (가림·반사)";
        return result;
    }
    result.found = true;
    return result;
}

// 1. NCC 로 대략 찾기. shift = 기준 위치에서 움직인 양 (정수 px).
bool ScrewFinder::locate_screw(const cv::Mat& gray, int screw_index, cv::Point expected_shift, cv::Point& shift, double& similarity) const {
    cv::Mat template8, scores;
    brightness_templates_[screw_index].convertTo(template8, CV_8U);

    // 먼저 QR 이 움직인 만큼 옮긴 자리 주변을 찾는다.
    const cv::Rect near_window = square_around(centers_[screw_index] + expected_shift, kPatchRadius + kNearSearch);
    if (inside_photo(near_window)) {
        cv::matchTemplate(gray(near_window), template8, scores, cv::TM_CCOEFF_NORMED);
        cv::Point peak;
        cv::minMaxLoc(scores, nullptr, &similarity, nullptr, &peak);
        const bool at_window_edge = peak.x <= 0 || peak.y <= 0 || peak.x >= scores.cols - 1 || peak.y >= scores.rows - 1;
        if (!at_window_edge && similarity >= kMinSimilarity) {
            shift = peak - cv::Point(kNearSearch, kNearSearch) + expected_shift;
            return true;
        }
    }
    // 못 찾으면 기준 위치 주변을 넓게 찾는다.
    const cv::Rect far_window = square_around(centers_[screw_index], kPatchRadius + kFarSearch);
    if (!inside_photo(far_window)) return false;
    cv::matchTemplate(gray(far_window), template8, scores, cv::TM_CCOEFF_NORMED);
    cv::Point peak;
    cv::minMaxLoc(scores, nullptr, &similarity, nullptr, &peak);
    shift = peak - cv::Point(kFarSearch, kFarSearch);
    // 나사는 QR 과 거의 같이 움직인다. QR 이동과 크게 다르면 다른 나사를 잡은 것이다.
    const bool moves_with_qr = std::abs(shift.x - expected_shift.x) <= kMaxAwayFromQr && std::abs(shift.y - expected_shift.y) <= kMaxAwayFromQr;
    return similarity >= kMinSimilarity && moves_with_qr;
}

// 2. 밝기를 맞춰 가며 위치 다듬기. 조명 차이는 매번 밝기 배율(gain)과 더하기(offset)로 맞춘다.
bool ScrewFinder::refine_by_brightness(const cv::Mat& gray, int screw_index, cv::Point start, double& dx, double& dy) const {
    const cv::Mat& reference_patch = brightness_templates_[screw_index];
    const cv::Point center = centers_[screw_index];
    dx = start.x;
    dy = start.y;
    bool settled = false;
    for (int step = 0; step < kBrightnessMaxSteps && !settled; ++step) {
        cv::Mat patch, slope_x, slope_y;
        if (!cut_patch(gray, center.x - kPatchRadius + dx, center.y - kPatchRadius + dy, patch)) return false;
        cv::Sobel(patch, slope_x, CV_64F, 1, 0, 3, 1.0 / 8);
        cv::Sobel(patch, slope_y, CV_64F, 0, 1, 3, 1.0 / 8);

        const double patch_mean = cv::mean(patch)[0], reference_mean = cv::mean(reference_patch)[0];
        const cv::Mat patch_centered = patch - patch_mean, reference_centered = reference_patch - reference_mean;
        const double gain = patch_centered.dot(reference_centered) / std::max(patch_centered.dot(patch_centered), 1e-12);
        const double offset = reference_mean - gain * patch_mean;
        const cv::Mat difference = gain * patch + offset - reference_patch;
        slope_x *= gain;
        slope_y *= gain;

        const double hxx = slope_x.dot(slope_x), hxy = slope_x.dot(slope_y), hyy = slope_y.dot(slope_y);
        const double determinant = hxx * hyy - hxy * hxy;
        if (determinant < 1e-12) return false;
        const double pull_x = -slope_x.dot(difference), pull_y = -slope_y.dot(difference);
        const double step_x = (hyy * pull_x - hxy * pull_y) / determinant;
        const double step_y = (hxx * pull_y - hxy * pull_x) / determinant;
        if (!std::isfinite(step_x + step_y) || std::abs(step_x) > kBrightnessMaxStep || std::abs(step_y) > kBrightnessMaxStep) return false;
        dx += step_x;
        dy += step_y;
        settled = std::abs(step_x) + std::abs(step_y) < kBrightnessDone;
    }
    return settled && std::abs(dx - start.x) <= kMaxDriftFromLocate && std::abs(dy - start.y) <= kMaxDriftFromLocate;
}

// 3. 테두리 그림으로 정밀하게 맞추기. start 만큼 옮긴 자리에서 조각을 잘라 ECC 로 남은 이동을 구한다.
bool ScrewFinder::refine_by_edges(const cv::Mat& gray, int screw_index, cv::Point start, double& dx, double& dy) const {
    const cv::Rect area = square_around(centers_[screw_index] + start, kEdgeHalf);
    if (!inside_photo(area)) return false;
    double remaining_x = 0, remaining_y = 0;
    if (!ecc_shift(edge_templates_[screw_index], edge_picture(gray(area)), edge_mask_, remaining_x, remaining_y)) return false;
    dx = start.x + remaining_x;
    dy = start.y + remaining_y;
    return std::isfinite(dx) && std::isfinite(dy);
}

}   // namespace alignment
