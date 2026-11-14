// 2단계: QR 이동량 근처에서 나사를 찾고 위치를 세밀하게 조정
// 처리: 닮은 위치 찾기 -> 밝기 차이 보정 -> 테두리 위치 맞추기
#include "steps/02_find_screws.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <vector>

namespace alignment {

namespace {

// 1. 초기 위치 탐색 설정
const int kPatchRadius = 45;                       // px. 나사 영상 영역의 반지름
const int kPatchSide = 2 * kPatchRadius + 1;       // 91 px
const int kNearSearch = 60;                        // QR 이동 위치 주변 탐색 반너비(px)
const int kFarSearch = 260;                        // 미검출 시 기준 위치 주변 탐색 반너비(px)
const double kMinSimilarity = 0.6;                 // 나사 검출을 허용하는 최소 닮은 정도
const int kMaxAwayFromQr = 100;                    // QR 이동량과의 최대 허용 차이(px)

// 2. 밝기 비교 설정
const int kBrightnessMaxSteps = 50;
const double kBrightnessDone = 1e-4;               // 반복 종료 이동량(px)
const double kBrightnessMaxStep = 3.0;             // 반복 1회 최대 이동량(px)
const double kMaxDriftFromLocate = 8.0;            // 초기 검출 위치 대비 최대 이동량(px)

// 3. 테두리 위치 맞춤 설정
const int kEdgeHalf = 70;                          // px. 윤곽선 영상의 반쪽 크기
const int kEdgeSide = 2 * kEdgeHalf + 1;           // 141 px
const int kEdgeMaskRadius = 45;                    // 위치 비교 대상 원의 반지름(px)
const double kEdgeBlurSigma = 1.5;                 // px. 테두리를 뽑기 전에 흐리는 정도
const int kEdgeSmoothing = 5;                      // px. 맞추기 전에 한 번 더 흐리는 창 크기
const int kEdgeMaxSteps = 200;
const double kEdgeDone = 1e-8;                     // 반복 종료 닮은 정도 변화량

// square_around: 중심 주변 사각 영역 계산
//   입력: center (중심), half (중심에서 한쪽 끝까지 px)
//   출력: 중심을 포함한 정사각 영역
cv::Rect square_around(cv::Point center, int half) {
    return cv::Rect(center.x - half, center.y - half, 2 * half + 1, 2 * half + 1);
}

//
// 밝기 비교용 소수 위치의 색값 계산
//

// cubic_weight: 주변 픽셀을 얼마나 반영할지 계산
//   입력: distance (픽셀과의 거리)
//   출력: 소수 위치 밝기 계산에 곱할 값
double cubic_weight(double distance) {
    const double a = -0.75;
    const double x = std::abs(distance);
    if (x <= 1) return (a + 2) * x * x * x - (a + 3) * x * x + 1;
    if (x < 2) return a * x * x * x - 5 * a * x * x + 8 * a * x - 4 * a;
    return 0;
}

// cut_patch: 소수 좌표에서 나사 영역 추출
//   입력: gray (회색 사진), left와 top (잘라낼 영역의 왼쪽 위 좌표)
//   출력: 성공 여부, patch (91×91). 영상 밖이면 false
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

//
// 테두리가 가장 잘 겹치는 위치 찾기
//

// edge_picture: 윤곽선 영상 생성
//   입력: patch (나사 영역)
//   출력: 밝기 변화량 영상
cv::Mat edge_picture(const cv::Mat& patch) {
    cv::Mat as_float, blurred, slope_x, slope_y, strength;
    patch.convertTo(as_float, CV_32F);
    cv::GaussianBlur(as_float, blurred, cv::Size(0, 0), kEdgeBlurSigma);
    cv::Sobel(blurred, slope_x, CV_32F, 1, 0, 3);
    cv::Sobel(blurred, slope_y, CV_32F, 0, 1, 3);
    cv::magnitude(slope_x, slope_y, strength);
    return strength;
}

// value_at: 주변 네 점으로 소수 위치의 밝기 계산
//   입력: picture (사진), x, y (소수 좌표)
//   출력: 계산한 밝기. 사진 밖은 0
double value_at(const cv::Mat& picture, double x, double y) {
    const int x0 = int(std::floor(x)), y0 = int(std::floor(y));
    const double fx = x - x0, fy = y - y0;
    auto pixel = [&](int yy, int xx) {
        return (xx < 0 || yy < 0 || xx >= picture.cols || yy >= picture.rows) ? 0.0 : double(picture.at<float>(yy, xx));
    };
    return (1 - fy) * ((1 - fx) * pixel(y0, x0) + fx * pixel(y0, x0 + 1)) + fy * ((1 - fx) * pixel(y0 + 1, x0) + fx * pixel(y0 + 1, x0 + 1));
}

//
// 기준 사진과 테두리가 가장 잘 겹치는 이동량 계산
// 계산 방식: OpenCV ECC(밝기 차이를 보정한 사진 비교)
// 소수 위치를 1/32 px로 반올림하지 않고 직접 계산
//
// 반복 계산 순서
// (가) read_shifted_patch: 옮긴 사진과 밝기 변화량 읽기
// (나) sum_ecc_terms: 평균 밝기를 빼고 두 사진 비교
// (다) update_ecc_shift: 밝기 차이를 줄이고 위치 조정
//

// 사진 비교 중 재사용할 자료. 픽셀 번호: y * kEdgeSide + x
struct EccWork {
    cv::Mat smooth, slope_x, slope_y;                           // 흐린 사진과 가로, 세로 밝기 변화량. 원 밖은 0
    std::vector<double> moved, moved_slope_x, moved_slope_y;    // 현재 이동량을 적용한 사진과 밝기 변화량
    std::vector<double> reference_centered;                     // 기준 영상 - 평균 (원 밖은 0)
    std::vector<unsigned char> in_mask;                         // 옮겨 읽은 자리가 원 안인지
    int low = 0, high = 0;                                      // 나사 주변 원 안의 계산 범위
};

// (나) 에서 구하는 합
struct EccSums {
    double moved_power = 0, reference_power = 0, product = 0;   // 이동한 영상², 기준 영상², 둘의 곱
    double hxx = 0, hxy = 0, hyy = 0, determinant = 0;          // 가로, 세로 위치 조정용 합계와 계산 가능 여부 확인값
    double moved_on_x = 0, moved_on_y = 0;                      // 밝기 변화량과 이동 사진을 같은 위치끼리 곱한 합
    double reference_on_x = 0, reference_on_y = 0;              // 밝기 변화량과 기준 사진을 같은 위치끼리 곱한 합
};

// prepare_ecc: 테두리 위치 비교에 쓸 자료 준비
//   입력: current (현재 사진), mask (비교할 영역)
//   출력: 흐린 사진, 밝기 변화량, 계산용 저장 공간
EccWork prepare_ecc(const cv::Mat& current, const cv::Mat& mask) {
    EccWork work;
    cv::GaussianBlur(current, work.smooth, cv::Size(kEdgeSmoothing, kEdgeSmoothing), 0, 0);
    const cv::Matx13f half_difference(-0.5f, 0.0f, 0.5f);
    cv::filter2D(work.smooth, work.slope_x, -1, half_difference);
    cv::filter2D(work.smooth, work.slope_y, -1, half_difference.t());
    work.slope_x.setTo(0, mask == 0);
    work.slope_y.setTo(0, mask == 0);
    const int count = kEdgeSide * kEdgeSide;
    work.moved.resize(count);
    work.moved_slope_x.resize(count);
    work.moved_slope_y.resize(count);
    work.reference_centered.resize(count);
    work.in_mask.resize(count);
    return work;
}

// read_shifted_patch: 이동한 사진과 밝기 변화량 읽기
//   입력: work (사진 비교 중간 자료), mask (비교할 영역 표시), shift_x (가로 이동량), shift_y (세로 이동량)
//   출력: 없음 (work에 이동한 사진과 밝기 변화량 저장)
void read_shifted_patch(EccWork& work, const cv::Mat& mask, double shift_x, double shift_y) {
    const int reach = kEdgeMaskRadius + 3 + int(std::ceil(std::max(std::abs(shift_x), std::abs(shift_y))));
    work.low = std::max(0, kEdgeHalf - reach);
    work.high = std::min(kEdgeSide - 1, kEdgeHalf + reach);
    for (int y = work.low; y <= work.high; ++y) {
        for (int x = work.low; x <= work.high; ++x) {
            const int i = y * kEdgeSide + x;
            const double source_x = x + shift_x, source_y = y + shift_y;
            work.moved[i] = value_at(work.smooth, source_x, source_y);
            work.moved_slope_x[i] = value_at(work.slope_x, source_x, source_y);
            work.moved_slope_y[i] = value_at(work.slope_y, source_x, source_y);
            const int mask_x = cvRound(source_x), mask_y = cvRound(source_y);
            work.in_mask[i] = (mask_x >= 0 && mask_y >= 0 && mask_x < kEdgeSide && mask_y < kEdgeSide && mask.at<uchar>(mask_y, mask_x)) ? 1 : 0;
        }
    }
}

// sum_ecc_terms: 두 사진의 닮은 정도와 위치 조정값 계산
//   입력: reference (기준 사진), work (사진 비교 중간 자료)
//   출력: 성공 여부, sums (위치 조정용 합계). 계산 불가 시 false
bool sum_ecc_terms(const cv::Mat& reference, EccWork& work, EccSums& sums) {
    double moved_sum = 0, reference_sum = 0;
    int inside = 0;
    for (int y = work.low; y <= work.high; ++y) {
        for (int x = work.low; x <= work.high; ++x) {
            const int i = y * kEdgeSide + x;
            if (work.in_mask[i]) { moved_sum += work.moved[i]; reference_sum += reference.at<float>(y, x); ++inside; }
        }
    }
    if (inside == 0) return false;
    const double moved_mean = moved_sum / inside, reference_mean = reference_sum / inside;

    sums = EccSums();
    for (int y = work.low; y <= work.high; ++y) {
        for (int x = work.low; x <= work.high; ++x) {
            const int i = y * kEdgeSide + x;
            if (work.in_mask[i]) {
                work.moved[i] -= moved_mean;
                work.reference_centered[i] = double(reference.at<float>(y, x)) - reference_mean;
                sums.moved_power += work.moved[i] * work.moved[i];
                sums.reference_power += work.reference_centered[i] * work.reference_centered[i];
                sums.product += work.reference_centered[i] * work.moved[i];
            } else {
                work.reference_centered[i] = 0;
            }
            sums.hxx += work.moved_slope_x[i] * work.moved_slope_x[i];
            sums.hxy += work.moved_slope_x[i] * work.moved_slope_y[i];
            sums.hyy += work.moved_slope_y[i] * work.moved_slope_y[i];
            sums.moved_on_x += work.moved_slope_x[i] * work.moved[i];
            sums.moved_on_y += work.moved_slope_y[i] * work.moved[i];
            sums.reference_on_x += work.moved_slope_x[i] * work.reference_centered[i];
            sums.reference_on_y += work.moved_slope_y[i] * work.reference_centered[i];
        }
    }
    sums.determinant = sums.hxx * sums.hyy - sums.hxy * sums.hxy;
    return sums.determinant > 0 && sums.moved_power > 0 && sums.reference_power > 0;
}

// update_ecc_shift: 밝기 배율 보정 후 이동량 갱신
//   입력: work (사진 비교 중간 자료), sums (위치 조정용 합계), shift_x (가로 이동량), shift_y (세로 이동량)
//   출력: 성공 여부, 갱신한 shift_x와 shift_y
bool update_ecc_shift(const EccWork& work, const EccSums& sums, double& shift_x, double& shift_y) {
    const double solved_x = (sums.hyy * sums.moved_on_x - sums.hxy * sums.moved_on_y) / sums.determinant;
    const double solved_y = (sums.hxx * sums.moved_on_y - sums.hxy * sums.moved_on_x) / sums.determinant;
    const double scale_top = sums.moved_power - (sums.moved_on_x * solved_x + sums.moved_on_y * solved_y);
    const double scale_bottom = sums.product - (sums.reference_on_x * solved_x + sums.reference_on_y * solved_y);
    if (scale_bottom <= 0) return false;
    const double scale = scale_top / scale_bottom;
    double error_on_x = 0, error_on_y = 0;
    for (int y = work.low; y <= work.high; ++y) {
        for (int x = work.low; x <= work.high; ++x) {
            const int i = y * kEdgeSide + x;
            const double error = scale * work.reference_centered[i] - work.moved[i];
            error_on_x += work.moved_slope_x[i] * error;
            error_on_y += work.moved_slope_y[i] * error;
        }
    }
    shift_x += (sums.hyy * error_on_x - sums.hxy * error_on_y) / sums.determinant;
    shift_y += (sums.hxx * error_on_y - sums.hxy * error_on_x) / sums.determinant;
    return true;
}

// ecc_shift: 닮은 정도가 더 변하지 않을 때까지 위치 조정
//   입력: reference와 current (기준과 현재 사진), mask (비교 영역), shift_x와 shift_y (초기 가로, 세로 이동)
//   출력: 성공 여부, 최종 shift_x와 shift_y
bool ecc_shift(const cv::Mat& reference, const cv::Mat& current, const cv::Mat& mask, double& shift_x, double& shift_y) {
    EccWork work = prepare_ecc(current, mask);
    shift_x = shift_y = 0;
    double likeness = -1, previous_likeness = -kEdgeDone;
    for (int step = 1; step <= kEdgeMaxSteps && std::abs(likeness - previous_likeness) >= kEdgeDone; ++step) {
        read_shifted_patch(work, mask, shift_x, shift_y);
        EccSums sums;
        if (!sum_ecc_terms(reference, work, sums)) return false;
        previous_likeness = likeness;
        likeness = sums.product / std::sqrt(sums.moved_power * sums.reference_power);
        if (std::isnan(likeness)) return false;
        if (!update_ecc_shift(work, sums, shift_x, shift_y)) return false;
    }
    return true;
}

// make_edge_mask: 나사 중심 주변만 비교할 원형 영역 생성
//   입력: 없음
//   출력: 비교할 픽셀을 표시한 영상. 가장자리 한 겹 제외
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

}   // 파일 내부

// ScrewFinder: 비교할 기준 나사 준비
//   입력: reference (기준 사진과 나사 위치)
//   출력: 나사별 밝기와 테두리 비교 사진 저장
ScrewFinder::ScrewFinder(const Reference& reference) {
    for (int k = 0; k < 4; ++k) {
        centers_[k] = reference.screw[k];
        locate_templates_[k] = reference.photo(square_around(centers_[k], kPatchRadius)).clone();
        locate_templates_[k].convertTo(brightness_templates_[k], CV_64F);
        cv::GaussianBlur(edge_picture(reference.photo(square_around(centers_[k], kEdgeHalf))), edge_templates_[k], cv::Size(kEdgeSmoothing, kEdgeSmoothing), 0, 0);
    }
    edge_mask_ = make_edge_mask();
}

// find_screw_shifts: 나사별 이동량 측정
//   입력: gray (회색 사진), qr_dx와 qr_dy (QR 이동량 px), wanted (사용할 나사)
//   출력: 나사 1~4의 이동량과 검출 결과
std::array<ScrewShift, 4> ScrewFinder::find_screw_shifts(const cv::Mat& gray, double qr_dx, double qr_dy, const std::array<bool, 4>& wanted) const {
    std::array<ScrewShift, 4> shifts;
    const cv::Point expected_shift(int(std::lround(qr_dx)), int(std::lround(qr_dy)));
    // 나사 4개의 위치를 각각 동시에 탐색
    cv::parallel_for_(cv::Range(0, 4), [&](const cv::Range& range) {
        for (int k = range.start; k < range.end; ++k) {
            if (wanted[k]) shifts[k] = find_one(gray, k, expected_shift);
            else shifts[k].problem = "설정에서 쓰지 않는 나사";
        }
    });
    return shifts;
}

// find_one: 나사 하나의 위치를 3단계로 맞추기
//   입력: gray (회색 사진), screw_index (0~3), expected_shift (예상 이동량 px)
//   출력: ScrewShift (이동량, 검출 여부, 실패 사유)
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
    // 밝기로 맞춘 위치에서 사진 자르기. 남은 이동은 테두리로 계산
    const cv::Point start(cvRound(rough_dx), cvRound(rough_dy));
    if (!refine_by_edges(gray, screw_index, start, result.dx, result.dy)) {
        result.problem = "나사 테두리를 맞추지 못함 (가림·반사)";
        return result;
    }
    result.found = true;
    return result;
}

// ScrewFinder::locate_screw: 기준 사진과 가장 닮은 위치 찾기
//   입력: gray (회색 사진), screw_index (나사 번호 0~3), expected_shift (예상 이동량 px)
//   출력: 성공 여부, shift (정수 이동량 px), similarity (닮은 정도)
bool ScrewFinder::locate_screw(const cv::Mat& gray, int screw_index, cv::Point expected_shift, cv::Point& shift, double& similarity) const {
    const cv::Mat& template8 = locate_templates_[screw_index];
    cv::Mat scores;

    // QR 이동 위치 주변 우선 탐색
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
    // 미검출 시 기준 위치 주변으로 탐색 확대
    const cv::Rect far_window = square_around(centers_[screw_index], kPatchRadius + kFarSearch);
    if (!inside_photo(far_window)) return false;
    cv::matchTemplate(gray(far_window), template8, scores, cv::TM_CCOEFF_NORMED);
    cv::Point peak;
    cv::minMaxLoc(scores, nullptr, &similarity, nullptr, &peak);
    shift = peak - cv::Point(kFarSearch, kFarSearch);
    // 오검출 방지용 QR 이동량 차이 제한
    const bool moves_with_qr = std::abs(shift.x - expected_shift.x) <= kMaxAwayFromQr && std::abs(shift.y - expected_shift.y) <= kMaxAwayFromQr;
    return similarity >= kMinSimilarity && moves_with_qr;
}

// ScrewFinder::refine_by_brightness: 밝기 차이를 줄여 위치 맞추기
//   입력: gray (회색 사진), screw_index (나사 번호 0~3), start (초기 이동량)
//   출력: 성공 여부, dx와 dy (px)
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

// ScrewFinder::refine_by_edges: 테두리를 맞추어 남은 이동량 계산
//   입력: gray (회색 사진), screw_index (나사 번호 0~3), start (초기 이동량)
//   출력: 성공 여부, dx와 dy (px)
bool ScrewFinder::refine_by_edges(const cv::Mat& gray, int screw_index, cv::Point start, double& dx, double& dy) const {
    const cv::Rect area = square_around(centers_[screw_index] + start, kEdgeHalf);
    if (!inside_photo(area)) return false;
    double remaining_x = 0, remaining_y = 0;
    if (!ecc_shift(edge_templates_[screw_index], edge_picture(gray(area)), edge_mask_, remaining_x, remaining_y)) return false;
    dx = start.x + remaining_x;
    dy = start.y + remaining_y;
    return std::isfinite(dx) && std::isfinite(dy);
}

}   // alignment
