// 새 기준의 QR 영역과 나사 중심 검출. 영점 연결: alignment.cpp
#include "setup/find_new_reference.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>

namespace alignment {

namespace {

const int kQrSearch = 120;                // 기존 QR 주변 탐색 반너비(px)
const double kQrMinSimilarity = 0.5;      // QR 최소 닮은 정도
const int kScrewHalf = 70;                // px. 나사 주변에서 테두리를 보는 범위 (±)
const double kEdgeBlurSigma = 1.5;        // 테두리 비교 전 흐림 정도(px)
const double kStrongEdge = 0.35;          // 가장 뚜렷한 테두리 대비 사용할 밝기 변화 비율
const int kCenterSteps = 3;               // 나사 중심을 고쳐 잡는 횟수

// find_qr_box: 가장 닮은 새 QR 위치 찾기
//   입력: old_photo (기준 사진), new_photo (새 사진), old_box (기존 QR 영역)
//   출력: 성공 여부, new_box (새 QR 영역)
bool find_qr_box(const cv::Mat& old_photo, const cv::Mat& new_photo, const cv::Rect& old_box, cv::Rect& new_box) {
    const cv::Rect window = cv::Rect(old_box.x - kQrSearch, old_box.y - kQrSearch, old_box.width + 2 * kQrSearch, old_box.height + 2 * kQrSearch) & kPhotoRect;
    cv::Mat similarity;
    cv::matchTemplate(new_photo(window), old_photo(old_box), similarity, cv::TM_CCOEFF_NORMED);
    double best = 0;
    cv::Point peak;
    cv::minMaxLoc(similarity, nullptr, &best, nullptr, &peak);
    new_box = cv::Rect(window.x + peak.x, window.y + peak.y, old_box.width, old_box.height);
    return best > kQrMinSimilarity;
}

// find_screw_center: 뚜렷한 테두리를 기준으로 나사 중심 찾기
//   입력: new_photo (새 사진), old_center (기존 나사 중심)
//   출력: 성공 여부, new_center (새 나사 중심)
bool find_screw_center(const cv::Mat& new_photo, cv::Point old_center, cv::Point& new_center) {
    new_center = old_center;
    for (int step = 0; step < kCenterSteps; ++step) {
        const cv::Rect area(new_center.x - kScrewHalf, new_center.y - kScrewHalf, 2 * kScrewHalf + 1, 2 * kScrewHalf + 1);
        if (!inside_photo(area)) return false;

        cv::Mat as_float, blurred, slope_x, slope_y, strength;
        new_photo(area).convertTo(as_float, CV_32F);
        cv::GaussianBlur(as_float, blurred, cv::Size(0, 0), kEdgeBlurSigma);
        cv::Sobel(blurred, slope_x, CV_32F, 1, 0, 3);
        cv::Sobel(blurred, slope_y, CV_32F, 0, 1, 3);
        cv::magnitude(slope_x, slope_y, strength);
        double strongest = 0;
        cv::minMaxLoc(strength, nullptr, &strongest);

        double total = 0, weighted_x = 0, weighted_y = 0;
        for (int y = 0; y < strength.rows; ++y) {
            for (int x = 0; x < strength.cols; ++x) {
                const double value = strength.at<float>(y, x);
                if (value < kStrongEdge * strongest) continue;
                total += value;
                weighted_x += value * (x - kScrewHalf);
                weighted_y += value * (y - kScrewHalf);
            }
        }
        if (!(total > 0)) return false;
        new_center.x += int(std::nearbyint(weighted_x / total));
        new_center.y += int(std::nearbyint(weighted_y / total));
    }
    return true;
}

}   // 파일 내부

// find_new_reference: 새 기준 위치 검출
//   입력: old_reference (기존 기준), new_photo (새 회색 사진)
//   출력: 성공 여부, new_reference (새 기준), error (실패 사유)
bool find_new_reference(const Reference& old_reference, const cv::Mat& new_photo, Reference& new_reference, std::string& error) {
    new_reference = old_reference;
    new_reference.photo = new_photo.clone();
    for (int k = 0; k < 3; ++k) {
        if (!find_qr_box(old_reference.photo, new_photo, old_reference.qr[k], new_reference.qr[k])) {
            error = "새 기준 사진에서 QR " + std::to_string(k + 1) + " 미검출";
            return false;
        }
    }
    // QR 3개의 이동량 중 가운뎃값만큼 나사 탐색 시작점도 옮긴다.
    // 화면이 크게 밀려도 기존 위치 주변의 다른 테두리를 나사로 잡지 않게 한다.
    int qr_dx[3], qr_dy[3];
    for (int k = 0; k < 3; ++k) {
        qr_dx[k] = new_reference.qr[k].x - old_reference.qr[k].x;
        qr_dy[k] = new_reference.qr[k].y - old_reference.qr[k].y;
    }
    std::sort(qr_dx, qr_dx + 3);
    std::sort(qr_dy, qr_dy + 3);
    const cv::Point common_shift(qr_dx[1], qr_dy[1]);
    for (int k = 0; k < 4; ++k) {
        if (!find_screw_center(new_photo, old_reference.screw[k] + common_shift, new_reference.screw[k])) {
            error = "새 기준 사진에서 나사 " + std::to_string(k + 1) + " 중심 미확보";
            return false;
        }
    }
    return true;
}

}   // alignment
