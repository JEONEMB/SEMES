#include "setup/find_new_reference.hpp"
#include <opencv2/imgproc.hpp>
#include <cmath>

namespace alignment {

namespace {

const cv::Rect kPhoto(0, 0, kPhotoWidth, kPhotoHeight);
const int kQrSearch = 120;                // px. 원래 QR 상자 주변을 이만큼 찾는다 (±)
const double kQrMinSimilarity = 0.5;      // 이보다 닮지 않았으면 QR 을 못 찾은 것으로 본다
const int kScrewHalf = 70;                // px. 나사 주변에서 테두리를 보는 범위 (±)
const double kEdgeBlurSigma = 1.5;        // px
const double kStrongEdge = 0.35;          // 가장 센 테두리의 이 비율보다 센 곳만 나사 테두리로 본다
const int kCenterSteps = 3;               // 나사 중심을 고쳐 잡는 횟수

// QR 상자: 원래 상자 주변에서 NCC 로 가장 닮은 자리를 찾는다.
bool find_qr_box(const cv::Mat& old_photo, const cv::Mat& new_photo, const cv::Rect& old_box, cv::Rect& new_box) {
    const cv::Rect window = cv::Rect(old_box.x - kQrSearch, old_box.y - kQrSearch, old_box.width + 2 * kQrSearch, old_box.height + 2 * kQrSearch) & kPhoto;
    cv::Mat similarity;
    cv::matchTemplate(new_photo(window), old_photo(old_box), similarity, cv::TM_CCOEFF_NORMED);
    double best = 0;
    cv::Point peak;
    cv::minMaxLoc(similarity, nullptr, &best, nullptr, &peak);
    new_box = cv::Rect(window.x + peak.x, window.y + peak.y, old_box.width, old_box.height);
    return best > kQrMinSimilarity;
}

// 나사 중심: 테두리가 센 화소들의 무게중심으로 중심을 몇 번 고쳐 잡는다.
bool find_screw_center(const cv::Mat& new_photo, cv::Point old_center, cv::Point& new_center) {
    new_center = old_center;
    for (int step = 0; step < kCenterSteps; ++step) {
        const cv::Rect area(new_center.x - kScrewHalf, new_center.y - kScrewHalf, 2 * kScrewHalf + 1, 2 * kScrewHalf + 1);
        if ((area & kPhoto) != area) return false;

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

}   // namespace

bool find_new_reference(const Reference& old_reference, const cv::Mat& new_photo, Reference& new_reference, std::string& error) {
    new_reference = old_reference;
    new_reference.photo = new_photo.clone();
    for (int k = 0; k < 3; ++k) {
        if (!find_qr_box(old_reference.photo, new_photo, old_reference.qr[k], new_reference.qr[k])) {
            error = "새 센터 사진에서 QR " + std::to_string(k + 1) + " 을 찾지 못했습니다";
            return false;
        }
    }
    for (int k = 0; k < 4; ++k) {
        if (!find_screw_center(new_photo, old_reference.screw[k], new_reference.screw[k])) {
            error = "새 센터 사진에서 나사 " + std::to_string(k + 1) + " 를 찾지 못했습니다";
            return false;
        }
    }
    return true;
}

}   // namespace alignment
