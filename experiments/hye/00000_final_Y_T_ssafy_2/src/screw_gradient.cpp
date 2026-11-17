#include "screw_gradient.hpp"
#include "site.hpp"
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace final00 {
namespace {
const int kHalf = 70, kSide = 2 * kHalf + 1, kMaskRadius = 45, kGaussian = 5, kIterations = 200;
const double kEpsilon = 1e-8;

cv::Mat gradient_magnitude(const cv::Mat& patch8) {
    cv::Mat a, blurred, gx, gy, magnitude;
    patch8.convertTo(a, CV_32F);
    cv::GaussianBlur(a, blurred, cv::Size(0, 0), 1.5);
    cv::Sobel(blurred, gx, CV_32F, 1, 0, 3);
    cv::Sobel(blurred, gy, CV_32F, 0, 1, 3);
    cv::magnitude(gx, gy, magnitude);
    return magnitude;
}

cv::Rect patch_rect(cv::Point center) { return {center.x - kHalf, center.y - kHalf, kSide, kSide}; }

// 영상 밖은 0 으로 보는 정확한 양선형 보간. cv::warpAffine 은 OpenCV 3.4.15 에서 좌표를 1/32 px 로 끊어
// 평행이동 추정에 ±0.018 px 의 계단 오차를 만든다. 그래서 직접 보간한다.
double bilinear(const cv::Mat& m, double x, double y) {
    const int x0 = int(std::floor(x)), y0 = int(std::floor(y));
    const double fx = x - x0, fy = y - y0;
    auto at = [&](int yy, int xx) { return (xx < 0 || yy < 0 || xx >= m.cols || yy >= m.rows) ? 0.0 : double(m.at<float>(yy, xx)); };
    return (1 - fy) * ((1 - fx) * at(y0, x0) + fx * at(y0, x0 + 1)) + fy * ((1 - fx) * at(y0 + 1, x0) + fx * at(y0 + 1, x0 + 1));
}

// OpenCV findTransformECC(MOTION_TRANSLATION, inputMask) 와 같은 식을 정확한 보간으로 푼다.
// 반환: 수렴하면 true, tx, ty 는 "현재 영상(x + tx, y + ty) ≈ 기준 영상(x, y)" 의 평행이동.
bool ecc_translation(const cv::Mat& template_blurred, const cv::Mat& current, const cv::Mat& mask, double& tx, double& ty) {
    cv::Mat image, gradient_x, gradient_y;
    cv::GaussianBlur(current, image, cv::Size(kGaussian, kGaussian), 0, 0);
    const cv::Matx13f dx(-0.5f, 0.0f, 0.5f);
    cv::filter2D(image, gradient_x, -1, dx);
    cv::filter2D(image, gradient_y, -1, dx.t());
    gradient_x.setTo(0, mask == 0);
    gradient_y.setTo(0, mask == 0);
    const int n = kSide * kSide;
    std::vector<double> warped(n), jx(n), jy(n), zero_mean_template(n);
    std::vector<unsigned char> inside(n);
    tx = ty = 0;
    double rho = -1, last_rho = -kEpsilon;
    for (int iteration = 1; iteration <= kIterations && std::abs(rho - last_rho) >= kEpsilon; ++iteration) {
        double image_sum = 0, template_sum = 0; int valid = 0;
        for (int y = 0; y < kSide; ++y)
            for (int x = 0; x < kSide; ++x) {
                const int i = y * kSide + x;
                const double sx = x + tx, sy = y + ty;
                warped[i] = bilinear(image, sx, sy); jx[i] = bilinear(gradient_x, sx, sy); jy[i] = bilinear(gradient_y, sx, sy);
                const int mx = cvRound(sx), my = cvRound(sy);
                inside[i] = (mx >= 0 && my >= 0 && mx < kSide && my < kSide && mask.at<uchar>(my, mx)) ? 1 : 0;
                if (inside[i]) { image_sum += warped[i]; template_sum += template_blurred.at<float>(y, x); ++valid; }
            }
        if (valid == 0) return false;
        const double image_mean = image_sum / valid, template_mean = template_sum / valid;
        double image_norm2 = 0, template_norm2 = 0, correlation = 0;
        double hxx = 0, hxy = 0, hyy = 0, ipx = 0, ipy = 0, tpx = 0, tpy = 0;
        for (int i = 0; i < n; ++i) {
            if (inside[i]) {
                warped[i] -= image_mean; zero_mean_template[i] = double(template_blurred.at<float>(i / kSide, i % kSide)) - template_mean;
                image_norm2 += warped[i] * warped[i]; template_norm2 += zero_mean_template[i] * zero_mean_template[i]; correlation += zero_mean_template[i] * warped[i];
            } else zero_mean_template[i] = 0;
            hxx += jx[i] * jx[i]; hxy += jx[i] * jy[i]; hyy += jy[i] * jy[i];
            ipx += jx[i] * warped[i]; ipy += jy[i] * warped[i]; tpx += jx[i] * zero_mean_template[i]; tpy += jy[i] * zero_mean_template[i];
        }
        const double det = hxx * hyy - hxy * hxy;
        if (!(det > 0) || !(image_norm2 > 0) || !(template_norm2 > 0)) return false;
        last_rho = rho; rho = correlation / std::sqrt(image_norm2 * template_norm2);
        if (std::isnan(rho)) return false;
        const double iphx = (hyy * ipx - hxy * ipy) / det, iphy = (hxx * ipy - hxy * ipx) / det;   // H^-1 · imageProjection
        const double lambda_n = image_norm2 - (ipx * iphx + ipy * iphy), lambda_d = correlation - (tpx * iphx + tpy * iphy);
        if (lambda_d <= 0) return false;
        const double lambda = lambda_n / lambda_d;
        double epx = 0, epy = 0;
        for (int i = 0; i < n; ++i) { const double e = lambda * zero_mean_template[i] - warped[i]; epx += jx[i] * e; epy += jy[i] * e; }
        tx += (hyy * epx - hxy * epy) / det; ty += (hxx * epy - hxy * epx) / det;
    }
    return true;
}
}

ScrewGradient::ScrewGradient(const cv::Mat& center) {
    if (center.type() != CV_8UC1 || center.size() != cv::Size(3840, 2160))
        throw std::invalid_argument("ScrewGradient: 센터 사진은 3840x2160 8비트 회색이어야 한다");
    for (int k = 0; k < 4; ++k)
        cv::GaussianBlur(gradient_magnitude(center(patch_rect(site::get().screws[k]))), templates_[k], cv::Size(kGaussian, kGaussian), 0, 0);
    cv::Mat disk = cv::Mat::zeros(kSide, kSide, CV_32F);
    for (int y = 0; y < kSide; ++y)
        for (int x = 0; x < kSide; ++x)
            if (std::hypot(double(x - kHalf), double(y - kHalf)) <= kMaskRadius) disk.at<float>(y, x) = 1;
    // findTransformECC 와 같은 마스크 처리: 흐린 뒤 0.95 보다 큰 곳만 남긴다(가장자리 한 겹을 뺀다).
    cv::GaussianBlur(disk, disk, cv::Size(kGaussian, kGaussian), 0, 0);
    disk *= (0.5 / 0.95);
    disk.convertTo(mask_, CV_8U);
}

cv::Mat ScrewGradient::measure(const cv::Mat& gray, const cv::Mat& initial8) const {
    if (gray.type() != CV_8UC1 || gray.size() != cv::Size(3840, 2160))
        throw std::invalid_argument("ScrewGradient: 사진은 3840x2160 8비트 회색이어야 한다");
    if (initial8.type() != CV_64F || initial8.total() != 8) throw std::invalid_argument("ScrewGradient: initial8 은 1x8 CV_64F");
    const double nan = std::numeric_limits<double>::quiet_NaN();
    cv::Mat out(1, 8, CV_64F, cv::Scalar(nan));
    for (int k = 0; k < 4; ++k) {
        const cv::Point offset(cvRound(initial8.at<double>(2 * k)), cvRound(initial8.at<double>(2 * k + 1)));
        const cv::Rect roi = patch_rect(site::get().screws[k] + offset);
        if ((roi & cv::Rect(0, 0, gray.cols, gray.rows)) != roi) continue;
        double tx = 0, ty = 0;
        if (!ecc_translation(templates_[k], gradient_magnitude(gray(roi)), mask_, tx, ty)) continue;   // 실패는 NaN 으로 둔다. 0 으로 숨기지 않는다.
        out.at<double>(2 * k) = offset.x + tx;
        out.at<double>(2 * k + 1) = offset.y + ty;
    }
    return out;
}
}
