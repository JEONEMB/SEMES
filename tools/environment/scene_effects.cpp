#include "scene_effects.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <fstream>
#include <random>
#include <vector>

namespace fs = std::filesystem;

namespace scene {

namespace {

cv::Mat as_float(const cv::Mat& gray) { cv::Mat f; gray.convertTo(f, CV_32F); return f; }
cv::Mat to_gray8(const cv::Mat& f) { cv::Mat g; f.convertTo(g, CV_8U); return g; }

cv::Mat warp_affine(const cv::Mat& gray, const cv::Mat& matrix) {
    cv::Mat out;
    cv::warpAffine(gray, out, matrix, gray.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    return out;
}

}   // namespace

// ---- 읽고 쓰기 ----

void save_png(const fs::path& path, const cv::Mat& picture) {
    std::vector<unsigned char> bytes;
    cv::imencode(".png", picture, bytes);
    std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
}

Layout read_layout(const fs::path& data_folder) {
    cv::FileStorage file(data_folder.u8string() + "/reference.json", cv::FileStorage::READ);
    Layout layout;
    for (int k = 0; k < 3; ++k) layout.qr[k] = cv::Rect(int(file["qr"][k][0]), int(file["qr"][k][1]), int(file["qr"][k][2]), int(file["qr"][k][3]));
    for (int k = 0; k < 4; ++k) layout.screw[k] = cv::Point(int(file["screws"][k][0]), int(file["screws"][k][1]));
    return layout;
}

cv::Mat labeled_view(const cv::Mat& picture, const std::string& title) {
    cv::Mat view, color;
    cv::resize(picture(kView), view, cv::Size(), 0.5, 0.5, cv::INTER_AREA);
    if (view.channels() == 1) cv::cvtColor(view, color, cv::COLOR_GRAY2BGR);
    else color = view;
    int baseline = 0;
    const cv::Size box = cv::getTextSize(title, cv::FONT_HERSHEY_SIMPLEX, 0.9, 2, &baseline);
    cv::rectangle(color, cv::Rect(0, 0, box.width + 20, box.height + 20), cv::Scalar(0, 0, 0), -1);
    cv::putText(color, title, cv::Point(10, box.height + 9), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    return color;
}

// ---- 카메라 위치, 거리, 렌즈 ----

cv::Mat scale_rotate(const cv::Mat& gray, double scale, double angle_deg) {
    return warp_affine(gray, cv::getRotationMatrix2D(cv::Point2f(gray.cols / 2.0f, gray.rows / 2.0f), angle_deg, scale));
}

cv::Mat shift(const cv::Mat& gray, double right_px, double down_px) {
    return warp_affine(gray, (cv::Mat_<double>(2, 3) << 1, 0, right_px, 0, 1, down_px));
}

cv::Mat tilt(const cv::Mat& gray, double tilt_deg, double focal_px) {
    const double w = gray.cols, h = gray.rows, a = tilt_deg * CV_PI / 180;
    const cv::Mat camera = (cv::Mat_<double>(3, 3) << focal_px, 0, w / 2, 0, focal_px, h / 2, 0, 0, 1);
    const cv::Mat rotation = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, std::cos(a), -std::sin(a), 0, std::sin(a), std::cos(a));
    cv::Mat out;
    cv::warpPerspective(gray, out, camera * rotation * camera.inv(), gray.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    return out;
}

cv::Mat distort(const cv::Mat& gray, double k1) {
    cv::Mat map_x(gray.size(), CV_32F), map_y(gray.size(), CV_32F);
    const double cx = gray.cols / 2.0, cy = gray.rows / 2.0, norm = gray.cols / 2.0;
    for (int y = 0; y < gray.rows; ++y)
        for (int x = 0; x < gray.cols; ++x) {
            const double u = (x - cx) / norm, v = (y - cy) / norm, f = 1 + k1 * (u * u + v * v);
            map_x.at<float>(y, x) = float(cx + u * f * norm);
            map_y.at<float>(y, x) = float(cy + v * f * norm);
        }
    cv::Mat out;
    cv::remap(gray, out, map_x, map_y, cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    return out;
}

// ---- 대상물 ----

cv::Mat replace_qr(const cv::Mat& gray, const cv::Rect& box, unsigned seed) {
    cv::Mat out = gray.clone();
    cv::Mat patch = out(box & cv::Rect(0, 0, gray.cols, gray.rows));
    const int modules = 25;   // QR 버전 2 (25×25 칸)
    const cv::Rect inner(int(box.width * 0.06), int(box.height * 0.06), int(box.width * 0.88), int(box.height * 0.88));
    const double cell = inner.width / double(modules);
    std::mt19937 random(seed);
    std::bernoulli_distribution dark(0.5);
    double black = 255, white = 0;
    cv::minMaxLoc(patch, &black, &white);
    auto in_finder = [&](int r, int c) { return (r < 8 && c < 8) || (r < 8 && c >= modules - 8) || (r >= modules - 8 && c < 8); };
    for (int r = 0; r < modules; ++r)
        for (int c = 0; c < modules; ++c) {
            const bool is_dark = dark(random);
            if (in_finder(r, c)) continue;
            const cv::Rect m(inner.x + int(c * cell), inner.y + int(r * cell), int(cell + 1), int(cell + 1));
            patch(m & cv::Rect(0, 0, patch.cols, patch.rows)).setTo(is_dark ? black + 25 : white - 25);
        }
    cv::GaussianBlur(patch, patch, cv::Size(0, 0), 1.2);
    return out;
}

cv::Rect scaled_box(const cv::Rect& box, double factor) {
    const double cx = box.x + box.width / 2.0, cy = box.y + box.height / 2.0;
    return cv::Rect(int(std::lround(cx - box.width * factor / 2)), int(std::lround(cy - box.height * factor / 2)),
                    int(std::lround(box.width * factor)), int(std::lround(box.height * factor)));
}

cv::Mat scale_qr(const cv::Mat& gray, const cv::Rect& box, double factor) {
    const cv::Rect photo(0, 0, gray.cols, gray.rows);
    cv::Mat out = gray.clone();
    // 원래 QR 자리를 판 색(QR 바로 위 띠의 평균)으로 덮는다
    const double plate = cv::mean(gray(cv::Rect(box.x, box.y - 20, box.width, 8) & photo))[0];
    out(cv::Rect(box.x - 3, box.y - 3, box.width + 6, box.height + 6) & photo).setTo(plate);
    // 자기 가운데 둘레로 키우거나 줄인 QR 을 붙인다
    const cv::Point2f center(box.x + box.width / 2.0f, box.y + box.height / 2.0f);
    const cv::Mat scaled = warp_affine(gray, cv::getRotationMatrix2D(center, 0, factor));
    const cv::Rect target = scaled_box(box, factor) & photo;
    scaled(target).copyTo(out(target));
    return out;
}

cv::Mat remove_screw(const cv::Mat& gray, cv::Point center) {
    cv::Mat out = gray.clone();
    const double plate = cv::mean(gray(cv::Rect(center.x - 70, center.y - 120, 140, 30)))[0];
    cv::circle(out, center, 34, cv::Scalar(plate), -1, cv::LINE_AA);
    cv::Mat area = out(cv::Rect(center.x - 70, center.y - 70, 140, 140));
    cv::GaussianBlur(area, area, cv::Size(0, 0), 2.0);
    return out;
}

cv::Mat different_screw(const cv::Mat& gray, cv::Point center) {
    cv::Mat out = remove_screw(gray, center);
    cv::circle(out, center, 30, cv::Scalar(150), -1, cv::LINE_AA);
    cv::circle(out, center, 30, cv::Scalar(90), 3, cv::LINE_AA);
    cv::line(out, center + cv::Point(-16, 0), center + cv::Point(16, 0), cv::Scalar(60), 6, cv::LINE_AA);
    cv::line(out, center + cv::Point(0, -16), center + cv::Point(0, 16), cv::Scalar(60), 6, cv::LINE_AA);
    return out;
}

// ---- 조명 ----

cv::Mat brightness(const cv::Mat& gray, double gain, double offset) {
    cv::Mat out;
    gray.convertTo(out, CV_8U, gain, offset);
    return out;
}

cv::Mat gamma(const cv::Mat& gray, double g) {
    cv::Mat table(1, 256, CV_8U);
    for (int i = 0; i < 256; ++i) table.at<uchar>(i) = cv::saturate_cast<uchar>(255.0 * std::pow(i / 255.0, g));
    cv::Mat out;
    cv::LUT(gray, table, out);
    return out;
}

cv::Mat side_light(const cv::Mat& gray, double left, double right) {
    cv::Mat f = as_float(gray);
    for (int x = 0; x < gray.cols; ++x) f.col(x) *= float(left + (right - left) * x / (gray.cols - 1.0));
    return to_gray8(f);
}

cv::Mat glare(const cv::Mat& gray, cv::Point center, double radius_px, double strength) {
    cv::Mat f = as_float(gray);
    const cv::Rect area = cv::Rect(int(center.x - 3 * radius_px), int(center.y - 3 * radius_px), int(6 * radius_px), int(6 * radius_px)) & cv::Rect(0, 0, gray.cols, gray.rows);
    for (int y = area.y; y < area.br().y; ++y)
        for (int x = area.x; x < area.br().x; ++x)
            f.at<float>(y, x) += float(strength * std::exp(-(std::pow(x - center.x, 2) + std::pow(y - center.y, 2)) / (radius_px * radius_px)));
    return to_gray8(f);
}

cv::Mat banding(const cv::Mat& gray, double depth, double period_px) {
    cv::Mat f = as_float(gray);
    for (int y = 0; y < gray.rows; ++y) f.row(y) *= float(1 + depth * std::sin(2 * CV_PI * y / period_px));
    return to_gray8(f);
}

// ---- 공정 환경 ----

cv::Mat defocus(const cv::Mat& gray, double sigma_px) {
    cv::Mat out;
    cv::GaussianBlur(gray, out, cv::Size(0, 0), sigma_px);
    return out;
}

cv::Mat motion_blur(const cv::Mat& gray, int length_px, bool horizontal) {
    cv::Mat kernel = cv::Mat::zeros(length_px, length_px, CV_32F);
    if (horizontal) kernel.row(length_px / 2).setTo(1.0f / length_px);
    else kernel.col(length_px / 2).setTo(1.0f / length_px);
    cv::Mat out;
    cv::filter2D(gray, out, -1, kernel);
    return out;
}

cv::Mat heat_shimmer(const cv::Mat& gray, double amplitude_px, double cell_px, unsigned seed) {
    std::mt19937 random(seed);
    std::normal_distribution<float> normal(0, 1);
    const cv::Size small(int(gray.cols / cell_px) + 2, int(gray.rows / cell_px) + 2);
    cv::Mat dx(small, CV_32F), dy(small, CV_32F);
    for (int y = 0; y < small.height; ++y)
        for (int x = 0; x < small.width; ++x) { dx.at<float>(y, x) = normal(random); dy.at<float>(y, x) = normal(random); }
    cv::resize(dx, dx, gray.size(), 0, 0, cv::INTER_CUBIC);
    cv::resize(dy, dy, gray.size(), 0, 0, cv::INTER_CUBIC);
    cv::Mat map_x(gray.size(), CV_32F), map_y(gray.size(), CV_32F);
    for (int y = 0; y < gray.rows; ++y)
        for (int x = 0; x < gray.cols; ++x) {
            map_x.at<float>(y, x) = float(x + amplitude_px * dx.at<float>(y, x));
            map_y.at<float>(y, x) = float(y + amplitude_px * dy.at<float>(y, x));
        }
    cv::Mat out;
    cv::remap(gray, out, map_x, map_y, cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    return out;
}

cv::Mat fog(const cv::Mat& gray, double amount, double sigma_px) {
    cv::Mat blurred, out;
    cv::GaussianBlur(gray, blurred, cv::Size(0, 0), sigma_px);
    cv::addWeighted(blurred, 1 - amount, cv::Mat(gray.size(), CV_8U, cv::Scalar(225)), amount, 0, out);
    return out;
}

cv::Mat spots(const cv::Mat& gray, int count, double min_radius, double max_radius, bool droplets, unsigned seed) {
    std::mt19937 random(seed);
    std::uniform_real_distribution<double> pick_x(kView.x, kView.br().x), pick_y(kView.y, kView.br().y), pick_r(min_radius, max_radius);
    cv::Mat layer = cv::Mat::zeros(gray.size(), CV_32F);
    for (int i = 0; i < count; ++i) {
        const cv::Point c(int(pick_x(random)), int(pick_y(random)));
        const int r = int(pick_r(random));
        if (droplets) {
            cv::circle(layer, c, r, cv::Scalar(-35), -1, cv::LINE_AA);
            cv::circle(layer, c, r, cv::Scalar(60), std::max(2, r / 6), cv::LINE_AA);
        } else {
            cv::circle(layer, c, r, cv::Scalar(-70), -1, cv::LINE_AA);
        }
    }
    cv::GaussianBlur(layer, layer, cv::Size(0, 0), droplets ? 3 : max_radius * 0.6);
    const cv::Mat sum = as_float(gray) + layer;
    return to_gray8(sum);
}

cv::Mat ghost(const cv::Mat& gray, double right_px, double down_px, double weight) {
    const cv::Mat mixed = (1 - weight) * as_float(gray) + weight * as_float(shift(gray, right_px, down_px));
    return to_gray8(mixed);
}

cv::Mat glow_from_top(const cv::Mat& gray, double strength, double center_y, double width) {
    cv::Mat f = as_float(gray);
    for (int y = 0; y < gray.rows; ++y) f.row(y) += float(strength * std::exp(-std::pow((y - center_y) / width, 2)));
    return to_gray8(f);
}

cv::Mat vignette(const cv::Mat& gray, cv::Point center, double radius_px) {
    cv::Mat f = as_float(gray);
    for (int y = 0; y < gray.rows; ++y)
        for (int x = 0; x < gray.cols; ++x) {
            const double r = std::hypot(x - center.x, y - center.y);
            if (r > radius_px) f.at<float>(y, x) *= float(std::max(0.15, 1 - (r - radius_px) / 120.0));
        }
    return to_gray8(f);
}

cv::Mat noise(const cv::Mat& gray, double sigma, unsigned seed) {
    cv::Mat n(gray.size(), CV_32F);
    cv::theRNG().state = seed;
    cv::randn(n, 0, sigma);
    const cv::Mat sum = as_float(gray) + n;
    return to_gray8(sum);
}

cv::Mat jpeg(const cv::Mat& gray, int quality) {
    std::vector<unsigned char> bytes;
    cv::imencode(".jpg", gray, bytes, {cv::IMWRITE_JPEG_QUALITY, quality});
    return cv::imdecode(bytes, cv::IMREAD_GRAYSCALE);
}

}   // namespace scene
