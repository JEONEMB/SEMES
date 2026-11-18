#include "theta_pipeline.hpp"
#include "utf8_io.hpp"
#include <opencv2/imgproc.hpp>
#include <climits>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace theta {
namespace {
double number(const cv::FileNode& n, const char* key) {
    const auto v = n[key];
    if (v.empty() || (!v.isInt() && !v.isReal()))
        throw std::runtime_error(std::string("missing/non-numeric config: ") + key);
    const double x = static_cast<double>(v);
    if (!std::isfinite(x)) throw std::runtime_error("non-finite config number");
    return x;
}
int integer(const cv::FileNode& n, const char* key) {
    const double x = number(n, key);
    if (std::floor(x) != x || x < INT_MIN || x > INT_MAX)
        throw std::runtime_error("non-integer config coordinate");
    return static_cast<int>(x);
}
Region region(const cv::FileNode& n) {
    Region r{integer(n,"x_first"), integer(n,"x_last"),
             integer(n,"y_first"), integer(n,"y_last"), {}};
    const auto intervals = n["excluded_x"];
    if (intervals.empty() || !intervals.isSeq()) throw std::runtime_error("missing occlusion mask");
    for (const auto& pair : intervals) {
        if (!pair.isSeq() || pair.size() != 2 || !pair[0].isInt() || !pair[1].isInt())
            throw std::runtime_error("invalid occlusion interval");
        r.excluded_x.emplace_back(static_cast<int>(pair[0]), static_cast<int>(pair[1]));
    }
    return r;
}
void checkRegion(const Region& r, int width, int height) {
    if (r.x_first < 0 || r.x_first > r.x_last || r.x_last >= width ||
        r.y_first < 0 || r.y_first > r.y_last || r.y_last >= height)
        throw std::runtime_error("ROI outside configured image");
    for (const auto& e : r.excluded_x)
        if (e[0] < r.x_first || e[1] > r.x_last || e[0] > e[1])
            throw std::runtime_error("occlusion mask outside ROI");
}
}

void validateConfig(const Config& c) {
    if (c.width <= 0 || c.height <= 0 || c.calibration_id.empty())
        throw std::runtime_error("missing image geometry/calibration identity");
    checkRegion(c.upper, c.width, c.height);
    checkRegion(c.lower, c.width, c.height);
    if (!std::isfinite(c.huber.t) || c.huber.t <= 0 || c.huber.max_iterations < 2 ||
        !std::isfinite(c.huber.tolerance) || c.huber.tolerance <= 0)
        throw std::runtime_error("invalid Huber config");
    decodeAngles(c.calibration.upper_intercept, c.calibration.lower_intercept, c.calibration);
}

Config loadConfig(const std::string& path) {
    cv::FileStorage fs(readText(path), cv::FileStorage::READ | cv::FileStorage::MEMORY |
                                     cv::FileStorage::FORMAT_JSON);
    if (!fs.isOpened()) throw std::runtime_error("cannot parse config JSON");
    if (integer(fs.root(), "schema_version") != 1) throw std::runtime_error("unsupported config schema");
    Config c;
    c.calibration_id = static_cast<std::string>(fs["calibration_id"]);
    c.width = integer(fs["image"], "width"); c.height = integer(fs["image"], "height");
    c.upper = region(fs["upper_region"]); c.lower = region(fs["lower_region"]);
    c.huber.t = number(fs["huber"], "t");
    c.huber.max_iterations = integer(fs["huber"], "max_iterations");
    c.huber.tolerance = number(fs["huber"], "tolerance");
    const auto m = fs["model"];
    if (static_cast<std::string>(m["type"]) != "affine_forward_two_angles")
        throw std::runtime_error("unsupported theta model");
    c.calibration.upper_intercept = number(m["upper"], "intercept_deg");
    c.calibration.upper_per_y_mm = number(m["upper"], "per_y_mm");
    c.calibration.upper_per_theta_deg = number(m["upper"], "per_theta_deg");
    c.calibration.lower_intercept = number(m["lower"], "intercept_deg");
    c.calibration.lower_per_y_mm = number(m["lower"], "per_y_mm");
    c.calibration.lower_per_theta_deg = number(m["lower"], "per_theta_deg");
    c.calibration.target_theta_deg = number(m, "target_theta_deg");
    validateConfig(c);
    return c;
}

Detector::Detector(Config c) : config_(std::move(c)) { validateConfig(config_); }

Result Detector::estimate(const cv::Mat& image) const {
    if (image.empty() || image.cols != config_.width || image.rows != config_.height ||
        image.depth() != CV_8U ||
        (image.channels() != 1 && image.channels() != 3 && image.channels() != 4))
        throw std::runtime_error("expected configured-size 8-bit gray/BGR/BGRA image");
    cv::Mat gray, gy;
    if (image.channels() == 1) gray = image;
    else cv::cvtColor(image, gray, image.channels() == 3 ? cv::COLOR_BGR2GRAY : cv::COLOR_BGRA2GRAY);
    cv::Sobel(gray, gy, CV_32F, 0, 1, 3, 1, 0, cv::BORDER_REFLECT_101);
    Result out;
    out.upper = fitHuberVertical(selectColumnMaxima(gy, config_.upper), config_.huber);
    out.lower = fitHuberVertical(selectColumnMaxima(gy, config_.lower), config_.huber);
    if (!out.upper.converged || !out.lower.converged)
        throw std::runtime_error("Huber iteration limit reached; no correction produced");
    out.pose = decodeAngles(out.upper.angle_deg, out.lower.angle_deg, config_.calibration);
    return out;
}
}
