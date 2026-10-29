#include "theta_alignment.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <vector>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

namespace semes {
namespace {
constexpr double pi = 3.14159265358979323846;
double clamp(double x, double low, double high) { return std::max(low, std::min(x, high)); }

#ifdef _WIN32
std::wstring widePath(const std::string& text) {
    const int n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text.c_str(), -1, nullptr, 0);
    if (n == 0) throw std::runtime_error("Invalid UTF-8 path");
    std::wstring result(n, L'\0');
    MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text.c_str(), -1, &result[0], n);
    result.resize(n - 1);
    return result;
}
#endif

cv::Mat grayscale(const cv::Mat& image, const ThetaConfig& config) {
    if (image.empty() || image.depth() != CV_8U || image.cols != config.width || image.rows != config.height)
        throw std::invalid_argument("Expected an 8-bit 3840x2160 image (or configured native size)");
    cv::Mat gray;
    if (image.channels() == 1) gray = image;
    else if (image.channels() == 3) cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    else if (image.channels() == 4) cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
    else throw std::invalid_argument("Expected gray, BGR or BGRA input");
    return gray;
}

// Continuous blurred-step fit, ported from the saved measurement implementation.
// The ridge/iteration/bounds below are numerical solver settings, not accuracy criteria.
double stepCenter(const cv::Mat& blurred, int x, int peak, double initial) {
    cv::Vec4d p(blurred.at<float>(peak - 5, x),
                blurred.at<float>(peak + 5, x) - blurred.at<float>(peak - 5, x), initial, 1.5);
    for (int iteration = 0; iteration < 30; ++iteration) {
        cv::Matx<double, 11, 4> jac;
        cv::Vec<double, 11> residual;
        for (int k = 0; k < 11; ++k) {
            const double t = (peak - 5 + k - p[2]) / p[3];
            const double cdf = 0.5 * (1.0 + std::erf(t / std::sqrt(2.0)));
            const double density = std::exp(-0.5 * t * t) / std::sqrt(2.0 * pi);
            residual[k] = p[0] + p[1] * cdf - blurred.at<float>(peak - 5 + k, x);
            jac(k, 0) = 1; jac(k, 1) = cdf;
            jac(k, 2) = -p[1] * density / p[3];
            jac(k, 3) = -p[1] * density * t / p[3];
        }
        auto solve = [&]() {
            cv::Matx44d normal = jac.t() * jac + cv::Matx44d::eye() * 1e-6;
            cv::Vec4d rhs = jac.t() * residual, delta;
            if (!cv::solve(normal, rhs, delta, cv::DECOMP_CHOLESKY))
                throw std::runtime_error("Edge fit linear solve failed");
            return delta;
        };
        cv::Vec4d delta = solve();
        const bool fixedCenter = (p[2] >= initial + 2 - 1e-8 && delta[2] < 0) ||
                                 (p[2] <= initial - 2 + 1e-8 && delta[2] > 0);
        const bool fixedWidth = (p[3] >= 4 - 1e-8 && delta[3] < 0) ||
                                (p[3] <= 0.6 + 1e-8 && delta[3] > 0);
        if (fixedCenter || fixedWidth) {
            for (int k = 0; k < 11; ++k) {
                if (fixedCenter) jac(k, 2) = 0;
                if (fixedWidth) jac(k, 3) = 0;
            }
            delta = solve();
        }
        delta[2] = clamp(delta[2], -0.5, 0.5);
        delta[3] = clamp(delta[3], -0.5, 0.5);
        p -= delta;
        p[2] = clamp(p[2], initial - 2, initial + 2);
        p[3] = clamp(p[3], 0.6, 4.0);
    }
    if (!std::isfinite(p[2])) throw std::runtime_error("Non-finite edge fit");
    return p[2];
}
} // namespace

std::string readTextUtf8(const std::string& path) {
#ifdef _WIN32
    std::ifstream input(widePath(path), std::ios::binary);
#else
    std::ifstream input(path, std::ios::binary);
#endif
    if (!input) throw std::runtime_error("Cannot open input: " + path);
    return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

void writeTextUtf8(const std::string& path, const std::string& text) {
#ifdef _WIN32
    std::ofstream output(widePath(path), std::ios::binary);
#else
    std::ofstream output(path, std::ios::binary);
#endif
    if (!output) throw std::runtime_error("Cannot open output: " + path);
    output.write(text.data(), static_cast<std::streamsize>(text.size()));
    if (!output) throw std::runtime_error("Cannot write output: " + path);
}

cv::Mat readGrayUtf8(const std::string& path) {
    const std::string bytes = readTextUtf8(path);
    std::vector<unsigned char> buffer(bytes.begin(), bytes.end());
    cv::Mat image = cv::imdecode(buffer, cv::IMREAD_GRAYSCALE);
    if (image.empty()) throw std::runtime_error("Cannot decode image: " + path);
    return image;
}

ThetaConfig ThetaConfig::load(const std::string& path) {
    cv::FileStorage fs(readTextUtf8(path), cv::FileStorage::READ | cv::FileStorage::MEMORY | cv::FileStorage::FORMAT_JSON);
    if (!fs.isOpened()) throw std::runtime_error("Cannot parse theta config");
    if (static_cast<int>(fs["schema_version"]) != 1) throw std::runtime_error("Unsupported theta schema");
    ThetaConfig c;
    fs["width"] >> c.width; fs["height"] >> c.height;
    std::vector<int> roi; fs["locator_roi_xywh"] >> roi;
    if (roi.size() != 4) throw std::runtime_error("Locator requires four coordinates");
    c.locatorRoi = cv::Rect(roi[0], roi[1], roi[2], roi[3]);
    fs["search_margin_px"] >> c.searchMargin;
    fs["tick_deg"] >> c.tickDeg; fs["tolerance_deg"] >> c.toleranceDeg;
    fs["max_step_deg"] >> c.maxStepDeg;
    const auto entries = fs["lines"];
    if (entries.size() != 2) throw std::runtime_error("Exactly two upper structure lines are required");
    int i = 0;
    for (auto it = entries.begin(); it != entries.end(); ++it, ++i) {
        const auto node = *it; auto& line = c.lines[i];
        node["name"] >> line.name;
        node["x0"] >> line.x0; node["x1"] >> line.x1;
        node["expected_y"] >> line.expectedY; node["half_height"] >> line.halfHeight;
        node["polarity"] >> line.polarity;
        node["group4_slope"] >> line.positiveSlope; node["group5_slope"] >> line.negativeSlope;
    }
    c.validate();
    return c;
}

void ThetaConfig::validate() const {
    if (width <= 0 || height <= 0 || searchMargin < 1 ||
        !(tickDeg > 0 && std::isfinite(tickDeg)) ||
        !(toleranceDeg >= 0 && std::isfinite(toleranceDeg)) ||
        !(maxStepDeg >= tickDeg && std::isfinite(maxStepDeg)))
        throw std::invalid_argument("Invalid theta motion/image settings");
    const cv::Rect whole(0, 0, width, height);
    const cv::Rect search(locatorRoi.x - searchMargin, locatorRoi.y - searchMargin,
                          locatorRoi.width + 2 * searchMargin, locatorRoi.height + 2 * searchMargin);
    if (locatorRoi.width < 2 || locatorRoi.height < 2 || (search & whole) != search)
        throw std::invalid_argument("Locator search outside configured image");
    for (const auto& line : lines) {
        if (line.x0 < 0 || line.x1 <= line.x0 || line.x1 >= width || line.halfHeight < 2 ||
            line.expectedY - line.halfHeight - 8 < 0 || line.expectedY + line.halfHeight + 9 > height ||
            (line.polarity != 1 && line.polarity != -1) ||
            !std::isfinite(line.positiveSlope) || !std::isfinite(line.negativeSlope) ||
            line.positiveSlope >= 0 || line.negativeSlope >= 0)
            throw std::invalid_argument("Invalid upper-line ROI or nonnegative/nonfinite slope");
    }
}

LineMeasurement ThetaEstimator::measureLine(const cv::Mat& image, const LineConfig& line, cv::Point2d shift) {
    if (image.empty() || image.type() != CV_8UC1) throw std::invalid_argument("Line extraction requires gray8");
    const int x0 = static_cast<int>(std::nearbyint(line.x0 + shift.x));
    const int x1 = static_cast<int>(std::nearbyint(line.x1 + shift.x));
    const int yc = static_cast<int>(std::nearbyint(line.expectedY + shift.y));
    const int y0 = yc - line.halfHeight - 8, y1 = yc + line.halfHeight + 9;
    if (x0 < 0 || x1 >= image.cols || y0 < 0 || y1 > image.rows)
        throw std::runtime_error("Tracked edge ROI outside image");
    cv::Mat patch, blurred;
    image(cv::Rect(x0, y0, x1 - x0 + 1, y1 - y0)).convertTo(patch, CV_32F);
    cv::GaussianBlur(patch, blurred, cv::Size(), 1.0, 1.0, cv::BORDER_REFLECT_101);
    const auto gradient = [&](int y, int x) {
        return static_cast<float>(line.polarity) *
               ((blurred.at<float>(y + 1, x) - blurred.at<float>(y - 1, x)) * 0.5f);
    };
    std::vector<cv::Point2d> edges;
    const int stride = x1 - x0 < 300 ? 1 : 2;
    for (int x = 2; x < patch.cols - 2; x += stride) {
        int peak = 8; float gb = gradient(peak, x);
        for (int y = 9; y < patch.rows - 8; ++y) {
            const float v = gradient(y, x);
            if (v > gb) { peak = y; gb = v; }
        }
        if (peak <= 8 || peak >= patch.rows - 9 || gb <= 0) continue;
        const double ga = gradient(peak - 1, x), gc = gradient(peak + 1, x);
        const double denominator = ga - 2 * gb + gc;
        const double offset = denominator < -1e-8 ? clamp(0.5 * (ga - gc) / denominator, -0.5, 0.5) : 0;
        edges.emplace_back(x0 + x, y0 + stepCenter(blurred, x, peak, peak + offset));
    }
    if (edges.size() < 20) throw std::runtime_error("Insufficient valid edge columns");
    cv::Point2d mean;
    for (const auto& p : edges) mean += p;
    mean *= 1.0 / edges.size();
    std::vector<cv::Point2f> centered;
    for (const auto& p : edges) centered.emplace_back(static_cast<float>(p.x - mean.x), static_cast<float>(p.y - mean.y));
    cv::Vec4f fit;
    cv::fitLine(centered, fit, cv::DIST_HUBER, 0, 1e-6, 1e-8);
    const double slope = static_cast<double>(fit[1]) / fit[0];
    const double intercept = mean.y + fit[3] - slope * (mean.x + fit[2]);
    double sse = 0;
    for (const auto& p : edges) {
        const double error = (p.y - slope * p.x - intercept) / std::sqrt(1 + slope * slope);
        sse += error * error;
    }
    LineMeasurement out;
    out.angleDeg = std::atan(slope) * 180.0 / pi;
    out.edgeRmsPx = std::sqrt(sse / edges.size());
    out.points = static_cast<int>(edges.size());
    if (!std::isfinite(out.angleDeg)) throw std::runtime_error("Non-finite line angle");
    return out;
}

ThetaResult thetaFromAngles(const std::array<double, 2>& angles,
                           const std::array<double, 2>& targets, const ThetaConfig& config) {
    config.validate();
    struct Candidate { double theta, cost; int group; } candidates[2];
    for (int k = 0; k < 2; ++k) {
        double numerator = 0, denominator = 0;
        for (int i = 0; i < 2; ++i) {
            if (!std::isfinite(angles[i]) || !std::isfinite(targets[i]))
                throw std::invalid_argument("Non-finite angle or target");
            const double a = k == 0 ? config.lines[i].positiveSlope : config.lines[i].negativeSlope;
            numerator += a * (angles[i] - targets[i]); denominator += a * a;
        }
        double theta = numerator / denominator;
        theta = k == 0 ? std::max(0.0, theta) : std::min(0.0, theta);
        double cost = 0;
        for (int i = 0; i < 2; ++i) {
            const double a = k == 0 ? config.lines[i].positiveSlope : config.lines[i].negativeSlope;
            const double residual = angles[i] - targets[i] - a * theta;
            cost += residual * residual;
        }
        candidates[k] = {theta, cost, k == 0 ? 4 : 5};
    }
    const auto best = candidates[candidates[1].cost < candidates[0].cost ? 1 : 0];
    ThetaResult r;
    r.angles = angles; r.targets = targets; r.estimatedThetaDeg = best.theta;
    r.group = best.theta == 0 ? 0 : best.group;
    r.fitResidualImageDeg = std::sqrt(best.cost / 2);
    r.continuousCorrectionDeg = -best.theta;
    r.outsideCalibration = std::abs(best.theta) > 1.0; // saved groups cover [-1, +1]
    r.thetaWithinTolerance = true;
    for (int i = 0; i < 2; ++i) {
        const double a = best.group == 4 ? config.lines[i].positiveSlope : config.lines[i].negativeSlope;
        r.perLineTheta[i] = (angles[i] - targets[i]) / a;
        r.thetaWithinTolerance = r.thetaWithinTolerance && std::abs(r.perLineTheta[i]) <= config.toleranceDeg + 1e-12;
    }
    if (r.thetaWithinTolerance) {
        r.status = "theta_within_tolerance";
    } else {
        // Clamp in integer ticks so rounding cannot exceed the configured cap.
        const double maxTicks = std::floor(config.maxStepDeg / config.tickDeg + 1e-12);
        const double ticks = clamp(std::round(r.continuousCorrectionDeg / config.tickDeg), -maxTicks, maxTicks);
        r.commandDeg = ticks * config.tickDeg;
        r.commandAvailable = ticks != 0;
        r.status = ticks == 0 ? "unresolved_at_tick_resolution" :
                   r.outsideCalibration ? "move_extrapolated" : "move";
    }
    return r;
}

ThetaEstimator::ThetaEstimator(const cv::Mat& reference, const ThetaConfig& config) : config_(config) {
    config_.validate();
    const cv::Mat gray = grayscale(reference, config_);
    locatorTemplate_ = gray(config_.locatorRoi).clone();
    cv::Scalar mean, sd; cv::meanStdDev(locatorTemplate_, mean, sd);
    if (sd[0] <= std::numeric_limits<double>::epsilon()) throw std::invalid_argument("Flat reference QR patch");
    for (int i = 0; i < 2; ++i) targets_[i] = measureLine(gray, config_.lines[i]).angleDeg;
}

ThetaResult ThetaEstimator::estimate(const cv::Mat& image) const {
    const cv::Mat gray = grayscale(image, config_);
    const auto roi = config_.locatorRoi; const int margin = config_.searchMargin;
    const cv::Rect search(roi.x - margin, roi.y - margin, roi.width + 2 * margin, roi.height + 2 * margin);
    cv::Scalar mean, sd; cv::meanStdDev(gray(search), mean, sd);
    if (sd[0] <= std::numeric_limits<double>::epsilon()) throw std::runtime_error("Flat locator search");
    cv::Mat scores; cv::matchTemplate(gray(search), locatorTemplate_, scores, cv::TM_CCOEFF_NORMED);
    cv::Point peak; double zncc;
    cv::minMaxLoc(scores, nullptr, &zncc, nullptr, &peak);
    if (peak.x == 0 || peak.y == 0 || peak.x == scores.cols - 1 || peak.y == scores.rows - 1)
        throw std::runtime_error("QR locator reached search boundary");
    const cv::Point coarse(peak.x - margin, peak.y - margin);
    cv::Mat warp = cv::Mat::eye(2, 3, CV_32F);
    const double ecc = cv::findTransformECC(locatorTemplate_, gray(cv::Rect(roi.x + coarse.x, roi.y + coarse.y, roi.width, roi.height)),
        warp, cv::MOTION_AFFINE, cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 60, 1e-6), cv::noArray(), 5);
    const double cx = (roi.width - 1) / 2.0, cy = (roi.height - 1) / 2.0;
    const cv::Point2d shift(coarse.x + warp.at<float>(0, 0) * cx + warp.at<float>(0, 1) * cy + warp.at<float>(0, 2) - cx,
                            coarse.y + warp.at<float>(1, 0) * cx + warp.at<float>(1, 1) * cy + warp.at<float>(1, 2) - cy);
    if (!std::isfinite(shift.x) || !std::isfinite(shift.y) || std::abs(shift.x) >= margin || std::abs(shift.y) >= margin)
        throw std::runtime_error("ECC locator left configured search");
    std::array<double, 2> angles;
    std::array<LineMeasurement, 2> lines;
    for (int i = 0; i < 2; ++i) { lines[i] = measureLine(gray, config_.lines[i], shift); angles[i] = lines[i].angleDeg; }
    ThetaResult r = thetaFromAngles(angles, targets_, config_);
    r.locatorShift = shift; r.zncc = zncc; r.ecc = ecc; r.measurements = lines;
    return r;
}

void writeThetaResult(cv::FileStorage& fs, const ThetaResult& r) {
    fs << "status" << r.status << "estimated_theta_deg" << r.estimatedThetaDeg
       << "continuous_correction_deg" << r.continuousCorrectionDeg << "command_deg" << r.commandDeg
       << "group" << r.group << "theta_within_tolerance" << static_cast<int>(r.thetaWithinTolerance)
       << "command_available" << static_cast<int>(r.commandAvailable)
       << "outside_calibration" << static_cast<int>(r.outsideCalibration)
       << "fit_residual_image_deg" << r.fitResidualImageDeg << "zncc" << r.zncc << "ecc" << r.ecc
       << "shift_xy_px" << "[" << r.locatorShift.x << r.locatorShift.y << "]";
    fs << "lines" << "[";
    for (int i = 0; i < 2; ++i) {
        fs << "{" << "angle_deg" << r.angles[i] << "target_deg" << r.targets[i]
           << "estimated_theta_deg" << r.perLineTheta[i]
           << "edge_rms_px" << r.measurements[i].edgeRmsPx << "points" << r.measurements[i].points << "}";
    }
    fs << "]";
}
} // namespace semes
