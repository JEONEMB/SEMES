#include "y_alignment.hpp"
#include "validation_utils.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace semes {
namespace {
cv::Mat grayImage(const cv::Mat& image, const YConfig& config) {
    if (image.empty() || image.depth() != CV_8U || image.cols != config.width || image.rows != config.height)
        throw std::invalid_argument("Y requires a native-size 8-bit image");
    cv::Mat gray;
    if (image.channels() == 1) gray = image;
    else if (image.channels() == 3) cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    else if (image.channels() == 4) cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
    else throw std::invalid_argument("Y requires gray/BGR/BGRA image");
    return gray;
}

// Roundoff guard only (about 5.5e-11 px at width 3840), not a new target band.
double pixelEpsilon(const YConfig& c) { return 64 * std::numeric_limits<double>::epsilon() * c.width; }
bool inside(double value, const YMarkerConfig& marker, double epsilon) {
    return value >= marker.lowPx - epsilon && value <= marker.highPx + epsilon;
}
} // namespace

YConfig YConfig::load(const std::string& path) {
    try {
        cv::FileStorage fs(readTextUtf8(path), cv::FileStorage::READ | cv::FileStorage::MEMORY | cv::FileStorage::FORMAT_JSON);
        detail::configCheck(fs.isOpened(), "Cannot parse Y configuration");
        const auto root = fs.root();
        detail::configCheck(detail::integer(root, "schema_version") == 1, "Invalid Y schema");
        YConfig c;
        c.width = detail::integer(root, "width"); c.height = detail::integer(root, "height");
        c.searchMargin = detail::integer(root, "search_margin_px"); c.tickMm = detail::number(root, "tick_mm");
        c.targetProfile = detail::string(root, "target_profile");
        detail::configCheck(detail::number(root, "additional_tolerance_px") == 0, "Additional Y tolerance is not supported");
        detail::configCheck(detail::integers(root, "slope_groups", 4) == std::vector<int>{2,3,6,7}, "Expected Y slope groups 2,3,6,7");
        const auto entries = detail::sequence(root, "markers", 3);
        int i = 0;
        for (auto it = entries.begin(); it != entries.end(); ++it, ++i) {
            const auto n = *it; auto& m = c.markers[i];
            m.name = detail::string(n, "name"); const auto roi = detail::integers(n, "roi_xywh", 4);
            m.roi = cv::Rect(roi[0], roi[1], roi[2], roi[3]);
            m.lowPx = detail::number(n, "low_px"); m.highPx = detail::number(n, "high_px");
            const auto slopes = detail::sequence(n, "slopes_px_per_mm", 4);
            int j = 0; for (auto a = slopes.begin(); a != slopes.end(); ++a, ++j)
                m.slopes[j] = detail::number(*a, std::string("slopes_px_per_mm"));
        }
        c.validate(); return c;
    } catch (const cv::Exception& e) { throw AlignmentError(ErrorCode::InvalidConfiguration, e.what()); }
}

void YConfig::validate() const {
    detail::configCheck(tickMm > 0 && std::isfinite(tickMm) && !targetProfile.empty(), "Invalid Y settings");
    const std::array<std::string, 3> names{{"L", "C", "R"}};
    for (int i = 0; i < 3; ++i) {
        const auto& m = markers[i]; detail::roiInside(m.roi, searchMargin, width, height);
        detail::configCheck(m.name == names[i] && std::isfinite(m.lowPx) && std::isfinite(m.highPx) &&
                            m.lowPx >= 0 && m.highPx < width && m.lowPx <= m.highPx, "Invalid Y target range/name");
        for (double a : m.slopes) detail::configCheck(a > 0 && std::isfinite(a), "Expected finite positive Y slope");
    }
}

YResult yFromPositions(const std::array<double, 3>& x, const YConfig& config) {
    config.validate();
    YResult r; r.x = x; r.withinTarget = true;
    const double eps = pixelEpsilon(config);
    r.commonLowMm = -std::numeric_limits<double>::infinity();
    r.commonHighMm = std::numeric_limits<double>::infinity();
    double minSlope = std::numeric_limits<double>::infinity();
    for (int i = 0; i < 3; ++i) {
        if (!std::isfinite(x[i]) || x[i] < 0 || x[i] >= config.width)
            throw std::invalid_argument("Invalid measured Y marker coordinate");
        const auto& m = config.markers[i];
        r.inside[i] = inside(x[i], m, eps);
        r.withinTarget = r.withinTarget && r.inside[i];
        double low = -std::numeric_limits<double>::infinity(), high = std::numeric_limits<double>::infinity();
        for (double a : m.slopes) {
            minSlope = std::min(minSlope, a);
            low = std::max(low, (m.lowPx - x[i]) / a);
            high = std::min(high, (m.highPx - x[i]) / a);
        }
        r.intervalLowMm[i] = low; r.intervalHighMm[i] = high;
        r.commonLowMm = std::max(r.commonLowMm, low);
        r.commonHighMm = std::min(r.commonHighMm, high);
    }
    if (!std::isfinite(r.commonLowMm) || !std::isfinite(r.commonHighMm))
        throw std::invalid_argument("Y movement interval overflow");
    if (r.withinTarget) {
        r.commandAvailable = true; r.status = "y_within_target";
        return r;
    }
    const double epsMm = eps / minSlope;
    if (r.commonLowMm > r.commonHighMm + epsMm) {
        r.errorCode = ErrorCode::YTargetConflict; r.status = errorCodeName(r.errorCode); return r;
    }
    const double firstTick = std::ceil((r.commonLowMm - epsMm) / config.tickMm);
    const double lastTick = std::floor((r.commonHighMm + epsMm) / config.tickMm);
    if (!std::isfinite(firstTick) || !std::isfinite(lastTick) || std::abs(firstTick) >= 9007199254740992.0 || std::abs(lastTick) >= 9007199254740992.0)
        throw std::invalid_argument("Y tick interval overflow");
    r.nearestTickBelowMm = std::floor(r.commonLowMm / config.tickMm) * config.tickMm;
    r.nearestTickAboveMm = std::ceil(r.commonHighMm / config.tickMm) * config.tickMm;
    r.tickDiagnosticsAvailable = true;
    if (firstTick > lastTick) { r.errorCode = ErrorCode::YNoReachableTick; r.status = errorCodeName(r.errorCode); return r; }
    const double ticks = std::max(firstTick, std::min(0.0, lastTick));
    const double command = ticks * config.tickMm;
    // Recheck all 12 model predictions after discretization; never clamp to an arbitrary endpoint.
    for (int i = 0; i < 3; ++i)
        for (double a : config.markers[i].slopes)
            if (!inside(x[i] + a * command, config.markers[i], eps)) {
                r.errorCode = ErrorCode::YNoReachableTick; r.status = errorCodeName(r.errorCode); return r;
            }
    if (ticks == 0) { r.errorCode = ErrorCode::YNoReachableTick; r.status = errorCodeName(r.errorCode); return r; }
    r.commandMm = command; r.commandAvailable = true; r.status = "move_y";
    return r;
}

YEstimator::YEstimator(const cv::Mat& reference, const YConfig& config) : config_(config) {
    config_.validate();
    const cv::Mat gray = grayImage(reference, config_);
    for (int i = 0; i < 3; ++i) {
        templates_[i] = gray(config_.markers[i].roi).clone();
        cv::Scalar mean, sd; cv::meanStdDev(templates_[i], mean, sd);
        if (sd[0] <= std::numeric_limits<double>::epsilon()) throw std::invalid_argument("Flat reference QR patch");
    }
}

YResult YEstimator::estimate(const cv::Mat& image) const {
    const cv::Mat gray = grayImage(image, config_);
    std::array<double, 3> x{}, zncc{}, ecc{};
    const int margin = config_.searchMargin;
    for (int i = 0; i < 3; ++i) {
        const auto roi = config_.markers[i].roi;
        const cv::Rect search(roi.x - margin, roi.y - margin, roi.width + 2 * margin, roi.height + 2 * margin);
        cv::Scalar mean, sd; cv::meanStdDev(gray(search), mean, sd);
        if (sd[0] <= std::numeric_limits<double>::epsilon()) throw std::runtime_error("Flat Y QR search");
        cv::Mat scores; cv::matchTemplate(gray(search), templates_[i], scores, cv::TM_CCOEFF_NORMED);
        cv::Point peak; cv::minMaxLoc(scores, nullptr, &zncc[i], nullptr, &peak);
        if (peak.x == 0 || peak.y == 0 || peak.x == scores.cols - 1 || peak.y == scores.rows - 1)
            throw std::runtime_error("Y QR match reached search boundary");
        const cv::Point coarse(peak.x - margin, peak.y - margin);
        cv::Mat warp = cv::Mat::eye(2, 3, CV_32F);
        ecc[i] = cv::findTransformECC(templates_[i], gray(cv::Rect(roi.x + coarse.x, roi.y + coarse.y, roi.width, roi.height)),
            warp, cv::MOTION_AFFINE, cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 60, 1e-6), cv::noArray(), 5);
        const double cx = (roi.width - 1) / 2.0, cy = (roi.height - 1) / 2.0;
        const double dx = coarse.x + warp.at<float>(0, 0) * cx + warp.at<float>(0, 1) * cy + warp.at<float>(0, 2) - cx;
        const double dy = coarse.y + warp.at<float>(1, 0) * cx + warp.at<float>(1, 1) * cy + warp.at<float>(1, 2) - cy;
        if (!std::isfinite(dx) || !std::isfinite(dy) || std::abs(dx) >= margin || std::abs(dy) >= margin)
            throw std::runtime_error("Y ECC locator left search region");
        x[i] = roi.x + cx + dx; // same zero-based ROI pixel-center convention as saved target ranges
    }
    YResult result = yFromPositions(x, config_);
    result.zncc = zncc; result.ecc = ecc;
    return result;
}

void writeYResult(cv::FileStorage& fs, const YResult& r) {
    fs << "status" << r.status << "error_code" << errorCodeName(r.errorCode)
       << "tick_diagnostics_available" << static_cast<int>(r.tickDiagnosticsAvailable)
       << "nearest_tick_below_mm" << r.nearestTickBelowMm << "nearest_tick_above_mm" << r.nearestTickAboveMm << "command_mm" << r.commandMm
       << "within_target" << static_cast<int>(r.withinTarget) << "command_available" << static_cast<int>(r.commandAvailable)
       << "common_interval_mm" << "[" << r.commonLowMm << r.commonHighMm << "]" << "markers" << "[";
    const std::array<std::string, 3> names{{"L", "C", "R"}};
    for (int i = 0; i < 3; ++i)
        fs << "{" << "name" << names[i] << "x_px" << r.x[i] << "inside" << static_cast<int>(r.inside[i])
           << "interval_mm" << "[" << r.intervalLowMm[i] << r.intervalHighMm[i] << "]"
           << "zncc" << r.zncc[i] << "ecc" << r.ecc[i] << "}";
    fs << "]";
}
} // namespace semes
