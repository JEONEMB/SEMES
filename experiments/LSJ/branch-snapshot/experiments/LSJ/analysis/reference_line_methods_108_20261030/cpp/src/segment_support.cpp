#include "segment_support.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace lm {
double segment_y_at(const DetectedSegment& s, double x) {
    return s.y1 + (x - s.x1) * ((s.y2 - s.y1) / (s.x2 - s.x1));
}
bool allowed_support_point(const TargetSpec& t, double x, double y) {
    return std::isfinite(x) && std::isfinite(y) && t.in_x(x) && t.in_band(y) && !t.excluded(x);
}
std::vector<SegmentPiece> clip_segment_support(const TargetSpec& t, const DetectedSegment& s) {
    if (!std::isfinite(s.x1) || !std::isfinite(s.x2) || !std::isfinite(s.y1) || !std::isfinite(s.y2))
        throw std::invalid_argument("non-finite segment endpoint");
    if (t.x1 < t.x0 || t.y1 < t.y0) throw std::invalid_argument("reversed target range");
    if (s.x1 == s.x2) return {}; // x-sampling cannot represent vertical/zero-width segments.
    double a = std::max(std::min(s.x1, s.x2), t.x0 - 0.5);
    double b = std::min(std::max(s.x1, s.x2), t.x1 + 0.5);
    if (b <= a) return {};
    double slope = (s.y2 - s.y1) / (s.x2 - s.x1);
    if (!std::isfinite(slope)) throw std::invalid_argument("non-finite segment slope");
    if (slope == 0) {
        if (!t.in_band(s.y1)) return {};
    } else {
        double x_at_low = s.x1 + (t.y0 - 0.5 - s.y1) / slope;
        double x_at_high = s.x1 + (t.y1 + 0.5 - s.y1) / slope;
        a = std::max(a, std::min(x_at_low, x_at_high));
        b = std::min(b, std::max(x_at_low, x_at_high));
        if (b <= a) return {};
        // Move a rounded-outside endpoint by one representable value, not an empirical pixel tolerance.
        if (!t.in_band(segment_y_at(s, a))) a = std::nextafter(a, b);
        if (!t.in_band(segment_y_at(s, b))) b = std::nextafter(b, a);
    }
    if (b <= a) return {};
    std::vector<SegmentPiece> current = {{a,b}};
    for (const auto& iv : t.exclude_x) {
        if (iv.b < iv.a) throw std::invalid_argument("reversed occlusion interval");
        double lo = iv.a - 0.5, hi = iv.b + 0.5;
        std::vector<SegmentPiece> next;
        for (const auto& pc : current) {
            if (pc.xb < lo || pc.xa > hi) { next.push_back(pc); continue; }
            // exclude_x includes its endpoints: preserve that definition without a pixel-sized gap.
            double left = std::min(pc.xb, std::nextafter(lo, -std::numeric_limits<double>::infinity()));
            double right = std::max(pc.xa, std::nextafter(hi, std::numeric_limits<double>::infinity()));
            if (left > pc.xa) next.push_back({pc.xa,left});
            if (pc.xb > right) next.push_back({right,pc.xb});
        }
        current = next;
    }
    return current;
}
std::vector<cv::Point2d> sample_segment_support(const TargetSpec& t, const DetectedSegment& s,
                                              const std::vector<SegmentPiece>& pieces, double x_step) {
    if (!std::isfinite(x_step) || x_step <= 0) throw std::invalid_argument("sample x step must be positive and finite");
    std::vector<cv::Point2d> points;
    for (const auto& pc : pieces) {
        for (double x = pc.xa; x <= pc.xb;) {
            double y = segment_y_at(s, x);
            if (allowed_support_point(t,x,y)) points.emplace_back(x,y);
            double next = x + x_step;
            if (!(next > x)) throw std::invalid_argument("sample x step cannot advance at this coordinate");
            x = next;
        }
    }
    return points;
}
cv::Point2d visible_support_midpoint(const DetectedSegment& s, const std::vector<SegmentPiece>& pieces) {
    double weighted_x = 0, length_x = 0;
    for (const auto& pc : pieces) {
        double w = pc.xb - pc.xa;
        weighted_x += w * (pc.xa + pc.xb) * 0.5; length_x += w;
    }
    if (!(length_x > 0)) throw std::invalid_argument("no visible segment support");
    double x = weighted_x / length_x;
    // All pieces have the same slope, so x-length and arc-length give the same centroid.
    return {x,segment_y_at(s,x)};
}
}
