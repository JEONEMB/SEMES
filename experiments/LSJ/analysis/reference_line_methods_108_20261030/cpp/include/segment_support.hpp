#pragma once
#include "common.hpp"

namespace lm {
struct DetectedSegment {
    double x1, y1, x2, y2;
    double score = 0;
    std::vector<cv::Point> support;
};
struct SegmentPiece { double xa, xb; };

double segment_y_at(const DetectedSegment& s, double x);
bool allowed_support_point(const TargetSpec& t, double x, double y);
std::vector<SegmentPiece> clip_segment_support(const TargetSpec& t, const DetectedSegment& s);
std::vector<cv::Point2d> sample_segment_support(const TargetSpec& t, const DetectedSegment& s,
                                              const std::vector<SegmentPiece>& pieces, double x_step);
cv::Point2d visible_support_midpoint(const DetectedSegment& s, const std::vector<SegmentPiece>& pieces);
// Shared selection, callable with supplied segments; detector algorithms are separate.
MethodResult select_and_fit_segments(MethodContext& c, std::vector<DetectedSegment> segs, const std::string& detector);
}
