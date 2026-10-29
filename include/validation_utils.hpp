#pragma once
#include "alignment_error.hpp"
#include <opencv2/core.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace semes { namespace detail {
inline void finite(double value, const char* what) {
    if (!std::isfinite(value)) throw AlignmentError(ErrorCode::NumericalFailure, what);
}
inline double finiteClamp(double value, double low, double high) {
    finite(value, "Non-finite value before clamp"); finite(low, "Non-finite lower bound"); finite(high, "Non-finite upper bound");
    if (low > high) throw AlignmentError(ErrorCode::NumericalFailure, "Reversed clamp bounds");
    return std::max(low, std::min(value, high));
}
// Shared by the edge solver and its fault-injection regression tests.
inline cv::Vec4d boundedEdgeUpdate(cv::Vec4d p, cv::Vec4d delta, double initial) {
    for (int j = 0; j < 4; ++j) {
        finite(p[j], "Non-finite edge parameters");
        finite(delta[j], "Non-finite edge solver update");
    }
    finite(initial, "Non-finite initial edge center");
    delta[2] = finiteClamp(delta[2], -0.5, 0.5);
    delta[3] = finiteClamp(delta[3], -0.5, 0.5);
    p -= delta;
    for (int j = 0; j < 4; ++j) finite(p[j], "Non-finite edge parameters before bounds");
    p[2] = finiteClamp(p[2], initial - 2, initial + 2);
    p[3] = finiteClamp(p[3], 0.6, 4.0);
    return p;
}
inline void configCheck(bool ok, const std::string& message) {
    if (!ok) throw AlignmentError(ErrorCode::InvalidConfiguration, message);
}
inline cv::FileNode required(const cv::FileNode& parent, const std::string& key) {
    configCheck(parent.isMap(), "Expected configuration object for " + key);
    const auto node = parent[key];
    configCheck(!node.isNone(), "Missing required field: " + key);
    return node;
}
inline double number(const cv::FileNode& node, const std::string& key) {
    configCheck(node.isInt() || node.isReal(), "Expected number: " + key);
    const double value = static_cast<double>(node);
    configCheck(std::isfinite(value), "Expected finite number: " + key);
    return value;
}
inline double number(const cv::FileNode& parent, const char* key) { return number(required(parent, key), std::string(key)); }
inline int integerNode(const cv::FileNode& node, const std::string& key) {
    configCheck(node.isInt(), "Expected integer: " + key);
    const double d = number(node, key);
    configCheck(d >= std::numeric_limits<int>::min() && d <= std::numeric_limits<int>::max(), "Integer overflow: " + key);
    return static_cast<int>(d);
}
inline int integer(const cv::FileNode& parent, const char* key) { return integerNode(required(parent, key), key); }
inline std::string string(const cv::FileNode& parent, const char* key) {
    const auto node = required(parent, key); configCheck(node.isString(), std::string("Expected string: ")+key);
    const std::string value = static_cast<std::string>(node); configCheck(!value.empty(), std::string("Empty field: ")+key);
    return value;
}
inline cv::FileNode sequence(const cv::FileNode& parent, const char* key, size_t size) {
    const auto n = required(parent, key);
    configCheck(n.isSeq() && n.size() == size, std::string("Invalid array: ")+key);
    return n;
}
inline std::vector<int> integers(const cv::FileNode& parent, const char* key, size_t size) {
    const auto n = sequence(parent, key, size); std::vector<int> out;
    for (auto it = n.begin(); it != n.end(); ++it) out.push_back(integerNode(*it, key));
    return out;
}
inline void roiInside(const cv::Rect& roi, int margin, int width, int height) {
    using I = int64_t;
    configCheck(width > 0 && height > 0 && margin >= 1 && roi.width >= 2 && roi.height >= 2 &&
                I(roi.x)-margin >= 0 && I(roi.y)-margin >= 0 &&
                I(roi.x)+roi.width+margin <= width && I(roi.y)+roi.height+margin <= height,
                "ROI/search outside configured image");
}
} } // namespace semes::detail
