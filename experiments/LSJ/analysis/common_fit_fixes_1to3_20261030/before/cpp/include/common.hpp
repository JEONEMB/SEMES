// Common types for the reference-image line extraction comparison tool.
// Coordinates: original 3840x2160 pixel centers, origin top-left, +x right, +y down.
#pragma once

#include <opencv2/core.hpp>
#include <nlohmann/json.hpp>

#include <functional>
#include <string>
#include <vector>

namespace lm {

using json = nlohmann::ordered_json;

struct Interval {
    int a = 0, b = 0;  // inclusive
};

struct TargetSpec {
    std::string id;    // upper_structure, lower_structure, qr_L, qr_C, qr_R
    std::string kind;  // "external" | "qr"
    std::string feature_definition;
    int x0 = 0, x1 = 0;  // support x range, inclusive
    int y0 = 0, y1 = 0;  // search band, inclusive
    std::vector<Interval> exclude_x;  // inclusive
    int polarity = 1;  // +1: intensity increases with y across the target boundary
    cv::Rect roi;      // QR: roi_xyxy (exclusive end). External: band x/y range (no margin).
    int margin = 0;    // detector context margin, px
    int column_step = 1;
    json raw;

    bool excluded(double x) const {
        for (const auto& iv : exclude_x)
            if (x >= iv.a - 0.5 && x <= iv.b + 0.5) return true;
        return false;
    }
    bool in_x(double x) const { return x >= x0 - 0.5 && x <= x1 + 0.5; }
    bool in_band(double y) const { return y >= y0 - 0.5 && y <= y1 + 0.5; }
    // Detector region: target area plus context margin, clamped to the image.
    cv::Rect detector_region(const cv::Size& image) const;
};

// Final line in original coordinates.
struct Line {
    double px = 0, py = 0;  // reference point
    double dx = 1, dy = 0;  // unit direction, dx > 0 (or dx == 0 and dy < 0)
};

Line normalize_line(double px, double py, double dx, double dy);
double line_angle_deg(const Line& l);            // [-90, 90)
double signed_distance(const Line& l, double x, double y);  // perpendicular, + toward normal (-dy, dx)
json line_json(const Line& l);                    // slope/intercept/general form/angle
json line_draw_endpoints(const Line& l, double xa, double xb);

struct FitParams {
    double huber_c = 1.345;
    int irls_max_iter = 100;
    double irls_tol = 1e-13;
    int trim_iterations = 2;
    double inlier_px = 2.0;
    int min_points = 20;
};

struct FitResult {
    bool ok = false;
    std::string reason;
    Line line;
    std::vector<char> inlier;  // per input point
    int n_all = 0, n_inlier = 0;
    double rms_inlier = 0, rms_all = 0, max_abs_inlier = 0;
    double initial_angle_deg = 0;  // Huber fit on all points before trimming
};

// Double-precision Huber IRLS orthogonal regression, then inlier trimming.
Line huber_line(const std::vector<cv::Point2d>& pts, const std::vector<char>* use, const FitParams& p);
FitResult robust_fit(const std::vector<cv::Point2d>& pts, const FitParams& p);

// Output of one method on one target in one image.
struct MethodResult {
    std::string status = "failed";       // ok | failed
    std::string failure_category;        // detection_failed | boundary_confusion | corner_not_identified | not_implemented | fit_failed
    std::string failure_reason;
    bool has_line = false;
    Line line;
    bool has_endpoints = false;
    double ex1 = 0, ey1 = 0, ex2 = 0, ey2 = 0;
    int n_candidates = 0;
    int n_selected = 0;
    std::string support_type;
    bool has_rms = false;
    double fit_rms = 0;
    std::string rms_null_reason;
    json diagnostics = json::object();
    json intermediates = json::object();
    json images = json::object();  // name -> relative path

    void fail(const std::string& category, const std::string& reason) {
        status = "failed";
        failure_category = category;
        failure_reason = reason;
    }
};

// Saves intermediate images under results/intermediates/<ref>/<method>/.
struct Sink {
    std::string abs_dir;  // UTF-8
    std::string rel_dir;  // relative to the results directory, '/' separators
    std::string prefix;   // target id
    std::string save_png(const std::string& name, const cv::Mat& image, const cv::Point& origin, json& images);
};

struct MethodContext {
    const cv::Mat& gray;  // CV_8UC1 full original image
    const TargetSpec& target;
    const json& params;   // method params for this target kind
    const json& common;   // common config
    FitParams fit;
    Sink& sink;
};

using MethodFn = std::function<MethodResult(MethodContext&)>;

struct MethodInfo {
    std::string id;
    std::string family;  // E or Q
    std::string name_ko;
    std::string implementation;  // file::function
    MethodFn fn;
};

const std::vector<MethodInfo>& registry();

// Helpers shared by method modules.
FitParams fit_params_for(const json& common, const std::string& kind);
void finish_with_points(MethodResult& r, const std::vector<cv::Point2d>& pts, const FitParams& fp,
                        const std::string& support_type);
double bilinear(const cv::Mat& f, double x, double y);  // CV_64F or CV_32F single channel
json round_pt(double x, double y, int digits = 4);
double rnd(double v, int digits = 4);

}  // namespace lm
