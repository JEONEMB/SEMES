#ifndef __ALIGNMENT_HPP__
#define __ALIGNMENT_HPP__

#include <opencv2/opencv.hpp>
#include <string>
#include <limits>

typedef double error_t;

typedef struct _alignment_t {
    error_t y;
    error_t t; // theta
} Alignment_t;

Alignment_t alignment(cv::Mat &src);

// First-stage initialization. Requires only the joint library and its assets.
void initialize_joint(const std::string &joint_assets);

// Initialization is explicit. Paths are UTF-8 and repository-relative or absolute.
// Initialize once before worker threads start; do not reinitialize concurrently.
void initialize_alignment(const std::string &joint_assets,
                          const std::string &control_assets = "",
                          double maximum_control_t = 0.09);

// Second-stage QR536 library. maximum_control_t is retained for source compatibility and unused.
// Independent of joint initialization. control_assets is a DIRECTORY containing model.json, points.json and center.bmp.
void initialize_control(const std::string &control_assets,
                        double maximum_control_t = 0.09);

// Both functions return ESTIMATES (y,t), never motion commands. Input is unchanged.
// Measurement/configuration failures throw std::exception; they never return fake0,0.
Alignment_t joint(cv::Mat &src);
Alignment_t control(cv::Mat &src);

struct AlignmentDiagnostics {
    std::string stage;
    bool measurement_valid;
    bool outside_training_range;
    bool control_precondition_met;
    double minimum_qr_correlation;
    double minimum_line_correlation;
    std::string message;
    std::string selected_axis;
    std::string selected_branch;
    int tracked_points = 0;
    int valid_points = 0;
    double model_cost = std::numeric_limits<double>::quiet_NaN();
    double other_axis_cost = std::numeric_limits<double>::quiet_NaN();
    double normalized_cost_margin = std::numeric_limits<double>::quiet_NaN();
    bool pose_stability_assessed = false;
};
// Per-thread diagnostics for the most recent call, including a failed call.
AlignmentDiagnostics alignment_diagnostics();

#endif // __ALIGNMENT_HPP__
