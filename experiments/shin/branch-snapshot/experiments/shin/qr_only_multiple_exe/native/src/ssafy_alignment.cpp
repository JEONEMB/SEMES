#include "qr_pose.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace qr {
namespace {
int bounded(double value, int limit) {
    return static_cast<int>(std::nearbyint(std::max(-double(limit), std::min(double(limit), value))));
}
}

const char* name(SsafyAlignmentAction action) {
    switch (action) {
        case SsafyAlignmentAction::center_candidate: return "qr_center_candidate";
        case SsafyAlignmentAction::theta_alignment: return "theta_alignment";
        case SsafyAlignmentAction::two_axis_alignment: return "two_axis_alignment";
    }
    throw std::invalid_argument("Unknown SSAFY alignment action");
}

SsafyAligner::SsafyAligner(const std::string& model_path) {
    const auto file = open_model(model_path);
    if (std::string(file["domain"]) != "ssafy")
        throw std::invalid_argument("Expected SSAFY alignment calibration");
    coefficients_ = matrix(file, "coefficients");
    const auto response = matrix(file, "horizontal_response");
    if (coefficients_.size() != cv::Size(2,4) || response.size() != cv::Size(2,1) ||
        !cv::checkRange(coefficients_) || !cv::checkRange(response) ||
        std::abs(response.at<double>(0,1)) < 1e-6)
        throw std::invalid_argument("Invalid SSAFY alignment calibration");
    horizontal_response_ = {response.at<double>(0,0), response.at<double>(0,1)};
    geometry_ = Geometry(file, centers(matrix(file, "centers")));
}

Features SsafyAligner::observe(const cv::Mat& frame) const {
    return geometry_.predict(frame);
}

Pose SsafyAligner::residual(Features observation, Pose target) const {
    for (double value : observation.val)
        if (!std::isfinite(value)) throw std::invalid_argument("Invalid SSAFY QR observation");
    for (double value : target.val)
        if (!std::isfinite(value)) throw std::invalid_argument("Invalid SSAFY target");
    const cv::Mat estimate = cv::Mat(observation).t() * coefficients_;
    return {estimate.at<double>(0) - target[0], estimate.at<double>(1) - target[1]};
}

double SsafyAligner::horizontal_error(Features observation, Pose target) const {
    return observation[0] - horizontal_response_.dot(target);
}

double SsafyAligner::rig_horizontal(Features observation, Pose target) const {
    return horizontal_error(observation, target) * rig_horizontal_slope_px_per_degree /
           horizontal_response_[1];
}

SsafyAlignmentStep SsafyAligner::step(Features observation, Pose target) const {
    const auto error = residual(observation, target);
    const auto dx = horizontal_error(observation, target);
    if (std::abs(dx) > .30)
        return {SsafyAlignmentAction::theta_alignment, error, dx,
                {0, bounded(-1000. * dx / horizontal_response_[1], 300)}};
    if (std::abs(error[0]) <= .045 && std::abs(error[1]) <= .01)
        return {SsafyAlignmentAction::center_candidate, error, dx, {0,0}};
    return {SsafyAlignmentAction::two_axis_alignment, error, dx,
            {bounded(-650. * error[0], 150), bounded(-650. * error[1], 50)}};
}

SsafyAlignmentResult run_ssafy_alignment(
    const SsafyAligner& aligner, Pose target, const std::function<Features()>& capture,
    const std::function<bool(Position)>& move_relative, int max_moves) {
    if (max_moves < 0 || max_moves > 20 || !capture || !move_relative)
        throw std::invalid_argument("Invalid SSAFY alignment budget or callbacks");
    int moves = 0, two_axis_moves = 0;
    for (;;) {
        const auto step = aligner.step(capture(), target);
        if (step.action == SsafyAlignmentAction::center_candidate)
            return {name(step.action), moves, two_axis_moves};
        if (step.relative_move == Position(0,0))
            return {"move_below_resolution", moves, two_axis_moves};
        const bool two_axis = step.action == SsafyAlignmentAction::two_axis_alignment;
        if (moves == max_moves || (two_axis && two_axis_moves == 3))
            return {"qr_correction_budget", moves, two_axis_moves};
        if (!move_relative(step.relative_move))
            return {"move_rejected", moves, two_axis_moves};
        ++moves;
        if (two_axis) ++two_axis_moves;
    }
}

ControlResult run_ssafy_rig_alignment(Stage& stage, const Settings& settings,
    const SsafyAligner& model, Pose visual_target, const std::function<Features()>& capture,
    const Event& event) {
    if (!capture || !std::isfinite(visual_target[0]) || !std::isfinite(visual_target[1]))
        throw std::invalid_argument("Invalid SSAFY capture or visual target");
    return run_correction(stage, settings,
        [&](){ return model.rig_horizontal(capture(), visual_target); },
        [&](){ return model.residual(capture(), visual_target); }, event);
}
}
