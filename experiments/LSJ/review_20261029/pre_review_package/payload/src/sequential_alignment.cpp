#include "sequential_alignment.hpp"
#include <cmath>
#include <stdexcept>

namespace semes {
AlignmentDecision chooseAlignmentCorrection(const ThetaResult& theta, const YResult* y) {
    AlignmentDecision result; result.theta = theta;
    if (!theta.commandAvailable) { result.status = "theta_" + theta.status; return result; }
    if (!theta.thetaWithinTolerance) {
        result.thetaDeg = theta.commandDeg; result.commandAvailable = true;
        result.status = theta.outsideCalibration ? "move_theta_extrapolated" : "move_theta";
        return result;
    }
    if (!y) throw std::logic_error("Y measurement required after theta acceptance");
    result.yEvaluated = true; result.y = *y;
    result.commandAvailable = y->commandAvailable;
    if (!y->commandAvailable) { result.status = y->status; return result; }
    result.yMm = y->commandMm; result.aligned = y->withinTarget;
    result.status = result.aligned ? "aligned" : "move_y";
    return result;
}

SequentialAligner::SequentialAligner(const cv::Mat& reference, const ThetaConfig& theta, const YConfig& y)
    : theta_(reference, theta), y_(reference, y) {
    if (theta.width != y.width || theta.height != y.height) throw std::invalid_argument("Theta and Y image sizes differ");
}

AlignmentDecision SequentialAligner::estimate(const cv::Mat& image) const {
    const auto theta = theta_.estimate(image);
    if (!theta.commandAvailable || !theta.thetaWithinTolerance) return chooseAlignmentCorrection(theta);
    const auto y = y_.estimate(image);
    return chooseAlignmentCorrection(theta, &y);
}

AlignmentDecision MoveBudget::issue(const AlignmentDecision& decision) {
    if (pending_) throw std::logic_error("Acknowledge the previous executed move, then supply a newly captured image");
    auto result = decision;
    const bool moving = result.yMm != 0 || result.thetaDeg != 0;
    if (!std::isfinite(result.yMm) || !std::isfinite(result.thetaDeg) ||
        (result.yMm != 0 && result.thetaDeg != 0) || (moving && (!result.commandAvailable || result.aligned)))
        throw std::logic_error("Invalid sequential movement decision");
    if (moving && completed_ >= 20) {
        result.yMm = 0; result.thetaDeg = 0; result.commandAvailable = false;
        result.status = "move_limit_reached";
    } else if (moving) pending_ = true;
    return result;
}

void MoveBudget::acknowledgeExecuted() {
    if (!pending_) throw std::logic_error("No pending movement to acknowledge");
    ++completed_; pending_ = false;
}

void writeAlignmentDecision(cv::FileStorage& fs, const AlignmentDecision& r) {
    fs << "status" << r.status << "command_y_mm" << r.yMm << "command_theta_deg" << r.thetaDeg
       << "command_available" << static_cast<int>(r.commandAvailable) << "aligned" << static_cast<int>(r.aligned)
       << "y_evaluated" << static_cast<int>(r.yEvaluated) << "theta" << "{";
    writeThetaResult(fs, r.theta); fs << "}";
    if (r.yEvaluated) { fs << "y" << "{"; writeYResult(fs, r.y); fs << "}"; }
}
} // namespace semes
