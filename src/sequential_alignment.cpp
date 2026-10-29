#include "sequential_alignment.hpp"
#include <cmath>
#include <stdexcept>

namespace semes {
AlignmentDecision chooseAlignmentCorrection(const ThetaResult& theta, const YResult* y) {
    AlignmentDecision result; result.theta = theta;
    if (!theta.commandAvailable) { result.status = theta.status; result.errorCode = theta.errorCode; return result; }
    if (!theta.thetaWithinTolerance) {
        result.thetaDeg = theta.commandDeg; result.commandAvailable = true;
        result.status = theta.outsideCalibration ? "move_theta_extrapolated" : "move_theta";
        return result;
    }
    if (!y) throw AlignmentError(ErrorCode::InvalidTransition, "Y measurement required after theta acceptance");
    result.yEvaluated = true; result.y = *y;
    result.commandAvailable = y->commandAvailable;
    if (!y->commandAvailable) { result.status = y->status; result.errorCode = y->errorCode; return result; }
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
    requireReady();
    auto result = decision;
    const bool moving = result.yMm != 0 || result.thetaDeg != 0;
    if (!std::isfinite(result.yMm) || !std::isfinite(result.thetaDeg) ||
        (result.yMm != 0 && result.thetaDeg != 0) || (moving && (!result.commandAvailable || result.aligned)))
        throw AlignmentError(ErrorCode::InvalidTransition, "Invalid sequential movement decision");
    if (moving && used_ >= 20) {
        result.yMm = 0; result.thetaDeg = 0; result.commandAvailable = false;
        result.status = "move_limit_reached";
        result.errorCode = ErrorCode::MoveLimitReached;
    } else if (moving) {
        if (result.thetaDeg != 0 && thetaHistory_.size() == 2 &&
            std::abs(thetaHistory_[0] - result.thetaDeg) < 1e-12 &&
            std::abs(thetaHistory_[1] + result.thetaDeg) < 1e-12) oscillationLatched_ = true;
        if (oscillationLatched_) {
            result.yMm = 0; result.thetaDeg = 0; result.commandAvailable = false;
            result.status = "theta_oscillation_detected"; result.errorCode = ErrorCode::ThetaOscillation;
        } else { pending_ = true; pendingDecision_ = result; }
    }
    return result;
}

void MoveBudget::acknowledgeExecuted() {
    if (!pending_) throw AlignmentError(ErrorCode::InvalidTransition, "No pending movement to acknowledge");
    ++completed_; ++used_; pending_ = false;
    if (pendingDecision_.thetaDeg != 0) {
        thetaHistory_.push_back(pendingDecision_.thetaDeg);
        if (thetaHistory_.size() > 2) thetaHistory_.erase(thetaHistory_.begin());
    } else thetaHistory_.clear();
}

void MoveBudget::requireReady() const {
    if (pending_) throw AlignmentError(ErrorCode::PendingMove, "Previous movement awaits execution outcome");
    if (needsInspection_) throw AlignmentError(ErrorCode::MovementUncertain, "Confirm robot state before resuming with a new image");
}

void MoveBudget::reportFailure(MoveFailure failure) {
    if (!pending_) throw AlignmentError(ErrorCode::InvalidTransition, "No pending movement to report");
    pending_ = false;
    if (failure == MoveFailure::PartialOrUnknown) {
        ++used_; needsInspection_ = true; thetaHistory_.clear();
    }
}

void MoveBudget::resumeAfterInspection() {
    if (pending_ || !needsInspection_) throw AlignmentError(ErrorCode::InvalidTransition, "No uncertain movement awaiting inspection");
    needsInspection_ = false;
}

void MoveBudget::startNewSession() {
    requireReady();
    completed_ = 0; used_ = 0; thetaHistory_.clear(); oscillationLatched_ = false;
}

void writeAlignmentDecision(cv::FileStorage& fs, const AlignmentDecision& r) {
    fs << "status" << r.status << "error_code" << errorCodeName(r.errorCode) << "command_y_mm" << r.yMm << "command_theta_deg" << r.thetaDeg
       << "command_available" << static_cast<int>(r.commandAvailable) << "aligned" << static_cast<int>(r.aligned)
       << "y_evaluated" << static_cast<int>(r.yEvaluated) << "theta" << "{";
    writeThetaResult(fs, r.theta); fs << "}";
    if (r.yEvaluated) { fs << "y" << "{"; writeYResult(fs, r.y); fs << "}"; }
}
} // namespace semes
