#pragma once
#include "y_alignment.hpp"
#include <vector>

namespace semes {
struct AlignmentDecision {
    double yMm = 0, thetaDeg = 0;
    bool commandAvailable = false, aligned = false, yEvaluated = false;
    std::string status;
    ErrorCode errorCode = ErrorCode::None;
    ThetaResult theta;
    YResult y;
};

AlignmentDecision chooseAlignmentCorrection(const ThetaResult& theta, const YResult* y = nullptr);

// Stateless per-frame estimator. Independent offline images do not count as machine moves.
class SequentialAligner {
public:
    SequentialAligner(const cv::Mat& reference, const ThetaConfig& theta, const YConfig& y);
    AlignmentDecision estimate(const cv::Mat& image) const;
private:
    ThetaEstimator theta_;
    YEstimator y_;
};

// Caller must acknowledge an actually completed move before submitting its new camera frame.
class MoveBudget {
public:
    AlignmentDecision issue(const AlignmentDecision& decision);
    void acknowledgeExecuted();
    void reportFailure(MoveFailure failure);
    void resumeAfterInspection();
    void startNewSession();
    void requireReady() const;
    int completed() const { return completed_; }
    int used() const { return used_; }
    bool pending() const { return pending_; }
    bool needsInspection() const { return needsInspection_; }
private:
    int completed_ = 0, used_ = 0;
    bool pending_ = false, needsInspection_ = false, oscillationLatched_ = false;
    AlignmentDecision pendingDecision_;
    std::vector<double> thetaHistory_;
};
void writeAlignmentDecision(cv::FileStorage& fs, const AlignmentDecision& result);
} // namespace semes
