#pragma once
#include "y_alignment.hpp"

namespace semes {
struct AlignmentDecision {
    double yMm = 0, thetaDeg = 0;
    bool commandAvailable = false, aligned = false, yEvaluated = false;
    std::string status;
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
    int completed() const { return completed_; }
    bool pending() const { return pending_; }
private:
    int completed_ = 0;
    bool pending_ = false;
};
void writeAlignmentDecision(cv::FileStorage& fs, const AlignmentDecision& result);
} // namespace semes
