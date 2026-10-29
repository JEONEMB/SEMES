#include "alignment.hpp"
#include "theta_alignment.hpp"
#include "sequential_alignment.hpp"
#include <memory>
#include <stdexcept>

namespace {
std::unique_ptr<semes::ThetaEstimator> estimator;
std::unique_ptr<semes::SequentialAligner> sequential;
semes::MoveBudget budget;
}

void configureAlignment(const cv::Mat& reference, const std::string& configPath) {
    auto next = std::make_unique<semes::ThetaEstimator>(reference, semes::ThetaConfig::load(configPath));
    estimator = std::move(next);
    sequential.reset();
    budget = semes::MoveBudget();
}

void configureAlignment(const cv::Mat& reference, const std::string& thetaConfigPath, const std::string& yConfigPath) {
    auto next = std::make_unique<semes::SequentialAligner>(reference, semes::ThetaConfig::load(thetaConfigPath),
                                                         semes::YConfig::load(yConfigPath));
    sequential = std::move(next); estimator.reset(); budget = semes::MoveBudget();
}

void acknowledgeAlignmentMoveExecuted() { budget.acknowledgeExecuted(); }
int completedAlignmentMoves() { return budget.completed(); }

Alignment_t alignment(cv::Mat &src) {
    if (sequential) {
        if (budget.pending()) throw std::logic_error("Previous movement is awaiting execution acknowledgement");
        const auto result = budget.issue(sequential->estimate(src));
        if (!result.commandAvailable) throw std::runtime_error(result.status);
        return Alignment_t{result.yMm, result.thetaDeg};
    }
    if (!estimator) throw std::logic_error("Call configureAlignment(reference, configPath) first");
    const auto result = estimator->estimate(src);
    if (!result.commandAvailable) throw std::runtime_error(result.status);
    return Alignment_t{0.0, result.commandDeg};
}
