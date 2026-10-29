#include "alignment.hpp"
#include "theta_alignment.hpp"
#include "sequential_alignment.hpp"
#include <memory>
#include <stdexcept>

namespace {
std::unique_ptr<semes::ThetaEstimator> estimator;
std::unique_ptr<semes::SequentialAligner> sequential;
semes::MoveBudget budget;
void requireConfigured() {
    if (!estimator && !sequential) throw semes::AlignmentError(semes::ErrorCode::NotConfigured, "Call configureAlignment first");
}
}

void configureAlignment(const cv::Mat& reference, const std::string& configPath) {
    try {
        auto next = std::make_unique<semes::ThetaEstimator>(reference, semes::ThetaConfig::load(configPath));
        estimator = std::move(next);
        sequential.reset();
    } catch (const semes::AlignmentError&) { throw; }
    catch (const std::invalid_argument& e) { throw semes::AlignmentError(semes::ErrorCode::InvalidInput, e.what()); }
    catch (const std::exception& e) { throw semes::AlignmentError(semes::ErrorCode::MeasurementFailed, e.what()); }
}

void configureAlignment(const cv::Mat& reference, const std::string& thetaConfigPath, const std::string& yConfigPath) {
    try {
        auto next = std::make_unique<semes::SequentialAligner>(reference, semes::ThetaConfig::load(thetaConfigPath),
                                                         semes::YConfig::load(yConfigPath));
        sequential = std::move(next); estimator.reset();
    } catch (const semes::AlignmentError&) { throw; }
    catch (const std::invalid_argument& e) { throw semes::AlignmentError(semes::ErrorCode::InvalidInput, e.what()); }
    catch (const std::exception& e) { throw semes::AlignmentError(semes::ErrorCode::MeasurementFailed, e.what()); }
}

void acknowledgeAlignmentMoveExecuted() { budget.acknowledgeExecuted(); }
int completedAlignmentMoves() { return budget.completed(); }
int usedAlignmentMoves() { return budget.used(); }
void reportAlignmentMoveFailure(semes::MoveFailure failure) { budget.reportFailure(failure); }
void resumeAlignmentAfterInspection() { budget.resumeAfterInspection(); }
void startNewAlignmentSession() { requireConfigured(); budget.startNewSession(); }

Alignment_t alignment(cv::Mat &src) {
    try {
        requireConfigured(); budget.requireReady();
        semes::AlignmentDecision decision;
        if (sequential) decision = sequential->estimate(src);
        else {
            const auto theta = estimator->estimate(src);
            decision.theta = theta; decision.thetaDeg = theta.commandDeg;
            decision.commandAvailable = theta.commandAvailable; decision.status = theta.status;
            decision.errorCode = theta.errorCode;
            // Theta-only acceptance is not a claim that Y is aligned.
        }
        const auto result = budget.issue(decision);
        if (!result.commandAvailable) throw semes::AlignmentError(result.errorCode, result.status);
        return Alignment_t{result.yMm, result.thetaDeg};
    } catch (const semes::AlignmentError&) { throw; }
    catch (const std::invalid_argument& e) { throw semes::AlignmentError(semes::ErrorCode::InvalidInput, e.what()); }
    catch (const std::exception& e) { throw semes::AlignmentError(semes::ErrorCode::MeasurementFailed, e.what()); }
}
