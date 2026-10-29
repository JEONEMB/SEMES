#include "alignment.hpp"
#include "sequential_alignment.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void near(double a, double b, double eps, const char* message) { require(std::abs(a-b) <= eps, message); }
template<class F> void throws(F action, const char* message) {
    bool caught = false; try { action(); } catch (const std::exception&) { caught = true; }
    require(caught, message);
}

void contracts(semes::YConfig config) {
    // Explicit physical units in numerical fixtures; no measured-image accuracy claims.
    const std::array<double, 3> x{{1000, 1500, 2000}};
    const std::array<double, 3> low{{0.10, 0.12, 0.14}}, high{{0.20, 0.18, 0.17}};
    for (int i = 0; i < 3; ++i) {
        config.markers[i].slopes.fill(10);
        config.markers[i].lowPx = x[i] + low[i]*10;
        config.markers[i].highPx = x[i] + high[i]*10;
    }
    auto result = semes::yFromPositions(x, config);
    near(result.commonLowMm, 0.14, 1e-10, "Common lower bound");
    near(result.commonHighMm, 0.17, 1e-10, "Common upper bound");
    near(result.commandMm, 0.15, 1e-12, "Select a feasible 0.05mm tick");
    require(result.commandAvailable && !result.withinTarget, "Movement is not already aligned");
    auto reached = x; for (auto& value : reached) value += 1.5;
    auto done = semes::yFromPositions(reached, config);
    require(done.withinTarget && done.commandAvailable && done.commandMm == 0, "All three inside stop");
    auto mismatch = reached; mismatch[2] += 5;
    auto bad = semes::yFromPositions(mismatch, config);
    require(!bad.withinTarget && !bad.commandAvailable && bad.status == "no_common_y_interval", "One bad QR cannot average away");

    auto narrower = config;
    for (int i = 0; i < 3; ++i) { narrower.markers[i].lowPx = x[i]+1.4; narrower.markers[i].highPx = x[i]+1.49; }
    bad = semes::yFromPositions(x, narrower);
    require(bad.status == "no_reachable_y_tick" && !bad.commandAvailable, "Interval can exist without any motor tick");
    for (int i = 0; i < 3; ++i) { narrower.markers[i].lowPx = x[i]-1.5; narrower.markers[i].highPx = x[i]-1.5; }
    result = semes::yFromPositions(x, narrower);
    near(result.commandMm, -0.15, 1e-12, "Exact inclusive negative tick at range endpoint");
    require(result.commandAvailable, "Exact tick accepted");
    auto conflictingSlopes = config;
    conflictingSlopes.markers[0].slopes[3] = 100;
    bad = semes::yFromPositions(x, conflictingSlopes);
    require(!bad.commandAvailable && bad.status == "no_common_y_interval", "Fourth run slope is not ignored or averaged");
    auto broad = config;
    for (int i = 0; i < 3; ++i) { broad.markers[i].lowPx = x[i]+1; broad.markers[i].highPx = x[i]+3; }
    result = semes::yFromPositions(x, broad);
    near(result.commandMm, 0.10, 1e-12, "Choose smallest movement among multiple feasible ticks, not midpoint");
    auto invalid = config; invalid.markers[0].lowPx = invalid.markers[0].highPx + 1;
    throws([&] { invalid.validate(); }, "Reject reversed target range");
    invalid = config; invalid.markers[0].slopes[2] = 0;
    throws([&] { invalid.validate(); }, "Reject invalid slope");
    invalid = config; invalid.tickMm = 0;
    throws([&] { invalid.validate(); }, "Reject invalid tick");
    auto nonfinite = x; nonfinite[0] = std::numeric_limits<double>::quiet_NaN();
    throws([&] { semes::yFromPositions(nonfinite, config); }, "Reject nonfinite observation");

    semes::ThetaResult theta; theta.commandDeg = -0.1;
    auto decision = semes::chooseAlignmentCorrection(theta, &done);
    require(decision.thetaDeg == -0.1 && decision.yMm == 0 && !decision.yEvaluated, "Theta takes priority");
    theta.thetaWithinTolerance = true; theta.commandDeg = 0;
    decision = semes::chooseAlignmentCorrection(theta, &result);
    require(decision.yMm == 0.10 && decision.thetaDeg == 0 && !decision.aligned, "Y after theta");
    decision = semes::chooseAlignmentCorrection(theta, &done);
    require(decision.aligned && decision.yMm == 0 && decision.thetaDeg == 0, "Both current measurements required for done");
    decision = semes::chooseAlignmentCorrection(theta, &bad);
    require(!decision.commandAvailable && !decision.aligned, "Failure is not zero command success");
    theta.commandAvailable = false;
    decision = semes::chooseAlignmentCorrection(theta, &done);
    require(!decision.commandAvailable && !decision.yEvaluated, "No Y after unresolved theta");

    semes::MoveBudget budget;
    semes::AlignmentDecision move; move.commandAvailable = true;
    throws([&] { budget.acknowledgeExecuted(); }, "Cannot acknowledge nonexistent move");
    for (int i = 0; i < 20; ++i) {
        move.yMm = i % 2 ? 0.05 : 0; move.thetaDeg = i % 2 ? 0 : 0.01;
        require(budget.issue(move).commandAvailable, "First 20 moves allowed");
        require(budget.completed() == i, "Issuing a command is not executing it");
        throws([&] { budget.issue(move); }, "Do not issue duplicate pending moves");
        budget.acknowledgeExecuted();
        throws([&] { budget.acknowledgeExecuted(); }, "Do not double count execution");
    }
    decision = budget.issue(move);
    require(decision.status == "move_limit_reached" && !decision.commandAvailable && decision.yMm == 0 && decision.thetaDeg == 0, "Block move 21 across both axes");
    semes::AlignmentDecision aligned; aligned.commandAvailable = true; aligned.aligned = true;
    require(budget.issue(aligned).aligned, "Still inspect final image after move 20");
}

void imageChecks(const char* thetaPath, const char* yPath, const char* referenceListPath) {
    const auto yc = semes::YConfig::load(yPath);
    const auto tc = semes::ThetaConfig::load(thetaPath);
    const cv::Mat reference = semes::readGrayUtf8(semes::readTextUtf8(referenceListPath));
    const semes::YEstimator estimator(reference, yc);
    auto result = estimator.estimate(reference);
    require(result.withinTarget, "Selected ranges contain the first center reference ROI positions");
    for (int i = 0; i < 3; ++i)
        near(result.x[i], yc.markers[i].roi.x + (yc.markers[i].roi.width-1)/2.0, 1e-5, "Preserve zero-based ROI pixel-center coordinates");
    cv::Mat translated;
    cv::warpAffine(reference, translated, cv::Mat(cv::Matx23d(1,0,5, 0,1,0)), reference.size(), cv::INTER_NEAREST, cv::BORDER_REFLECT_101);
    result = estimator.estimate(translated);
    for (int i = 0; i < 3; ++i)
        near(result.x[i], yc.markers[i].roi.x + (yc.markers[i].roi.width-1)/2.0 + 5, 0.001, "Known +5 pixel shift recovered in each ROI");
    require(result.commandAvailable && result.commandMm < 0, "Positive image shift requires negative nominal Y correction");
    // Exercise preserved skeleton adapter, actual image routing, and execution acknowledgement.
    configureAlignment(reference, thetaPath, yPath);
    cv::Mat current = reference.clone();
    auto command = alignment(current);
    require(command.y == 0 && command.t == 0 && completedAlignmentMoves() == 0, "Reference aligned through skeleton");
    command = alignment(translated);
    require(command.y < 0 && command.t == 0, "Translated image reaches Y stage through skeleton");
    throws([&] { alignment(translated); }, "Wait for completed movement before next image");
    acknowledgeAlignmentMoveExecuted();
    require(completedAlignmentMoves() == 1, "External execution acknowledgement counted");
    current = reference.clone(); command = alignment(current);
    require(command.y == 0 && command.t == 0, "Fresh aligned image stops session");
    cv::Mat wrongSize(10, 10, CV_8UC1, cv::Scalar(0));
    throws([&] { alignment(wrongSize); }, "Reject wrong native size");
    cv::Mat blank(reference.size(), CV_8UC1, cv::Scalar(0));
    throws([&] { estimator.estimate(blank); }, "Reject textureless matching image");
    std::cout << "PASS native reference, known 5px translation, QR coordinates, skeleton integration and acknowledgement\n";
}

int main(int argc, char** argv) {
    try {
        require(argc == 3 || argc == 4, "Expected theta config, Y config, optional UTF-8 reference-path file");
        cv::setNumThreads(2);
        contracts(semes::YConfig::load(argv[2]));
        std::cout << "PASS Y intervals, all 12 slopes, tick feasibility, axis sequencing and 20 executed-move cap\n";
        if (argc == 4) imageChecks(argv[1], argv[2], argv[3]);
        return 0;
    } catch (const std::exception& e) { std::cerr << "FAIL: " << e.what() << '\n'; return 1; }
}
