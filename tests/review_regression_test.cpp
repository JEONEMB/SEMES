#include "alignment.hpp"
#include "sequential_alignment.hpp"
#include "validation_utils.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

using namespace semes;
void check(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
template<class F> void expect(ErrorCode code, F action) {
    try { action(); } catch (const AlignmentError& e) { check(e.code() == code, "Wrong structured error"); return; }
    throw std::runtime_error("Expected structured failure");
}

void numericalFaults() {
    const cv::Vec4d p(15, 220, 10, 1.5), delta(0, 0, 0, 0);
    const double nan = std::numeric_limits<double>::quiet_NaN(), inf = std::numeric_limits<double>::infinity();
    for (double bad : {nan, inf, -inf}) {
        for (int i = 0; i < 4; ++i) {
            auto d = delta; d[i] = bad;
            expect(ErrorCode::NumericalFailure, [&] { detail::boundedEdgeUpdate(p, d, 10); });
            auto state = p; state[i] = bad;
            expect(ErrorCode::NumericalFailure, [&] { detail::boundedEdgeUpdate(state, delta, 10); });
        }
        expect(ErrorCode::NumericalFailure, [&] { detail::finiteClamp(bad, -0.5, 0.5); });
    }
    auto large = p, d = delta;
    large[0] = std::numeric_limits<double>::max(); d[0] = -large[0];
    expect(ErrorCode::NumericalFailure, [&] { detail::boundedEdgeUpdate(large, d, 10); });
    const auto bounded = detail::boundedEdgeUpdate(p, cv::Vec4d(0,0,-100,100), 10);
    check(bounded[2] == 10.5 && bounded[3] == 1.0, "Finite clamp behavior changed");
}

void thetaChecks(ThetaConfig config) {
    const std::array<double,2> targets{{0,0}};
    // Equal slopes give an independent analytic interval: intersect [-tol-qi, tol-qi].
    auto equal = config;
    for (auto& line : equal.lines) line.positiveSlope = line.negativeSlope = -0.1;
    int cases = 0;
    for (int a = -25; a <= 25; ++a) for (int b = -25; b <= 25; ++b) {
        const double q1 = a * 0.001, q2 = b * 0.001;
        const auto r = thetaFromAngles({{-0.1*q1, -0.1*q2}}, targets, equal);
        const bool already = std::abs(q1) <= 0.005+1e-12 && std::abs(q2) <= 0.005+1e-12;
        const double lo = std::max(-0.005-q1, -0.005-q2), hi = std::min(0.005-q1, 0.005-q2);
        const double first = std::ceil((lo-1e-12)/0.01), last = std::floor((hi+1e-12)/0.01);
        check(r.thetaWithinTolerance == already, "Both-line acceptance changed");
        check(r.commandAvailable == (already || first <= last), "Tick feasibility differs from analytic interval");
        if (r.commandAvailable && !already) {
            check(r.commandDeg >= lo-1e-12 && r.commandDeg <= hi+1e-12, "Command misses a line target");
            check(r.predictedWithinTolerance && !r.thetaWithinTolerance, "Prediction must not claim current acceptance");
            check(std::abs(r.commandDeg - std::max(first,std::min(0.0,last))*0.01)<1e-12, "Not nearest feasible tick");
        }
        ++cases;
    }
    auto r = thetaFromAngles({{-0.1*0.004, -0.1*0.0065}}, targets, equal);
    check(!r.commandAvailable && r.errorCode == ErrorCode::ThetaNoReachableTick, "Disagreeing near-half-tick pair should stop");
    r = thetaFromAngles({{-0.1*0.02, 0.1*0.02}}, targets, equal);
    check(!r.commandAvailable && r.errorCode == ErrorCode::ThetaTargetConflict, "Disjoint line intervals should stop");
    for (double q : {-0.006, 0.006, -0.505, 0.505, -2.0, 2.0}) {
        std::array<double,2> angles;
        for (int i = 0; i < 2; ++i) angles[i] = q*(q>=0 ? config.lines[i].positiveSlope : config.lines[i].negativeSlope);
        r = thetaFromAngles(angles, targets, config);
        check(r.commandAvailable && r.commandDeg*q < 0, "Saved signed calibration cannot correct pose");
        check(std::abs(r.commandDeg)<=config.maxStepDeg+1e-12, "Configured cap exceeded");
        const double endpoint = q+r.plannedTotalCorrectionDeg;
        for (int i = 0; i < 2; ++i) angles[i] = endpoint*(endpoint>=0 ? config.lines[i].positiveSlope : config.lines[i].negativeSlope);
        check(thetaFromAngles(angles, targets, config).thetaWithinTolerance, "Cross-zero endpoint fails saved branch model");
    }
    std::cout << "PASS theta analytic interval cases: " << cases << ", saved branch crossing and cap\n";
}

void budgetChecks() {
    MoveBudget budget; AlignmentDecision move; move.commandAvailable = true; move.yMm = 0.05;
    budget.issue(move);
    expect(ErrorCode::PendingMove, [&] { budget.startNewSession(); });
    budget.reportFailure(MoveFailure::NoMotion);
    check(budget.used()==0 && budget.completed()==0 && !budget.pending(), "No-motion rejection must not count as a move");
    budget.issue(move); budget.reportFailure(MoveFailure::PartialOrUnknown);
    check(budget.used()==1 && budget.completed()==0 && budget.needsInspection(), "Uncertain movement must consume a slot");
    expect(ErrorCode::MovementUncertain, [&] { budget.issue(move); });
    expect(ErrorCode::MovementUncertain, [&] { budget.startNewSession(); });
    budget.resumeAfterInspection();
    check(budget.used()==1, "Recovery reset budget");
    for (int i=1; i<20; ++i) { check(budget.issue(move).commandAvailable, "Unexpected early limit"); budget.acknowledgeExecuted(); }
    const auto blocked = budget.issue(move);
    check(blocked.errorCode==ErrorCode::MoveLimitReached && !blocked.commandAvailable, "Unknown movement escaped 20-slot cap");
    AlignmentDecision done; done.commandAvailable=true; done.aligned=true;
    check(budget.issue(done).aligned, "Final measurement after 20 must remain possible");
    budget.startNewSession();
    check(budget.used()==0 && budget.completed()==0, "Explicit new task did not reset");
    move.yMm=0; move.thetaDeg=0.01; budget.issue(move); budget.acknowledgeExecuted();
    move.thetaDeg=-0.01; budget.issue(move); budget.acknowledgeExecuted();
    move.thetaDeg=0.01; const auto cycle=budget.issue(move);
    check(cycle.errorCode==ErrorCode::ThetaOscillation && !cycle.commandAvailable && !cycle.aligned && !budget.pending(), "Third reversal not stopped");
    check(budget.used()==2, "Blocked reversal consumed a slot");
    move.thetaDeg=0.02;
    check(budget.issue(move).errorCode==ErrorCode::ThetaOscillation, "Oscillation stop not latched");
    std::cout << "PASS execution outcomes, recovery, combined 20-slot budget and reversal stop\n";
}

void yDiagnostics(const YConfig& config) {
    std::array<double,3> x;
    for(int i=0;i<3;++i) x[i]=config.markers[i].roi.x+(config.markers[i].roi.width-1)/2.0+0.2;
    const auto r=yFromPositions(x,config);
    check(r.errorCode==ErrorCode::YNoReachableTick && !r.commandAvailable && !r.withinTarget, "Narrow original Y interval must not be relaxed");
    check(r.tickDiagnosticsAvailable && std::abs(r.nearestTickBelowMm+0.05)<1e-12 && r.nearestTickAboveMm==0, "Nearest Y ticks incorrect");
}

void adapterChecks(const char* tc, const char* yc, const char* pathFile) {
    cv::Mat reference = readGrayUtf8(readTextUtf8(pathFile)), shifted;
    cv::warpAffine(reference, shifted, cv::Mat(cv::Matx23d(1,0,5,0,1,0)), reference.size(), cv::INTER_NEAREST, cv::BORDER_REFLECT_101);
    configureAlignment(reference, tc, yc); startNewAlignmentSession();
    auto cmd=alignment(shifted); check(cmd.y<0 && cmd.t==0, "Native Y routing failed");
    cv::Mat invalidReference;
    expect(ErrorCode::InvalidInput, [&] { configureAlignment(invalidReference, tc, yc); });
    expect(ErrorCode::PendingMove, [&] { alignment(reference); });
    configureAlignment(reference, tc, yc);
    expect(ErrorCode::PendingMove, [&] { alignment(reference); });
    acknowledgeAlignmentMoveExecuted();
    configureAlignment(reference, tc, yc);
    check(completedAlignmentMoves()==1 && usedAlignmentMoves()==1, "Reload erased acknowledged count");
    alignment(shifted); reportAlignmentMoveFailure(MoveFailure::PartialOrUnknown);
    configureAlignment(reference, tc, yc);
    expect(ErrorCode::MovementUncertain, [&] { alignment(reference); });
    resumeAlignmentAfterInspection();
    check(completedAlignmentMoves()==1 && usedAlignmentMoves()==2, "Inspection erased movement count");
    alignment(shifted); reportAlignmentMoveFailure(MoveFailure::NoMotion);
    check(usedAlignmentMoves()==2, "Known no-motion rejected command counted");
    for(int i=2;i<20;++i) { alignment(shifted); acknowledgeAlignmentMoveExecuted(); }
    // Reusing a synthetic frame tests the adapter's lifecycle, not a simulated robot trajectory.
    expect(ErrorCode::MoveLimitReached, [&] { alignment(shifted); });
    cmd=alignment(reference); check(cmd.y==0 && cmd.t==0, "Final observation blocked after 20");
    startNewAlignmentSession(); check(usedAlignmentMoves()==0, "New task reset failed");
    cv::Mat wrongSize(10,10,CV_8UC1,cv::Scalar(0));
    expect(ErrorCode::InvalidInput, [&] { alignment(wrongSize); });
    std::cout << "PASS native adapter reload/pending/failure/recovery, final-frame check and 20-slot cap\n";
}

int main(int argc,char** argv) {
    try {
        cv::setNumThreads(2);
        if(argc==4 && std::string(argv[1])=="--config-probe") {
            try {
                if(std::string(argv[2])=="theta") ThetaConfig::load(argv[3]); else YConfig::load(argv[3]);
            } catch(const AlignmentError& e) { std::cout << errorCodeName(e.code()) << '\n'; return 3; }
            std::cout << "valid\n"; return 0;
        }
        check(argc==3 || argc==4,"Expected theta config, Y config, optional reference-path file");
        numericalFaults(); thetaChecks(ThetaConfig::load(argv[1])); budgetChecks(); yDiagnostics(YConfig::load(argv[2]));
        if(argc==4) adapterChecks(argv[1],argv[2],argv[3]);
        std::cout << "PASS review regression suite\n"; return 0;
    } catch(const std::exception& e) { std::cerr << "FAIL: " << e.what() << '\n'; return 1; }
}
