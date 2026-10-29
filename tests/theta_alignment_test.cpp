#include "alignment.hpp"
#include "theta_alignment.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void near(double a, double b, double epsilon, const char* message) {
    require(std::abs(a - b) <= epsilon, message);
}
template<class F> void throws(F action, const char* message) {
    bool caught = false; try { action(); } catch (const std::exception&) { caught = true; }
    require(caught, message);
}

int main(int argc, char** argv) {
    try {
        require(argc == 2, "Expected config path");
        auto config = semes::ThetaConfig::load(argv[1]);
        const std::array<double, 2> targets{{0.22, 0.31}};
        auto observed = [&](double theta) {
            auto angles = targets;
            for (int i = 0; i < 2; ++i) angles[i] += theta * (theta >= 0 ? config.lines[i].positiveSlope : config.lines[i].negativeSlope);
            return angles;
        };
        auto result = semes::thetaFromAngles(targets, targets, config);
        require(result.thetaWithinTolerance && result.commandAvailable && result.commandDeg == 0, "Target must stop theta");
        result = semes::thetaFromAngles(observed(0.4), targets, config);
        require(result.group == 4, "Positive pose must use group4 (not command sign)");
        near(result.commandDeg, -0.4, 1e-12, "Positive pose requires negative correction");
        result = semes::thetaFromAngles(observed(-0.3), targets, config);
        require(result.group == 5, "Negative pose must use group5");
        near(result.commandDeg, 0.3, 1e-12, "Negative pose requires positive correction");
        result = semes::thetaFromAngles(observed(0.005), targets, config);
        require(result.thetaWithinTolerance && result.commandDeg == 0, "Inclusive half tick tolerance");
        result = semes::thetaFromAngles(observed(-0.006), targets, config);
        near(result.commandDeg, 0.01, 1e-12, "Quantize above tolerance");
        result = semes::thetaFromAngles(observed(2.0), targets, config);
        require(result.outsideCalibration, "Extrapolation must be visible");
        near(result.commandDeg, -1.0, 1e-12, "Do not exceed configured step cap");

        // Opposite line errors must not cancel into a false aligned result.
        auto conflicting = targets;
        conflicting[0] += config.lines[0].positiveSlope * 0.02;
        conflicting[1] -= config.lines[1].positiveSlope * 0.02;
        result = semes::thetaFromAngles(conflicting, targets, config);
        require(!result.thetaWithinTolerance, "Disagreement is not theta aligned");
        require(!result.commandAvailable, "Sub-tick disagreement must be explicit");

        auto invalid = config; invalid.lines[0].positiveSlope = 0;
        throws([&] { invalid.validate(); }, "Reject zero slope");
        invalid = config; invalid.tickDeg = 0;
        throws([&] { invalid.validate(); }, "Reject zero tick");
        auto nan = targets; nan[0] = std::numeric_limits<double>::quiet_NaN();
        throws([&] { semes::thetaFromAngles(nan, targets, config); }, "Reject nonfinite feature");
        cv::Mat empty;
        throws([&] { alignment(empty); }, "Skeleton must require reference initialization");
        throws([&] { semes::ThetaEstimator estimator(empty, config); }, "Reject empty reference");
        cv::Mat flat(config.height, config.width, CV_8UC1, cv::Scalar(0));
        throws([&] { semes::ThetaEstimator estimator(flat, config); }, "Reject flat reference");

        // Known synthetic blurred edge tests the actual pixel estimator, not the command formula.
        const auto line = config.lines[0];
        cv::Mat synthetic(config.height, config.width, CV_8UC1, cv::Scalar(15));
        const double expectedAngle = 0.2, slope = std::tan(expectedAngle * CV_PI / 180.0);
        const double xc = (line.x0 + line.x1) / 2.0;
        for (int y = line.expectedY - 30; y <= line.expectedY + 30; ++y)
            for (int x = line.x0; x <= line.x1; ++x) {
                const double edge = line.expectedY + slope * (x - xc);
                synthetic.at<unsigned char>(y, x) = cv::saturate_cast<unsigned char>(15 + 220 * 0.5 * (1 + std::erf((y - edge) / std::sqrt(2.0))));
            }
        const auto measured = semes::ThetaEstimator::measureLine(synthetic, line);
        // Numerical pixel-estimator regression tolerance only, not hardware accuracy.
        near(measured.angleDeg, expectedAngle, 0.001, "Known synthetic edge angle");
        require(measured.points > 400, "Expected dense edge samples");
        std::cout << "PASS theta conversion, branches, quantization, limits, failure states, synthetic edge\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n'; return 1;
    }
}
