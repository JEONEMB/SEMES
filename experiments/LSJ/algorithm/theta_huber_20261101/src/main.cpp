#include "theta_pipeline.hpp"
#include "utf8_io.hpp"
#include <opencv2/imgcodecs.hpp>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace {
void line(const theta::LineResult& r) {
    std::cout << "{\"slope\":" << r.slope << ",\"intercept_px\":" << r.intercept_px
              << ",\"angle_deg\":" << r.angle_deg << ",\"mad_scale_px\":" << r.mad_scale_px
              << ",\"tau_px\":" << r.tau_px << ",\"point_count\":" << r.point_count
              << ",\"iterations\":" << r.iterations << ",\"deviance_delta\":" << r.deviance_delta
              << ",\"downweighted_points\":" << r.downweighted_points << ",\"converged\":true}";
}
int run(const std::vector<std::string>& args) {
    if (args.size() != 3) {
        std::cerr << "Usage: theta_alignment CONFIG.json IMAGE.bmp\n";
        return 2;
    }
    const theta::Detector detector(theta::loadConfig(args[1]));
    const auto image = cv::imdecode(theta::readBytes(args[2]), cv::IMREAD_UNCHANGED);
    const auto result = detector.estimate(image);
    std::cout << std::setprecision(17) << "{\"upper\":"; line(result.upper);
    std::cout << ",\"lower\":"; line(result.lower);
    std::cout << ",\"theta_estimate_deg\":" << result.pose.theta_deg
              << ",\"theta_correction_deg\":" << result.pose.theta_correction_deg
              << ",\"nuisance_y_estimate_mm\":" << result.pose.nuisance_y_mm
              << ",\"y_correction_mm\":0,\"y_command_available\":false"
              << ",\"sensitivity_determinant\":" << result.pose.determinant << "}\n";
    return 0;
}
}

#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
#else
int main(int argc, char** argv) {
#endif
    try {
        std::vector<std::string> args;
        for (int i = 0; i < argc; ++i) {
#ifdef _WIN32
            args.push_back(theta::toUtf8(argv[i]));
#else
            args.emplace_back(argv[i]);
#endif
        }
        return run(args);
    } catch (const std::exception& e) {
        std::cerr << "theta_alignment: " << e.what() << '\n';
        return 1;
    }
}
