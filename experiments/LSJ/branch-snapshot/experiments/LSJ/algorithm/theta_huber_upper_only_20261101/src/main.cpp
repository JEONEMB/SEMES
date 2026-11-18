#include "upper_theta.hpp"
#include "utf8_io.hpp"
#include <opencv2/imgcodecs.hpp>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace {
int run(const std::vector<std::string>& args) {
    if(args.size()!=3) { std::cerr<<"Usage: theta_upper_alignment CONFIG.json IMAGE.bmp\n"; return 2; }
    const theta::UpperDetector detector(theta::loadUpperConfig(args[1]));
    auto image=cv::imdecode(theta::readBytes(args[2]),cv::IMREAD_UNCHANGED);
    const auto r=detector.estimate(image);
    const auto& l=r.upper;
    std::cout<<std::setprecision(17)<<"{\"upper\":{\"slope\":"<<l.slope
      <<",\"intercept_px\":"<<l.intercept_px<<",\"angle_deg\":"<<l.angle_deg
      <<",\"mad_scale_px\":"<<l.mad_scale_px<<",\"tau_px\":"<<l.tau_px
      <<",\"point_count\":"<<l.point_count<<",\"iterations\":"<<l.iterations
      <<",\"deviance_delta\":"<<l.deviance_delta<<",\"downweighted_points\":"<<l.downweighted_points
      <<",\"converged\":true},\"lower_used\":false,\"y_effect_corrected\":false"
      <<",\"theta_estimate_deg\":"<<r.theta_deg<<",\"theta_correction_deg\":"<<r.theta_correction_deg
      <<",\"y_correction_mm\":0,\"y_command_available\":false}\n";
    return 0;
}
}
#ifdef _WIN32
int wmain(int argc,wchar_t** argv) {
#else
int main(int argc,char** argv) {
#endif
    try {
        std::vector<std::string> args;
        for(int i=0;i<argc;++i) {
#ifdef _WIN32
            args.push_back(theta::toUtf8(argv[i]));
#else
            args.emplace_back(argv[i]);
#endif
        }
        return run(args);
    } catch(const std::exception& e) { std::cerr<<"theta_upper_alignment: "<<e.what()<<'\n'; return 1; }
}
