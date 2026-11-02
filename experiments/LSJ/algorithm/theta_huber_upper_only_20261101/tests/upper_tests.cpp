#include "upper_theta.hpp"
#include "alignment.hpp"
#include "utf8_io.hpp"
#include <opencv2/imgcodecs.hpp>
#include <cmath>
#include <iostream>
#include <functional>
#include <stdexcept>

namespace {
int checks=0;
void require(bool c,const char* message) { ++checks; if(!c) throw std::runtime_error(message); }
void throws(const std::function<void()>& f,const char* message) {
    bool ok=false; try{f();}catch(const std::exception&){ok=true;} require(ok,message);
}
int run(const std::vector<std::string>& args) {
    theta::UpperConfig c;
    c.intercept_deg=.3; c.per_theta_deg=-.08;
    for(double t:{-.7,0.,.3,1.7})
        require(std::abs(theta::decodeUpperAngle(.3-.08*t,c)-t)<1e-12,"known theta inverse");
    throws([&]{theta::decodeUpperAngle(NAN,c);},"NaN angle rejected");
    auto zero=c; zero.per_theta_deg=0;
    throws([&]{theta::decodeUpperAngle(.3,zero);},"zero sensitivity rejected");
    cv::Mat empty;
    throws([&]{alignment(empty);},"unconfigured skeleton rejected");
    if(args.size()==3) {
        auto config=theta::loadUpperConfig(args[1]);
        theta::UpperDetector detector(config);
        auto image=cv::imdecode(theta::readBytes(args[2]),cv::IMREAD_GRAYSCALE);
        auto original=detector.estimate(image);
        auto changed=image.clone();
        // Change every pixel strictly below the upper ROI plus Sobel support.
        // Includes the complete lower structure, without touching the source file.
        changed.rowRange(config.upper.y_last+2,changed.rows).setTo(0);
        auto removed=detector.estimate(changed);
        require(removed.upper.slope==original.upper.slope,"lower pixels cannot change upper slope");
        require(removed.theta_correction_deg==original.theta_correction_deg,"lower pixels cannot change correction");
        changed.rowRange(config.upper.y_last+2,changed.rows).setTo(255);
        auto bright=detector.estimate(changed);
        require(bright.theta_correction_deg==original.theta_correction_deg,"bright lower pixels cannot change correction");
        configureThetaAlignment(args[1]);
        auto cmd=alignment(image);
        require(cmd.y==0 && cmd.t==original.theta_correction_deg,"skeleton output matches upper detector");
        throws([&]{detector.estimate(cv::Mat());},"empty input rejected");
        throws([&]{detector.estimate(cv::Mat(10,10,CV_8UC1));},"wrong size rejected");
        std::cout<<"lower pixel independence checks passed; ";
    } else if(args.size()!=1) throw std::runtime_error("Usage: upper_tests [CONFIG.json IMAGE.bmp]");
    std::cout<<checks<<" checks passed\n";
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
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
