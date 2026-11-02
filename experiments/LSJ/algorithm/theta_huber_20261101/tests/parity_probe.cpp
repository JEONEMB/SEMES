// Implementation check only; fixtures contain calibration images (no held-out images).
#include "theta_pipeline.hpp"
#include "alignment.hpp"
#include "utf8_io.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace {
int run(const std::string& path) {
    cv::FileStorage fixture(theta::readText(path),cv::FileStorage::READ|cv::FileStorage::MEMORY|cv::FileStorage::FORMAT_JSON);
    const std::string configPath=static_cast<std::string>(fixture["config_path"]);
    auto config=theta::loadConfig(configPath);
    theta::Detector detector(config);
    configureThetaAlignment(configPath);
    std::cout << std::setprecision(17) << "[";
    bool first=true;
    for(const auto& item: fixture["images"]) {
        const std::string id=static_cast<std::string>(item["id"]);
        auto image=cv::imdecode(theta::readBytes(static_cast<std::string>(item["path"])),cv::IMREAD_GRAYSCALE);
        cv::Mat gy; cv::Sobel(image,gy,CV_32F,0,1,3,1,0,cv::BORDER_REFLECT_101);
        auto result=detector.estimate(image);
        auto command=alignment(image);
        if(command.y!=0 || command.t!=result.pose.theta_correction_deg)
            throw std::runtime_error("skeleton command mismatch");
        for(const std::string target:{"upper","lower"}) {
            auto points=theta::selectColumnMaxima(gy,target=="upper"?config.upper:config.lower);
            auto expected=item[target+"_points"];
            if(points.size()!=expected.size()) throw std::runtime_error("point count mismatch");
            int i=0;
            for(const auto& p:expected) {
                if(points[i].x!=static_cast<double>(p[0]) || points[i].y!=static_cast<double>(p[1]))
                    throw std::runtime_error("Sobel point mismatch: "+id+" "+target);
                ++i;
            }
            auto line=target=="upper"?result.upper:result.lower;
            if(!first) std::cout << ','; first=false;
            std::cout << "{\"id\":\""<<id<<"\",\"target\":\""<<target
              <<"\",\"slope\":"<<line.slope<<",\"intercept_px\":"<<line.intercept_px
              <<",\"angle_deg\":"<<line.angle_deg<<",\"scale_px\":"<<line.mad_scale_px
              <<",\"iterations\":"<<line.iterations<<",\"point_count\":"<<points.size()
              <<",\"deviance_delta\":"<<line.deviance_delta<<",\"theta_correction_deg\":"<<command.t<<"}";
        }
    }
    std::cout << "]\n";
    return 0;
}
}
#ifdef _WIN32
int wmain(int argc,wchar_t** argv) {
#else
int main(int argc,char** argv) {
#endif
    try {
        if(argc!=2) throw std::runtime_error("Usage: theta_parity_probe FIXTURE.json");
#ifdef _WIN32
        return run(theta::toUtf8(argv[1]));
#else
        return run(argv[1]);
#endif
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
