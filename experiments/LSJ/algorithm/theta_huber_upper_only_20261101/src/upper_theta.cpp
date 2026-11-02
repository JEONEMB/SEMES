#include "upper_theta.hpp"
#include "utf8_io.hpp"
#include <opencv2/imgproc.hpp>
#include <climits>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace theta {
namespace {
double number(const cv::FileNode& n, const char* key) {
    auto v=n[key];
    if(v.empty() || (!v.isInt() && !v.isReal()))
        throw std::runtime_error(std::string("missing/nonnumeric config: ")+key);
    double x=static_cast<double>(v);
    if(!std::isfinite(x)) throw std::runtime_error("non-finite config");
    return x;
}
int integer(const cv::FileNode& n, const char* key) {
    double x=number(n,key);
    if(std::floor(x)!=x || x<INT_MIN || x>INT_MAX)
        throw std::runtime_error("noninteger config coordinate");
    return static_cast<int>(x);
}
void validate(const UpperConfig& c) {
    const auto& r=c.upper;
    if(c.calibration_id.empty() || c.width<=0 || c.height<=0 ||
       r.x_first<0 || r.x_first>r.x_last || r.x_last>=c.width ||
       r.y_first<0 || r.y_first>r.y_last || r.y_last>=c.height)
        throw std::runtime_error("invalid image geometry/upper ROI");
    for(const auto& e:r.excluded_x)
        if(e[0]<r.x_first || e[1]>r.x_last || e[0]>e[1])
            throw std::runtime_error("invalid upper occlusion mask");
    if(!std::isfinite(c.huber.t) || c.huber.t<=0 ||
       !std::isfinite(c.huber.tolerance) || c.huber.tolerance<=0 || c.huber.max_iterations<2)
        throw std::runtime_error("invalid Huber settings");
    decodeUpperAngle(c.intercept_deg,c);
}
}

double decodeUpperAngle(double angle, const UpperConfig& c) {
    if(!std::isfinite(angle) || !std::isfinite(c.intercept_deg) ||
       !std::isfinite(c.per_theta_deg) || c.per_theta_deg==0 || !std::isfinite(c.target_theta_deg))
        throw std::runtime_error("invalid upper-only calibration/angle");
    double theta=(angle-c.intercept_deg)/c.per_theta_deg;
    if(!std::isfinite(theta) || !std::isfinite(c.target_theta_deg-theta))
        throw std::runtime_error("non-finite theta estimate");
    return theta;
}

UpperConfig loadUpperConfig(const std::string& path) {
    cv::FileStorage fs(readText(path),cv::FileStorage::READ|cv::FileStorage::MEMORY|cv::FileStorage::FORMAT_JSON);
    if(!fs.isOpened() || integer(fs.root(),"schema_version")!=1)
        throw std::runtime_error("unsupported upper config schema");
    UpperConfig c;
    c.calibration_id=static_cast<std::string>(fs["calibration_id"]);
    c.width=integer(fs["image"],"width"); c.height=integer(fs["image"],"height");
    const auto r=fs["upper_region"];
    c.upper={integer(r,"x_first"),integer(r,"x_last"),integer(r,"y_first"),integer(r,"y_last"),{}};
    const auto intervals=r["excluded_x"];
    if(intervals.empty() || !intervals.isSeq()) throw std::runtime_error("missing upper occlusion mask");
    for(const auto& e:intervals) {
        if(!e.isSeq() || e.size()!=2 || !e[0].isInt() || !e[1].isInt())
            throw std::runtime_error("invalid upper occlusion interval");
        c.upper.excluded_x.emplace_back(static_cast<int>(e[0]),static_cast<int>(e[1]));
    }
    c.huber.t=number(fs["huber"],"t");
    c.huber.max_iterations=integer(fs["huber"],"max_iterations");
    c.huber.tolerance=number(fs["huber"],"tolerance");
    const auto m=fs["model"];
    if(static_cast<std::string>(m["type"])!="affine_forward_upper_only")
        throw std::runtime_error("unsupported upper-only model");
    c.intercept_deg=number(m,"intercept_deg");
    c.per_theta_deg=number(m,"per_theta_deg");
    c.target_theta_deg=number(m,"target_theta_deg");
    validate(c);
    return c;
}

UpperDetector::UpperDetector(UpperConfig c):config_(std::move(c)) { validate(config_); }

UpperResult UpperDetector::estimate(const cv::Mat& image) const {
    if(image.empty() || image.cols!=config_.width || image.rows!=config_.height ||
       image.depth()!=CV_8U || (image.channels()!=1 && image.channels()!=3 && image.channels()!=4))
        throw std::runtime_error("expected configured-size 8-bit gray/BGR/BGRA image");
    cv::Mat gray,gy;
    if(image.channels()==1) gray=image;
    else cv::cvtColor(image,gray,image.channels()==3?cv::COLOR_BGR2GRAY:cv::COLOR_BGRA2GRAY);
    // Identical full-image Sobel operation; only the UPPER ROI is selected/fitted.
    cv::Sobel(gray,gy,CV_32F,0,1,3,1,0,cv::BORDER_REFLECT_101);
    UpperResult result;
    result.upper=fitHuberVertical(selectColumnMaxima(gy,config_.upper),config_.huber);
    if(!result.upper.converged)
        throw std::runtime_error("Huber iteration limit reached; no correction produced");
    result.theta_deg=decodeUpperAngle(result.upper.angle_deg,config_);
    result.theta_correction_deg=config_.target_theta_deg-result.theta_deg;
    return result;
}
}
