#include "theta_pipeline.hpp"
#include "alignment.hpp"
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
int checks = 0;
void require(bool value, const char* message) {
    ++checks;
    if (!value) throw std::runtime_error(message);
}
void throws(const std::function<void()>& action, const char* message) {
    bool did_throw = false;
    try { action(); } catch (const std::exception&) { did_throw = true; }
    require(did_throw, message);
}
}

int main() {
    try {
        // Known two-axis physical model, independent of the fitted dataset.
        theta::Calibration c{0.3, 0.02, -0.1, 0.2, -0.01, 0.2, 0.0};
        for (double y : {-1.0, 0.0, 0.7}) for (double t : {-0.4, 0.0, 0.6}) {
            auto r = theta::decodeAngles(0.3+0.02*y-0.1*t, 0.2-0.01*y+0.2*t, c);
            require(std::abs(r.theta_deg-t)<1e-12, "two-axis theta recovery");
            require(std::abs(r.nuisance_y_mm-y)<1e-12, "two-axis Y recovery");
            require(std::abs(r.theta_correction_deg+t)<1e-12, "correction sign");
        }
        auto r = theta::decodeAngles(c.upper_intercept, c.lower_intercept, c);
        require(r.theta_deg == 0 && r.theta_correction_deg == 0, "nominal zero");
        auto shifted = c; shifted.target_theta_deg = 0.5;
        require(theta::decodeAngles(0.3,0.2,shifted).theta_correction_deg == 0.5, "configurable target");
        auto singular=c; singular.lower_per_y_mm=0.04; singular.lower_per_theta_deg=-0.2;
        throws([&]{theta::decodeAngles(0,0,singular);}, "singular model must fail");
        throws([&]{theta::decodeAngles(NAN,0,c);}, "NaN must fail");

        cv::Mat gy(4,5,CV_32F,cv::Scalar(-5));
        gy.at<float>(1,0)=10; gy.at<float>(2,0)=10;  // first tie wins
        gy.at<float>(3,1)=20;                      // boundary retained
        gy.at<float>(2,3)=-1;                      // negative maximum retained
        gy.at<float>(0,4)=4;                       // no response-8 threshold
        theta::Region region{0,4,0,3,{cv::Vec2i(2,2)}};
        auto p=theta::selectColumnMaxima(gy,region);
        require(p.size()==4, "only occluded column removed");
        require(p[0]==cv::Point2d(0,1), "first maximum tie");
        require(p[1]==cv::Point2d(1,3), "band boundary kept");
        require(p[2]==cv::Point2d(3,2), "negative maximum kept");
        require(p[3]==cv::Point2d(4,0), "weak response kept");
        gy.at<float>(0,0)=NAN;
        throws([&]{theta::selectColumnMaxima(gy,region);}, "non-finite Sobel must fail");

        std::vector<cv::Point2d> points;
        for(int i=0;i<100;++i) points.emplace_back(i, 4+0.2*i+0.3*std::sin(i));
        points[20].y+=50; points[70].y-=50;
        auto fit=theta::fitHuberVertical(points);
        require(fit.converged, "Huber convergence on noisy line");
        require(std::abs(fit.slope-0.2)<0.002, "Huber synthetic slope recovery");
        require(fit.downweighted_points>=2, "outliers downweighted, not deleted");
        require(fit.point_count==100, "all points retained");
        auto limited=theta::HuberSettings(); limited.max_iterations=2;
        auto stopped=theta::fitHuberVertical(points,limited);
        require(!stopped.converged && stopped.stop_reason=="iteration_limit", "limit vs convergence distinguished");
        throws([&]{theta::fitHuberVertical({{0,0},{1,1}});}, "undefined residual degrees of freedom");
        throws([&]{theta::fitHuberVertical({{1,0},{1,1},{1,2}});}, "rank-deficient line");
        throws([&]{theta::fitHuberVertical({{0,NAN},{1,1},{2,3}});}, "invalid points");
        throws([&]{theta::fitHuberVertical({{0,0},{1,0},{2,0}});}, "zero scale explicit handling");
        theta::Config small;
        small.width=5; small.height=4; small.upper=region; small.lower=region;
        small.calibration=c; small.calibration_id="synthetic_component_test";
        theta::Detector detector(small);
        throws([&]{detector.estimate(cv::Mat());}, "empty image rejected");
        throws([&]{detector.estimate(cv::Mat(5,5,CV_8UC1));}, "wrong image size rejected");
        throws([&]{detector.estimate(cv::Mat(4,5,CV_32FC1));}, "wrong image depth rejected");
        throws([&]{detector.estimate(cv::Mat(4,5,CV_8UC2));}, "wrong image channels rejected");
        small.upper.x_last=5;
        throws([&]{theta::Detector invalid(small);}, "invalid ROI rejected");
        cv::Mat image;
        throws([&]{alignment(image);}, "unconfigured skeleton cannot return zero command");
        std::cout << checks << " component checks passed\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr << "FAILED: " << e.what() << '\n';
        return 1;
    }
}
