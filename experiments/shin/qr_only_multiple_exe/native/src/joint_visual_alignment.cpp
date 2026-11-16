#include "qr_pose.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace qr {
JointVisualModel::JointVisualModel(const std::string& path) {
    const auto model=open_model(path);
    if (std::string(model["kind"])!="qr_affine_joint")
        throw std::invalid_argument("Expected a joint visual QR calibration");
    coefficients_=matrix(model,"coefficients");
    centers_=centers(matrix(model,"centers"));
    if (coefficients_.size()!=cv::Size(2,18) || !cv::checkRange(coefficients_))
        throw std::invalid_argument("Invalid joint visual QR coefficients");
    if (!model["intercept"].empty()) {
        const auto offset=matrix(model,"intercept");
        if (offset.size()!=cv::Size(2,1) || !cv::checkRange(offset))
            throw std::invalid_argument("Invalid joint visual QR intercept");
        intercept_={offset.at<double>(0,0),offset.at<double>(0,1)};
    }
}

Pose JointVisualModel::predict(const AffineFeatures& observation) const {
    for (double value : observation.val)
        if (!std::isfinite(value)) throw std::invalid_argument("Invalid QR affine observation");
    const cv::Mat pose=cv::Mat(observation).t()*coefficients_;
    return Pose(pose.at<double>(0),pose.at<double>(1))+intercept_;
}

JointVisualStep joint_visual_step(const JointVisualModel& model,const AffineFeatures& observation) {
    const auto pose=model.predict(observation);
    if (std::abs(pose[0])<=.008 && std::abs(pose[1])<=.002) return {pose,{0,0},true};
    const auto command=[](double error,double cap){
        return static_cast<int>(std::nearbyint(1000.*std::max(-cap,std::min(cap,-.8*error))));
    };
    return {pose,{command(pose[0],.45),command(pose[1],.30)},false};
}

VisionResult run_joint_visual_alignment(const JointVisualModel& model,
    const std::function<AffineFeatures()>& capture, const std::function<bool(Position)>& move_relative,
    int max_moves, const VisionEvent& event) {
    if (!capture || !move_relative || max_moves<0 || max_moves>20)
        throw std::invalid_argument("Invalid vision callbacks or motion budget");
    for (int moves=0;;++moves) {
        const auto step=joint_visual_step(model,capture());
        if (event) event("vision_estimate",step.estimate);
        if (step.center) return {"qr_center_candidate",moves};
        if (moves==max_moves) return {"move_budget",moves};
        if (step.delta==Position(0,0)) return {"below_resolution",moves};
        if (!move_relative(step.delta)) return {"move_rejected",moves};
        if (event) event("move_relative",{step.delta[0]/1000.,step.delta[1]/1000.});
    }
}
}
