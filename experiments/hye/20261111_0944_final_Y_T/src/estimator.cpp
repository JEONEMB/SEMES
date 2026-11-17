#include "final04.hpp"
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <stdexcept>
namespace final04 {
namespace fs=std::filesystem;
Estimator::Estimator(const std::string& assets)
    :qr_(fs::u8path(assets)),pins_(alignment_detail::read_gray(fs::u8path(assets)/"center.bmp")),
     block_x_(qr_.block_centers()) {
    const auto root=fs::u8path(assets);
    cv::FileStorage f((root/"provenance.json").u8string(),cv::FileStorage::READ);
    if(!f.isOpened()) throw std::runtime_error("Missing final04 provenance.json");
    std::string id; f["model_id"]>>id;
    if(id!="final04_huber66_715_irls6_r1e-7" || int(f["features"])!=66)
        throw std::runtime_error("Unexpected final04 model identity");
    for(const std::string name:{"center.bmp","geometry.json","line_protocol.json","huber_linear66.yml"}) {
        std::string expected; f[name+"_sha256"]>>expected;
        if(expected.size()!=64 || alignment_detail::sha256(root/name)!=expected)
            throw std::runtime_error("Asset hash mismatch: "+name);
    }
    for(const auto* key:{"training_min","training_max"})
        if(!f[key].isSeq() || f[key].size()!=2) throw std::runtime_error("Invalid training bounds");
    for(int k=0;k<2;++k) {
        minimum_[k]=double(f["training_min"][k]); maximum_[k]=double(f["training_max"][k]);
        if(!std::isfinite(minimum_[k]) || !std::isfinite(maximum_[k]) || minimum_[k]>maximum_[k])
            throw std::runtime_error("Invalid training interval");
    }
    model_.load((root/"huber_linear66.yml").u8string());
}
Result Estimator::predict(const cv::Mat& input) const {
    if(input.empty() || input.depth()!=CV_8U || input.size()!=cv::Size(3840,2160))
        throw std::invalid_argument("Input must be3840x2160 uint8 image");
    cv::Mat gray;
    if(input.channels()==1) gray=input;
    else if(input.channels()==3) cv::cvtColor(input,gray,cv::COLOR_BGR2GRAY);
    else if(input.channels()==4) cv::cvtColor(input,gray,cv::COLOR_BGRA2GRAY);
    else throw std::invalid_argument("Expected1/3/4 channels");
    const auto qr=qr_.measure(gray); const auto pins=pins_.measure(gray);
    const auto p=model_.predict(features(qr.values,pins.displacement,block_x_));
    Result result; result.pose={p.at<double>(0),p.at<double>(1)};
    cv::hconcat(qr.values,pins.displacement,result.raw110);
    result.qr_transforms=qr.qr_transforms;
    result.pin_quality=pins.quality; result.qr_correlation=qr.qr_correlation;
    result.line_correlation=qr.line_correlation;
    result.outside_training_range=result.pose.y<minimum_[0] || result.pose.y>maximum_[0] ||
        result.pose.t<minimum_[1] || result.pose.t>maximum_[1];
    return result;
}
}
