#include "t_estimator.hpp"
#include "t_features.hpp"
#include <stdexcept>
#include <cmath>
namespace t_only {
struct Estimator::Impl {
    std::filesystem::path assets;
    Model model;
    cv::Mat reference;
    qrwithbar::detail::Warps warps;
    qrwithbar::detail::Observation observation;
    Impl(const std::string& a,const std::string& m):assets(std::filesystem::u8path(a)),model(load(m)) {
        reference=qrwithbar::detail::read_gray(assets/"reference.png");
        if(reference.cols!=3840||reference.rows!=2160)throw std::runtime_error("Wrong T reference size");
        warps=qrwithbar::detail::align_qrs(reference,assets);
        observation=qrwithbar::detail::analytic_observe(reference,warps);
        if(observation.failures)throw std::runtime_error("T reference initialization failed");
    }
};
Estimator::Estimator(const std::string& a,const std::string& m):impl_(std::make_shared<Impl>(a,m)){}
Estimator::~Estimator()=default;
Estimate Estimator::estimate(const cv::Mat& source)const {
    using namespace qrwithbar::detail;
    if(source.empty()||source.depth()!=CV_8U||source.cols!=3840||source.rows!=2160)throw std::invalid_argument("T estimator expects 3840x2160 uint8 image");
    cv::Mat gray;
    if(source.channels()==1)gray=source;
    else if(source.channels()==3)cv::cvtColor(source,gray,cv::COLOR_BGR2GRAY);
    else if(source.channels()==4)cv::cvtColor(source,gray,cv::COLOR_BGRA2GRAY);
    else throw std::invalid_argument("Expected 1,3 or4 channels");
    auto& c=*impl_;auto warps=align_qrs(gray,c.assets);auto initial=analytic_observe(gray,warps);
    if(initial.failures)throw std::runtime_error("T observation initialization failed");
    auto observed=interpolated_observe(gray,c.reference,warps,c.warps,initial,c.observation,2);
    if(observed.failures)throw std::runtime_error("T Lanczos registration failed");
    for(double x:observed.features)if(!std::isfinite(x))throw std::runtime_error("Nonfinite T feature");
    auto features=enriched_features(observed,c.observation);
    double t=predict(c.model,features);
    return {t,t/tick,features,t<c.model.minimum_t||t>c.model.maximum_t,c.model.requires_y_fixed_at_reference};
}
}
