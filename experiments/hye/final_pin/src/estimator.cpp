#include "final_pin.hpp"
#include "observer_internal.hpp"
namespace final_pin {
struct Estimator::Impl {
    alignment_detail::Observer qr;
    Observer pin;
    Regressor model;
    Impl(const std::string& assets,const std::string& file):
        qr(std::filesystem::u8path(assets)),
        pin(alignment_detail::read_gray(std::filesystem::u8path(assets)/"center.bmp")) {
        model.load(file);
    }
};
Estimator::Estimator(const std::string& assets,const std::string& file):impl_(new Impl(assets,file)){}
Estimator::~Estimator()=default;
cv::Vec2d Estimator::predict(const cv::Mat& gray) const {
    auto qr=impl_->qr.measure(gray);auto pin=impl_->pin.measure(gray);
    return impl_->model.predict(relative_features(qr.values,pin.displacement));
}
}
