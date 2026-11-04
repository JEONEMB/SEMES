#include "final_pin.hpp"
#include "observer_internal.hpp"
namespace final_pin {
struct Estimator::Impl {
    alignment_detail::Observer qr;
    Observer pin;
    Regressor model;
    std::vector<double> block_x;
    Impl(const std::string& assets,const std::string& file):
        qr(std::filesystem::u8path(assets)),
        pin(alignment_detail::read_gray(std::filesystem::u8path(assets)/"center.bmp")) {
        model.load(file);
        if(model.feature_count()!=14&&model.feature_count()!=18)throw std::invalid_argument("Expected QR-pin 14 or QR-pin-bars 18 feature model");
        block_x=qr.block_centers();
    }
};
Estimator::Estimator(const std::string& assets,const std::string& file):impl_(new Impl(assets,file)){}
Estimator::~Estimator()=default;
cv::Vec2d Estimator::predict(const cv::Mat& gray) const {
    auto qr=impl_->qr.measure(gray);auto pin=impl_->pin.measure(gray);
    return impl_->model.predict(impl_->model.feature_count()==18?
        with_bars_features(qr.values,pin.displacement,impl_->block_x):
        relative_features(qr.values,pin.displacement));
}
}
