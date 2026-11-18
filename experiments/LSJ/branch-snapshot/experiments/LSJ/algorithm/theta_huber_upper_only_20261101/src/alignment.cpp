#include "alignment.hpp" // original skeleton signature, from baseline include directory
#include "upper_theta.hpp"
#include <memory>
#include <mutex>
#include <stdexcept>

namespace {
std::mutex configuration_mutex;
std::shared_ptr<const theta::UpperDetector> detector;
}
void configureThetaAlignment(const std::string& path) {
    auto prepared=std::make_shared<theta::UpperDetector>(theta::loadUpperConfig(path));
    std::lock_guard<std::mutex> guard(configuration_mutex);
    detector=std::move(prepared);
}
Alignment_t alignment(cv::Mat& image) {
    std::shared_ptr<const theta::UpperDetector> current;
    { std::lock_guard<std::mutex> guard(configuration_mutex); current=detector; }
    if(!current) throw std::runtime_error("configureThetaAlignment must be called first");
    const auto result=current->estimate(image);
    return Alignment_t{0.0,result.theta_correction_deg};
}
