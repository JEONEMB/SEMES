#include "alignment.hpp"
#include "theta_pipeline.hpp"
#include <memory>
#include <mutex>
#include <stdexcept>

namespace {
std::mutex configuration_mutex;
std::shared_ptr<const theta::Detector> detector;
}

void configureThetaAlignment(const std::string& path) {
    auto prepared = std::make_shared<theta::Detector>(theta::loadConfig(path));
    std::lock_guard<std::mutex> guard(configuration_mutex);
    detector = std::move(prepared);
}

Alignment_t alignment(cv::Mat &src) {
    std::shared_ptr<const theta::Detector> current;
    {
        std::lock_guard<std::mutex> guard(configuration_mutex);
        current = detector;
    }
    if (!current) throw std::runtime_error("configureThetaAlignment must be called first");
    const auto result = current->estimate(src);
    return Alignment_t{0.0, result.pose.theta_correction_deg};
}
