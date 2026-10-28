#include "alignment.hpp"
#include <stdexcept>

void initialize_alignment(const std::string& assets,
                          const std::string& control_model,
                          double maximum_control_t)
{
#ifndef SEMES_HAS_CONTROL
    // Reject unsupported setup before changing the joint context.
    if (!control_model.empty()) {
        throw std::logic_error("This build contains joint only. Enable SEMES_BUILD_CONTROL for stage2.");
    }
    (void)maximum_control_t;
#endif
    initialize_joint(assets);
#ifdef SEMES_HAS_CONTROL
    initialize_control(control_model, maximum_control_t);
#endif
}

Alignment_t alignment(cv::Mat &src)
{
    // Compatibility entry: the first stage always uses the joint observation.
    return joint(src);
}
