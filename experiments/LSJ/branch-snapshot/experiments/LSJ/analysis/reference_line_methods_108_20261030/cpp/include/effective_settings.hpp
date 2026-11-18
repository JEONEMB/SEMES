#pragma once
#include "common.hpp"
namespace lm {
bool uses_common_fit(const std::string& method_id);
bool uses_segment_selection(const std::string& method_id);
void validate_method_parameters(const std::string& method_id, const json& parameters);
// No image reads or measurements. Describes the runtime branch and rejects unsupported settings.
json effective_settings(const std::string& method_id, const json& parameters, const json& common,
                        const TargetSpec& target, cv::Size frame_size);
}
