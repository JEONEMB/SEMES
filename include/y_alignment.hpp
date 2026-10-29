#pragma once

#include "theta_alignment.hpp"
#include <array>
#include <string>

namespace semes {

struct YMarkerConfig {
    std::string name;
    cv::Rect roi;
    double lowPx = 0, highPx = 0;
    // Saved Y capture runs 2, 3, 6, 7. No averaging of markers or runs.
    std::array<double, 4> slopes{};
};

struct YConfig {
    int width = 3840, height = 2160, searchMargin = 110;
    double tickMm = 0.05;
    std::string targetProfile;
    std::array<YMarkerConfig, 3> markers;
    static YConfig load(const std::string& utf8Path);
    void validate() const;
};

struct YResult {
    std::array<double, 3> x{}, intervalLowMm{}, intervalHighMm{}, zncc{}, ecc{};
    std::array<bool, 3> inside{};
    double commonLowMm = 0, commonHighMm = 0, commandMm = 0;
    bool withinTarget = false, commandAvailable = false;
    std::string status;
};

// Signed displacement in the original filename/nominal Y coordinate system.
YResult yFromPositions(const std::array<double, 3>& x, const YConfig& config);

class YEstimator {
public:
    YEstimator(const cv::Mat& reference, const YConfig& config);
    YResult estimate(const cv::Mat& image) const;
private:
    YConfig config_;
    std::array<cv::Mat, 3> templates_;
};

void writeYResult(cv::FileStorage& fs, const YResult& result);

} // namespace semes
