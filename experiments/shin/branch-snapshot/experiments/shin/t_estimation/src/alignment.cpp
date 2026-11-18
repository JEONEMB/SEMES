#include "pipeline_internal.hpp"
#include "qrwithbar_lanczos.hpp"
#include <mutex>
#include <memory>
#include <filesystem>
namespace {
std::mutex estimatorMutex;
std::shared_ptr<const qrwithbar::Estimator> estimator;
std::string estimatorAssets;
std::shared_ptr<const qrwithbar::Estimator> getEstimator(const std::string& requested)
{
    std::lock_guard<std::mutex> lock(estimatorMutex);
    if (requested.empty() && estimator) return estimator;
    const auto path = requested.empty() ? qrwithbar::default_assets_directory() : requested;
    if (!estimator || path != estimatorAssets) {
        auto next = std::make_shared<const qrwithbar::Estimator>(path);
        estimator = std::move(next);
        estimatorAssets = path;
    }
    return estimator;
}
}
void initialize_lanczos_alignment(const std::string& path)
{
    (void)getEstimator(path);
}
Alignment_t alignment(cv::Mat &source)
{
    AlignmentOptions options;
    options.visualize = false; // Production estimate must not open debug windows.
    return alignment(source, options, nullptr);
}
Alignment_t alignment(cv::Mat &source, const AlignmentOptions &options,
                      PlateDetectionResult *output)
{
    using namespace alignment_detail;
    if (output)
    {
        *output = PlateDetectionResult{};
    }
    if (source.empty() || source.depth() != CV_8U)
    {
        CV_Error(cv::Error::StsBadArg, "Expected nonempty 8-bit image");
    }
    Alignment_t pose;
    qrwithbar::Estimate estimated{};
    if (options.estimatePose) {
        estimated = getEstimator(options.poseAssetsDirectory)->estimate(source);
        pose.y = estimated.y_mm;
        pose.t = estimated.yaw_deg;
        if (!options.visualize && !output) return pose;
    }
    cv::Mat image;
    if (source.channels() == 1)
    {
        cv::cvtColor(source, image, cv::COLOR_GRAY2BGR);
    }
    else if (source.channels() == 4)
    {
        cv::cvtColor(source, image, cv::COLOR_BGRA2BGR);
    }
    else if (source.channels() == 3)
    {
        image = source;
    }
    else
    {
        CV_Error(cv::Error::StsBadArg, "Expected 1, 3 or 4 channels");
    }
    auto geometry = detectPlate(image, options);
    PlateDetectionResult result;
    result.poseEstimated = options.estimatePose;
    if (options.estimatePose) {
        result.poseFeatures = estimated.features;
        result.poseMethod = "Lanczos-3 QR630 + common horizontal boundaries";
    }
    result.found = geometry.found;
    result.score = geometry.score;
    result.searchRoi = geometry.searchRoi;
    result.corners = geometry.corners;
    result.edges = geometry.edges;
    result.hough = geometry.hough;
    result.visualization = image.clone();
    if (result.found)
    {
        std::vector<cv::Point> polygon;
        for (const auto &p : result.corners)
        {
            polygon.emplace_back(cvRound(p.x), cvRound(p.y));
        }
        cv::polylines(result.visualization, std::vector<std::vector<cv::Point>>{polygon}, true,
                      {0, 0, 255}, 4, cv::LINE_AA);
    }
    else
    {
        cv::putText(result.visualization, "PLATE NOT FOUND", {30, 60}, cv::FONT_HERSHEY_SIMPLEX,
                    1.5, {0, 0, 255}, 3);
    }
    StageWriter stage(options);
    stage("_03_plate", "3 - QR plate boundary", result.visualization);
    if (result.found)
    {
        createQrRois(image, options, result);
        for (int i = 0; i < 3; ++i)
        {
            if (result.qrRois[i].valid)
            {
                analyzeQrFeatures(image, options, result.qrRois[i], i);
            }
        }
    }
    if (output)
    {
        *output = std::move(result);
    }
    return pose;
}
