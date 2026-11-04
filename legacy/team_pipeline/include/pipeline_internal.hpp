#ifndef PIPELINE_INTERNAL_HPP
#define PIPELINE_INTERNAL_HPP
#include "alignment.hpp"
namespace alignment_detail
{
struct PlateGeometry
{
    bool found = false;
    double score = -1;
    cv::Rect searchRoi;
    std::array<cv::Point2f, 4> corners{};
    cv::Mat edges, hough;
};
struct DecodeResult
{
    std::string payload;
    cv::Mat modules, input;
    std::vector<cv::Point2f> corners;
};
class StageWriter
{
    const AlignmentOptions &options;
    std::string label;

  public:
    StageWriter(const AlignmentOptions &value, const std::string &suffix = "")
        : options(value), label(suffix)
    {
    }
    void save(const std::string &suffix, const cv::Mat &image) const;
    void operator()(const std::string &suffix, const std::string &title,
                    const cv::Mat &image) const;
};
PlateGeometry detectPlate(const cv::Mat &image, const AlignmentOptions &options);
void createQrRois(const cv::Mat &image, const AlignmentOptions &options,
                  PlateDetectionResult &result);
bool decodeQr(const cv::Mat &image, DecodeResult &result);
bool regenerateQrModel(const std::string &payload, const cv::Mat &observed, cv::Mat &model,
                       double &mismatch);
} // namespace alignment_detail
#endif
