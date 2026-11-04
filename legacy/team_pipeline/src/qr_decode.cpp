#include "pipeline_internal.hpp"
#include "pipeline_settings.hpp"
namespace alignment_detail
{
bool decodeQr(const cv::Mat &image, DecodeResult &result)
{
    result = DecodeResult{};
    cv::Mat gray, smoothed;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, smoothed, {settings::DecodeKernel, settings::DecodeKernel},
                     settings::DecodeSigma);
    cv::resize(smoothed, result.input, {}, settings::DecodeScale, settings::DecodeScale,
               cv::INTER_LINEAR);
    cv::copyMakeBorder(result.input, result.input, settings::DecodeBorder, settings::DecodeBorder,
                       settings::DecodeBorder, settings::DecodeBorder, cv::BORDER_CONSTANT,
                       cv::Scalar(255));
    cv::QRCodeDetector detector;
    cv::Mat points;
    result.payload = detector.detectAndDecode(result.input, points, result.modules);
    if (result.payload.empty() || points.total() != 4 || result.modules.empty())
    {
        return false;
    }
    cv::Mat flat = points.reshape(2, 4);
    for (int i = 0; i < 4; ++i)
    {
        result.corners.push_back(
            (flat.at<cv::Point2f>(i) -
             cv::Point2f(float(settings::DecodeBorder), float(settings::DecodeBorder))) /
            float(settings::DecodeScale));
    }
    return true;
}
} // namespace alignment_detail
