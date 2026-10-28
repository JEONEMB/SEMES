#include "pipeline_internal.hpp"
namespace alignment_detail
{
void StageWriter::save(const std::string &suffix, const cv::Mat &image) const
{
    if (!options.outputPrefix.empty() &&
        !cv::imwrite(options.outputPrefix + suffix + (label.empty() ? "" : "_" + label) + ".png",
                     image))
    {
        CV_Error(cv::Error::StsError, "Cannot save stage image");
    }
}
void StageWriter::operator()(const std::string &suffix, const std::string &title,
                             const cv::Mat &image) const
{
    save(suffix, image);
    if (!options.visualize)
    {
        return;
    }
    const auto name = title + (label.empty() ? "" : " / " + label);
    cv::namedWindow(name, cv::WINDOW_NORMAL);
    cv::imshow(name, image);
    cv::resizeWindow(name, 1000, 650);
    cv::waitKey(0);
    cv::destroyWindow(name);
}
} // namespace alignment_detail
