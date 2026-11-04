#include "pipeline_internal.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
namespace alignment_detail
{
void createQrRois(const cv::Mat &image, const AlignmentOptions &options,
                  PlateDetectionResult &result)
{
    StageWriter stage(options);
    auto &qrRois = result.qrRois;
    auto &plateCenter = result.center;
    auto &roiVisualization = result.roiVisualization;
    roiVisualization = result.visualization.clone();
    // Detection returns cyclic vertices in arbitrary starting order.
    // At the supported +/-8 degree rotation, sorting rows is unambiguous.
    auto ordered = result.corners;
    std::sort(ordered.begin(), ordered.end(),
              [](const cv::Point2f &a, const cv::Point2f &b) { return a.y < b.y; });
    if (ordered[0].x > ordered[1].x)
    {
        std::swap(ordered[0], ordered[1]);
    }
    if (ordered[2].x < ordered[3].x)
    {
        std::swap(ordered[2], ordered[3]);
    }
    const std::array<cv::Point2f, 4> model{{{-.5f, -.5f}, {.5f, -.5f}, {.5f, .5f}, {-.5f, .5f}}};
    cv::Mat modelToImage = cv::getPerspectiveTransform(model.data(), ordered.data());
    std::vector<cv::Point2f> centerModel{{0, 0}}, centerImage;
    cv::perspectiveTransform(centerModel, centerImage, modelToImage);
    plateCenter = centerImage[0];
    cv::drawMarker(roiVisualization, plateCenter, {0, 255, 255}, cv::MARKER_CROSS, 24, 2);
    const double plateWidth =
        (cv::norm(ordered[1] - ordered[0]) + cv::norm(ordered[2] - ordered[3])) * .5;
    const double plateHeight =
        (cv::norm(ordered[3] - ordered[0]) + cv::norm(ordered[2] - ordered[1])) * .5;
    if (!std::isfinite(options.qrWidthRatio) || !std::isfinite(options.qrHeightRatio) ||
        options.qrWidthRatio <= 0 || options.qrHeightRatio <= 0 || options.qrWidthRatio > 2 ||
        options.qrHeightRatio > 2)
    {
        CV_Error(cv::Error::StsBadArg, "QR ROI ratios must be positive and at most 2");
    }
    cv::Size size(std::max(2, cvRound(plateWidth * options.qrWidthRatio)),
                  std::max(2, cvRound(plateHeight * options.qrHeightRatio)));
    const std::array<cv::Point2f, 4> target{{{0, 0},
                                             {float(size.width - 1), 0},
                                             {float(size.width - 1), float(size.height - 1)},
                                             {0, float(size.height - 1)}}};
    const char *names[] = {"L", "C", "R"};
    cv::Mat montage(size.height, size.width * 3, CV_8UC3, cv::Scalar(255, 255, 255));
    for (int i = 0; i < 3; ++i)
    {
        const float x = options.qrCenterOffsetsX[i], y = options.qrCenterOffsetY;
        if (!std::isfinite(x) || !std::isfinite(y) || std::abs(x) > 2 || std::abs(y) > 2)
        {
            CV_Error(cv::Error::StsBadArg, "Invalid QR ROI center offset");
        }
        float dx = options.qrWidthRatio * .5f, dy = options.qrHeightRatio * .5f;
        std::vector<cv::Point2f> local{
            {x - dx, y - dy}, {x + dx, y - dy}, {x + dx, y + dy}, {x - dx, y + dy}, {x, y}},
            projected;
        cv::perspectiveTransform(local, projected, modelToImage);
        auto &qr = qrRois[i];
        qr.center = projected[4];
        qr.valid = true;
        std::vector<cv::Point> polygon;
        for (int k = 0; k < 4; ++k)
        {
            qr.corners[k] = projected[k];
            polygon.emplace_back(cvRound(projected[k].x), cvRound(projected[k].y));
            if (!std::isfinite(projected[k].x) || !std::isfinite(projected[k].y) ||
                projected[k].x < 0 || projected[k].y < 0 || projected[k].x > image.cols - 1 ||
                projected[k].y > image.rows - 1)
            {
                qr.valid = false;
            }
        }
        if (!qr.valid)
        {
            std::cerr << "QR ROI " << names[i] << " extends outside the image; crop skipped\n";
            continue;
        }
        cv::Mat transform = cv::getPerspectiveTransform(qr.corners.data(), target.data());
        cv::warpPerspective(image, qr.image, transform, size, cv::INTER_LINEAR);
        stage.save(std::string("_05_qr_") + names[i], qr.image);
        qr.image.copyTo(montage(cv::Rect(i * size.width, 0, size.width, size.height)));
        cv::polylines(roiVisualization, std::vector<std::vector<cv::Point>>{polygon}, true,
                      {255, 100, 0}, 3, cv::LINE_AA);
        cv::putText(roiVisualization, names[i], polygon[0], cv::FONT_HERSHEY_SIMPLEX, .8,
                    {255, 100, 0}, 2);
    }
    stage("_04_qr_rois", "4 - Plate center and QR ROIs", roiVisualization);
    stage("_05_qr_crops", "5 - QR crops: L / C / R", montage);
}
} // namespace alignment_detail
