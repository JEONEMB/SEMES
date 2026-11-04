#include "pipeline_internal.hpp"
#include "pipeline_settings.hpp"
#include "qr_refinement.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>

namespace
{
cv::Point2f project(const cv::Point2f &p, const cv::Mat &h)
{
    double z = h.at<double>(2, 0) * p.x + h.at<double>(2, 1) * p.y + h.at<double>(2, 2);
    return {float((h.at<double>(0, 0) * p.x + h.at<double>(0, 1) * p.y + h.at<double>(0, 2)) / z),
            float((h.at<double>(1, 0) * p.x + h.at<double>(1, 1) * p.y + h.at<double>(1, 2)) / z)};
}
bool inBounds(const cv::Point2f &p, const cv::Mat &image, int margin = 0)
{
    return std::isfinite(p.x) && std::isfinite(p.y) && p.x >= margin && p.y >= margin &&
           p.x < image.cols - margin && p.y < image.rows - margin;
}
} // namespace

static void refineCandidates(QrRoiResult &qr, const cv::Mat &gray, const cv::Mat &quality,
                             const cv::Mat &h, double pitch, int radius, QrRefinementMethod method)
{
    for (auto &f : qr.features)
    {
        if (!inBounds(f.initial, gray, radius + 3))
        {
            continue;
        }
        if (method == QrRefinementMethod::CornerSubPix)
        {
            std::vector<cv::Point2f> point{f.initial};
            cv::cornerSubPix(gray, point, {radius, radius}, {-1, -1},
                             {cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 40, 0.001});
            f.refined = point[0];
            f.refinementConverged = true; // OpenCV does not expose the termination reason.
        }
        else
        {
            subpixel::Fit fit;
            if (method == QrRefinementMethod::ImageFormation)
            {
                auto origin = project(f.model, h);
                auto dx = (project(f.model + cv::Point2f(.01f, 0), h) - origin) * 100;
                auto dy = (project(f.model + cv::Point2f(0, .01f), h) - origin) * 100;
                double a1 = std::atan2(dy.y, dy.x) - subpixel::pi / 2;
                double a2 = std::atan2(dx.y, dx.x) + subpixel::pi / 2;
                fit = subpixel::modelFit(subpixel::patch(gray, f.initial, 3), a1, a2, pitch * .45);
            }
            else
            {
                // Eq. (23) adaptation: subtract the fixed patch mean before fitting.
                fit = subpixel::fourierFit(
                    subpixel::patch(gray, f.initial,
                                    alignment_detail::settings::FourierRefinementRadius),
                    alignment_detail::settings::FourierRefinementAlpha,
                    alignment_detail::settings::FourierRefinementDemean, pitch * .45);
            }
            f.refinementConverged = fit.converged;
            f.refinementBoundaryMask = fit.boundaryMask;
            f.refinementCondition = fit.condition;
            f.refinementResidual = fit.rms;
            f.refined = f.initial + cv::Point2f(float(fit.p.x), float(fit.p.y));
            if (!fit.valid)
            {
                f.shift = cv::norm(f.refined - f.initial);
                continue;
            }
        }
        f.shift = cv::norm(f.refined - f.initial);
        if (!inBounds(f.refined, gray, radius + 3))
        {
            continue;
        }
        f.minEigenvalue = quality.at<float>(cvRound(f.refined.y), cvRound(f.refined.x));
        // Verify the expected four-quadrant polarity on the observed original pixels.
        double blackMean = 0, whiteMean = 0;
        int nb = 0, nw = 0;
        for (int dy = -1; dy <= 1; dy += 2)
        {
            for (int dx = -1; dx <= 1; dx += 2)
            {
                cv::Point2f sample = project(f.model + cv::Point2f(dx * .28f, dy * .28f), h) +
                                     (f.refined - f.initial);
                if (!inBounds(sample, gray, 1))
                {
                    continue;
                }
                cv::Mat patch;
                cv::getRectSubPix(gray, {1, 1}, sample, patch);
                int row = int(f.model.y) + (dy < 0 ? -1 : 0),
                    col = int(f.model.x) + (dx < 0 ? -1 : 0);
                if (qr.idealModules.at<uchar>(row, col) < 128)
                {
                    blackMean += patch.at<uchar>(0);
                    ++nb;
                }
                else
                {
                    whiteMean += patch.at<uchar>(0);
                    ++nw;
                }
            }
        }
        f.contrast = nb == 2 && nw == 2 ? whiteMean / nw - blackMean / nb : 0;
        f.localAccepted = f.shift <= pitch * .45 && f.minEigenvalue > 0.0001 && f.contrast > 25;
    }
}
static int validateGeometry(QrRoiResult &qr, double n, double pitch)
{
    std::vector<cv::Point2f> modelPoints, imagePoints;
    for (const auto &f : qr.features)
    {
        if (f.localAccepted)
        {
            modelPoints.push_back(f.model);
            imagePoints.push_back(f.refined);
        }
    }
    cv::Mat inliers, fit;
    double limit = std::max(.75, pitch * .15);
    if (modelPoints.size() >= 8)
    {
        fit = cv::findHomography(modelPoints, imagePoints, cv::RANSAC, limit, inliers);
    }
    // Leave each point out of its own geometric prediction. Do not score a point using a fit to
    // itself.
    int accepted = 0, localIndex = 0;
    for (auto &f : qr.features)
    {
        if (f.localAccepted)
        {
            int thisIndex = localIndex++;
            if (fit.empty() || !inliers.at<uchar>(thisIndex))
            {
                continue;
            }
            std::vector<cv::Point2f> otherModel, otherImage;
            for (size_t j = 0; j < modelPoints.size(); ++j)
            {
                if (int(j) != thisIndex && inliers.at<uchar>(int(j)))
                {
                    otherModel.push_back(modelPoints[j]);
                    otherImage.push_back(imagePoints[j]);
                }
            }
            if (otherModel.size() < 7)
            {
                continue;
            }
            cv::Mat heldOut = cv::findHomography(otherModel, otherImage, 0);
            if (heldOut.empty())
            {
                continue;
            }
            f.reprojectionError = cv::norm(f.refined - project(f.model, heldOut));
            f.accepted = std::isfinite(f.reprojectionError) && f.reprojectionError <= limit;
            accepted += f.accepted;
        }
    }
    std::vector<cv::Point2f> acceptedModel;
    for (const auto &f : qr.features)
    {
        if (f.accepted)
        {
            acceptedModel.push_back(f.model);
        }
    }
    cv::Rect coverage = acceptedModel.empty() ? cv::Rect() : cv::boundingRect(acceptedModel);
    qr.featuresValid = accepted >= 8 && coverage.width >= n * .4 && coverage.height >= n * .4;
    qr.featureStatus = qr.featuresValid ? "ok" : "insufficient_consistent_features";
    return accepted;
}

void analyzeQrFeatures(const cv::Mat &source, const AlignmentOptions &options, QrRoiResult &qr,
                       int index)
{
    qr.features.clear();
    qr.featuresValid = false;
    qr.modelMatched = false;
    qr.modelMismatch = 1;
    qr.moduleCount = 0;
    const std::string label = std::string(1, "LCR"[index]);
    const char *refinementName =
        options.qrRefinement == QrRefinementMethod::CornerSubPix     ? "cornerSubPix"
        : options.qrRefinement == QrRefinementMethod::ImageFormation ? "imageFormation"
                                                                     : "fourierCentered";
    alignment_detail::StageWriter stage(options, label);
    alignment_detail::DecodeResult decoded;
    qr.decoded = alignment_detail::decodeQr(qr.image, decoded);
    qr.payload = decoded.payload;
    qr.decodeUsedGaussian = true;
    qr.decodeGaussianSigma = alignment_detail::settings::DecodeSigma;
    qr.decodedVisualization = qr.image.clone();
    const auto &decodedCorners = decoded.corners;
    for (size_t i = 0; i < decodedCorners.size(); ++i)
    {
        cv::line(qr.decodedVisualization, decodedCorners[i], decodedCorners[(i + 1) % 4],
                 {0, 255, 0}, 1);
    }
    cv::putText(qr.decodedVisualization, qr.decoded ? qr.payload : "DECODE FAILED", {4, 18},
                cv::FONT_HERSHEY_SIMPLEX, .45, {0, 0, 255}, 1);
    stage("_06_decode_input", "6a - Fixed decode input", decoded.input);
    stage("_06_decode", "6 - QR decode", qr.decodedVisualization);
    if (!qr.decoded)
    {
        qr.featureStatus = "decode_failed";
        return;
    }
    qr.moduleCount = decoded.modules.rows;
    qr.modelMatched = alignment_detail::regenerateQrModel(qr.payload, decoded.modules,
                                                          qr.idealModules, qr.modelMismatch);
    if (!qr.modelMatched)
    {
        qr.featureStatus = "model_mismatch_or_ambiguous";
        return;
    }

    // Crop pixels -> original image pixels. Keep subpixel refinement out of resampled crop.
    std::array<cv::Point2f, 4> cropBounds{{{0, 0},
                                           {float(qr.image.cols - 1), 0},
                                           {float(qr.image.cols - 1), float(qr.image.rows - 1)},
                                           {0, float(qr.image.rows - 1)}}};
    cv::Mat cropToSource = cv::getPerspectiveTransform(cropBounds.data(), qr.corners.data());
    std::vector<cv::Point2f> sourceCorners;
    cv::perspectiveTransform(decodedCorners, sourceCorners, cropToSource);
    float n = float(qr.moduleCount);
    std::array<cv::Point2f, 4> modelBounds{{{0, 0}, {n, 0}, {n, n}, {0, n}}};
    cv::Mat h = cv::getPerspectiveTransform(modelBounds.data(), sourceCorners.data());
    cv::Mat sourceToCrop = cropToSource.inv(), gray, quality;
    cv::cvtColor(source, gray, cv::COLOR_BGR2GRAY);
    cv::cornerMinEigenVal(gray, quality, 3, 3);
    double pitch = 0;
    for (int i = 0; i < 4; ++i)
    {
        pitch += cv::norm(sourceCorners[i] - sourceCorners[(i + 1) % 4]) / (4 * n);
    }
    int radius = std::max(1, int(std::floor(pitch * .30)));
    if (pitch < 4)
    {
        qr.featureStatus = "insufficient_pixels_per_module";
        return;
    }
    cv::Mat ideal;
    cv::resize(qr.idealModules, ideal, {}, 14, 14, cv::INTER_NEAREST);
    cv::cvtColor(ideal, ideal, cv::COLOR_GRAY2BGR);
    qr.candidateVisualization = qr.image.clone();
    for (int row = 1; row < qr.moduleCount; ++row)
    {
        for (int col = 1; col < qr.moduleCount; ++col)
        {
            auto black = [&](int y, int x) { return qr.idealModules.at<uchar>(y, x) < 128; };
            // Only isolated checkerboard-type intersections; do not treat all grid vertices as
            // corners.
            bool a = black(row - 1, col - 1), b = black(row - 1, col), c = black(row, col - 1),
                 d = black(row, col);
            if (!(a == d && b == c && a != b))
            {
                continue;
            }
            QrFeature f;
            f.id = row * (qr.moduleCount + 1) + col;
            f.model = {float(col), float(row)};
            f.initial = project(f.model, h);
            f.refined = f.initial;
            qr.features.push_back(f);
            auto p = project(f.initial, sourceToCrop);
            cv::circle(qr.candidateVisualization, p, 2, {255, 255, 0}, 1, cv::LINE_AA);
            cv::circle(ideal, {col * 14, row * 14}, 3, {0, 0, 255}, 1);
        }
    }
    stage("_07_model", "7 - Verified ideal QR and X-corner candidates", ideal);
    stage("_08_candidates", "8 - Projected candidates (cyan)", qr.candidateVisualization);
    refineCandidates(qr, gray, quality, h, pitch, radius, options.qrRefinement);
    const int accepted = validateGeometry(qr, n, pitch);
    qr.featureVisualization = qr.image.clone();
    cv::resize(qr.featureVisualization, qr.featureVisualization, {}, 3, 3, cv::INTER_NEAREST);
    for (const auto &f : qr.features)
    {
        auto initial = project(f.initial, sourceToCrop) * 3,
             refined = project(f.refined, sourceToCrop) * 3;
        cv::drawMarker(qr.featureVisualization, initial, {255, 255, 0}, cv::MARKER_CROSS, 6, 1);
        cv::line(qr.featureVisualization, initial, refined, {0, 255, 255}, 1, cv::LINE_AA);
        cv::circle(qr.featureVisualization, refined, 3,
                   f.accepted ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
    }
    cv::putText(qr.featureVisualization,
                qr.payload + " accepted " + std::to_string(accepted) + "/" +
                    std::to_string(qr.features.size()),
                {8, 22}, cv::FONT_HERSHEY_SIMPLEX, .55, {255, 0, 0}, 1);
    double meanResidual = 0;
    for (const auto &f : qr.features)
    {
        if (f.accepted)
        {
            meanResidual += f.reprojectionError;
        }
    }
    if (accepted)
    {
        meanResidual /= accepted;
    }
    const int footerY = qr.featureVisualization.rows;
    cv::copyMakeBorder(qr.featureVisualization, qr.featureVisualization, 0, 55, 0, 0,
                       cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));
    cv::putText(qr.featureVisualization,
                "Held-out mean residual: " + std::to_string(meanResidual) + " px",
                {8, footerY + 20}, cv::FONT_HERSHEY_SIMPLEX, .48, {0, 0, 0}, 1);
    cv::putText(qr.featureVisualization,
                "Cyan: initial  Green: accepted  Red: rejected (not a probability)",
                {8, footerY + 43}, cv::FONT_HERSHEY_SIMPLEX, .40, {0, 0, 0}, 1);
    stage("_09_refined", "9 - Subpixel: cyan initial, green accepted, red rejected",
          qr.featureVisualization);
    if (!options.outputPrefix.empty())
    {
        std::ofstream csv(options.outputPrefix + "_10_features_" + label + ".csv");
        if (!csv)
        {
            CV_Error(cv::Error::StsError, "Cannot save feature CSV");
        }
        csv << "id,model_x,model_y,initial_x,initial_y,refined_x,refined_y,shift_px,min_eigenvalue,"
               "contrast,heldout_residual_px,local_accepted,accepted,refinement_converged,"
               "refinement_boundary_mask,refinement_condition,refinement_residual,refinement_"
               "method\n";
        csv.precision(10);
        for (const auto &f : qr.features)
        {
            csv << f.id << ',' << f.model.x << ',' << f.model.y << ',' << f.initial.x << ','
                << f.initial.y << ',' << f.refined.x << ',' << f.refined.y << ',' << f.shift << ','
                << f.minEigenvalue << ',' << f.contrast << ',' << f.reprojectionError << ','
                << f.localAccepted << ',' << f.accepted << ',' << f.refinementConverged << ','
                << f.refinementBoundaryMask << ',' << f.refinementCondition << ','
                << f.refinementResidual << ',' << refinementName << '\n';
        }
    }
    std::cout << "QR " << label << ": " << qr.payload << ", model mismatch=" << qr.modelMismatch
              << ", accepted=" << accepted << "/" << qr.features.size() << "\n";
}
