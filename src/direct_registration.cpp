#include "direct_registration.hpp"
#include <map>

DirectRegistrationResult registerQrPixels(const cv::Mat &reference, const cv::Mat &current,
                                         const PlateDetectionResult &rd,
                                         const PlateDetectionResult &cd, int motion)
{
    DirectRegistrationResult result;
    if (reference.empty() || current.empty() || reference.size() != current.size() ||
        reference.type() != CV_8UC3 || current.type() != CV_8UC3 || !rd.found || !cd.found ||
        (motion != cv::MOTION_AFFINE && motion != cv::MOTION_HOMOGRAPHY))
    {
        result.status = "invalid_input";
        return result;
    }
    std::vector<cv::Point2f> a, b, bounds;
    cv::Mat fullMask = cv::Mat::zeros(current.size(), CV_8U);
    int slots = 0;
    for (const auto &rq : rd.qrRois)
    {
        if (!rq.featuresValid) continue;
        for (const auto &cq : cd.qrRois)
        {
            if (!cq.featuresValid || rq.payload != cq.payload) continue;
            std::map<int, cv::Point2f> found;
            for (const auto &f : cq.features) if (f.accepted) found[f.id] = f.refined;
            int count = 0;
            for (const auto &f : rq.features)
            {
                auto it = found.find(f.id);
                if (!f.accepted || it == found.end()) continue;
                a.push_back(f.refined);
                b.push_back(it->second);
                ++count;
            }
            if (count < 6)
            {
                result.status = "insufficient_qr_correspondence";
                return result;
            }
            ++slots;
            bounds.insert(bounds.end(), rq.corners.begin(), rq.corners.end());
            bounds.insert(bounds.end(), cq.corners.begin(), cq.corners.end());
            std::vector<cv::Point> poly;
            for (const auto &p : cq.corners) poly.emplace_back(cvRound(p.x), cvRound(p.y));
            cv::fillConvexPoly(fullMask, poly, cv::Scalar(255));
        }
    }
    if (slots != 3 || a.size() < 24)
    {
        result.status = "three_qr_required";
        return result;
    }
    cv::Rect roi = cv::boundingRect(bounds);
    roi.x -= 12; roi.y -= 12; roi.width += 24; roi.height += 24;
    roi &= cv::Rect(0, 0, reference.cols, reference.rows);
    cv::Mat h = cv::findHomography(a, b, cv::RANSAC, 1.0);
    if (h.empty()) { result.status = "initialization_failed"; return result; }
    double error = 0;
    std::vector<cv::Point2f> projected;
    cv::perspectiveTransform(a, projected, h);
    for (size_t i = 0; i < a.size(); ++i) error += cv::norm(projected[i] - b[i]);
    result.initialResidual = error / a.size();
    cv::Mat toFull = (cv::Mat_<double>(3, 3) << 1, 0, roi.x, 0, 1, roi.y, 0, 0, 1);
    cv::Mat local = toFull.inv() * h * toFull;
    local /= local.at<double>(2, 2);
    if (motion == cv::MOTION_AFFINE)
    {
        std::vector<cv::Point2f> la, lb;
        for (size_t i = 0; i < a.size(); ++i)
        {
            la.push_back(a[i] - cv::Point2f(float(roi.x), float(roi.y)));
            lb.push_back(b[i] - cv::Point2f(float(roi.x), float(roi.y)));
        }
        local = cv::estimateAffine2D(la, lb);
        if (local.empty()) { result.status = "affine_initialization_failed"; return result; }
    }
    cv::Mat rg, cg, warp;
    cv::cvtColor(reference(roi), rg, cv::COLOR_BGR2GRAY);
    cv::cvtColor(current(roi), cg, cv::COLOR_BGR2GRAY);
    local.convertTo(warp, CV_32F);
    cv::Mat mask = fullMask(roi).clone();
    cv::erode(mask, mask, cv::Mat(), cv::Point(-1, -1), 3);
    try
    {
        result.correlation = cv::findTransformECC(rg, cg, warp, motion,
            cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 80, 1e-7), mask);
    }
    catch (const cv::Exception &)
    {
        result.status = "ecc_failed";
        return result;
    }
    cv::Mat wh = cv::Mat::eye(3, 3, CV_64F), converted;
    warp.convertTo(converted, CV_64F);
    converted.copyTo(wh(cv::Rect(0, 0, 3, converted.rows)));
    result.referenceToCurrent = toFull * wh * toFull.inv();
    result.referenceToCurrent /= result.referenceToCurrent.at<double>(2, 2);
    if (!cv::checkRange(result.referenceToCurrent) || result.correlation < .95)
    {
        result.status = "low_correlation_or_nonfinite";
        return result;
    }
    cv::perspectiveTransform(a, projected, result.referenceToCurrent);
    double maximum = 0;
    for (size_t i = 0; i < a.size(); ++i) maximum = std::max(maximum, cv::norm(projected[i] - b[i]));
    result.valid = maximum < 5.;
    result.status = result.valid ? "ok" : "warp_disagrees_with_features";
    return result;
}
