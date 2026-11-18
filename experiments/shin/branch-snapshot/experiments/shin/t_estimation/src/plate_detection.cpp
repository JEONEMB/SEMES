#include "pipeline_internal.hpp"
#include "pipeline_settings.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
namespace alignment_detail
{
namespace
{
struct Line
{
    cv::Point2f p, v;
    double angle, length;
};
bool intersect(const Line &a, const Line &b, cv::Point2f &p)
{
    double d = a.v.cross(b.v);
    if (std::abs(d) < 1e-6)
    {
        return false;
    }
    p = a.p + a.v * float((b.p - a.p).cross(b.v) / d);
    return true;
}
double support(const cv::Mat &distance, cv::Point2f a, cv::Point2f b)
{
    int hit = 0, total = 0;
    // Rounded corners are intentionally omitted from the straight-edge score.
    for (int i = 8; i <= 92; ++i)
    {
        cv::Point2f p = a + (b - a) * (i / 100.f);
        int x = cvRound(p.x), y = cvRound(p.y);
        if (x < 0 || y < 0 || x >= distance.cols || y >= distance.rows)
        {
            return 0;
        }
        hit += distance.at<float>(y, x) <= 3.f;
        ++total;
    }
    return double(hit) / total;
}
std::vector<Line> mergeLines(std::vector<Line> input, double tolerance, double gap)
{
    std::sort(input.begin(), input.end(),
              [](const Line &a, const Line &b) { return a.length > b.length; });
    std::vector<Line> output;
    for (const auto &segment : input)
    {
        bool merged = false;
        for (auto &line : output)
        {
            if (std::abs(line.v.cross(segment.v)) > std::sin(CV_PI / 180))
            {
                continue;
            }
            cv::Point2f a = segment.p, b = segment.p + segment.v * float(segment.length);
            if (std::max(std::abs((a - line.p).cross(line.v)),
                         std::abs((b - line.p).cross(line.v))) > tolerance)
            {
                continue;
            }
            double ta = (a - line.p).dot(line.v), tb = (b - line.p).dot(line.v);
            if (ta > tb)
            {
                std::swap(ta, tb);
            }
            if (ta > line.length + gap || tb < -gap)
            {
                continue;
            }
            double low = std::min(0., ta), high = std::max(line.length, tb);
            line.p += line.v * float(low);
            line.length = high - low;
            merged = true;
            break;
        }
        if (!merged)
        {
            output.push_back(segment);
        }
    }
    return output;
}
} // namespace
PlateGeometry detectPlate(const cv::Mat &image, const AlignmentOptions &options)
{
    StageWriter stage(options);
    // One broad search ROI, not three QR-specific crops. Scale with input resolution.
    double sx = image.cols / 3840., sy = image.rows / 2160.;
    cv::Rect roi(cvRound(1200 * sx), cvRound(800 * sy), cvRound(1400 * sx), cvRound(550 * sy));
    roi &= cv::Rect(0, 0, image.cols, image.rows);
    if (roi.empty())
    {
        CV_Error(cv::Error::StsBadArg, "Search ROI is empty");
    }
    cv::Mat crop = image(roi), gray, blurred, edges;
    cv::cvtColor(crop, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 1.2);
    cv::Canny(blurred, edges, 40, 120);
    stage("_01_canny", "1 - Canny edges", edges);

    std::vector<cv::Vec4i> segments;
    cv::HoughLinesP(edges, segments, 1, CV_PI / 720, 30, 50 * sx, 10 * sx);
    cv::Mat hough = crop.clone();
    std::vector<Line> horizontal, vertical;
    for (auto s : segments)
    {
        cv::line(hough, {s[0], s[1]}, {s[2], s[3]}, cv::Scalar(0, 200, 0), 1, cv::LINE_AA);
        cv::Point2f p{float(s[0]), float(s[1])}, v{float(s[2] - s[0]), float(s[3] - s[1])};
        double length = cv::norm(v);
        v *= float(1. / length);
        double angle = std::atan2(v.y, v.x) * 180 / CV_PI;
        while (angle < 0)
        {
            angle += 180;
        }
        Line l{p, v, angle, length};
        // Keep fragmented plate borders; rectangle geometry and edge support still validate
        // candidates.
        if (std::min(angle, 180 - angle) < 8 && length > 50 * sx)
        {
            horizontal.push_back(l);
        }
        if (std::abs(angle - 90) < 8 && length > 50 * sy)
        {
            vertical.push_back(l);
        }
    }
    stage("_02_hough", "2 - Hough line segments", hough);
    auto trim = [](std::vector<Line> &lines)
    {
        std::sort(lines.begin(), lines.end(),
                  [](const Line &a, const Line &b) { return a.length > b.length; });
        if (lines.size() > 40)
        {
            lines.resize(40);
        }
    };
    horizontal =
        mergeLines(horizontal, settings::MergeDistance * sx, settings::HorizontalMergeGap * sx);
    vertical = mergeLines(vertical, settings::MergeDistance * sx, settings::VerticalMergeGap * sy);
    horizontal.erase(std::remove_if(horizontal.begin(), horizontal.end(), [&](const Line &l)
                                    { return l.length < settings::MinimumHorizontalLength * sx; }),
                     horizontal.end());
    trim(horizontal);
    trim(vertical);
    cv::Mat merged = crop.clone();
    for (const auto &list : {horizontal, vertical})
    {
        for (const auto &line : list)
        {
            cv::line(merged, line.p, line.p + line.v * float(line.length), {0, 180, 255}, 2,
                     cv::LINE_AA);
        }
    }
    stage("_02_merged", "2b - Merged collinear segments", merged);

    std::cout << "Horizontal/vertical candidates: " << horizontal.size() << "/" << vertical.size()
              << "\n";
    cv::Mat inverse, distance;
    cv::bitwise_not(edges, inverse);
    cv::distanceTransform(inverse, distance, cv::DIST_L2, 3);
    double best = -1;
    std::array<cv::Point2f, 4> corners{};
    // Preserve existing successful fits. Use shorter segments only if the strict pass fails.
    auto searchRectangle = [&](const std::vector<Line> &candidates)
    {
        for (size_t i = 0; i < candidates.size(); ++i)
        {
            for (size_t j = i + 1; j < candidates.size(); ++j)
            {
                for (size_t k = 0; k < vertical.size(); ++k)
                {
                    for (size_t l = k + 1; l < vertical.size(); ++l)
                    {
                        std::array<cv::Point2f, 4> q;
                        if (!intersect(candidates[i], vertical[k], q[0]) ||
                            !intersect(candidates[i], vertical[l], q[1]) ||
                            !intersect(candidates[j], vertical[l], q[2]) ||
                            !intersect(candidates[j], vertical[k], q[3]))
                        {
                            continue;
                        }
                        double w1 = cv::norm(q[1] - q[0]) / sx, w2 = cv::norm(q[2] - q[3]) / sx;
                        double h1 = cv::norm(q[3] - q[0]) / sy, h2 = cv::norm(q[2] - q[1]) / sy;
                        // Size prior for the black plate surrounding all three QR labels.
                        if (std::min(w1, w2) < 850 || std::max(w1, w2) > 1100 ||
                            std::min(h1, h2) < 180 || std::max(h1, h2) > 300)
                        {
                            continue;
                        }
                        if (std::abs(w1 - w2) > 40 || std::abs(h1 - h2) > 40)
                        {
                            continue;
                        }
                        double score = 0, minSupport = 1;
                        for (int t = 0; t < 4; ++t)
                        {
                            double s = support(distance, q[t], q[(t + 1) % 4]);
                            score += s;
                            minSupport = std::min(minSupport, s);
                        }
                        if (minSupport < 0.45)
                        {
                            continue;
                        }
                        score /= 4;
                        // Prefer the dark plate boundary over nearby metal housing edges.
                        cv::Point2f center = (q[0] + q[1] + q[2] + q[3]) * 0.25f;
                        double contrast = 0;
                        int samples = 0;
                        for (int t = 0; t < 4; ++t)
                        {
                            for (int z = 20; z <= 80; z += 5)
                            {
                                cv::Point2f p = q[t] + (q[(t + 1) % 4] - q[t]) * (z / 100.f);
                                cv::Point2f n = center - p;
                                n *= float(10 * sx / cv::norm(n));
                                cv::Point inside(cvRound(p.x + n.x), cvRound(p.y + n.y));
                                cv::Point outside(cvRound(p.x - n.x), cvRound(p.y - n.y));
                                if (cv::Rect(0, 0, gray.cols, gray.rows).contains(inside) &&
                                    cv::Rect(0, 0, gray.cols, gray.rows).contains(outside))
                                {
                                    contrast +=
                                        int(gray.at<uchar>(outside)) - int(gray.at<uchar>(inside));
                                    ++samples;
                                }
                            }
                        }
                        if (samples)
                        {
                            score += 0.4 * contrast / (255 * samples);
                        }
                        if (score > best)
                        {
                            best = score;
                            corners = q;
                        }
                    }
                }
            }
        }
    };
    searchRectangle(horizontal);
    PlateGeometry result;
    result.found = best >= 0.65;
    result.score = best;
    result.searchRoi = roi;
    result.edges = edges;
    result.hough = hough;
    if (result.found)
    {
        for (int i = 0; i < 4; ++i)
        {
            result.corners[i] = corners[i] + cv::Point2f(float(roi.x), float(roi.y));
        }
    }
    return result;
}
} // namespace alignment_detail
