#include "qr_pose.hpp"
#include <opencv2/imgproc.hpp>

#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int width = 3840;
constexpr int height = 2160;
constexpr int point_count = 42;
constexpr int feature_count = 4 * point_count + 1;
const cv::Point coarse_point(1907, 1130);

struct Model {
    cv::Mat coarse;
    cv::Mat coefficients;
    cv::Mat zero_shifts;
    std::vector<cv::Point> points;
    std::vector<cv::Mat> patches;
};

Model load_model()
{
    const char *configured = std::getenv("QR_POSE_MODEL");
    const std::string path = configured && *configured ? configured : QR_POSE_DEFAULT_MODEL;
    cv::FileStorage file(path, cv::FileStorage::READ);
    if (!file.isOpened())
        throw std::runtime_error("Cannot open QR pose calibration: " + path);

    Model model;
    cv::Mat points;
    file["coarse"] >> model.coarse;
    file["points"] >> points;
    file["coefficients"] >> model.coefficients;
    file["zero_shifts"] >> model.zero_shifts;
    if (static_cast<int>(file["version"]) != 1 ||
        model.coarse.size() != cv::Size(151, 151) || model.coarse.type() != CV_8U ||
        points.size() != cv::Size(2, point_count) || points.type() != CV_32S ||
        model.coefficients.size() != cv::Size(2, feature_count) ||
        model.coefficients.type() != CV_64F || !cv::checkRange(model.coefficients) ||
        model.zero_shifts.size() != cv::Size(2, point_count) ||
        model.zero_shifts.type() != CV_64F || !cv::checkRange(model.zero_shifts))
        throw std::runtime_error("Invalid QR pose calibration: " + path);

    const cv::FileNode patches = file["patches"];
    if (!patches.isSeq() || patches.size() != point_count)
        throw std::runtime_error("Invalid QR patch list: " + path);
    int i = 0;
    for (const auto &node : patches) {
        cv::Mat patch;
        node >> patch;
        const cv::Point point(points.at<int>(i, 0), points.at<int>(i, 1));
        if (patch.size() != cv::Size(35, 35) || patch.type() != CV_8U ||
            point.x < 200 || point.x >= width - 200 ||
            point.y < 60 || point.y >= height - 60)
            throw std::runtime_error("Invalid QR patch: " + path);
        model.points.push_back(point);
        model.patches.push_back(patch);
        ++i;
    }
    return model;
}

double peak_offset(double a, double center, double b)
{
    const double curvature = a + b - 2 * center;
    if (curvature >= -1e-9)
        throw std::runtime_error("QR patch has an ambiguous peak");
    return 0.5 * (a - b) / curvature;
}

cv::Point2d displacement(const cv::Mat &gray, const cv::Mat &patch, cv::Point point,
                         int search_x, int search_y)
{
    const int radius = patch.cols / 2;
    const cv::Rect window(point.x - radius - search_x, point.y - radius - search_y,
                          patch.cols + 2 * search_x, patch.rows + 2 * search_y);
    cv::Mat response;
    cv::matchTemplate(gray(window), patch, response, cv::TM_CCOEFF_NORMED);
    cv::Point p;
    double score;
    cv::minMaxLoc(response, nullptr, &score, nullptr, &p);
    if (!std::isfinite(score) || score < 0.90 ||
        p.x <= 0 || p.x >= response.cols - 1 || p.y <= 0 || p.y >= response.rows - 1)
        throw std::runtime_error("QR patch was not found reliably");

    const float center = response.at<float>(p);
    return {
        p.x - search_x + peak_offset(response.at<float>(p.y, p.x - 1), center,
                                    response.at<float>(p.y, p.x + 1)),
        p.y - search_y + peak_offset(response.at<float>(p.y - 1, p.x), center,
                                    response.at<float>(p.y + 1, p.x))
    };
}
} // namespace

qr_pose::Pose qr_pose::estimate(const cv::Mat &src)
{
    if (src.size() != cv::Size(width, height) ||
        (src.type() != CV_8UC1 && src.type() != CV_8UC3))
        throw std::invalid_argument("QR pose expects a rectified 3840x2160 gray or BGR image");

    static const Model model = load_model();
    cv::Mat gray;
    if (src.channels() == 3)
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    else
        gray = src;

    const auto coarse = displacement(gray, model.coarse, coarse_point, 140, 22);
    const cv::Point origin(static_cast<int>(std::nearbyint(coarse.x)),
                           static_cast<int>(std::nearbyint(coarse.y)));
    std::vector<cv::Point2d> shifts;
    cv::Point2d mean;
    for (int i = 0; i < point_count; ++i) {
        const auto local = displacement(gray, model.patches[i], model.points[i] + origin, 9, 7);
        shifts.emplace_back(local.x + origin.x - model.zero_shifts.at<double>(i, 0),
                            local.y + origin.y - model.zero_shifts.at<double>(i, 1));
        mean += shifts.back();
    }
    mean *= 1.0 / point_count;

    // Relative QR deformation provides a second signal beyond common horizontal travel.
    const double travel = mean.x / 110.0;
    cv::Mat input(1, feature_count, CV_64F);
    auto *value = input.ptr<double>();
    value[2 * point_count] = 10.0 * travel * travel;
    for (int i = 0; i < point_count; ++i) {
        value[2 * i] = shifts[i].x;
        value[2 * i + 1] = shifts[i].y;
        value[2 * point_count + 1 + 2 * i] = travel * (shifts[i].x - mean.x);
        value[2 * point_count + 2 + 2 * i] = travel * (shifts[i].y - mean.y);
    }
    const cv::Mat pose = input * model.coefficients;
    return {pose.at<double>(0, 0), pose.at<double>(0, 1)};
}
