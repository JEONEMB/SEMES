#include "pose_model.hpp"
#include <opencv2/core.hpp>
#include <algorithm>
#include <cmath>

namespace tools {

const char* const kValueNames[kValueCount] = {"qr", "screw1", "screw2", "screw3", "screw4"};
const char* const kScrewSetNames[16] = {"1234", "234", "134", "124", "123", "12", "13", "14", "23", "24", "34", "1", "2", "3", "4", "none"};
const char* const kCheckSetNames[11] = {"1234", "234", "134", "124", "123", "12", "13", "14", "23", "24", "34"};

namespace {

const int kHuberRounds = 6;
const double kRidgePerPhoto = 1e-7;

}   // namespace

bool passes(double error_y_mm, double error_t_deg) {
    return std::abs(error_y_mm) <= kToleranceYmm && std::abs(error_t_deg) <= kToleranceTdeg;
}

std::vector<int> used_values_of(const std::string& screws) {
    std::vector<int> used = {0};   // QR 은 언제나 쓴다
    if (screws == "none") return used;
    for (char c : screws) used.push_back(c - '0');
    return used;
}

void PoseModel::fit(const std::vector<int>& used_values, const std::vector<Values>& values, const std::vector<double>& truth_y, const std::vector<double>& truth_t) {
    used = used_values;
    const int n = int(values.size()), p = int(used.size());
    mean.assign(p, 0);
    scale.assign(p, 1);
    for (int j = 0; j < p; ++j) {
        double sum = 0, square = 0;
        for (const Values& v : values) sum += v[used[j]];
        mean[j] = sum / n;
        for (const Values& v : values) square += std::pow(v[used[j]] - mean[j], 2);
        scale[j] = std::max(std::sqrt(square / n), 1e-6);
    }

    cv::Mat design(n, p + 1, CV_64F), truth(n, 2, CV_64F);
    for (int i = 0; i < n; ++i) {
        design.at<double>(i, 0) = 1;
        for (int j = 0; j < p; ++j) design.at<double>(i, j + 1) = (values[i][used[j]] - mean[j]) / scale[j];
        truth.at<double>(i, 0) = truth_y[i];
        truth.at<double>(i, 1) = truth_t[i];
    }
    cv::Mat ridge = cv::Mat::eye(p + 1, p + 1, CV_64F) * (kRidgePerPhoto * n);
    ridge.at<double>(0, 0) = 0;   // 절편에는 ridge 를 걸지 않는다

    std::vector<double> weight(n, 1.0);
    cv::Mat coefficients;
    for (int round = 0; round < kHuberRounds; ++round) {
        cv::Mat weighted = design.clone();
        for (int i = 0; i < n; ++i) weighted.row(i) *= weight[i];
        cv::solve(weighted.t() * design + ridge, weighted.t() * truth, coefficients, cv::DECOMP_SVD);
        const cv::Mat predicted = design * coefficients;
        double weight_sum = 0;
        for (int i = 0; i < n; ++i) {
            const double badness = std::max(std::abs(predicted.at<double>(i, 0) - truth_y[i]) / kToleranceYmm,
                                            std::abs(predicted.at<double>(i, 1) - truth_t[i]) / kToleranceTdeg);
            weight[i] = std::min(1.0, 1.0 / std::max(badness, 1e-12));
            weight_sum += weight[i];
        }
        for (double& w : weight) w *= n / weight_sum;
    }

    intercept_y = coefficients.at<double>(0, 0);
    intercept_t = coefficients.at<double>(0, 1);
    weight_y.assign(p, 0);
    weight_t.assign(p, 0);
    for (int j = 0; j < p; ++j) {
        weight_y[j] = coefficients.at<double>(j + 1, 0);
        weight_t[j] = coefficients.at<double>(j + 1, 1);
    }
}

void PoseModel::predict(const Values& values, double& y_mm, double& t_deg) const {
    y_mm = intercept_y;
    t_deg = intercept_t;
    for (size_t j = 0; j < used.size(); ++j) {
        const double normalized = (values[used[j]] - mean[j]) / scale[j];
        y_mm += weight_y[j] * normalized;
        t_deg += weight_t[j] * normalized;
    }
}

}   // namespace tools
