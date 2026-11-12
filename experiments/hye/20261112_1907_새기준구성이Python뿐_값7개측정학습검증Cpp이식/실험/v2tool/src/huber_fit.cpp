#include "huber_fit.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace v2 {
namespace {
cv::Mat design(const cv::Mat& x, const cv::Mat& mean, const cv::Mat& scale) {
    cv::Mat a(x.rows, x.cols + 1, CV_64F);
    for (int i = 0; i < x.rows; ++i) {
        a.at<double>(i, 0) = 1.0;
        for (int j = 0; j < x.cols; ++j) a.at<double>(i, j + 1) = (x.at<double>(i, j) - mean.at<double>(j)) / scale.at<double>(j);
    }
    return a;
}
}

void LinearModel::fit(const cv::Mat& x, const cv::Mat& targets) {
    if (x.type() != CV_64F || targets.type() != CV_64F || targets.cols != 2 || x.rows != targets.rows || x.rows < 2)
        throw std::invalid_argument("LinearModel::fit: x 는 n×p, targets 는 n×2 CV_64F");
    const int n = x.rows, p = x.cols;
    mean = cv::Mat::zeros(1, p, CV_64F); scale = cv::Mat::zeros(1, p, CV_64F);
    for (int j = 0; j < p; ++j) {
        double m = 0, v = 0;
        for (int i = 0; i < n; ++i) m += x.at<double>(i, j);
        m /= n;
        for (int i = 0; i < n; ++i) v += std::pow(x.at<double>(i, j) - m, 2);
        mean.at<double>(j) = m; scale.at<double>(j) = std::max(std::sqrt(v / n), 1e-6);   // 모표준편차
    }
    const cv::Mat a = design(x, mean, scale);
    cv::Mat penalty = cv::Mat::eye(p + 1, p + 1, CV_64F) * (1e-7 * n);
    penalty.at<double>(0, 0) = 0;
    std::vector<double> w(n, 1.0);
    for (int iteration = 0; iteration < 6; ++iteration) {
        cv::Mat aw = a.clone();
        for (int i = 0; i < n; ++i) aw.row(i) *= w[i];
        const cv::Mat lhs = aw.t() * a + penalty, rhs = aw.t() * targets;
        if (!cv::solve(lhs, rhs, weights, cv::DECOMP_SVD)) throw std::runtime_error("LinearModel::fit: 정규방정식을 풀지 못했다");
        const cv::Mat predicted = a * weights;
        double sum = 0;
        for (int i = 0; i < n; ++i) {
            const double r = std::max(std::abs(predicted.at<double>(i, 0) - targets.at<double>(i, 0)) / 0.045,
                                      std::abs(predicted.at<double>(i, 1) - targets.at<double>(i, 1)) / 0.01);
            w[i] = std::min(1.0, 1.0 / std::max(r, 1e-12)); sum += w[i];
        }
        for (double& v : w) v *= n / sum;
    }
}

cv::Mat LinearModel::predict(const cv::Mat& x) const {
    if (x.type() != CV_64F || x.cols != mean.cols || weights.empty()) throw std::invalid_argument("LinearModel::predict: 입력 크기가 모델과 다르다");
    return design(x, mean, scale) * weights;
}
}
