#include "coefficient_fit.hpp"
#include <opencv2/core.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>

namespace tools {

const char* const kScrewSetNames[16] = {"1234", "234", "134", "124", "123", "12", "13", "14", "23", "24", "34", "1", "2", "3", "4", "none"};
const char* const kCheckSetNames[11] = {"1234", "234", "134", "124", "123", "12", "13", "14", "23", "24", "34"};

namespace {

const int kHuberRounds = 6;
const double kRidgePerPhoto = 1e-7;
const double kRobustScale = 1.4826;          // 가운데값 기반 흔들림을 표준편차 단위로
const double kThresholdQuantile = 0.999;
const double kThresholdMargin = 1.5;
const int kCheckTerms = 6;                   // 점검식의 항: 1, Y, T, Y², YT, T²

double median_of(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

double quantile_of(std::vector<double> v, double q) {
    std::sort(v.begin(), v.end());
    const double position = q * (v.size() - 1);
    const size_t low = size_t(std::floor(position));
    return low + 1 < v.size() ? v[low] + (position - low) * (v[low + 1] - v[low]) : v[low];
}

// 조합이 쓰는 값 번호 (QR 은 언제나, 나사는 이름에 든 번호)
std::vector<int> used_values_of(const std::string& screws) {
    std::vector<int> used = {0};
    if (screws == "none") return used;
    for (char c : screws) used.push_back(c - '0');
    return used;
}

// 쓰는 값마다 평균과 표준편차(모표준편차)를 set 에 넣는다.
void standardize(alignment::CoefficientSet& set, const std::vector<int>& used, const std::vector<Values>& values) {
    const int n = int(values.size());
    for (int v : used) {
        double sum = 0, square = 0;
        for (const Values& one : values) sum += one[v];
        set.mean[v] = sum / n;
        for (const Values& one : values) square += std::pow(one[v] - set.mean[v], 2);
        set.scale[v] = std::max(std::sqrt(square / n), 1e-6);
    }
}

// Huber 가중 최소제곱. 돌려주는 행렬: 줄 = 절편 + 쓰는 값, 칸 = Y, T
cv::Mat solve_huber(const cv::Mat& design, const std::vector<double>& truth_y, const std::vector<double>& truth_t) {
    const int n = design.rows;
    cv::Mat truth(n, 2, CV_64F);
    for (int i = 0; i < n; ++i) {
        truth.at<double>(i, 0) = truth_y[i];
        truth.at<double>(i, 1) = truth_t[i];
    }
    cv::Mat ridge = cv::Mat::eye(design.cols, design.cols, CV_64F) * (kRidgePerPhoto * n);
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
    return coefficients;
}

std::string screws_text(const std::string& screws) {
    if (screws == "none") return "QR 만";
    std::string text = "QR + 나사 ";
    for (size_t i = 0; i < screws.size(); ++i) text += (i ? ", " : "") + std::string(1, screws[i]);
    return text;
}

}   // namespace

bool passes(double error_y_mm, double error_t_deg) {
    return std::abs(error_y_mm) <= kToleranceYmm && std::abs(error_t_deg) <= kToleranceTdeg;
}

bool has_agreement_check(const std::string& screws) {
    return std::find(std::begin(kCheckSetNames), std::end(kCheckSetNames), screws) != std::end(kCheckSetNames);
}

alignment::CoefficientSet fit_coefficient_set(const std::string& screws, const std::vector<Values>& values,
                                              const std::vector<double>& truth_y, const std::vector<double>& truth_t) {
    alignment::CoefficientSet set;
    set.screws = screws;
    const std::vector<int> used = used_values_of(screws);
    for (int v : used)
        if (v >= 1) set.uses_screw[v - 1] = true;
    set.precise = (set.uses_screw[0] || set.uses_screw[2]) && (set.uses_screw[1] || set.uses_screw[3]);   // 왼쪽(1, 3)과 오른쪽(2, 4)이 모두 있음
    standardize(set, used, values);

    const int n = int(values.size()), p = int(used.size());
    cv::Mat design(n, p + 1, CV_64F);
    for (int i = 0; i < n; ++i) {
        design.at<double>(i, 0) = 1;
        for (int j = 0; j < p; ++j) design.at<double>(i, j + 1) = (values[i][used[j]] - set.mean[used[j]]) / set.scale[used[j]];
    }
    const cv::Mat coefficients = solve_huber(design, truth_y, truth_t);
    set.intercept_y = coefficients.at<double>(0, 0);
    set.intercept_t = coefficients.at<double>(0, 1);
    for (int j = 0; j < p; ++j) {
        set.weight_y[used[j]] = coefficients.at<double>(j + 1, 0);
        set.weight_t[used[j]] = coefficients.at<double>(j + 1, 1);
    }
    return set;
}

alignment::YT predict(const alignment::CoefficientSet& set, const Values& values) {
    return alignment::compute_yt(set, values.data());
}

alignment::AgreementCheck fit_agreement_check(const alignment::CoefficientSet& set, const std::vector<Values>& values) {
    const int n = int(values.size());
    cv::Mat design(n, kCheckTerms, CV_64F);
    for (int i = 0; i < n; ++i) {
        const alignment::YT pose = predict(set, values[i]);
        const double terms[kCheckTerms] = {1, pose.y_mm, pose.t_deg, pose.y_mm * pose.y_mm, pose.y_mm * pose.t_deg, pose.t_deg * pose.t_deg};
        for (int q = 0; q < kCheckTerms; ++q) design.at<double>(i, q) = terms[q];
    }
    alignment::AgreementCheck check;
    check.screws = set.screws;
    std::vector<double> score(n, 0);   // 사진마다 점검값 = 가장 크게 어긋난 값의 (어긋난 양 ÷ 흔들림)
    for (int v : used_values_of(set.screws)) {
        cv::Mat target(n, 1, CV_64F), solved;
        for (int i = 0; i < n; ++i) target.at<double>(i) = values[i][v];
        cv::solve(design, target, solved, cv::DECOMP_SVD);
        alignment::AgreementCheck::Row row;
        row.value_index = v;
        for (int q = 0; q < kCheckTerms; ++q) row.c[q] = solved.at<double>(q);
        std::vector<double> residual(n), spread(n);
        for (int i = 0; i < n; ++i) {
            double expected = 0;
            for (int q = 0; q < kCheckTerms; ++q) expected += row.c[q] * design.at<double>(i, q);
            residual[i] = values[i][v] - expected;
        }
        const double middle = median_of(residual);
        for (int i = 0; i < n; ++i) spread[i] = std::abs(residual[i] - middle);
        row.usual_spread = kRobustScale * median_of(spread);
        for (int i = 0; i < n; ++i) score[i] = std::max(score[i], std::abs(residual[i]) / row.usual_spread);
        check.rows.push_back(row);
    }
    check.limit = kThresholdMargin * quantile_of(score, kThresholdQuantile);
    return check;
}

void write_coefficient_file(const std::filesystem::path& path, const alignment::CoefficientSet& set, int photo_count) {
    std::ofstream file(path, std::ios::binary);
    file << std::setprecision(17);
    file << "# Y·T 계산 계수. 쓰는 값: " << screws_text(set.screws) << ". 학습 사진 " << photo_count << "장 (Huber 선형식, tools/fit_coefficients).\n"
         << "# Y = intercept_Y + sum(weight_Y * (값 - mean) / scale), T 도 같은 식\n"
         << "name\tmean\tscale\tweight_Y\tweight_T\n"
         << "intercept\t0\t1\t" << set.intercept_y << '\t' << set.intercept_t << '\n';
    for (int v : used_values_of(set.screws))
        file << kValueNames[v] << '\t' << set.mean[v] << '\t' << set.scale[v] << '\t' << set.weight_y[v] << '\t' << set.weight_t[v] << '\n';
}

void write_agreement_file(const std::filesystem::path& path, const alignment::AgreementCheck& check, int photo_count) {
    std::ofstream file(path, std::ios::binary);
    file << std::setprecision(17);
    file << "# 값 일치 점검식. 값마다 기대값 = c1 + cY*Y + cT*T + cYY*Y^2 + cYT*Y*T + cTT*T^2 (Y, T = 영점을 빼기 전 계산값). 학습 사진 " << photo_count << "장\n"
         << "# 점검값 = max |측정값 - 기대값 - 기준 잔차| / scale. threshold 를 넘으면 값 하나가 다른 값과 맞지 않는다.\n"
         << "threshold\t" << check.limit << '\n'
         << "name\tc1\tcY\tcT\tcYY\tcYT\tcTT\tscale\n";
    for (const alignment::AgreementCheck::Row& row : check.rows) {
        file << kValueNames[row.value_index];
        for (double c : row.c) file << '\t' << c;
        file << '\t' << row.usual_spread << '\n';
    }
}

}   // namespace tools
