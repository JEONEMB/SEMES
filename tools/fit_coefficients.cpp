// 학습: dataset/prepared/ → 새 data 폴더 (coefficients/, checks/, reference.json, center.bmp)
//
// 사용: fit_coefficients <prepared 폴더> <새 data 폴더> [--data <지금 data 폴더>]
//   새 data 폴더는 비어 있어야 한다 (덮어쓰지 않는다). 평가로 확인한 뒤 사람이 data/ 에 복사한다.
//   --data 를 안 주면 "data". 기준 사진과 QR, 나사 위치는 그대로 가져온다 (측정값을 잰 기준이라 바꾸면 안 된다).
//
// 하는 일
//   1. 계수: 나사 조합 16가지마다 Y, T 선형식 (common/pose_model.hpp 의 식). 학습 사진 = prepared 의 usable 사진 전부
//   2. 점검식: 나사 2개 이상 조합 11가지마다, 값마다 "이 Y, T 라면 이 값이어야 한다" 2차식
//        기대값 = c1 + cY·Y + cT·T + cYY·Y² + cYT·Y·T + cTT·T²   (Y, T = 그 조합의 계수로 낸 값)
//        scale  = 1.4826 × (잔차 − 잔차 가운데값)의 절댓값 가운데값   (정상 사진에서 흔들리는 정도)
//        기준값 = 1.5 × (학습 사진 점검값의 99.9% 지점)
//   3. reference.json 의 학습 범위 = 학습 사진 정답의 최소, 최대
#include "common/dataset.hpp"
#include "common/pose_model.hpp"
#include "common/tables.hpp"
#include <opencv2/core.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace fs = std::filesystem;
using namespace tools;

namespace {

const double kRobustScale = 1.4826;          // 가운데값 기반 흔들림을 표준편차 단위로
const double kThresholdQuantile = 0.999;
const double kThresholdMargin = 1.5;

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

std::string screws_text(const std::string& screws) {
    if (screws == "none") return "QR 만";
    std::string text = "QR + 나사 ";
    for (size_t i = 0; i < screws.size(); ++i) text += (i ? ", " : "") + std::string(1, screws[i]);
    return text;
}

void write_coefficients(const fs::path& path, const std::string& screws, const PoseModel& model, int photo_count) {
    std::ofstream file(path, std::ios::binary);
    file << std::setprecision(17);
    file << "# Y·T 계산 계수. 쓰는 값: " << screws_text(screws) << ". 학습 사진 " << photo_count << "장 (Huber 선형식, tools/fit_coefficients).\n"
         << "# Y = intercept_Y + sum(weight_Y * (값 - mean) / scale), T 도 같은 식\n"
         << "name\tmean\tscale\tweight_Y\tweight_T\n"
         << "intercept\t0\t1\t" << model.intercept_y << '\t' << model.intercept_t << '\n';
    for (size_t j = 0; j < model.used.size(); ++j)
        file << kValueNames[model.used[j]] << '\t' << model.mean[j] << '\t' << model.scale[j] << '\t' << model.weight_y[j] << '\t' << model.weight_t[j] << '\n';
}

struct AgreementFit {
    double threshold = 0;
    std::vector<int> value;
    std::vector<std::array<double, 6>> coefficient;
    std::vector<double> scale;
};

std::array<double, 6> pose_terms(double y, double t) { return {1, y, t, y * y, y * t, t * t}; }

AgreementFit fit_agreement(const PoseModel& model, const std::vector<Values>& values) {
    const int n = int(values.size());
    cv::Mat design(n, 6, CV_64F);
    for (int i = 0; i < n; ++i) {
        double y, t;
        model.predict(values[i], y, t);
        const auto terms = pose_terms(y, t);
        for (int q = 0; q < 6; ++q) design.at<double>(i, q) = terms[q];
    }
    AgreementFit fit;
    std::vector<std::vector<double>> residual(n);
    for (int v : model.used) {
        cv::Mat target(n, 1, CV_64F), solved;
        for (int i = 0; i < n; ++i) target.at<double>(i) = values[i][v];
        cv::solve(design, target, solved, cv::DECOMP_SVD);
        std::array<double, 6> c;
        for (int q = 0; q < 6; ++q) c[q] = solved.at<double>(q);
        std::vector<double> r(n);
        for (int i = 0; i < n; ++i) {
            double expected = 0;
            for (int q = 0; q < 6; ++q) expected += c[q] * design.at<double>(i, q);
            r[i] = values[i][v] - expected;
            residual[i].push_back(r[i]);
        }
        const double middle = median_of(r);
        std::vector<double> spread(n);
        for (int i = 0; i < n; ++i) spread[i] = std::abs(r[i] - middle);
        fit.value.push_back(v);
        fit.coefficient.push_back(c);
        fit.scale.push_back(kRobustScale * median_of(spread));
    }
    std::vector<double> score(n, 0);
    for (int i = 0; i < n; ++i)
        for (size_t k = 0; k < fit.value.size(); ++k) score[i] = std::max(score[i], std::abs(residual[i][k]) / fit.scale[k]);
    fit.threshold = kThresholdMargin * quantile_of(score, kThresholdQuantile);
    return fit;
}

void write_agreement(const fs::path& path, const AgreementFit& fit, int photo_count) {
    std::ofstream file(path, std::ios::binary);
    file << std::setprecision(17);
    file << "# 값 일치 점검식. 값마다 기대값 = c1 + cY*Y + cT*T + cYY*Y^2 + cYT*Y*T + cTT*T^2 (Y, T = 영점을 빼기 전 계산값). 학습 사진 " << photo_count << "장\n"
         << "# 점검값 = max |측정값 - 기대값 - 기준 잔차| / scale. threshold 를 넘으면 값 하나가 다른 값과 맞지 않는다.\n"
         << "threshold\t" << fit.threshold << '\n'
         << "name\tc1\tcY\tcT\tcYY\tcYT\tcTT\tscale\n";
    for (size_t k = 0; k < fit.value.size(); ++k) {
        file << kValueNames[fit.value[k]];
        for (double c : fit.coefficient[k]) file << '\t' << c;
        file << '\t' << fit.scale[k] << '\n';
    }
}

// 기준 사진과 위치는 그대로, 학습 범위만 새로 적는다.
bool write_reference(const fs::path& from_data, const fs::path& to_data, double y_min, double y_max, double t_min, double t_max, std::string& error) {
    cv::FileStorage old_file(from_data.u8string() + "/reference.json", cv::FileStorage::READ);
    if (!old_file.isOpened()) { error = "reference.json 을 읽지 못했습니다: " + from_data.u8string(); return false; }
    const cv::FileNode qr = old_file["qr"], screws = old_file["screws"];
    std::ofstream file(to_data / "reference.json", std::ios::binary);
    file << std::setprecision(6) << "{\n"
         << " \"about\": \"기준 사진(center.bmp)에서 QR 3개와 나사 4개의 위치. 단위 px, 사진 왼쪽 위가 (0, 0).\",\n \"qr\": [";
    for (int k = 0; k < 3; ++k) file << (k ? ", " : "") << "[" << int(qr[k][0]) << ", " << int(qr[k][1]) << ", " << int(qr[k][2]) << ", " << int(qr[k][3]) << "]";
    file << "],\n \"qr_about\": \"[x, y, 너비, 높이]. 왼쪽부터 QR 1, 2, 3\",\n \"screws\": [";
    for (int k = 0; k < 4; ++k) file << (k ? ", " : "") << "[" << int(screws[k][0]) << ", " << int(screws[k][1]) << "]";
    file << "],\n \"screws_about\": \"[중심 x, 중심 y]. 1 왼쪽 위, 2 오른쪽 위, 3 왼쪽 아래, 4 오른쪽 아래\",\n"
         << " \"trained_y_mm\": [" << y_min << ", " << y_max << "],\n"
         << " \"trained_t_deg\": [" << t_min << ", " << t_max << "],\n"
         << " \"trained_about\": \"계수를 학습한 사진의 Y, T 범위. 이 밖의 값은 오차가 커질 수 있어 경고한다\"\n}\n";
    return true;
}

}   // namespace

int main(int argc, char** argv) {
    if (argc < 3) { std::cerr << "사용: fit_coefficients <prepared 폴더> <새 data 폴더> [--data <지금 data 폴더>]\n"; return 2; }
    const fs::path prepared = fs::u8path(argv[1]), output = fs::u8path(argv[2]);
    fs::path current_data = "data";
    for (int a = 3; a + 1 < argc; ++a) if (std::string(argv[a]) == "--data") current_data = fs::u8path(argv[a + 1]);

    std::vector<DatasetPhoto> photos;
    std::vector<SessionZero> sessions;
    std::string error;
    if (!load_prepared(prepared, photos, sessions, error)) { std::cerr << error << "\n"; return 1; }
    if (!make_empty_folder(output, error)) { std::cerr << error << "\n"; return 1; }

    std::vector<Values> values;
    std::vector<double> truth_y, truth_t;
    for (const DatasetPhoto& p : photos) {
        if (!p.usable) continue;
        values.push_back(p.values);
        truth_y.push_back(p.label_y_mm);
        truth_t.push_back(p.label_t_deg);
    }
    const int n = int(values.size());
    if (n < 20) { std::cerr << "학습할 사진이 너무 적습니다 (" << n << "장)\n"; return 1; }

    fs::create_directories(output / "coefficients");
    fs::create_directories(output / "checks");
    std::ofstream report(output / "fit_report.txt", std::ios::binary);
    report << "학습 사진 " << n << "장\n\n조합      학습 사진 통과율 (학습에 쓴 사진이라 낙관적)\n";
    for (const char* name : kScrewSetNames) {
        PoseModel model;
        model.fit(used_values_of(name), values, truth_y, truth_t);
        write_coefficients(output / "coefficients" / (std::string("screws_") + name + ".tsv"), name, model, n);
        int passed = 0;
        for (int i = 0; i < n; ++i) { double y, t; model.predict(values[i], y, t); passed += passes(y - truth_y[i], t - truth_t[i]); }
        report << "  " << std::left << std::setw(8) << name << format_rate(passed, n) << "\n";
        if (std::find_if(std::begin(kCheckSetNames), std::end(kCheckSetNames), [&](const char* c) { return std::string(c) == name; }) != std::end(kCheckSetNames))
            write_agreement(output / "checks" / (std::string("screws_") + name + ".tsv"), fit_agreement(model, values), n);
    }

    const double y_min = *std::min_element(truth_y.begin(), truth_y.end()), y_max = *std::max_element(truth_y.begin(), truth_y.end());
    const double t_min = *std::min_element(truth_t.begin(), truth_t.end()), t_max = *std::max_element(truth_t.begin(), truth_t.end());
    if (!write_reference(current_data, output, y_min, y_max, t_min, t_max, error)) { std::cerr << error << "\n"; return 1; }
    fs::copy_file(current_data / "center.bmp", output / "center.bmp");
    report << "\n학습 범위: Y " << y_min << " ~ " << y_max << " mm, T " << t_min << " ~ " << t_max << " 도\n"
           << "기준 사진과 QR, 나사 위치는 " << current_data.u8string() << " 에서 그대로 가져왔다.\n";
    report.close();
    std::ifstream echo(output / "fit_report.txt", std::ios::binary);
    std::cout << echo.rdbuf();
    return 0;
}
