// 학습: data/new_photos/tables/ → 새 model 폴더 (coefficients/, checks/, reference.json, center.bmp)
//
// 사용: fit_coefficients <tables 폴더> <새 model 폴더> [--data <지금 model 폴더>]
//   새 model 폴더는 비어 있어야 한다 (덮어쓰지 않는다). 평가로 확인한 뒤 사람이 data/model/ 에 복사한다.
//   --data 를 안 주면 "data/model". 기준 사진과 QR, 나사 위치는 그대로 가져온다 (측정값을 잰 기준이라 바꾸면 안 된다).
//
// 하는 일 (식은 common/coefficient_fit.hpp)
//   1. 계수: 나사 조합 16가지마다 Y, T 선형식. 학습 사진 = tables 의 usable 사진 전부
//   2. 점검식: 나사 2개 이상 조합 11가지마다, 값마다 "이 Y, T 라면 이 값이어야 한다" 2차식
//   3. reference.json 의 학습 범위 = 학습 사진 정답의 최소, 최대
#include "common/coefficient_fit.hpp"
#include "common/dataset.hpp"
#include "common/tables.hpp"
#include <opencv2/core.hpp>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace fs = std::filesystem;
using namespace tools;

namespace {

const int kMinTrainingPhotos = 20;

struct Arguments {
    fs::path tables, output;
    fs::path current_model = "data/model";
};

bool read_arguments(int argc, char** argv, Arguments& arguments) {
    if (argc < 3) return false;
    arguments.tables = fs::u8path(argv[1]);
    arguments.output = fs::u8path(argv[2]);
    for (int a = 3; a + 1 < argc; ++a)
        if (std::string(argv[a]) == "--data") arguments.current_model = fs::u8path(argv[a + 1]);
    return true;
}

// 학습에 쓰는 사진의 값과 정답
struct TrainingPhotos {
    std::vector<Values> values;
    std::vector<double> truth_y, truth_t;
    int count() const { return int(values.size()); }
};

TrainingPhotos collect_training_photos(const std::vector<DatasetPhoto>& photos) {
    TrainingPhotos training;
    for (const DatasetPhoto& p : photos) {
        if (!p.usable) continue;
        training.values.push_back(p.values);
        training.truth_y.push_back(p.label_y_mm);
        training.truth_t.push_back(p.label_t_deg);
    }
    return training;
}

// 조합마다 계수와 점검식을 학습해 파일로 쓰고, 학습 사진 통과율을 보고서에 적는다.
void fit_all_sets(const TrainingPhotos& training, const fs::path& output, std::ostream& report) {
    const int n = training.count();
    fs::create_directories(output / "coefficients");
    fs::create_directories(output / "checks");
    report << "학습 사진 " << n << "장\n\n조합      학습 사진 통과율 (학습에 쓴 사진이라 낙관적)\n";
    for (const char* name : kScrewSetNames) {
        const std::string file_name = std::string("screws_") + name + ".tsv";
        const alignment::CoefficientSet set = fit_coefficient_set(name, training.values, training.truth_y, training.truth_t);
        write_coefficient_file(output / "coefficients" / file_name, set, n);
        int passed = 0;
        for (int i = 0; i < n; ++i) {
            const alignment::YT pose = predict(set, training.values[i]);
            passed += passes(pose.y_mm - training.truth_y[i], pose.t_deg - training.truth_t[i]);
        }
        report << "  " << std::left << std::setw(8) << name << format_rate(passed, n) << "\n";
        if (has_agreement_check(name)) write_agreement_file(output / "checks" / file_name, fit_agreement_check(set, training.values), n);
    }
}

// 기준 사진과 위치는 그대로, 학습 범위만 새로 적는다.
bool write_reference_file(const fs::path& from_model, const fs::path& to_model, const double range_y[2], const double range_t[2], std::string& error) {
    cv::FileStorage old_file(from_model.u8string() + "/reference.json", cv::FileStorage::READ);
    if (!old_file.isOpened()) { error = "reference.json 을 읽지 못했습니다: " + from_model.u8string(); return false; }
    const cv::FileNode qr = old_file["qr"], screws = old_file["screws"];
    std::ofstream file(to_model / "reference.json", std::ios::binary);
    file << std::setprecision(6) << "{\n"
         << " \"about\": \"기준 사진(center.bmp)에서 QR 3개와 나사 4개의 위치. 단위 px, 사진 왼쪽 위가 (0, 0).\",\n \"qr\": [";
    for (int k = 0; k < 3; ++k) file << (k ? ", " : "") << "[" << int(qr[k][0]) << ", " << int(qr[k][1]) << ", " << int(qr[k][2]) << ", " << int(qr[k][3]) << "]";
    file << "],\n \"qr_about\": \"[x, y, 너비, 높이]. 왼쪽부터 QR 1, 2, 3\",\n \"screws\": [";
    for (int k = 0; k < 4; ++k) file << (k ? ", " : "") << "[" << int(screws[k][0]) << ", " << int(screws[k][1]) << "]";
    file << "],\n \"screws_about\": \"[중심 x, 중심 y]. 1 왼쪽 위, 2 오른쪽 위, 3 왼쪽 아래, 4 오른쪽 아래\",\n"
         << " \"trained_y_mm\": [" << range_y[0] << ", " << range_y[1] << "],\n"
         << " \"trained_t_deg\": [" << range_t[0] << ", " << range_t[1] << "],\n"
         << " \"trained_about\": \"계수를 학습한 사진의 Y, T 범위. 이 밖의 값은 오차가 커질 수 있어 경고한다\"\n}\n";
    return true;
}

}   // namespace

int main(int argc, char** argv) {
    Arguments arguments;
    if (!read_arguments(argc, argv, arguments)) { std::cerr << "사용: fit_coefficients <tables 폴더> <새 model 폴더> [--data <지금 model 폴더>]\n"; return 2; }
    std::vector<DatasetPhoto> photos;
    std::vector<SessionZero> sessions;
    std::string error;
    if (!load_tables(arguments.tables, photos, sessions, error)) { std::cerr << error << "\n"; return 1; }
    if (!make_empty_folder(arguments.output, error)) { std::cerr << error << "\n"; return 1; }

    const TrainingPhotos training = collect_training_photos(photos);
    if (training.count() < kMinTrainingPhotos) { std::cerr << "학습할 사진이 너무 적습니다 (" << training.count() << "장)\n"; return 1; }

    std::ofstream report(arguments.output / "fit_report.txt", std::ios::binary);
    fit_all_sets(training, arguments.output, report);

    const auto y = std::minmax_element(training.truth_y.begin(), training.truth_y.end());
    const auto t = std::minmax_element(training.truth_t.begin(), training.truth_t.end());
    const double range_y[2] = {*y.first, *y.second}, range_t[2] = {*t.first, *t.second};
    if (!write_reference_file(arguments.current_model, arguments.output, range_y, range_t, error)) { std::cerr << error << "\n"; return 1; }
    fs::copy_file(arguments.current_model / "center.bmp", arguments.output / "center.bmp");
    report << "\n학습 범위: Y " << range_y[0] << " ~ " << range_y[1] << " mm, T " << range_t[0] << " ~ " << range_t[1] << " 도\n"
           << "기준 사진과 QR, 나사 위치는 " << arguments.current_model.u8string() << " 에서 그대로 가져왔다.\n";
    report.close();
    std::ifstream echo(arguments.output / "fit_report.txt", std::ios::binary);
    std::cout << echo.rdbuf();
    return 0;
}
