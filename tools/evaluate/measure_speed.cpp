// 평가: 속도
//
// 사용: measure_speed <model 폴더> <사진 폴더 또는 tables 폴더> <출력 폴더> [--count 사진 수]
//
// 재는 것 (무엇을 쟀는지 분명하게)
//   - initialize 한 번 걸리는 시간
//   - 사진마다: 사진 읽기(파일 → 회색 사진), 측정(measure_alignment), 둘을 더한 것
//   - OpenCV 스레드 1개, 4개 각각. 스레드 설정마다 첫 1장은 준비 시간이라 뺀다
//   - 중앙값, 95% 지점, 최대
#include "alignment.hpp"
#include "common/photo_files.hpp"
#include "common/tables.hpp"
#include <opencv2/core/utility.hpp>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace fs = std::filesystem;
using namespace tools;
using Clock = std::chrono::steady_clock;

namespace {

const int kThreadSettings[] = {1, 4};

double milliseconds_since(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

double quantile_of(std::vector<double> v, double q) {
    std::sort(v.begin(), v.end());
    const double position = q * (v.size() - 1);
    const size_t low = size_t(position);
    return low + 1 < v.size() ? v[low] + (position - low) * (v[low + 1] - v[low]) : v[low];
}

std::string summary(const std::vector<double>& times) {
    std::ostringstream text;
    text << std::fixed << std::setprecision(1) << "중앙값 " << quantile_of(times, 0.5) << " ms, 95% " << quantile_of(times, 0.95)
         << " ms, 최대 " << *std::max_element(times.begin(), times.end()) << " ms";
    return text.str();
}

std::vector<fs::path> photo_paths(const fs::path& input) {
    std::vector<fs::path> paths;
    Table table;
    if (read_table(input / "photos.tsv", table)) {
        const int column = table.column("path");
        for (const auto& row : table.rows) paths.push_back(fs::u8path(row.at(column)));
    } else {
        for (const PhotoFile& p : find_photos(input)) paths.push_back(p.path);
    }
    return paths;
}

std::string environment(const char* name) { const char* value = std::getenv(name); return value ? value : "(알 수 없음)"; }

}   // namespace

int main(int argc, char** argv) {
    if (argc < 4) { std::cerr << "사용: measure_speed <model 폴더> <사진 폴더 또는 tables 폴더> <출력 폴더> [--count 사진 수]\n"; return 2; }
    const std::string data_folder = argv[1];
    const fs::path input = fs::u8path(argv[2]), output = fs::u8path(argv[3]);
    size_t count = 0;
    for (int a = 4; a + 1 < argc; ++a) if (std::string(argv[a]) == "--count") count = size_t(std::stoul(argv[a + 1]));
    std::string error;
    if (!make_empty_folder(output, error)) { std::cerr << error << "\n"; return 1; }

    std::vector<fs::path> paths = photo_paths(input);
    if (count > 0 && paths.size() > count) paths.resize(count);
    if (paths.size() < 2) { std::cerr << "사진이 2장 이상 필요합니다\n"; return 1; }

    std::ostringstream report;
    report << "컴퓨터: " << environment("PROCESSOR_IDENTIFIER") << ", 논리 코어 " << environment("NUMBER_OF_PROCESSORS") << "개\n"
           << "OpenCV " << CV_VERSION << ", 사진 " << paths.size() << "장 (스레드 설정마다 첫 1장 제외)\n\n";

    const auto init_start = Clock::now();
    if (alignment::initialize(data_folder).status == alignment::Status::Failed) return 1;
    report << "initialize 한 번: " << std::fixed << std::setprecision(1) << milliseconds_since(init_start) << " ms\n";

    std::ofstream table(output / "times.tsv", std::ios::binary);
    table << "threads\tphoto\tread_ms\tmeasure_ms\n";
    for (int threads : kThreadSettings) {
        cv::setNumThreads(threads);
        std::vector<double> read_times, measure_times, total_times;
        for (size_t i = 0; i < paths.size(); ++i) {
            const auto read_start = Clock::now();
            const cv::Mat photo = read_gray_photo(paths[i]);
            const double read_ms = milliseconds_since(read_start);
            const auto measure_start = Clock::now();
            alignment::measure_alignment(photo);
            const double measure_ms = milliseconds_since(measure_start);
            table << threads << '\t' << paths[i].filename().u8string() << '\t' << read_ms << '\t' << measure_ms << '\n';
            if (i == 0) continue;   // 준비 시간
            read_times.push_back(read_ms);
            measure_times.push_back(measure_ms);
            total_times.push_back(read_ms + measure_ms);
        }
        report << "\n스레드 " << threads << "개\n"
               << "  측정만 (사진 읽기 제외): " << summary(measure_times) << "\n"
               << "  사진 읽기만:             " << summary(read_times) << "\n"
               << "  읽기 + 측정:             " << summary(total_times) << "\n";
    }
    std::cout << report.str();
    std::ofstream(output / "speed_report.txt", std::ios::binary) << report.str();
    return 0;
}
