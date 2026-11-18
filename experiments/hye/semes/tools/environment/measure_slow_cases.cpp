// 느린 경우의 측정 속도 평가
// 사용: measure_slow_cases <model 폴더> <tables 폴더> <출력 폴더> [--count 사진 수]
// 방법: 실제 사진에 가림, 흐림, 이동 같은 변화를 합성한 뒤 측정 한 번의 시간을 잰다. 합성 시간은 제외
// 조건: 동시 작업 수 1개와 4개 비교. 영점 잡기(사진 5장, 10장)의 시간도 함께 잰다
// 주의: 경고와 실패가 나면 측정 함수가 알림 한 줄을 화면에 쓴다. 그 시간도 포함된다 (실제 사용과 같음)
// 출력: slow_cases_report.txt (상황별 평균, 최소, 최대, 정상 대비 배율), times.tsv (사진마다의 시간)
#include "alignment.hpp"
#include "common/console.hpp"
#include "common/tables.hpp"
#include "scene_effects.hpp"
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>

namespace fs = std::filesystem;
using namespace scene;
using Clock = std::chrono::steady_clock;

namespace {

const int kThreadSettings[] = {1, 4};
const size_t kDefaultPhotoCount = 20;      // 상황마다 쓰는 실제 사진 수
const size_t kZeroPhotoCounts[] = {5, 10}; // 영점 잡기에 쓰는 사진 수
const int kZeroRepeats = 5;                // 영점 잡기 반복 횟수

struct Case {
    std::string group, name;
    std::function<cv::Mat(const cv::Mat&)> make;
};

struct Timing {
    std::vector<double> ms;
    std::map<std::string, int> outcomes;   // "상태 코드" 별 사진 수
};

// milliseconds_since: 경과 시간 계산
//   입력: start (시작 시각)
//   출력: 경과 시간(ms)
double milliseconds_since(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

// cover: 사각 영역을 어두운 물체로 가림
//   입력: gray (회색 사진), box (가릴 영역)
//   출력: 가림 영상
cv::Mat cover(const cv::Mat& gray, const cv::Rect& box) {
    cv::Mat out = gray.clone();
    cv::rectangle(out, box, cv::Scalar(45), -1);
    return out;
}

// remove_screws: 나사 여러 개 제거
//   입력: gray (회색 사진), layout (나사 위치), count (왼쪽부터 제거할 개수)
//   출력: 나사 영역을 덮은 영상
cv::Mat remove_screws(const cv::Mat& gray, const Layout& layout, int count) {
    cv::Mat out = gray;
    for (int k = 0; k < count; ++k) out = remove_screw(out, layout.screw[k]);
    return out;
}

// list_cases: 느려질 수 있는 상황 구성
//   입력: L (QR과 나사 위치)
//   출력: 상황 목록. 첫 항목은 변화 없는 정상 사진
std::vector<Case> list_cases(const Layout& L) {
    const cv::Rect all_qr = L.qr[0] | L.qr[1] | L.qr[2];
    return {
        {"기준", "정상 (변화 없음)", [=](const cv::Mat& g) { return g.clone(); }},
        {"나사", "나사 1개 없음", [=](const cv::Mat& g) { return remove_screws(g, L, 1); }},
        {"나사", "나사 2개 없음", [=](const cv::Mat& g) { return remove_screws(g, L, 2); }},
        {"나사", "나사 4개 모두 없음", [=](const cv::Mat& g) { return remove_screws(g, L, 4); }},
        {"나사", "나사 2 모양 다름", [=](const cv::Mat& g) { return different_screw(g, L.screw[1]); }},
        {"나사", "나사 2 반사", [=](const cv::Mat& g) { return glare(g, L.screw[1] + cv::Point(10, -10), 25, 230); }},
        {"QR", "QR 2 반사", [=](const cv::Mat& g) { return glare(g, L.qr[1].tl() + cv::Point(90, 100), 45, 230); }},
        {"QR", "QR 2 가림", [=](const cv::Mat& g) { return cover(g, L.qr[1]); }},
        {"QR", "QR 3개 모두 가림", [=](const cv::Mat& g) { return cover(g, all_qr); }},
        {"화질", "초점 흐림 4 px", [=](const cv::Mat& g) { return defocus(g, 4); }},
        {"화질", "이동 흐림 15 px (가로)", [=](const cv::Mat& g) { return motion_blur(g, 15, true); }},
        {"화질", "잡음 12", [=](const cv::Mat& g) { return noise(g, 12, kSeed); }},
        {"화질", "안개", [=](const cv::Mat& g) { return fog(g, 0.45, 3); }},
        {"화질", "물방울 25개", [=](const cv::Mat& g) { return spots(g, 25, 12, 40, true, kSeed); }},
        {"빛", "어두움 0.4배", [=](const cv::Mat& g) { return brightness(g, 0.4, 0); }},
        {"빛", "밝음 1.8배 (포화)", [=](const cv::Mat& g) { return brightness(g, 1.8, 0); }},
        {"위치", "아래로 40 px", [=](const cv::Mat& g) { return shift(g, 0, 40); }},
        {"위치", "오른쪽으로 200 px", [=](const cv::Mat& g) { return shift(g, 200, 0); }},
        {"위치", "회전 2도", [=](const cv::Mat& g) { return scale_rotate(g, 1, 2); }},
        {"위치", "크기 +5%", [=](const cv::Mat& g) { return scale_rotate(g, 1.05, 0); }},
        {"입력", "컬러 사진 (BGR)", [=](const cv::Mat& g) { cv::Mat color; cv::cvtColor(g, color, cv::COLOR_GRAY2BGR); return color; }},
        {"입력", "크기가 다른 사진 (1920x1080)", [=](const cv::Mat& g) { cv::Mat half; cv::resize(g, half, cv::Size(1920, 1080)); return half; }},
        {"입력", "빈 사진", [=](const cv::Mat&) { return cv::Mat(); }},
    };
}

// status_name: 상태 이름
//   입력: status (측정 상태)
//   출력: Ok, Warning, Failed 중 하나
const char* status_name(alignment::Status status) {
    return status == alignment::Status::Ok ? "Ok" : status == alignment::Status::Warning ? "Warning" : "Failed";
}

// most_common: 가장 많이 나온 결과
//   입력: outcomes (결과별 사진 수), total (전체 사진 수)
//   출력: 결과 이름과 사진 수. 예: Warning W_SCREW_MISSING (20/20)
std::string most_common(const std::map<std::string, int>& outcomes, size_t total) {
    const auto top = std::max_element(outcomes.begin(), outcomes.end(), [](const auto& a, const auto& b) { return a.second < b.second; });
    return top->first + " (" + std::to_string(top->second) + "/" + std::to_string(total) + ")";
}

// mean_of: 평균 계산
//   입력: v (숫자 목록)
//   출력: 평균
double mean_of(const std::vector<double>& v) { return std::accumulate(v.begin(), v.end(), 0.0) / v.size(); }

// pick_evenly: 목록에서 고르게 뽑기
//   입력: paths (전체 경로), count (뽑을 수)
//   출력: 같은 간격으로 뽑은 경로 목록
std::vector<std::string> pick_evenly(const std::vector<std::string>& paths, size_t count) {
    if (paths.size() <= count) return paths;
    std::vector<std::string> picked;
    for (size_t i = 0; i < count; ++i) picked.push_back(paths[i * paths.size() / count]);
    return picked;
}

// time_zero_setting: 영점 잡기 시간 측정
//   입력: center (센터 사진), report (보고서 출력)
//   출력: 없음 (보고서에 사진 수별 평균, 최소, 최대 기록)
//   받은 사진에 명령 (0, 0) 사진이 없어 센터 사진을 여러 장으로 복사해 쓴다. 평균 계산 시간은 사진 내용과 무관
void time_zero_setting(const cv::Mat& center, std::ostream& report) {
    report << "\n영점 잡기 (set_zero_from_photos, 센터 사진을 복사한 같은 사진 여러 장, 사진 읽기 제외, " << kZeroRepeats << "번 반복)\n";
    for (size_t count : kZeroPhotoCounts) {
        const std::vector<cv::Mat> photos(count, center);
        for (int threads : kThreadSettings) {
            cv::setNumThreads(threads);
            std::vector<double> ms;
            std::string outcome;
            for (int repeat = 0; repeat < kZeroRepeats; ++repeat) {
                const auto start = Clock::now();
                const alignment::Result r = alignment::set_zero_from_photos(photos);
                ms.push_back(milliseconds_since(start));
                outcome = std::string(status_name(r.status)) + (r.code.empty() ? "" : " " + r.code);
            }
            report << "  사진 " << count << "장, 스레드 " << threads << "개: 평균 " << mean_of(ms) << " ms, 최소 " << *std::min_element(ms.begin(), ms.end())
                   << " ms, 최대 " << *std::max_element(ms.begin(), ms.end()) << " ms, 결과 " << outcome << "\n";
        }
    }
    alignment::clear_zero();
}

}   // 파일 내부

// main: 느린 경우의 측정 속도 평가
//   입력: 실행할 때 지정한 경로와 옵션(파일 상단 사용법)
//   출력: 실행 결과와 성공 여부를 나타내는 종료 코드
int main(int argc, char** argv) {
    tools::use_korean_console();
    if (argc < 4) { std::cerr << "사용: measure_slow_cases <model 폴더> <tables 폴더> <출력 폴더> [--count 사진 수]\n"; return 2; }
    const fs::path data = fs::u8path(argv[1]), tables = fs::u8path(argv[2]), output = fs::u8path(argv[3]);
    size_t count = kDefaultPhotoCount;
    for (int a = 4; a + 1 < argc; ++a) if (std::string(argv[a]) == "--count") count = size_t(std::stoul(argv[a + 1]));
    std::string error;
    if (!tools::make_empty_folder(output, error)) { std::cerr << error << "\n"; return 1; }

    tools::Table list;
    if (!tools::read_table(tables / "photos.tsv", list)) { std::cerr << "photos.tsv 를 읽을 수 없습니다\n"; return 1; }
    const int path_column = list.column("path");
    if (path_column < 0) { std::cerr << "photos.tsv 에 path 열이 필요합니다\n"; return 1; }
    std::vector<std::string> all_paths;
    for (const auto& row : list.rows) all_paths.push_back(row.at(path_column));
    if (all_paths.size() < 2 || count < 2) { std::cerr << "사진이 2장 이상 필요합니다\n"; return 1; }

    if (alignment::initialize(data.u8string()).status == alignment::Status::Failed) { std::cerr << "initialize 실패\n"; return 1; }
    alignment::clear_zero();
    std::vector<cv::Mat> photos;
    for (const std::string& path : pick_evenly(all_paths, count)) {
        photos.push_back(alignment::read_photo(path));
        if (photos.back().empty()) { std::cerr << "사진을 읽을 수 없습니다: " << path << "\n"; return 1; }
    }
    const std::vector<Case> cases = list_cases(read_layout(data));

    const char* processor = std::getenv("PROCESSOR_IDENTIFIER");
    const char* cores = std::getenv("NUMBER_OF_PROCESSORS");
    std::ostringstream report;
    report << std::fixed << std::setprecision(1)
           << "컴퓨터: " << (processor ? processor : "(알 수 없음)") << ", 논리 코어 " << (cores ? cores : "(알 수 없음)") << "개\n"
           << "OpenCV " << CV_VERSION << ", 상황마다 실제 사진 " << photos.size() << "장에 변화를 합성 (합성 시간과 사진 읽기 제외)\n"
           << "측정 함수: measure_diagnostic (영점 없이 측정 전체 실행)\n";

    std::ofstream table(output / "times.tsv", std::ios::binary);
    table << "threads\tgroup\tcase\tphoto\tmeasure_ms\tstatus\tcode\tscrews_used\n";
    for (int threads : kThreadSettings) {
        cv::setNumThreads(threads);
        alignment::measure_diagnostic(photos[0]);   // 첫 실행의 준비 시간 제외
        std::vector<Timing> timings(cases.size());
        // 사진을 바깥 반복에 두어 상황마다 같은 순서, 같은 열 상태에서 잰다
        for (size_t p = 0; p < photos.size(); ++p) {
            for (size_t c = 0; c < cases.size(); ++c) {
                const cv::Mat image = cases[c].make(photos[p]);
                const auto start = Clock::now();
                const alignment::Result r = alignment::measure_diagnostic(image);
                const double ms = milliseconds_since(start);
                timings[c].ms.push_back(ms);
                ++timings[c].outcomes[std::string(status_name(r.status)) + (r.code.empty() ? "" : " " + r.code)];
                table << threads << '\t' << cases[c].group << '\t' << cases[c].name << '\t' << p << '\t' << ms << '\t'
                      << status_name(r.status) << '\t' << r.code << '\t' << r.screws_used << '\n';
            }
            std::cout << "스레드 " << threads << "개: 사진 " << (p + 1) << "/" << photos.size() << "\r" << std::flush;
        }
        const double normal = mean_of(timings[0].ms);
        report << "\n스레드 " << threads << "개\n"
               << "묶음\t상황\t평균 ms\t최소 ms\t최대 ms\t정상 대비 배율\t가장 많은 결과\n";
        for (size_t c = 0; c < cases.size(); ++c) {
            const std::vector<double>& ms = timings[c].ms;
            report << cases[c].group << '\t' << cases[c].name << '\t' << mean_of(ms) << '\t' << *std::min_element(ms.begin(), ms.end()) << '\t'
                   << *std::max_element(ms.begin(), ms.end()) << '\t' << std::setprecision(2) << mean_of(ms) / normal << std::setprecision(1) << '\t'
                   << most_common(timings[c].outcomes, ms.size()) << '\n';
        }
    }
    const cv::Mat center = alignment::read_photo((data / "center.bmp").u8string());
    if (center.empty()) report << "\n영점 잡기: center.bmp 를 읽을 수 없어 재지 않음\n";
    else time_zero_setting(center, report);

    std::cout << "\n" << report.str();
    std::ofstream(output / "slow_cases_report.txt", std::ios::binary) << report.str();
    return 0;
}
