// 평가: 명령값 기준 채점 (엑셀로 열어 손으로 확인하는 표)
//
// 사용: score_photos <사진 폴더> <출력 폴더> [--data <model 폴더>] [--tolerance-y 0.045] [--tolerance-t 0.01] [--answers 정답표.tsv]
//
// 정답
//   - 기본: 파일 이름의 명령값 ("Y+0.05 T+0.01 (2).bmp" → Y 0.05, T 0.01)
//   - --answers: 탭으로 나눈 표 (머리줄: file	y_mm	t_deg). file 은 파일 이름
// 영점
//   - 정답이 (0, 0) 인 사진이 있으면 그 사진들로 영점을 잡고 채점한다 (실사용에서 매일 영점을 잡는 것과 같은 조건)
//     영점에 쓴 사진은 채점에서 뺀다
//   - 없으면 영점 없이 채점하고, 결과 맨 위에 경고를 적는다
//   - 두 점수(영점 없음, 영점 잡음)를 항상 같이 낸다
// 출력
//   results.csv  한 줄 = 사진 한 장 (엑셀에서 바로 열림)
//   summary.txt  전체, 폴더별 통과율, 최대 오차, 상태별 개수
#include "alignment.hpp"
#include "common/coefficient_fit.hpp"   // 통과 기준 kToleranceYmm, kToleranceTdeg
#include "common/photo_files.hpp"
#include "common/tables.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>

namespace fs = std::filesystem;
using namespace tools;

namespace {

struct Arguments {
    fs::path photo_folder, output, answers_file;
    std::string model_folder = "data/model";
    double tolerance_y = kToleranceYmm, tolerance_t = kToleranceTdeg;   // 기본은 학습, 평가와 같은 통과 기준
};

bool read_arguments(int argc, char** argv, Arguments& arguments) {
    if (argc < 3) return false;
    arguments.photo_folder = fs::u8path(argv[1]);
    arguments.output = fs::u8path(argv[2]);
    for (int a = 3; a + 1 < argc; ++a) {
        const std::string option = argv[a];
        if (option == "--data") arguments.model_folder = argv[a + 1];
        if (option == "--tolerance-y") arguments.tolerance_y = std::stod(argv[a + 1]);
        if (option == "--tolerance-t") arguments.tolerance_t = std::stod(argv[a + 1]);
        if (option == "--answers") arguments.answers_file = fs::u8path(argv[a + 1]);
    }
    return true;
}

struct Counts { int passed = 0, total = 0; };

std::string csv_cell(const std::string& text) {
    std::string quoted = "\"";
    for (char c : text) quoted += (c == '"') ? std::string("\"\"") : std::string(1, c);
    return quoted + "\"";
}

std::string number(double value, int digits) {
    if (!std::isfinite(value)) return "";
    std::ostringstream text;
    text << std::fixed << std::setprecision(digits) << value;
    return text.str();
}

const char* status_name(alignment::Status status) {
    return status == alignment::Status::Ok ? "정상" : status == alignment::Status::Warning ? "경고" : "실패";
}

// 정답표(--answers)의 값으로 사진의 정답을 바꾼다. 표에 없는 사진은 정답 없음이 된다.
bool read_answer_table(const fs::path& answers_file, std::vector<PhotoFile>& photos) {
    Table answers;
    if (!read_table(answers_file, answers)) return false;
    std::map<std::string, std::pair<double, double>> by_name;
    for (const auto& row : answers.rows) by_name[row.at(0)] = {std::stod(row.at(1)), std::stod(row.at(2))};
    for (PhotoFile& p : photos) {
        const auto it = by_name.find(p.name);
        p.has_command = it != by_name.end();
        if (p.has_command) {
            p.command_y_mm = it->second.first;
            p.command_t_deg = it->second.second;
        }
    }
    return true;
}

// 사진을 채점할 것, 영점용(정답 0, 0), 정답 없는 것으로 나눈 것
struct PhotoGroups {
    std::vector<PhotoFile> scored, zero_photos, no_answer;
};

PhotoGroups group_photos(const std::vector<PhotoFile>& photos) {
    PhotoGroups groups;
    for (const PhotoFile& p : photos) (p.has_command ? (is_zero_command(p) ? groups.zero_photos : groups.scored) : groups.no_answer).push_back(p);
    return groups;
}

struct Zero {
    double y_mm = 0, t_deg = 0;   // 잡은 영점 (못 잡았으면 0, 0)
    std::string note;             // 요약 맨 위에 적는 설명 또는 경고
};

// 정답 (0, 0) 사진들로 영점을 잡는다.
Zero set_zero(const std::vector<PhotoFile>& zero_photos) {
    Zero zero;
    if (zero_photos.empty()) {
        zero.note = "경고: 정답 (0, 0) 사진이 없어 영점 없이 채점했다. 그날 로봇 0 이 절대 영점에서 어긋난 만큼(우리 사진에서 Y 0.04 ~ 0.13 mm) 치우칠 수 있다";
        return zero;
    }
    const std::string count = std::to_string(zero_photos.size());
    std::vector<cv::Mat> images;
    for (const PhotoFile& p : zero_photos) images.push_back(alignment::read_photo(p.path.u8string()));
    const alignment::Result result = alignment::set_zero_from_photos(images);
    if (result.status == alignment::Status::Failed) {
        zero.note = "경고: 정답 (0, 0) 사진 " + count + "장으로 영점을 잡지 못했다 (" + result.message + "). 영점 없이 채점했다";
        return zero;
    }
    zero.y_mm = result.y_mm;
    zero.t_deg = result.t_deg;
    zero.note = "영점: 정답 (0, 0) 사진 " + count + "장으로 잡음 (Y " + number(zero.y_mm, 4) + " mm, T " + number(zero.t_deg, 5) + " 도). 이 사진들은 채점에서 뺐다";
    return zero;
}

struct Scores {
    Counts with_zero, without_zero;
    std::map<std::string, Counts> by_folder;
    std::map<std::string, int> by_status, by_code;
    double worst_y = 0, worst_t = 0;   // 최대 오차 (영점 잡음, 실패 제외)
};

// 사진을 한 장씩 재서 results.csv 에 한 줄씩 쓰고 점수를 모은다.
Scores measure_photos(const std::vector<PhotoFile>& scored, const Zero& zero, const Arguments& arguments) {
    std::ofstream csv(arguments.output / "results.csv", std::ios::binary);
    csv << "\xEF\xBB\xBF";   // 엑셀이 한글을 읽게
    csv << "사진,폴더,정답 Y,정답 T,측정 Y,측정 T,오차 Y,오차 T,통과,영점 없이 측정 Y,영점 없이 측정 T,영점 없이 통과,상태,코드,메시지,시간 ms\n";
    Scores scores;
    for (const PhotoFile& p : scored) {
        const cv::Mat image = alignment::read_photo(p.path.u8string());
        const auto start = std::chrono::steady_clock::now();
        const alignment::Result r = alignment::measure_alignment(image);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        const std::string folder = fs::relative(p.path.parent_path(), arguments.photo_folder).u8string();
        const bool failed = r.status == alignment::Status::Failed;
        const double error_y = r.y_mm - p.command_y_mm, error_t = r.t_deg - p.command_t_deg;
        const double raw_y = r.y_mm + zero.y_mm, raw_t = r.t_deg + zero.t_deg;   // 영점을 빼기 전 값
        const bool ok = !failed && std::abs(error_y) <= arguments.tolerance_y && std::abs(error_t) <= arguments.tolerance_t;
        const bool ok_raw = !failed && std::abs(raw_y - p.command_y_mm) <= arguments.tolerance_y && std::abs(raw_t - p.command_t_deg) <= arguments.tolerance_t;
        ++scores.with_zero.total;
        scores.with_zero.passed += ok;
        ++scores.without_zero.total;
        scores.without_zero.passed += ok_raw;
        ++scores.by_folder[folder].total;
        scores.by_folder[folder].passed += ok;
        ++scores.by_status[status_name(r.status)];
        if (!r.code.empty()) ++scores.by_code[r.code];
        if (!failed) {
            scores.worst_y = std::max(scores.worst_y, std::abs(error_y));
            scores.worst_t = std::max(scores.worst_t, std::abs(error_t));
        }
        csv << csv_cell(p.name) << ',' << csv_cell(folder) << ',' << number(p.command_y_mm, 4) << ',' << number(p.command_t_deg, 4) << ','
            << number(r.y_mm, 4) << ',' << number(r.t_deg, 5) << ',' << number(error_y, 4) << ',' << number(error_t, 5) << ',' << (ok ? 1 : 0) << ','
            << number(raw_y, 4) << ',' << number(raw_t, 5) << ',' << (ok_raw ? 1 : 0) << ',' << status_name(r.status) << ',' << csv_cell(r.code) << ','
            << csv_cell(r.message) << ',' << number(ms, 1) << '\n';
    }
    return scores;
}

std::string summary_text(const PhotoGroups& groups, const Zero& zero, const Scores& scores, const Arguments& arguments) {
    std::ostringstream summary;
    summary << zero.note << "\n\n"
            << "통과 기준: |Y 오차| <= " << arguments.tolerance_y << " mm 그리고 |T 오차| <= " << arguments.tolerance_t << " 도\n"
            << "채점한 사진 " << groups.scored.size() << "장 (영점 사진 " << groups.zero_photos.size() << "장, 정답이 없어 뺀 사진 " << groups.no_answer.size() << "장)\n\n"
            << "통과율 (영점 잡음): " << format_rate(scores.with_zero.passed, scores.with_zero.total) << "\n"
            << "통과율 (영점 없음): " << format_rate(scores.without_zero.passed, scores.without_zero.total) << "\n"
            << "최대 오차 (영점 잡음, 실패 제외): Y " << number(scores.worst_y, 4) << " mm, T " << number(scores.worst_t, 5) << " 도\n\n폴더별 (영점 잡음)\n";
    for (const auto& f : scores.by_folder) summary << "  " << f.first << "  " << format_rate(f.second.passed, f.second.total) << "\n";
    summary << "\n상태별\n";
    for (const auto& s : scores.by_status) summary << "  " << s.first << " " << s.second << "\n";
    summary << "\n코드별 (README 5절)\n";
    for (const auto& c : scores.by_code) summary << "  " << c.first << " " << c.second << "\n";
    if (!groups.no_answer.empty()) {
        summary << "\n정답이 없어 뺀 사진\n";
        for (const PhotoFile& p : groups.no_answer) summary << "  " << p.name << "\n";
    }
    return summary.str();
}

}   // namespace

int main(int argc, char** argv) {
    Arguments arguments;
    if (!read_arguments(argc, argv, arguments)) {
        std::cerr << "사용: score_photos <사진 폴더> <출력 폴더> [--data <model 폴더>] [--tolerance-y 0.045] [--tolerance-t 0.01] [--answers 정답표.tsv]\n";
        return 2;
    }
    std::string error;
    if (!make_empty_folder(arguments.output, error)) { std::cerr << error << "\n"; return 1; }
    if (alignment::initialize(arguments.model_folder).status == alignment::Status::Failed) return 1;

    std::vector<PhotoFile> photos = find_photos(arguments.photo_folder);
    if (!arguments.answers_file.empty() && !read_answer_table(arguments.answers_file, photos)) {
        std::cerr << "정답표를 읽지 못했습니다: " << arguments.answers_file.u8string() << "\n";
        return 1;
    }
    const PhotoGroups groups = group_photos(photos);
    const Zero zero = set_zero(groups.zero_photos);
    const Scores scores = measure_photos(groups.scored, zero, arguments);

    const std::string summary = summary_text(groups, zero, scores, arguments);
    std::cout << summary;
    std::ofstream(arguments.output / "summary.txt", std::ios::binary) << summary;
    return 0;
}
