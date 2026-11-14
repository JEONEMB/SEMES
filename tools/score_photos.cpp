// 평가: 명령값 기준 채점 (엑셀로 열어 손으로 확인하는 표)
//
// 사용: score_photos <사진 폴더> <출력 폴더> [--data <data 폴더>] [--tolerance-y 0.045] [--tolerance-t 0.01] [--answers 정답표.tsv]
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

}   // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "사용: score_photos <사진 폴더> <출력 폴더> [--data <data 폴더>] [--tolerance-y 0.045] [--tolerance-t 0.01] [--answers 정답표.tsv]\n";
        return 2;
    }
    const fs::path photo_folder = fs::u8path(argv[1]), output = fs::u8path(argv[2]);
    std::string data_folder = "data";
    double tolerance_y = 0.045, tolerance_t = 0.01;
    fs::path answers_file;
    for (int a = 3; a + 1 < argc; ++a) {
        const std::string option = argv[a];
        if (option == "--data") data_folder = argv[a + 1];
        if (option == "--tolerance-y") tolerance_y = std::stod(argv[a + 1]);
        if (option == "--tolerance-t") tolerance_t = std::stod(argv[a + 1]);
        if (option == "--answers") answers_file = fs::u8path(argv[a + 1]);
    }
    std::string error;
    if (!make_empty_folder(output, error)) { std::cerr << error << "\n"; return 1; }
    if (alignment::initialize(data_folder).status == alignment::Status::Failed) return 1;

    // 정답 붙이기
    std::vector<PhotoFile> photos = find_photos(photo_folder);
    if (!answers_file.empty()) {
        Table answers;
        if (!read_table(answers_file, answers)) { std::cerr << "정답표를 읽지 못했습니다: " << answers_file.u8string() << "\n"; return 1; }
        std::map<std::string, std::pair<double, double>> by_name;
        for (const auto& row : answers.rows) by_name[row.at(0)] = {std::stod(row.at(1)), std::stod(row.at(2))};
        for (PhotoFile& p : photos) {
            const auto it = by_name.find(p.name);
            p.has_command = it != by_name.end();
            if (p.has_command) { p.command_y_mm = it->second.first; p.command_t_deg = it->second.second; }
        }
    }
    std::vector<PhotoFile> scored, zero_photos, no_answer;
    for (const PhotoFile& p : photos) (p.has_command ? (is_zero_command(p) ? zero_photos : scored) : no_answer).push_back(p);

    // 영점
    double zero_y = 0, zero_t = 0;
    std::string zero_note;
    if (!zero_photos.empty()) {
        std::vector<cv::Mat> images;
        for (const PhotoFile& p : zero_photos) images.push_back(read_gray_photo(p.path));
        const alignment::Result zero = alignment::set_zero_from_photos(images);
        if (zero.status == alignment::Status::Failed) {
            zero_note = "경고: 정답 (0, 0) 사진 " + std::to_string(zero_photos.size()) + "장으로 영점을 잡지 못했다 (" + zero.message + "). 영점 없이 채점했다";
        } else {
            zero_y = zero.y_mm;
            zero_t = zero.t_deg;
            zero_note = "영점: 정답 (0, 0) 사진 " + std::to_string(zero_photos.size()) + "장으로 잡음 (Y " + number(zero_y, 4) + " mm, T " + number(zero_t, 5) + " 도). 이 사진들은 채점에서 뺐다";
        }
    } else {
        zero_note = "경고: 정답 (0, 0) 사진이 없어 영점 없이 채점했다. 그날 로봇 0 이 절대 영점에서 어긋난 만큼(우리 사진에서 Y 0.04 ~ 0.13 mm) 치우칠 수 있다";
    }

    // 채점
    std::ofstream csv(output / "results.csv", std::ios::binary);
    csv << "\xEF\xBB\xBF";   // 엑셀이 한글을 읽게
    csv << "사진,폴더,정답 Y,정답 T,측정 Y,측정 T,오차 Y,오차 T,통과,영점 없이 측정 Y,영점 없이 측정 T,영점 없이 통과,상태,코드,메시지,시간 ms\n";
    Counts with_zero, without_zero;
    std::map<std::string, Counts> by_folder;
    std::map<std::string, int> by_status, by_code;
    double worst_y = 0, worst_t = 0;
    for (const PhotoFile& p : scored) {
        const cv::Mat image = read_gray_photo(p.path);
        const auto start = std::chrono::steady_clock::now();
        const alignment::Result r = alignment::measure_alignment(image);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        const std::string folder = fs::relative(p.path.parent_path(), photo_folder).u8string();
        const bool failed = r.status == alignment::Status::Failed;
        const double error_y = r.y_mm - p.command_y_mm, error_t = r.t_deg - p.command_t_deg;
        const double raw_y = r.y_mm + zero_y, raw_t = r.t_deg + zero_t;   // 영점을 빼기 전 값
        const bool ok = !failed && std::abs(error_y) <= tolerance_y && std::abs(error_t) <= tolerance_t;
        const bool ok_raw = !failed && std::abs(raw_y - p.command_y_mm) <= tolerance_y && std::abs(raw_t - p.command_t_deg) <= tolerance_t;
        ++with_zero.total; with_zero.passed += ok;
        ++without_zero.total; without_zero.passed += ok_raw;
        ++by_folder[folder].total; by_folder[folder].passed += ok;
        ++by_status[status_name(r.status)];
        if (!r.code.empty()) ++by_code[r.code];
        if (!failed) { worst_y = std::max(worst_y, std::abs(error_y)); worst_t = std::max(worst_t, std::abs(error_t)); }
        csv << csv_cell(p.name) << ',' << csv_cell(folder) << ',' << number(p.command_y_mm, 4) << ',' << number(p.command_t_deg, 4) << ','
            << number(r.y_mm, 4) << ',' << number(r.t_deg, 5) << ',' << number(error_y, 4) << ',' << number(error_t, 5) << ',' << (ok ? 1 : 0) << ','
            << number(raw_y, 4) << ',' << number(raw_t, 5) << ',' << (ok_raw ? 1 : 0) << ',' << status_name(r.status) << ',' << csv_cell(r.code) << ','
            << csv_cell(r.message) << ',' << number(ms, 1) << '\n';
    }

    std::ostringstream summary;
    summary << zero_note << "\n\n"
            << "통과 기준: |Y 오차| <= " << tolerance_y << " mm 그리고 |T 오차| <= " << tolerance_t << " 도\n"
            << "채점한 사진 " << scored.size() << "장 (영점 사진 " << zero_photos.size() << "장, 정답이 없어 뺀 사진 " << no_answer.size() << "장)\n\n"
            << "통과율 (영점 잡음): " << format_rate(with_zero.passed, with_zero.total) << "\n"
            << "통과율 (영점 없음): " << format_rate(without_zero.passed, without_zero.total) << "\n"
            << "최대 오차 (영점 잡음, 실패 제외): Y " << number(worst_y, 4) << " mm, T " << number(worst_t, 5) << " 도\n\n폴더별 (영점 잡음)\n";
    for (const auto& f : by_folder) summary << "  " << f.first << "  " << format_rate(f.second.passed, f.second.total) << "\n";
    summary << "\n상태별\n";
    for (const auto& s : by_status) summary << "  " << s.first << " " << s.second << "\n";
    summary << "\n코드별 (README 5절)\n";
    for (const auto& c : by_code) summary << "  " << c.first << " " << c.second << "\n";
    if (!no_answer.empty()) {
        summary << "\n정답이 없어 뺀 사진\n";
        for (const PhotoFile& p : no_answer) summary << "  " << p.name << "\n";
    }
    std::cout << summary.str();
    std::ofstream(output / "summary.txt", std::ios::binary) << summary.str();
    return 0;
}
