// 확인: 사진 여러 장을 재서 표로 쓰고, 기대값 표가 있으면 비교한다.
// 사용: check_photos <data 폴더> <사진 목록.tsv> <출력.tsv> [기대값.tsv] [가로로 옮길 px]
//   사진 목록.tsv : 번호<TAB>사진 경로  (머리줄 없음)
//   기대값.tsv    : 번호<TAB>Y<TAB>T    (머리줄 없음)
//   가로로 옮길 px: 사진을 통째로 옮겨 큰 이동을 흉내 낼 때 (기본 0)
#include "alignment.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>

namespace {

cv::Mat read_photo(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    const std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    return bytes.empty() ? cv::Mat() : cv::imdecode(bytes, cv::IMREAD_GRAYSCALE);
}

std::vector<std::vector<std::string>> read_table(const std::string& path) {
    std::ifstream file(path);
    std::vector<std::vector<std::string>> rows;
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (rows.empty() && line.size() >= 3 && (unsigned char)line[0] == 0xEF) line = line.substr(3);   // UTF-8 표시 3바이트
        if (line.empty()) continue;
        std::vector<std::string> cells;
        std::string cell;
        std::istringstream stream(line);
        while (std::getline(stream, cell, '\t')) cells.push_back(cell);
        rows.push_back(cells);
    }
    return rows;
}

cv::Mat moved_sideways(const cv::Mat& photo, int pixels) {
    if (pixels == 0 || photo.empty()) return photo;
    const cv::Mat move = (cv::Mat_<double>(2, 3) << 1, 0, pixels, 0, 1, 0);
    cv::Mat moved;
    cv::warpAffine(photo, moved, move, photo.size(), cv::INTER_NEAREST, cv::BORDER_REPLICATE);
    return moved;
}

}   // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "사용: check_photos <data 폴더> <사진 목록.tsv> <출력.tsv> [기대값.tsv] [가로로 옮길 px]\n";
        return 2;
    }
    if (alignment::initialize(argv[1]).status == alignment::Status::Failed) return 1;
    const auto photos = read_table(argv[2]);
    std::map<std::string, std::pair<double, double>> expected;
    if (argc >= 5 && std::string(argv[4]) != "-")
        for (const auto& row : read_table(argv[4])) expected[row.at(0)] = {std::stod(row.at(1)), std::stod(row.at(2))};
    const int sideways = argc >= 6 ? std::stoi(argv[5]) : 0;

    std::ofstream out(argv[3], std::ios::binary);
    out << std::setprecision(10) << "id\tY_mm\tT_deg\tstatus\tcode\tqr_px\tscrew1_px\tscrew2_px\tscrew3_px\tscrew4_px\tscrews_used\tcheck_value\tcheck_limit\tms\n";
    int ok = 0, warned = 0, failed = 0, compared = 0;
    double worst_y = 0, worst_t = 0;
    std::vector<double> times;
    std::map<std::string, int> codes;
    for (const auto& row : photos) {
        const cv::Mat photo = moved_sideways(read_photo(row.at(1)), sideways);
        const auto start = std::chrono::steady_clock::now();
        const alignment::Result r = alignment::measure_alignment(photo);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        times.push_back(ms);
        (r.status == alignment::Status::Ok ? ok : r.status == alignment::Status::Warning ? warned : failed) += 1;
        if (!r.code.empty()) ++codes[r.code];
        out << row[0] << '\t' << r.y_mm << '\t' << r.t_deg << '\t' << int(r.status) << '\t' << r.code << '\t' << r.qr_shift_px;
        for (double shift : r.screw_shift_px) out << '\t' << shift;
        out << '\t' << r.screws_used << '\t' << r.check_value << '\t' << r.check_limit << '\t' << ms << '\n';
        const auto match = expected.find(row[0]);
        if (match != expected.end() && r.status != alignment::Status::Failed) {
            ++compared;
            worst_y = std::max(worst_y, std::abs(r.y_mm - match->second.first));
            worst_t = std::max(worst_t, std::abs(r.t_deg - match->second.second));
        }
    }
    std::sort(times.begin(), times.end());
    std::cout << "사진 " << photos.size() << "장: 정상 " << ok << ", 경고 " << warned << ", 실패 " << failed
              << "  |  한 장 시간 중앙값 " << std::fixed << std::setprecision(1) << (times.empty() ? 0.0 : times[times.size() / 2]) << " ms\n";
    for (const auto& code : codes) std::cout << "  " << code.first << " " << code.second << "\n";
    if (!expected.empty())
        std::cout << "기대값과 비교 " << compared << "장: Y 차이 최대 " << std::setprecision(8) << worst_y << " mm, T 차이 최대 " << worst_t << " 도\n";
    return 0;
}
