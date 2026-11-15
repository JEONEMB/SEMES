// 사진 목록 측정과 예상값 비교
// 사용: check_photos <model 폴더> <사진 목록.tsv> <출력.tsv> [예상값.tsv] [오른쪽 px] [아래쪽 px]
// 목록: 번호<TAB>경로. 예상값: 번호<TAB>Y<TAB>T. 모두 헤더 없음
// 이동량 기본값: 0. 예상값 생략 후 이동량 지정 시 - 사용
#include "alignment.hpp"
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

// read_table: 사진 목록 또는 정답표 읽기
//   입력: path (표 경로)
//   출력: 행별 문자열 목록
std::vector<std::vector<std::string>> read_table(const std::string& path) {
    std::ifstream file(path);
    std::vector<std::vector<std::string>> rows;
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (rows.empty() && line.size() >= 3 && (unsigned char)line[0] == 0xEF) line = line.substr(3);   // UTF-8 BOM 3바이트
        if (line.empty()) continue;
        std::vector<std::string> cells;
        std::string cell;
        std::istringstream stream(line);
        while (std::getline(stream, cell, '\t')) cells.push_back(cell);
        rows.push_back(cells);
    }
    return rows;
}

// moved: 사진 전체 이동
//   입력: photo (원본), right (오른쪽 px), down (아래쪽 px)
//   출력: 이동한 사진
cv::Mat moved(const cv::Mat& photo, int right, int down) {
    if ((right == 0 && down == 0) || photo.empty()) return photo;
    const cv::Mat move = (cv::Mat_<double>(2, 3) << 1, 0, right, 0, 1, down);
    cv::Mat moved;
    cv::warpAffine(photo, moved, move, photo.size(), cv::INTER_NEAREST, cv::BORDER_REPLICATE);
    return moved;
}

}   // 파일 내부

// main: 사진 목록 측정과 예상값 비교
//   입력: 실행할 때 지정한 경로와 옵션(파일 상단 사용법)
//   출력: 실행 결과와 성공 여부를 나타내는 종료 코드
int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "사용: check_photos <model 폴더> <사진 목록.tsv> <출력.tsv> [기대값.tsv] [가로로 옮길 px] [아래로 옮길 px]\n";
        return 2;
    }
    if (alignment::initialize(argv[1]).status == alignment::Status::Failed) return 1;
    const auto photos = read_table(argv[2]);
    std::map<std::string, std::pair<double, double>> expected;
    if (argc >= 5 && std::string(argv[4]) != "-")
        for (const auto& row : read_table(argv[4])) expected[row.at(0)] = {std::stod(row.at(1)), std::stod(row.at(2))};
    const int sideways = argc >= 6 ? std::stoi(argv[5]) : 0;
    const int down = argc >= 7 ? std::stoi(argv[6]) : 0;

    std::ofstream out(argv[3], std::ios::binary);
    out << std::setprecision(10) << "id\tY_mm\tT_deg\tstatus\tcode\tqr_px\tscrew1_px\tscrew2_px\tscrew3_px\tscrew4_px\tscrew1_area_px\tscrew2_area_px\tscrew3_area_px\tscrew4_area_px\tscrews_used\tcheck_value\tcheck_limit\tms\n";
    int ok = 0, warned = 0, failed = 0, compared = 0;
    double worst_y = 0, worst_t = 0;
    std::vector<double> times;
    std::map<std::string, int> codes;
    for (const auto& row : photos) {
        const cv::Mat photo = moved(alignment::read_photo(row.at(1)), sideways, down);
        const auto start = std::chrono::steady_clock::now();
        const alignment::Result r = alignment::measure_alignment(photo);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        times.push_back(ms);
        (r.status == alignment::Status::Ok ? ok : r.status == alignment::Status::Warning ? warned : failed) += 1;
        if (!r.code.empty()) ++codes[r.code];
        out << row[0] << '\t' << r.y_mm << '\t' << r.t_deg << '\t' << int(r.status) << '\t' << r.code << '\t' << r.qr_shift_px;
        for (double shift : r.screw_shift_px) out << '\t' << shift;
        for (double shift : r.screw_area_shift_px) out << '\t' << shift;
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
