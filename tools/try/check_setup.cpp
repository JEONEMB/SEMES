// 확인: 영점 잡기, 기준 다시 잡기, 나사 가림이 설명대로 동작하는지 본다.
// 사용: check_setup <model 폴더> <사진 목록.tsv>
//   사진 목록.tsv : 번호<TAB>사진 경로  (머리줄 없음). 기준 사진은 model 폴더의 center.bmp 를 쓴다.
#include "alignment.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>

namespace {

const int kScreenShiftDown = 40;                                      // px. 2 에서 화면을 아래로 미는 양
const cv::Rect kScrew2Cover(2864 - 200, 1358 - 80, 400, 160);         // 3 에서 덮는 곳: 나사 2 (중심 2864, 1358) 가 움직이는 범위를 통째로

using Photos = std::vector<cv::Mat>;
using Results = std::vector<alignment::Result>;

std::vector<std::string> read_photo_paths(const std::string& list) {
    std::ifstream file(list);
    std::vector<std::string> paths;
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const size_t tab = line.find('\t');
        if (tab != std::string::npos) paths.push_back(line.substr(tab + 1));
    }
    return paths;
}

cv::Mat moved(const cv::Mat& photo, int right, int down) {
    const cv::Mat move = (cv::Mat_<double>(2, 3) << 1, 0, right, 0, 1, down);
    cv::Mat out;
    cv::warpAffine(photo, out, move, photo.size(), cv::INTER_NEAREST, cv::BORDER_REPLICATE);
    return out;
}

// 정상, 경고, 실패 수와 코드별 수
struct Tally {
    int ok = 0, warned = 0, failed = 0;
    std::map<std::string, int> codes;
    void add(const alignment::Result& r) {
        (r.status == alignment::Status::Ok ? ok : r.status == alignment::Status::Warning ? warned : failed) += 1;
        if (!r.code.empty()) ++codes[r.code];
    }
    void print(const std::string& title) const {
        std::cout << title << ": 정상 " << ok << ", 경고 " << warned << ", 실패 " << failed;
        for (const auto& code : codes) std::cout << "  " << code.first << " " << code.second;
        std::cout << "\n";
    }
};

// 처음 값(plain)과 가장 크게 다른 정도. 둘 중 하나라도 실패한 사진은 뺀다.
struct WorstDifference {
    double y = 0, t = 0;
    void add(const alignment::Result& now, const alignment::Result& plain) {
        if (now.status == alignment::Status::Failed || plain.status == alignment::Status::Failed) return;
        y = std::max(y, std::abs(now.y_mm - plain.y_mm));
        t = std::max(t, std::abs(now.t_deg - plain.t_deg));
    }
};

// 1. 영점 잡기: 첫 사진을 영점으로 삼으면 그 사진이 (0, 0), 다른 사진은 그만큼 빠진다
void check_zero(const Photos& photos, const Results& plain) {
    const alignment::Result zero = alignment::set_zero_from_photos({photos[0]});
    const alignment::Result same = alignment::measure_alignment(photos[0]);
    double worst = 0;
    for (size_t i = 0; i < photos.size(); ++i) {
        const alignment::Result r = alignment::measure_alignment(photos[i]);
        worst = std::max({worst, std::abs(r.y_mm - (plain[i].y_mm - zero.y_mm)), std::abs(r.t_deg - (plain[i].t_deg - zero.t_deg))});
    }
    std::cout << "1. 영점 잡기: 영점 (" << zero.y_mm << ", " << zero.t_deg << "), 그 사진의 값 (" << same.y_mm << ", " << same.t_deg << "), 다른 사진이 (처음 값 − 영점) 과 다른 정도 최대 " << worst << "\n";
    alignment::clear_zero();
}

// 2. 기준 다시 잡기: 화면이 아래로 밀리면 QR 을 못 찾는다 → 밀린 센터 사진으로 기준을 다시 잡으면 처음 값과 같아진다
void check_reference_reset(const Photos& photos, const Results& plain, const cv::Mat& center) {
    Tally before, after;
    for (const auto& photo : photos) before.add(alignment::measure_alignment(moved(photo, 0, kScreenShiftDown)));
    const alignment::Result set = alignment::set_reference_from_photos({moved(center, 0, kScreenShiftDown)});
    WorstDifference worst;
    for (size_t i = 0; i < photos.size(); ++i) {
        const alignment::Result r = alignment::measure_alignment(moved(photos[i], 0, kScreenShiftDown));
        after.add(r);
        worst.add(r, plain[i]);
    }
    before.print("2. 화면이 아래로 " + std::to_string(kScreenShiftDown) + " px 밀림, 기준 그대로");
    std::cout << "   기준 다시 잡기 결과: " << (set.status == alignment::Status::Failed ? "실패 " + set.message : "성공") << "\n";
    after.print("   기준 다시 잡은 뒤");
    std::cout << "   처음 값과 다른 정도 최대: Y " << worst.y << " mm, T " << worst.t << " 도\n";
}

// 3. 나사 가림: 나사 2 자리를 회색으로 덮는다 → 나머지 나사로 계산하고 경고
void check_covered_screw(const Photos& photos, const Results& plain) {
    Tally covered;
    WorstDifference worst;
    std::map<std::string, int> used;
    for (size_t i = 0; i < photos.size(); ++i) {
        cv::Mat photo = photos[i].clone();
        photo(kScrew2Cover).setTo(128);
        const alignment::Result r = alignment::measure_alignment(photo);
        covered.add(r);
        ++used[r.screws_used];
        worst.add(r, plain[i]);
    }
    covered.print("3. 나사 2 를 가림");
    std::cout << "   쓴 나사:";
    for (const auto& u : used) std::cout << " " << u.first << " " << u.second << "장";
    std::cout << "\n   가리기 전 값과 다른 정도 최대: Y " << worst.y << " mm, T " << worst.t << " 도\n";
}

}   // namespace

int main(int argc, char** argv) {
    if (argc != 3) { std::cerr << "사용: check_setup <model 폴더> <사진 목록.tsv>\n"; return 2; }
    const std::string model_folder = argv[1];
    const cv::Mat center = alignment::read_photo(model_folder + "/center.bmp");
    Photos photos;
    for (const auto& path : read_photo_paths(argv[2])) photos.push_back(alignment::read_photo(path));
    std::cout << std::fixed << std::setprecision(5);

    // 처음 상태의 값 (비교 기준)
    alignment::initialize(model_folder);
    Results plain;
    for (const auto& photo : photos) plain.push_back(alignment::measure_alignment(photo));

    check_zero(photos, plain);
    check_reference_reset(photos, plain, center);
    alignment::initialize(model_folder);   // 2 에서 바꾼 기준을 처음으로 되돌린다
    check_covered_screw(photos, plain);
    return 0;
}
