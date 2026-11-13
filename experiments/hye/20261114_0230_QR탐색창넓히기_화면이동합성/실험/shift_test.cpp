// QR 탐색 창 넓히기 시험: 사진을 가로로 통째로 옮겨 큰 이동을 흉내 내고, 탐색 창 ±110 / ±250 에서 값을 내는지,
// 엉뚱한 곳을 잡지 않는지, 점검 경고와 측정 시간을 잰다(계획.md).
// 사용: shift_test <base_assets> <photos.tsv> <labels.tsv> <출력 폴더>
//   base_assets: site.json 이 없는 SEMES assets(값 5개 흐림 NCC 계수와 점검식). 출력 폴더에 탐색 창별 assets 를 만든다.
// 인자 경로는 ASCII.
#include "final00.hpp"
#include "site.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utility.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
namespace fs = std::filesystem;

namespace {
std::vector<std::vector<std::string>> read_table(const fs::path& file, bool header) {
    std::ifstream f(file); if (!f) throw std::runtime_error("파일을 열 수 없다: " + file.u8string());
    std::vector<std::vector<std::string>> rows; std::string line; bool first = true;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (first && line.size() >= 3 && (unsigned char)line[0] == 0xEF) line = line.substr(3);
        if (first && header) { first = false; continue; }
        first = false; if (line.empty()) continue;
        std::vector<std::string> r; std::string item; std::istringstream s(line);
        while (std::getline(s, item, '\t')) r.push_back(item);
        rows.push_back(r);
    }
    return rows;
}
cv::Mat read_image(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return bytes.empty() ? cv::Mat() : cv::imdecode(bytes, cv::IMREAD_GRAYSCALE);
}
std::string rect_json(const cv::Rect& r) { std::ostringstream o; o << '[' << r.x << ", " << r.y << ", " << r.width << ", " << r.height << ']'; return o.str(); }

// base_assets 를 복사하고 탐색 창만 바꾼 site.json 을 쓴다. 탐색 창이 full_crop 안에 들도록 full_crop 을 가로로 넓힌다.
fs::path make_assets(const fs::path& base, const fs::path& out, int search_x) {
    const site::Geometry g = site::read(base);   // site.json 이 없으면 원래 SEMES 상수
    const fs::path dir = out / ("assets_w" + std::to_string(search_x));
    fs::create_directories(dir);
    for (auto& e : fs::directory_iterator(base)) fs::copy_file(e.path(), dir / e.path().filename(), fs::copy_options::overwrite_existing);
    int x0 = g.full_crop.x, x1 = g.full_crop.x + g.full_crop.width;
    for (const auto& q : g.qr) { x0 = std::min(x0, q.x - search_x); x1 = std::max(x1, q.x + q.width + search_x); }
    const cv::Rect crop(x0, g.full_crop.y, x1 - x0, g.full_crop.height);
    std::ofstream s(dir / "site.json", std::ios::binary);
    s << "{\n \"center_sha256\": \"" << g.center_sha256 << "\",\n \"full_crop\": " << rect_json(crop) << ",\n \"qr\": [" << rect_json(g.qr[0]) << ", " << rect_json(g.qr[1]) << ", " << rect_json(g.qr[2])
      << "],\n \"screws\": [";
    for (int k = 0; k < 4; ++k) s << (k ? ", " : "") << '[' << g.screws[k].x << ", " << g.screws[k].y << ']';
    s << "],\n \"search_x\": " << search_x << ",\n \"search_y\": " << g.search_y << ",\n \"qr_method\": \"ncc\",\n \"note\": \"experiment 20261114_0230 search window test\"\n}\n";
    return dir;
}
const double kNaN = std::numeric_limits<double>::quiet_NaN();
struct Row { int id; int shift; bool valid; double y, t, health, thr, ms; std::string message; };
}

int main(int argc, char** argv) {
    try {
        if (argc != 5) { std::cerr << "shift_test <base_assets> <photos.tsv> <labels.tsv> <출력 폴더>\n"; return 2; }
        const fs::path base = fs::u8path(argv[1]), out = fs::u8path(argv[4]);
        if (fs::exists(out / "shift_rows.tsv")) { std::cerr << "출력이 이미 있다\n"; return 3; }
        std::map<int, std::pair<double, double>> lab;
        for (auto& r : read_table(fs::u8path(argv[3]), true)) lab[std::stoi(r.at(0))] = {std::stod(r.at(4)), std::stod(r.at(5))};
        const auto photos = read_table(fs::u8path(argv[2]), false);
        const std::vector<int> shifts = {0, -50, 50, -100, 100, -150, 150, -200, 200, -250, 250};
        std::ofstream rows_out(out.empty() ? "shift_rows.tsv" : (fs::create_directories(out), out / "shift_rows.tsv"), std::ios::binary);
        rows_out << std::setprecision(10) << "window\tid\tY_abs\tT_abs\tshift_px\tvalid\tY\tT\thealth\thealth_threshold\tms\tmessage\n";
        std::ostringstream summary; summary << std::fixed;
        for (int window : {110, 250}) {
            const fs::path assets = make_assets(base, out, window);
            final00::Estimator est(assets.u8string());
            std::cout << "window " << window << " method " << est.qr_method() << " model " << est.model_id() << std::endl;
            const int n = int(photos.size()), m = int(shifts.size());
            std::vector<Row> rows(size_t(n) * m);
            cv::setNumThreads(1);
#pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < n; ++i) {
                const int id = std::stoi(photos[i].at(0));
                const cv::Mat gray = read_image(fs::u8path(photos[i].at(1)));
                for (int k = 0; k < m; ++k) {
                    Row r{id, shifts[k], false, kNaN, kNaN, kNaN, kNaN, 0, ""};
                    try {
                        const cv::Mat move = (cv::Mat_<double>(2, 3) << 1, 0, shifts[k], 0, 1, 0);   // 정수 이동이라 보간이 없다
                        cv::Mat moved; cv::warpAffine(gray, moved, move, gray.size(), cv::INTER_NEAREST, cv::BORDER_REPLICATE);
                        const auto t0 = std::chrono::steady_clock::now();
                        const final00::Result res = est.predict(moved);
                        r.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                        r.valid = true; r.y = res.pose.y; r.t = res.pose.t; r.health = res.health; r.thr = res.health_threshold;
                    } catch (const std::exception& e) { r.message = e.what(); for (char& c : r.message) if (c == '\t' || c == '\n') c = ' '; }
                    rows[size_t(i) * m + k] = r;
                }
            }
            for (const auto& r : rows) { const auto l = lab.at(r.id);
                rows_out << window << '\t' << r.id << '\t' << l.first << '\t' << l.second << '\t' << r.shift << '\t' << r.valid << '\t' << r.y << '\t' << r.t << '\t' << r.health << '\t' << r.thr << '\t' << r.ms << '\t' << r.message << '\n'; }
            // 요약: 이동 전 값(0 px) 기준 ΔY, ΔT. "T / px" 기울기는 ±50 px 에서 맞춘다.
            std::map<int, std::pair<double, double>> base0; for (const auto& r : rows) if (r.shift == 0 && r.valid) base0[r.id] = {r.y, r.t};
            double sxy = 0, sxx = 0; for (const auto& r : rows) if (r.valid && std::abs(r.shift) == 50 && base0.count(r.id)) { sxy += r.shift * (r.t - base0[r.id].second); sxx += double(r.shift) * r.shift; }
            const double slope = sxy / sxx;
            summary << "window " << window << "  T per px (fit at ±50) " << std::setprecision(6) << slope << "\n";
            summary << "  shift   measured   wrong_lock(|dT-slope*shift|>0.1)   mean dY   sd dY   max|dT-slope*shift|   health_warn   median_ms   main_error\n";
            for (int s : shifts) {
                int total = 0, ok = 0, wrong = 0, warn = 0; double sum = 0, sum2 = 0, dev = 0; std::vector<double> ms; std::map<std::string, int> err;
                for (const auto& r : rows) {
                    if (r.shift != s) continue; ++total;
                    if (!r.valid) { ++err[r.message.substr(0, 40)]; continue; }
                    ++ok; ms.push_back(r.ms); warn += r.health > r.thr;
                    if (!base0.count(r.id)) continue;
                    const double dy = r.y - base0[r.id].first, d = std::abs(r.t - base0[r.id].second - slope * s);
                    sum += dy; sum2 += dy * dy; dev = std::max(dev, d); wrong += d > 0.1;
                }
                std::sort(ms.begin(), ms.end());
                std::string top; int most = 0; for (auto& e : err) if (e.second > most) { most = e.second; top = e.first; }
                const double mean = ok ? sum / ok : kNaN, sd = ok ? std::sqrt(std::max(0.0, sum2 / ok - mean * mean)) : kNaN;
                summary << std::setw(7) << s << std::setw(7) << ok << "/" << total << std::setw(10) << wrong << std::setprecision(5) << std::setw(28) << mean << std::setw(9) << sd << std::setw(12) << dev
                        << std::setw(14) << warn << std::setprecision(1) << std::setw(12) << (ms.empty() ? kNaN : ms[ms.size() / 2]) << "   " << top << "\n";
            }
        }
        { std::ofstream t(out / "shift_summary.txt", std::ios::binary); t << summary.str(); }
        std::cout << summary.str();
        return 0;
    } catch (const std::exception& e) { std::cerr << "오류: " << e.what() << "\n"; return 1; }
}
