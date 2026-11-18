// QR 탐색 창 방식 비교: A 늘 좁게(±110), B 늘 넓게(±250), C 좁게 먼저 재고 실패하면 넓게 다시(계획.md).
// 사용: window_policy <shift_rows.tsv> <assets_w110> <assets_w250> <photos.tsv> <출력.txt>
//   shift_rows.tsv: 실험 20261114_0230 의 사진별 결과(창, 사진, 이동, 성공, Y, T ...)
// 인자 경로는 ASCII.
#include "final00.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utility.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
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
double median(std::vector<double> v) { if (v.empty()) return 0; std::sort(v.begin(), v.end()); return v[v.size() / 2]; }
struct Result { bool ok; double y, t; };
}

int main(int argc, char** argv) {
    try {
        if (argc != 6) { std::cerr << "window_policy <shift_rows.tsv> <assets_w110> <assets_w250> <photos.tsv> <출력.txt>\n"; return 2; }
        std::ostringstream o; o << std::fixed;
        // 1. 사진별 결과: 창 → (사진, 이동) → 결과
        std::map<int, std::map<std::pair<int, int>, Result>> res;
        for (auto& r : read_table(fs::u8path(argv[1]), true))
            res[std::stoi(r[0])][{std::stoi(r[1]), std::stoi(r[4])}] = {r[5] == "1", std::stod(r[6] == "nan" ? "0" : r[6]), std::stod(r[7] == "nan" ? "0" : r[7])};
        auto& narrow = res[110]; auto& wide = res[250];
        // 정확도: 두 창이 모두 값을 낸 경우 차이
        double dy = 0, dt = 0; int both = 0;
        for (auto& [key, a] : narrow) { const auto& b = wide.at(key); if (a.ok && b.ok) { ++both; dy = std::max(dy, std::abs(a.y - b.y)); dt = std::max(dt, std::abs(a.t - b.t)); } }
        o << "정확도: 두 창이 모두 값을 낸 " << both << " 경우, Y 차이 최대 " << std::setprecision(10) << dy << " mm, T 차이 최대 " << dt << "\n";
        // 측정률
        std::map<int, std::array<int, 4>> rate;   // 이동 → 전체, A, B, C
        for (auto& [key, a] : narrow) { const auto& b = wide.at(key); auto& c = rate[key.second]; ++c[0]; c[1] += a.ok; c[2] += b.ok; c[3] += a.ok || b.ok; }
        o << "\n이동(px) | 늘 좁게 | 늘 넓게 | 자동 넓히기 (702장 중 값을 낸 수)\n";
        for (auto& [s, c] : rate) o << s << " | " << c[1] << " | " << c[2] << " | " << c[3] << "\n";

        // 2. 시간 실측 (OpenCV 4스레드, 사진 읽기 제외)
        // 화면 위치(site)는 모듈 전체에 하나라, 창마다 측정기를 따로 만들고 그 창의 시간을 다 잰 뒤 다음 창으로 넘어간다.
        cv::setNumThreads(4);
        const auto photos = read_table(fs::u8path(argv[4]), false);
        std::vector<cv::Mat> images;   // 100장 × 이동 {0, +150}
        for (size_t i = 0; i < photos.size(); i += 7) {
            const cv::Mat gray = read_image(fs::u8path(photos[i].at(1)));
            for (int shift : {0, 150}) {
                const cv::Mat move = (cv::Mat_<double>(2, 3) << 1, 0, shift, 0, 1, 0);
                cv::Mat moved; cv::warpAffine(gray, moved, move, gray.size(), cv::INTER_NEAREST, cv::BORDER_REPLICATE);
                images.push_back(moved);
            }
        }
        std::vector<double> n_ok, n_fail, w_ok, w_fail;
        auto time_all = [&](const char* assets, std::vector<double>& ok_ms, std::vector<double>& fail_ms) {
            final00::Estimator est(assets);
            for (const auto& im : images) for (int k = 0; k < 2; ++k) {   // 첫 번째는 캐시 데우기, 두 번째를 잰다
                const auto t0 = std::chrono::steady_clock::now(); bool ok = true;
                try { est.predict(im); } catch (const std::exception&) { ok = false; }
                const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                if (k) (ok ? ok_ms : fail_ms).push_back(ms);
            }
        };
        time_all(argv[2], n_ok, n_fail);
        time_all(argv[3], w_ok, w_fail);
        const double tn = median(n_ok), tf = median(n_fail), tw = median(w_ok);
        o << "\n시간 중앙값 (ms, 4스레드): 좁게 성공 " << std::setprecision(2) << tn << " (" << n_ok.size() << "번), 좁게 실패 " << tf << " (" << n_fail.size() << "번), 넓게 성공 " << tw << " (" << w_ok.size() << "번), 넓게 실패 " << median(w_fail) << " (" << w_fail.size() << "번)\n";

        // 3. 이동 분포별 평균 시간과 측정률
        auto summarize = [&](const std::string& name, const std::vector<int>& shifts_px) {
            // shifts_px: 각 자세의 QR 이동(px). 702장 이동 0 결과에 자세 이동만큼 더한 것이 아니라, 좁게/넓게 성공 여부를 |이동| 으로 판단한다
            double ta = 0, tb = 0, tc = 0; int oka = 0, okb = 0, okc = 0;
            for (int s : shifts_px) {
                const bool a = std::abs(s) <= 110 - 6, b = std::abs(s) <= 250 - 6;   // 봉우리가 창 끝 1 px 안이면 실패, 포물선용 여유
                ta += a ? tn : tf; tb += tw; tc += a ? tn : tf + tw;
                oka += a; okb += b; okc += a || b;
            }
            const double n = double(shifts_px.size());
            o << name << " (" << shifts_px.size() << " 자세)\n"
              << "  늘 좁게   : 값을 낸 비율 " << std::setprecision(1) << 100 * oka / n << "%, 평균 " << std::setprecision(2) << ta / n << " ms (실패도 시간에 포함)\n"
              << "  늘 넓게   : 값을 낸 비율 " << std::setprecision(1) << 100 * okb / n << "%, 평균 " << std::setprecision(2) << tb / n << " ms\n"
              << "  자동 넓히기: 값을 낸 비율 " << std::setprecision(1) << 100 * okc / n << "%, 평균 " << std::setprecision(2) << tc / n << " ms\n";
        };
        // SEMES 학습 자료: 각 사진의 QR 이동 = 이동 0 결과의 T, Y 로 환산(14·Y + 80·T)
        std::vector<int> semes;
        for (auto& [key, a] : wide) if (key.second == 0 && a.ok) semes.push_back(int(std::lround(14 * a.y + 80 * a.t)));
        // 성현 님 평가표 35 자세 (Y mm, T 도)
        const double ksh[35][2] = {{0.1,0.16},{1.1,0.16},{1.1,-1.6},{2,0.6},{2,1},{2.1,0.16},{2.1,-0.16},{7,1},{-2,-1},{-2,-1.2},{-2,-1.4},{-2,-1.6},{-3,-1},{-3,-1.6},{-4,-1},{-4,-1.6},{-6,-1},
                                   {-1.5,1.2},{-2.5,1},{-3,0.5},{-3.5,0.8},{-4,1},{-5,1},{-6,2},{2,-0.8},{3,-0.7},{3,-1.5},{4,-2},{4,-1},{5,-0.5},{6,-2},{7,-1},{-0.1,0.04},{-0.1,0.02},{-0.1,0.01}};
        std::vector<int> table; for (auto& p : ksh) table.push_back(int(std::lround(14 * p[0] + 80 * p[1])));
        o << "\n"; summarize("SEMES 학습 자료", semes); summarize("성현 님 평가표", table);
        std::ofstream(fs::u8path(argv[5]), std::ios::binary) << o.str();
        std::cout << o.str();
        return 0;
    } catch (const std::exception& e) { std::cerr << "오류: " << e.what() << "\n"; return 1; }
}
