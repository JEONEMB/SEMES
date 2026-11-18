// 나사 측정 단계별 속도와, 위상 상관을 ECC 초기값으로 썼을 때 값이 같은지 C++(OpenCV 3.4.15)로 잰다.
// 단계 (나사 4개 합, 한 스레드, 사진 읽기 제외)
//   pin_hint : 지금 모듈의 초기값. 00004 핀 NCC 를 QR 이동 주변 ±60 px 에서 찾기
//   pin_full : 같은 핀 NCC 를 ±260 px 전체에서 찾기(예전 방식, 참고)
//   ecc      : 지금 모듈의 기울기 영상 ECC (ScrewGradient::measure)
//   pc141    : cv::phaseCorrelate, 141 조각(핀 초기값 위치), Hann 창
//   pc257    : cv::phaseCorrelate, 257 조각(초기값 없이 나사 기준 위치), Hann 창
//   ecc_pc   : pc257 의 결과를 초기값으로 준 ECC
// 사용: pc_bench <assets> <photos.tsv> <features7.tsv> <출력 tsv>
#include "screw_gradient.hpp"
#include "pin_observer.hpp"
#include "site.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>

namespace fs = std::filesystem;
using clk = std::chrono::steady_clock;
static double ms(clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }

static cv::Mat read_gray(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    std::vector<unsigned char> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    cv::Mat im = cv::imdecode(buf, cv::IMREAD_GRAYSCALE);
    if (im.empty()) throw std::runtime_error("사진을 읽지 못했다: " + p.u8string());
    return im;
}

int main(int argc, char** argv) {
    if (argc != 5) { std::cerr << "pc_bench <assets> <photos.tsv> <features7.tsv> <out.tsv>\n"; return 2; }
    cv::setNumThreads(1);
    const fs::path assets = fs::u8path(argv[1]);
    site::load(assets);
    const cv::Mat center = read_gray(assets / "center.bmp");
    final00::ScrewGradient screws(center);
    final04_pin::Observer pins(center);
    const auto& g = site::get();
    const int H141 = 70, H257 = 128;
    cv::Mat ref141[4], ref257[4], win141, win257;
    cv::createHanningWindow(win141, cv::Size(141, 141), CV_64F);
    cv::createHanningWindow(win257, cv::Size(257, 257), CV_64F);
    for (int k = 0; k < 4; ++k) {
        const cv::Point c = g.screws[k];
        center(cv::Rect(c.x - H141, c.y - H141, 141, 141)).convertTo(ref141[k], CV_64F);
        center(cv::Rect(c.x - H257, c.y - H257, 257, 257)).convertTo(ref257[k], CV_64F);
    }
    std::map<int, double> qrdx;
    { std::ifstream f(argv[3]); std::string line; std::getline(f, line);
      while (std::getline(f, line)) { std::istringstream s(line); int id; double v; s >> id >> v; qrdx[id] = v; } }
    std::ifstream list(argv[2]); std::ofstream out(fs::u8path(argv[4]));
    out << "id\tpin_hint\tpin_full\tecc\tpc141\tpc257\tecc_pc\tmax_diff_ecc_pc\tpin_same\n";
    std::vector<double> T[6]; double worst = 0; int n = 0, nan_cnt = 0;
    std::string line;
    while (std::getline(list, line)) {
        std::istringstream s(line); int id; std::string path; s >> id; std::getline(s >> std::ws, path);
        const cv::Mat gray = read_gray(fs::u8path(path));
        const cv::Point hint(int(std::lround(qrdx.at(id))), 0);
        auto t0 = clk::now(); const auto pin = pins.measure(gray, hint);
        auto t1 = clk::now(); const auto pinf = pins.measure(gray);
        auto t2 = clk::now(); const cv::Mat e = screws.measure(gray, pin.displacement);
        auto t3 = clk::now();
        cv::Mat a, dummy;
        for (int k = 0; k < 4; ++k) {
            const cv::Point c = g.screws[k] + cv::Point(int(std::lround(pin.displacement.at<double>(2 * k))), int(std::lround(pin.displacement.at<double>(2 * k + 1))));
            gray(cv::Rect(c.x - H141, c.y - H141, 141, 141)).convertTo(a, CV_64F);
            cv::phaseCorrelate(ref141[k], a, win141);
        }
        auto t4 = clk::now();
        cv::Mat init(1, 8, CV_64F);
        for (int k = 0; k < 4; ++k) {
            const cv::Point c = g.screws[k];
            gray(cv::Rect(c.x - H257, c.y - H257, 257, 257)).convertTo(a, CV_64F);
            const cv::Point2d d = cv::phaseCorrelate(ref257[k], a, win257);
            init.at<double>(2 * k) = d.x; init.at<double>(2 * k + 1) = d.y;
        }
        auto t5 = clk::now(); const cv::Mat e2 = screws.measure(gray, init);
        auto t6 = clk::now();
        double diff = 0; bool bad = !cv::checkRange(e) || !cv::checkRange(e2);
        if (!bad) diff = cv::norm(e, e2, cv::NORM_INF); else ++nan_cnt;
        worst = std::max(worst, diff);
        bool same = true;
        for (int k = 0; k < 8; ++k) same &= std::lround(pin.displacement.at<double>(k)) == std::lround(init.at<double>(k));
        const double v[6] = {ms(t0, t1), ms(t1, t2), ms(t2, t3), ms(t3, t4), ms(t4, t5), ms(t5, t6)};
        for (int i = 0; i < 6; ++i) T[i].push_back(v[i]);
        out << id; for (double x : v) out << '\t' << x; out << '\t' << diff << '\t' << same << '\n';
        if (++n % 100 == 0) std::cerr << n << "\n";
    }
    const char* names[6] = {"pin_hint", "pin_full", "ecc", "pc141", "pc257", "ecc_pc"};
    for (int i = 0; i < 6; ++i) {
        auto v = T[i]; std::sort(v.begin(), v.end());
        double mean = 0; for (double x : v) mean += x / v.size();
        std::cout << names[i] << "\tmedian " << v[v.size() / 2] << " ms\tmean " << mean << " ms\tp95 " << v[v.size() * 95 / 100] << " ms\n";
    }
    std::cout << "photos " << n << "\tmax |ecc - ecc_pc| " << worst << " px\tNaN " << nan_cnt << "\n";
}
