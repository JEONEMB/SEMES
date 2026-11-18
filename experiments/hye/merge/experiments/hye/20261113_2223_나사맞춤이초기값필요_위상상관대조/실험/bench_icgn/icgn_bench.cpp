// 나사 4개 이동을 KSH 의 IC-GN(ksh::fine_alignment, origin/KSH 429b5533 복사본)으로 재고, 지금 방식(기울기 영상 ECC)과 속도를 함께 잰다.
// 시작 위치는 두 방식 모두 지금 모듈과 같은 핀 NCC(QR 이동 주변 ±60 px)다. 핀 단계 시간은 따로 잰다.
// 후보 (이름: 영상, 커널, 반폭)
//   raw_omoms_60  : 밝기, O-MOMS 3차, 121×121 (KSH 설정 그대로: σ 0, 60회, eps 1e-5, 여유 28)
//   raw_omoms_70  : 밝기, O-MOMS 3차, 141×141 (지금 조각 크기)
//   raw_linear_60 : 밝기, 선형, 121×121 (보간 효과 분리)
//   grad_omoms_60 : 기울기 영상(흐림 1.5 뒤 Sobel 크기, 지금과 같은 전처리), O-MOMS, 121×121
//   grad_omoms_70 : 기울기 영상, O-MOMS, 141×141
//   raw_omoms_45, grad_omoms_45, grad_linear_45 : 91×91 (지금 방식의 반지름 45 원과 비슷한 크기)
// 큰 영상 전체를 실수로 바꾸지 않도록 나사 주변만 같은 원점으로 잘라 넘긴다(현재 조각 원점 = 기준 원점 + 정수 시작 이동).
// 사용: icgn_bench <assets> <photos.tsv> <features7.tsv> <출력 tsv>
#include "screw_gradient.hpp"
#include "pin_observer.hpp"
#include "site.hpp"
#include "ksh/fine_align.hpp"
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
    if (im.empty()) throw std::runtime_error("사진을 읽지 못했다");
    return im;
}
static cv::Mat grad32(const cv::Mat& a8) {
    cv::Mat a, b, gx, gy, m;
    a8.convertTo(a, CV_32F); cv::GaussianBlur(a, b, cv::Size(0, 0), 1.5);
    cv::Sobel(b, gx, CV_32F, 1, 0, 3); cv::Sobel(b, gy, CV_32F, 0, 1, 3); cv::magnitude(gx, gy, m); return m;
}

struct Variant { const char* name; bool grad; ksh::Kernel kernel; int hw; };
static const Variant V[] = {{"raw_omoms_60", false, ksh::Kernel::omoms3, 60}, {"raw_omoms_70", false, ksh::Kernel::omoms3, 70},
                            {"raw_linear_60", false, ksh::Kernel::linear, 60}, {"grad_omoms_60", true, ksh::Kernel::omoms3, 60},
                            {"grad_omoms_70", true, ksh::Kernel::omoms3, 70},
                            {"raw_omoms_45", false, ksh::Kernel::omoms3, 45}, {"grad_omoms_45", true, ksh::Kernel::omoms3, 45},
                            {"grad_linear_45", true, ksh::Kernel::linear, 45}};
static const int NV = 8, S = 112;   // 잘라 오는 반폭 = 70 + 여유 28 + 커널·미분 여백 14

int main(int argc, char** argv) {
    if (argc != 5) { std::cerr << "icgn_bench <assets> <photos.tsv> <features7.tsv> <out.tsv>\n"; return 2; }
    cv::setNumThreads(1);
    const fs::path assets = fs::u8path(argv[1]);
    site::load(assets);
    const cv::Mat center = read_gray(assets / "center.bmp");
    final00::ScrewGradient screws(center);
    final04_pin::Observer pins(center);
    const auto& g = site::get();
    cv::Mat ref_raw[4], ref_grad[4];
    for (int k = 0; k < 4; ++k) {
        const cv::Rect r(g.screws[k].x - S, g.screws[k].y - S, 2 * S + 1, 2 * S + 1);
        center(r).convertTo(ref_raw[k], CV_32F); ref_grad[k] = grad32(center(r));
    }
    ksh::FineParams fp;   // KSH 기본값: σ 0, 60회, eps 1e-5, 여유 28, min_zncc 0
    std::map<int, double> qrdx;
    { std::ifstream f(argv[3]); std::string line; std::getline(f, line);
      while (std::getline(f, line)) { std::istringstream s(line); int id; double v; s >> id >> v; qrdx[id] = v; } }
    std::ifstream list(argv[2]); std::ofstream out(fs::u8path(argv[4]));
    out << "id\tt_pin\tt_ecc";
    for (const auto& v : V) out << "\tt_" << v.name;
    for (const auto& v : V) for (int k = 1; k <= 4; ++k) out << '\t' << v.name << "_s" << k << "x\t" << v.name << "_s" << k << "y";
    for (const auto& v : V) out << "\tfail_" << v.name << "\tminzncc_" << v.name;
    out << '\n';
    std::vector<double> T[NV + 2]; int fails[NV] = {0}, n = 0;
    std::string line;
    while (std::getline(list, line)) {
        std::istringstream s(line); int id; std::string path; s >> id; std::getline(s >> std::ws, path);
        const cv::Mat gray = read_gray(fs::u8path(path));
        auto t0 = clk::now();
        const auto pin = pins.measure(gray, cv::Point(int(std::lround(qrdx.at(id))), 0));
        auto t1 = clk::now();
        const cv::Mat e = screws.measure(gray, pin.displacement);
        auto t2 = clk::now();
        T[0].push_back(ms(t0, t1)); T[1].push_back(ms(t1, t2));
        double val[NV][8]; int fail[NV] = {0}; double minz[NV];
        for (int vi = 0; vi < NV; ++vi) {
            const Variant& v = V[vi]; minz[vi] = 1;
            auto a = clk::now();
            for (int k = 0; k < 4; ++k) {
                const cv::Point2d init(pin.displacement.at<double>(2 * k), pin.displacement.at<double>(2 * k + 1));
                const cv::Point oi(int(std::lround(init.x)), int(std::lround(init.y)));
                const cv::Rect rc(g.screws[k].x + oi.x - S, g.screws[k].y + oi.y - S, 2 * S + 1, 2 * S + 1);
                cv::Mat cur;
                if (v.grad) cur = grad32(gray(rc)); else gray(rc).convertTo(cur, CV_32F);
                const cv::Mat& ref = v.grad ? ref_grad[k] : ref_raw[k];
                ksh::FineParams p = fp; p.kernel = v.kernel;
                ksh::FineResult r;
                const cv::Rect roi(S - v.hw, S - v.hw, 2 * v.hw + 1, 2 * v.hw + 1);
                const bool ok = ksh::fine_alignment(ref, cur, roi, init - cv::Point2d(oi), p, r);
                if (!ok) { ++fail[vi]; val[vi][2 * k] = val[vi][2 * k + 1] = std::nan(""); continue; }
                val[vi][2 * k] = oi.x + r.shift_px.x; val[vi][2 * k + 1] = oi.y + r.shift_px.y; minz[vi] = std::min(minz[vi], r.zncc);
            }
            T[2 + vi].push_back(ms(a, clk::now()));
            if (fail[vi]) ++fails[vi];
        }
        out << id << '\t' << T[0].back() << '\t' << T[1].back();
        for (int vi = 0; vi < NV; ++vi) out << '\t' << T[2 + vi].back();
        out.precision(10);
        for (int vi = 0; vi < NV; ++vi) for (int k = 0; k < 8; ++k) out << '\t' << val[vi][k];
        for (int vi = 0; vi < NV; ++vi) out << '\t' << fail[vi] << '\t' << minz[vi];
        out << '\n';
        if (++n % 100 == 0) std::cerr << n << "\n";
    }
    auto report = [&](const char* name, std::vector<double> v) {
        std::sort(v.begin(), v.end()); double mean = 0; for (double x : v) mean += x / v.size();
        std::cout << name << "\tmedian " << v[v.size() / 2] << " ms\tmean " << mean << " ms\tp95 " << v[v.size() * 95 / 100] << " ms\n";
    };
    report("pin_hint", T[0]); report("ecc(지금)", T[1]);
    for (int vi = 0; vi < NV; ++vi) { report(V[vi].name, T[2 + vi]); std::cout << "\t실패 사진 " << fails[vi] << "\n"; }
    std::cout << "photos " << n << "\n";
}
