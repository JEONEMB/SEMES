// A19: joint 측정(QR ECC 54 + 위·아래 가로선 48)이 안정적으로 뽑히는지 점검한다.
//  1부: 575장 전수 측정. 실패, 탐색 범위 확장, QR NCC·ECC 상관, 정밀화 이동량, 선 블록 유효·상관을 기록하고 측정값(102)을 캐시로 남긴다.
//  2부: 대표 30장에 알려진 교란(정확한 FFT 이동, 밝기, 흐림, 잡음, 조명 기울기)을 넣고 측정값과 frozen joint 의 (Y,T) 가 얼마나 바뀌는지 본다.
//  측정 코드는 00000_final_Y_T_joint 를 복사했다(바꾼 것: macOS 용 sha256, 진단을 내는 measure_detailed 추가). 원본은 수정하지 않았다.
#include "observer_internal.hpp"
#include <opencv2/opencv.hpp>
#include <atomic>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <thread>

namespace fs = std::filesystem;
using namespace alignment_detail;

struct Shot { int idx; std::string set, rel; double y, t; int group; };
static std::vector<std::string> split(const std::string& s) { std::vector<std::string> r; std::string c; std::stringstream ss(s); while (std::getline(ss, c, ',')) r.push_back(c); return r; }

struct Frozen {
    cv::Mat origin, projector, initial; double scale[2];
    explicit Frozen(const fs::path& p) {
        cv::FileStorage f(p.u8string(), cv::FileStorage::READ); auto m = f["model"];
        origin = node_matrix(m["origin"]); projector = node_matrix(m["projector"]); initial = node_matrix(m["initial"]);
        scale[0] = double(m["scale"][0]); scale[1] = double(m["scale"][1]);
    }
    cv::Point2d predict(const cv::Mat& x) const { cv::Mat q = (feature_map(x) - origin) * projector * initial; return {q.at<double>(0, 0) * scale[0], q.at<double>(0, 1) * scale[1]}; }
};

static cv::Mat fft_shift(const cv::Mat& g, double dx, double dy) {
    cv::Mat f; g.convertTo(f, CV_32F); cv::Mat F; cv::dft(f, F, cv::DFT_COMPLEX_OUTPUT);
    for (int v = 0; v < F.rows; ++v) { double fy = (v <= F.rows / 2 ? v : v - F.rows) / double(F.rows);
        auto* row = F.ptr<cv::Vec2f>(v);
        for (int u = 0; u < F.cols; ++u) { double fx = (u <= F.cols / 2 ? u : u - F.cols) / double(F.cols);
            double ph = -2 * CV_PI * (fx * dx + fy * dy), c = std::cos(ph), s = std::sin(ph); auto z = row[u];
            row[u] = cv::Vec2f(float(z[0] * c - z[1] * s), float(z[0] * s + z[1] * c)); } }
    cv::Mat out; cv::idft(F, out, cv::DFT_REAL_OUTPUT | cv::DFT_SCALE); cv::Mat u8; out.convertTo(u8, CV_8U); return u8;
}

int main(int argc, char** argv) {
    const fs::path repo = fs::u8path(REPO_DIR), res = fs::u8path(EXPERIMENT_DIR);
    fs::create_directories(res / "results"); fs::create_directories(res / "cache");
    const fs::path assets = repo / fs::u8path(u8"experiments/hye/00000_final_Y_T_joint/assets");
    Observer obs(assets); Frozen model(assets / "both_model.json");
    std::vector<Shot> shots;
    { std::ifstream in(repo / fs::u8path(u8"experiments/hye/20261102_124309_A0_분할측정고정_계층분할과점측정캐시/결과/cache/shots.csv")); std::string l; std::getline(in, l);
      while (std::getline(in, l)) { auto r = split(l); shots.push_back({std::stoi(r[0]), r[1], r[2], std::stod(r[3]), std::stod(r[4]), std::stoi(r[7])}); } }
    auto path_of = [&](const Shot& s) { return repo / "data" / "clean" / fs::u8path(s.set) / fs::u8path(s.rel); };
    const int N = int(shots.size());
    std::string mode = argc > 1 ? argv[1] : "all";

    if (mode == "all" || mode == "extract") {
        std::vector<DetailedObservation> D(N); std::vector<double> ms(N); std::vector<std::string> err(N);
        std::atomic<int> next{0};
        auto work = [&]() { for (int i; (i = next++) < N;) {
            try { cv::Mat g = cv::imread(path_of(shots[i]).u8string(), cv::IMREAD_GRAYSCALE); if (g.empty()) throw std::runtime_error("read");
                  auto t0 = cv::getTickCount(); D[i] = obs.measure_detailed(g); ms[i] = (cv::getTickCount() - t0) * 1000.0 / cv::getTickFrequency(); }
            catch (const std::exception& e) { err[i] = e.what(); } } };
        std::vector<std::thread> th; for (int k = 0; k < 6; ++k) th.emplace_back(work); for (auto& t : th) t.join();
        std::ofstream o(res / "results" / "per_shot.csv");
        o << "idx,set,rel,Y,T,group,error,pad,qr_ok,ncc0,ncc1,ncc2,ecc0,ecc1,ecc2,refine0,refine1,refine2,lines_ok,line_corr_min,joint_Y,joint_T,eY,eT,ms\n";
        std::ofstream lc(res / "results" / "line_blocks.csv"); lc << "idx"; for (int b = 0; b < 48; ++b) lc << ",shift" << b; for (int b = 0; b < 48; ++b) lc << ",corr" << b; for (int b = 0; b < 48; ++b) lc << ",valid" << b; lc << "\n";
        std::vector<double> cache(size_t(N) * 102, std::numeric_limits<double>::quiet_NaN());
        for (int i = 0; i < N; ++i) {
            auto& s = shots[i]; auto& d = D[i];
            double lmin = 1; for (int b = 0; b < 48; ++b) lmin = std::min(lmin, d.line_corr[b]);
            cv::Point2d p(NAN, NAN);
            if (err[i].empty() && d.qr_ok && d.lines_ok == 48) p = model.predict(d.values);
            o << s.idx << "," << s.set << "," << s.rel << "," << s.y << "," << s.t << "," << s.group << "," << (err[i].empty() ? "" : err[i]) << "," << d.pad << "," << d.qr_ok;
            for (double v : d.ncc) o << "," << v; for (double v : d.ecc) o << "," << v; for (double v : d.refine_px) o << "," << v;
            o << "," << d.lines_ok << "," << lmin << "," << p.x << "," << p.y << "," << p.x - s.y << "," << p.y - s.t << "," << ms[i] << "\n";
            lc << s.idx; for (int b = 0; b < 48; ++b) lc << "," << (d.values.empty() ? NAN : d.values.at<double>(54 + b)); for (int b = 0; b < 48; ++b) lc << "," << d.line_corr[b]; for (int b = 0; b < 48; ++b) lc << "," << d.line_valid[b]; lc << "\n";
            if (!d.values.empty()) for (int j = 0; j < 102; ++j) cache[size_t(i) * 102 + j] = d.values.at<double>(j);
        }
        std::ofstream cb(res / "cache" / "obs102.f64", std::ios::binary); cb.write(reinterpret_cast<const char*>(cache.data()), cache.size() * sizeof(double));
        std::cout << "extract done\n";
    }
    if (mode == "all" || mode == "perturb") {
        std::vector<int> pick; for (int i = 0; i < N; i += 19) pick.push_back(i);
        struct P { std::string name; std::function<cv::Mat(const cv::Mat&)> f; double qx, ly; };
        cv::RNG rng(7);
        std::vector<P> ps = {
            {"이동 x+0.25", [](const cv::Mat& g) { return fft_shift(g, 0.25, 0); }, 0.25, 0},
            {"이동 x+0.50", [](const cv::Mat& g) { return fft_shift(g, 0.5, 0); }, 0.5, 0},
            {"이동 y+0.25", [](const cv::Mat& g) { return fft_shift(g, 0, 0.25); }, 0, 0.25},
            {"이동 y+0.50", [](const cv::Mat& g) { return fft_shift(g, 0, 0.5); }, 0, 0.5},
            {"밝기 x0.8", [](const cv::Mat& g) { cv::Mat o; g.convertTo(o, CV_8U, 0.8); return o; }, 0, 0},
            {"밝기 x1.2 -20", [](const cv::Mat& g) { cv::Mat o; g.convertTo(o, CV_8U, 1.2, -20); return o; }, 0, 0},
            {"흐림 σ0.7", [](const cv::Mat& g) { cv::Mat o; cv::GaussianBlur(g, o, {0, 0}, 0.7); return o; }, 0, 0},
            {"흐림 σ1.2", [](const cv::Mat& g) { cv::Mat o; cv::GaussianBlur(g, o, {0, 0}, 1.2); return o; }, 0, 0},
            {"잡음 σ2", [&](const cv::Mat& g) { cv::Mat n(g.size(), CV_32F); rng.fill(n, cv::RNG::NORMAL, 0, 2); cv::Mat f; g.convertTo(f, CV_32F); f += n; cv::Mat o; f.convertTo(o, CV_8U); return o; }, 0, 0},
            {"잡음 σ4", [&](const cv::Mat& g) { cv::Mat n(g.size(), CV_32F); rng.fill(n, cv::RNG::NORMAL, 0, 4); cv::Mat f; g.convertTo(f, CV_32F); f += n; cv::Mat o; f.convertTo(o, CV_8U); return o; }, 0, 0},
            {"조명 기울기 ±10%", [](const cv::Mat& g) { cv::Mat f; g.convertTo(f, CV_32F); for (int y = 0; y < f.rows; ++y) { auto* r = f.ptr<float>(y); for (int x = 0; x < f.cols; ++x) r[x] *= float(0.9 + 0.2 * x / f.cols); } cv::Mat o; f.convertTo(o, CV_8U); return o; }, 0, 0},
        };
        std::ofstream o(res / "results" / "perturb.csv");
        o << "idx,rel,교란,측정실패,lines_ok,QR x 변화 평균,QR x 기대,QR x 오차 최대,QR y 변화 평균,선 변화 평균,선 기대,선 오차 최대,ΔY,ΔT\n";
        std::mutex mu; std::atomic<int> next{0};
        auto work = [&]() { for (int k; (k = next++) < int(pick.size());) {
            int i = pick[k]; cv::Mat g = cv::imread(path_of(shots[i]).u8string(), cv::IMREAD_GRAYSCALE);
            DetailedObservation b0; cv::Point2d p0;
            try { b0 = obs.measure_detailed(fft_shift(g, 0, 0)); p0 = model.predict(b0.values); } catch (...) { continue; }  // 기준도 같은 FFT 왕복을 거친 영상
            for (auto& p : ps) {
                std::ostringstream line; line << i << "," << shots[i].rel << "," << p.name;
                try { auto d = obs.measure_detailed(p.f(g)); bool fail = !d.qr_ok || d.lines_ok < 48; cv::Point2d q(NAN, NAN); if (!fail) q = model.predict(d.values);
                    double qx = 0, qy = 0, qxe = 0, ls = 0, lse = 0;
                    for (int j = 0; j < 54; j += 2) { double vx = d.values.at<double>(j) - b0.values.at<double>(j), vy = d.values.at<double>(j + 1) - b0.values.at<double>(j + 1); qx += vx / 27; qy += vy / 27; qxe = std::max(qxe, std::abs(vx - p.qx)); }
                    for (int j = 54; j < 102; ++j) { double v = d.values.at<double>(j) - b0.values.at<double>(j); ls += v / 48; lse = std::max(lse, std::abs(v - p.ly)); }
                    line << "," << fail << "," << d.lines_ok << "," << qx << "," << p.qx << "," << qxe << "," << qy << "," << ls << "," << p.ly << "," << lse << "," << q.x - p0.x << "," << q.y - p0.y; }
                catch (const std::exception& e) { line << ",1,,,,,,,,,,"; }
                std::lock_guard<std::mutex> lk(mu); o << line.str() << "\n";
            } } };
        std::vector<std::thread> th; for (int k = 0; k < 6; ++k) th.emplace_back(work); for (auto& t : th) t.join();
        std::cout << "perturb done " << pick.size() << "\n";
    }
}
