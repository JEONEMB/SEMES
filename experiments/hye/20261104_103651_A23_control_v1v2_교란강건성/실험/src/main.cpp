// A23: control v1 과 v2 에 같은 교란을 넣어 Y 출력 변화(ΔY)를 비교한다.
//  대상: 2차 운용 영역(남은 T 작음)의 사진. 1014 Y 단독 80장 중 매 4번째 + center, 1020 의 |T|<=0.03 사진 (총 약 40장).
//  교란: FFT 정확 이동(가로·세로 0.25/0.5 px = 카메라 이동), 밝기, 흐림, 잡음, 조명 기울기 (A19·A21 과 같음).
#include "alignment.hpp"
#include <opencv2/opencv.hpp>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <mutex>
#include <sstream>
namespace fs = std::filesystem;
static cv::Mat fft_shift(const cv::Mat& g, double dx, double dy) {
    cv::Mat f; g.convertTo(f, CV_32F); cv::Mat F; cv::dft(f, F, cv::DFT_COMPLEX_OUTPUT);
    for (int v = 0; v < F.rows; ++v) { double fy = (v <= F.rows / 2 ? v : v - F.rows) / double(F.rows); auto* row = F.ptr<cv::Vec2f>(v);
        for (int u = 0; u < F.cols; ++u) { double fx = (u <= F.cols / 2 ? u : u - F.cols) / double(F.cols); double ph = -2 * CV_PI * (fx * dx + fy * dy), c = std::cos(ph), s = std::sin(ph); auto z = row[u];
            row[u] = cv::Vec2f(float(z[0] * c - z[1] * s), float(z[0] * s + z[1] * c)); } }
    cv::Mat out; cv::idft(F, out, cv::DFT_REAL_OUTPUT | cv::DFT_SCALE); cv::Mat u8; out.convertTo(u8, CV_8U); return u8;
}
int main() {
    const fs::path repo = fs::u8path(REPO_DIR), res = fs::u8path(EXPERIMENT_DIR) / "results"; fs::create_directories(res);
    initialize_control(ASSETS_DIR);
    std::vector<std::pair<std::string, std::string>> pick;
    { std::ifstream in(repo / fs::u8path(u8"experiments/hye/00000_final_v2_Y_T control/assets/training_list.csv")); std::string l; std::getline(in, l); int k = 0;
      while (std::getline(in, l)) { std::stringstream ss(l); std::string set, rel, y, t, g, cond; std::getline(ss, set, ','); std::getline(ss, rel, ','); std::getline(ss, y, ','); std::getline(ss, t, ','); std::getline(ss, g, ','); std::getline(ss, cond, ',');
        bool take = (set.find("1014") != std::string::npos && (cond == "center" || (cond == "Y" && (k++ % 4 == 0)))) || (set.find("1020") != std::string::npos && cond != "center" && std::abs(std::stod(t)) <= 0.03 + 1e-9);
        if (take) pick.push_back({set, rel}); } }
    cv::RNG rng(7);
    std::vector<std::pair<std::string, std::function<cv::Mat(const cv::Mat&)>>> ps = {
        {"없음 (FFT 왕복)", [](const cv::Mat& g) { return fft_shift(g, 0, 0); }},
        {"이동 x+0.25", [](const cv::Mat& g) { return fft_shift(g, 0.25, 0); }}, {"이동 x+0.50", [](const cv::Mat& g) { return fft_shift(g, 0.5, 0); }},
        {"이동 y+0.25", [](const cv::Mat& g) { return fft_shift(g, 0, 0.25); }}, {"이동 y+0.50", [](const cv::Mat& g) { return fft_shift(g, 0, 0.5); }},
        {"밝기 x0.8", [](const cv::Mat& g) { cv::Mat o; g.convertTo(o, CV_8U, 0.8); return o; }}, {"밝기 x1.2 -20", [](const cv::Mat& g) { cv::Mat o; g.convertTo(o, CV_8U, 1.2, -20); return o; }},
        {"흐림 σ0.7", [](const cv::Mat& g) { cv::Mat o; cv::GaussianBlur(g, o, {0, 0}, 0.7); return o; }}, {"흐림 σ1.2", [](const cv::Mat& g) { cv::Mat o; cv::GaussianBlur(g, o, {0, 0}, 1.2); return o; }},
        {"잡음 σ2", [&](const cv::Mat& g) { cv::Mat n(g.size(), CV_32F); rng.fill(n, cv::RNG::NORMAL, 0, 2); cv::Mat f; g.convertTo(f, CV_32F); f += n; cv::Mat o; f.convertTo(o, CV_8U); return o; }},
        {"잡음 σ4", [&](const cv::Mat& g) { cv::Mat n(g.size(), CV_32F); rng.fill(n, cv::RNG::NORMAL, 0, 4); cv::Mat f; g.convertTo(f, CV_32F); f += n; cv::Mat o; f.convertTo(o, CV_8U); return o; }},
        {"조명 기울기 ±10%", [](const cv::Mat& g) { cv::Mat f; g.convertTo(f, CV_32F); for (int y = 0; y < f.rows; ++y) { auto* r = f.ptr<float>(y); for (int x = 0; x < f.cols; ++x) r[x] *= float(0.9 + 0.2 * x / f.cols); } cv::Mat o; f.convertTo(o, CV_8U); return o; }},
    };
    std::ofstream o(res / (std::string("perturb_") + VERSION_NAME + ".csv")); o.precision(8); o << "set,rel,교란,y,t,error\n";
    for (auto& [set, rel] : pick) { cv::Mat g = cv::imread((repo / "data" / "clean" / fs::u8path(set) / fs::u8path(rel)).u8string(), cv::IMREAD_GRAYSCALE);
        for (auto& [name, f] : ps) { cv::Mat im = f(g); try { auto q = control(im); o << set << "," << rel << "," << name << "," << q.y << "," << q.t << ",\n"; } catch (const std::exception& e) { o << set << "," << rel << "," << name << ",,," << e.what() << "\n"; } } }
    std::cout << VERSION_NAME << " done " << pick.size() << "\n";
}
