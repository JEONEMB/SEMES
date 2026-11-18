// 사진 목록을 받아 고정 630점 이동(dx, dy)을 저장한다. A0 캐시(disp.f32)와 같은 측정기·같은 형식이다.
// 사용: measure630 <control assets 폴더> <사진 목록.txt (UTF-8, 한 줄에 경로 하나)> <출력 폴더>
// 출력: disp.f32 (n × 630 × 2 float32), shots.csv (idx, path, valid_count, valid_selected, ncc0..2, measured)
// 기존 출력이 있으면 덮어쓰지 않고 멈춘다.
#include "control_internal.hpp"
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
namespace fs = std::filesystem;
static cv::Mat imread_gray(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    std::vector<unsigned char> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return b.empty() ? cv::Mat() : cv::imdecode(b, cv::IMREAD_GRAYSCALE);
}
int main(int argc, char** argv) {
    try {
        if (argc != 4) { std::cerr << "usage: measure630 ASSETS LIST OUTDIR\n"; return 2; }
        const fs::path assets = fs::u8path(argv[1]), list = fs::u8path(argv[2]), out = fs::u8path(argv[3]);
        if (fs::exists(out / "disp.f32")) { std::cerr << "출력이 이미 있다: " << (out / "disp.f32").u8string() << "\n"; return 3; }
        fs::create_directories(out);
        std::vector<std::string> paths; { std::ifstream f(list); std::string s; while (std::getline(f, s)) { if (!s.empty() && s.back() == '\r') s.pop_back(); if (!s.empty()) paths.push_back(s); } }
        legacy_control::Tracker tracker(assets);
        const int n = int(paths.size()), kPoints = 630;
        std::vector<float> disp(size_t(n) * kPoints * 2, 0);
        std::vector<int> vc(n, 0), vs(n, 0), ok(n, 0); std::vector<std::array<double, 3>> ncc(n);
        cv::setNumThreads(1);
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < n; ++i) {
            try {
                cv::Mat img = imread_gray(fs::u8path(paths[i]));
                if (img.empty()) throw std::runtime_error("read");
                auto tr = tracker.measure(img);
                for (int p = 0; p < kPoints; ++p) {
                    disp[size_t(i) * kPoints * 2 + p * 2] = tr.displacement.at<float>(p, 0);
                    disp[size_t(i) * kPoints * 2 + p * 2 + 1] = tr.displacement.at<float>(p, 1);
                }
                vc[i] = int(std::count(tr.valid.begin(), tr.valid.end(), 1)); vs[i] = tracker.valid_selected(tr); ncc[i] = tr.ncc; ok[i] = 1;
            } catch (const std::exception&) { ok[i] = 0; }
        }
        { std::ofstream f(out / "disp.f32", std::ios::binary); f.write(reinterpret_cast<const char*>(disp.data()), std::streamsize(disp.size() * sizeof(float))); }
        std::ofstream c(out / "shots.csv"); c << "idx,path,valid_count,valid_selected,ncc0,ncc1,ncc2,measured\n";
        int good = 0;
        for (int i = 0; i < n; ++i) { c << i << ",\"" << paths[i] << "\"," << vc[i] << "," << vs[i] << "," << ncc[i][0] << "," << ncc[i][1] << "," << ncc[i][2] << "," << ok[i] << "\n"; good += ok[i]; }
        std::cout << "measured " << good << "/" << n << "\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << "error: " << e.what() << "\n"; return 1; }
}
