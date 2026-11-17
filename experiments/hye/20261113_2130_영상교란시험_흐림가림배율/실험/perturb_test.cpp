// 영상 교란 시험: 사진에 미리 정한 교란(계획.md)을 넣고 지금 모듈(Estimator, 절대 영점)의 출력 변화를 잰다.
// 사용: perturb_test <assets> <photos.tsv> <labels.tsv> <출력.tsv> [method=ncc|track536]
//   photos.tsv: id<TAB>경로 (머리줄 없음), labels.tsv: id source Y_cmd T_cmd Y_abs T_abs (머리줄 있음)
//   출력: 사진 × 교란마다 한 줄 (id, 교란, 멈춤 여부, Y, T, 라벨, 메시지)
#include "final00.hpp"
#include "site.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utility.hpp>
#include <filesystem>
#include <fstream>
#include <functional>
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
        if (first && line.size() >= 3 && (unsigned char)line[0] == 0xEF) line = line.substr(3);
        if (first && header) { first = false; continue; }
        first = false;
        if (line.empty()) continue;
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

using Op = std::function<cv::Mat(const cv::Mat&, int)>;   // (사진, 사진 번호) → 교란 사진
struct Perturb { std::string name; Op op; };

cv::Mat affine(const cv::Mat& g, const cv::Mat& m) { cv::Mat o; cv::warpAffine(g, o, m, g.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE); return o; }
cv::Mat shift(const cv::Mat& g, double dx, double dy) { return affine(g, (cv::Mat_<double>(2, 3) << 1, 0, dx, 0, 1, dy)); }
cv::Mat about_center(const cv::Mat& g, double angle_deg, double scale) { return affine(g, cv::getRotationMatrix2D(cv::Point2f(g.cols / 2.f, g.rows / 2.f), angle_deg, scale)); }
cv::Mat lut(const cv::Mat& g, const std::function<double(double)>& f) {
    cv::Mat t(1, 256, CV_8U); for (int i = 0; i < 256; ++i) t.at<uchar>(i) = cv::saturate_cast<uchar>(f(i));
    cv::Mat o; cv::LUT(g, t, o); return o;
}

std::vector<Perturb> make_perturbs(const site::Geometry& geo) {
    std::vector<Perturb> p;
    p.push_back({"none", [](const cv::Mat& g, int) { return g; }});
    for (double s : {0.5, 1.0, 2.0, 3.0, 5.0}) p.push_back({"blur_s" + std::to_string(s).substr(0, 3), [s](const cv::Mat& g, int) { cv::Mat o; cv::GaussianBlur(g, o, cv::Size(0, 0), s); return o; }});
    for (int n : {5, 9}) p.push_back({"motion_x" + std::to_string(n), [n](const cv::Mat& g, int) { cv::Mat k = cv::Mat::ones(1, n, CV_32F) / float(n), o; cv::filter2D(g, o, -1, k, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE); return o; }});
    for (double a : {0.6, 1.4}) p.push_back({"gain" + std::to_string(a).substr(0, 3), [a](const cv::Mat& g, int) { return lut(g, [a](double v) { return v * a; }); }});
    for (int b : {-40, 40}) p.push_back({"offset" + std::to_string(b), [b](const cv::Mat& g, int) { return lut(g, [b](double v) { return v + b; }); }});
    for (double gm : {0.7, 1.4}) p.push_back({"gamma" + std::to_string(gm).substr(0, 3), [gm](const cv::Mat& g, int) { return lut(g, [gm](double v) { return 255.0 * std::pow(v / 255.0, gm); }); }});
    for (double s : {3.0, 8.0}) p.push_back({"noise" + std::to_string(int(s)), [s](const cv::Mat& g, int idx) {
        cv::RNG rng(uint64(12345 + idx)); cv::Mat n(g.size(), CV_32F); rng.fill(n, cv::RNG::NORMAL, 0, s);
        cv::Mat f; g.convertTo(f, CV_32F); f += n; cv::Mat o; f.convertTo(o, CV_8U); return o; }});
    for (double d : {0.5, 2.0}) p.push_back({"shift_x" + std::to_string(d).substr(0, 3), [d](const cv::Mat& g, int) { return shift(g, d, 0); }});
    for (double d : {10.0, 25.0}) p.push_back({"shift_y" + std::to_string(int(d)), [d](const cv::Mat& g, int) { return shift(g, 0, d); }});
    for (double s : {0.998, 1.002, 1.005}) p.push_back({"scale" + std::to_string(s).substr(0, 5), [s](const cv::Mat& g, int) { return about_center(g, 0, s); }});
    for (double a : {0.05, 0.2}) p.push_back({"rotate" + std::to_string(a).substr(0, 4), [a](const cv::Mat& g, int) { return about_center(g, a, 1); }});
    const cv::Rect q = geo.qr[0];
    p.push_back({"qr1_cover_quarter", [q](const cv::Mat& g, int) { cv::Mat o = g.clone(); o(cv::Rect(q.x, q.y, q.width / 2, q.height / 2)).setTo(128); return o; }});
    p.push_back({"qr1_cover_half", [q](const cv::Mat& g, int) { cv::Mat o = g.clone(); o(cv::Rect(q.x, q.y, q.width / 2, q.height)).setTo(128); return o; }});
    p.push_back({"qr1_glare", [q](const cv::Mat& g, int) { cv::Mat o = g.clone(); cv::circle(o, (q.tl() + q.br()) / 2, 30, cv::Scalar(255), -1); return o; }});
    const cv::Point s1 = geo.screws[0];
    p.push_back({"screw1_cover_half", [s1](const cv::Mat& g, int) { cv::Mat o = g.clone(); o(cv::Rect(s1.x - 60, s1.y - 60, 60, 120)).setTo(128); return o; }});
    p.push_back({"screw1_glare", [s1](const cv::Mat& g, int) { cv::Mat o = g.clone(); cv::circle(o, s1, 20, cv::Scalar(255), -1); return o; }});
    // 판–나사 밀림: QR 쪽 영역은 원래대로, 나머지 화면만 가로 이동
    cv::Rect keep = geo.full_crop;
    for (int k = 0; k < 3; ++k) keep |= cv::Rect(geo.qr[k].x - geo.search_x - 8, geo.qr[k].y - geo.search_y - 8, geo.qr[k].width + 2 * (geo.search_x + 8), geo.qr[k].height + 2 * (geo.search_y + 8));
    keep &= cv::Rect(0, 0, 3840, 2160);
    for (double d : {0.1, 0.3}) p.push_back({"plate_screw" + std::to_string(d).substr(0, 3), [d, keep](const cv::Mat& g, int) { cv::Mat o = shift(g, d, 0); g(keep).copyTo(o(keep)); return o; }});
    return p;
}
}

int main(int argc, char** argv) {
    try {
        std::string method; std::vector<std::string> a;
        for (int i = 1; i < argc; ++i) { std::string s(argv[i]); if (s.rfind("method=", 0) == 0) method = s.substr(7); else a.push_back(s); }
        if (a.size() != 4) { std::cerr << "perturb_test <assets> <photos.tsv> <labels.tsv> <출력.tsv> [method=...]\n"; return 2; }
        const fs::path out = fs::u8path(a[3]);
        if (fs::exists(out)) { std::cerr << "출력이 이미 있다\n"; return 3; }
        final00::Estimator est(a[0], method);
        const site::Geometry geo = site::get();
        const auto perturbs = make_perturbs(geo);
        std::map<int, std::pair<double, double>> lab;
        for (auto& r : read_table(fs::u8path(a[2]), true)) lab[std::stoi(r.at(0))] = {std::stod(r.at(4)), std::stod(r.at(5))};
        const auto rows = read_table(fs::u8path(a[1]), false);
        const int n = int(rows.size()), m = int(perturbs.size());
        std::cout << "method " << est.qr_method() << " model " << est.model_id() << " photos " << n << " perturbations " << m << std::endl;
        std::vector<std::string> lines(size_t(n) * m);
        cv::setNumThreads(1);
        int done = 0;
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < n; ++i) {
            const int id = std::stoi(rows[i].at(0)); const auto l = lab.at(id);
            const cv::Mat gray = read_image(fs::u8path(rows[i].at(1)));
            for (int k = 0; k < m; ++k) {
                std::ostringstream o; o << std::setprecision(10) << id << '\t' << perturbs[k].name << '\t';
                try {
                    if (gray.empty()) throw std::runtime_error("사진을 읽지 못했다");
                    const final00::Result r = est.predict(perturbs[k].op(gray, i));
                    o << "1\t" << r.pose.y << '\t' << r.pose.t << '\t' << l.first << '\t' << l.second << '\t' << r.health << '\t' << r.health_threshold << '\t' << r.health_worst << "\t";
                } catch (const std::exception& e) {
                    std::string msg = e.what(); for (char& c : msg) if (c == '\t' || c == '\n') c = ' ';
                    o << "0\tnan\tnan\t" << l.first << '\t' << l.second << "\tnan\tnan\t-1\t" << msg;
                }
                lines[size_t(i) * m + k] = o.str();
            }
#pragma omp critical
            { if (++done % 100 == 0) std::cout << done << "/" << n << std::endl; }
        }
        std::ofstream f(out, std::ios::binary);
        f << "id\tperturb\tvalid\tY\tT\tY_abs\tT_abs\thealth\thealth_threshold\thealth_worst\tmessage\n";   // health: 측정값 일치 점검값(2026-11-13 밤 추가)
        for (auto& s : lines) f << s << '\n';
        std::cout << "written " << out.u8string() << std::endl;
        return 0;
    } catch (const std::exception& e) { std::cerr << "오류: " << e.what() << "\n"; return 1; }
}
