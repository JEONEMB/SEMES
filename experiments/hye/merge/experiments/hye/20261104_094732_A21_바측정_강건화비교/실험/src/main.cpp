// A21: 위·아래 바(가로선) 측정 강건화 비교.
//  목적: joint T 를 카메라 이동·밝기·잡음·조명에 강건하게 하려면 바의 "기울기"를 안정적으로 재야 한다(A19, A20).
//  후보
//   S0 지금 방식: 블록 48개(64px 폭) 프로파일 정합 → 바마다 24블록 보통 최소제곱 직선의 기울기
//   S1 같은 블록 + Huber 강건 직선 적합
//   S2 열 단위 에지 부화소(8px 띠 평균 → 미분 → 극값 3점 포물선) + Huber 직선 (바마다 띠 192개)
//   S3 바 구간 4개(768x49)를 ECC 세로 이동으로 정합 → 좌·우 구간 이동으로 기울기 (joint 와 같은 호환 ECC, 마스크 지정)
//   S4 지금 정합기 그대로, 블록을 16px 좁은 띠 4개로 나눠 바마다 96개 + Huber
//  검증: (1) 교란 11종 × 31장에서 기울기 차 d 의 변화를 T 로 환산, (2) 575장 전수에서 [s, d] 2특징 모델의 T 정확도
//        (1014만 학습 → 1020·1029, 날짜 층화 5-fold 10분할, 목표 근처), (3) 처리 시간.
#include "observer_internal.hpp"
#include "ecc_compat.hpp"
#include <opencv2/opencv.hpp>
#include <atomic>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <sstream>
#include <thread>
namespace fs = std::filesystem;
using namespace alignment_detail;

struct Shot { int idx; std::string set, rel, cond; double y, t; int group; };
static std::vector<std::string> split(const std::string& s) { std::vector<std::string> r; std::string c; std::stringstream ss(s); while (std::getline(ss, c, ',')) r.push_back(c); return r; }

struct Seg { int x0, x1, y0, y1, pol; bool upper; };
// line_protocol.json 의 구간과 블록 read_box 세로 범위(49줄)에 맞춘다.
static const Seg kSeg[4] = {{400, 1168, 686, 735, 1, true}, {2700, 3468, 697, 746, 1, true}, {400, 1168, 1761, 1810, -1, false}, {2600, 3368, 1770, 1819, -1, false}};
static const double kXc = 1934.0;

// Huber IRLS 직선 y = a + b*(x-kXc)/1000
static std::pair<double, double> fit_line(const std::vector<double>& x, const std::vector<double>& y, bool huber) {
    std::vector<double> w(x.size(), 1.0); double a = 0, b = 0;
    for (int it = 0; it < (huber ? 20 : 1); ++it) {
        double sw = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
        for (size_t i = 0; i < x.size(); ++i) { double u = (x[i] - kXc) / 1000; sw += w[i]; sx += w[i] * u; sy += w[i] * y[i]; sxx += w[i] * u * u; sxy += w[i] * u * y[i]; }
        double den = sw * sxx - sx * sx; b = (sw * sxy - sx * sy) / den; a = (sy - b * sx) / sw;
        if (!huber) break;
        std::vector<double> r(x.size()); for (size_t i = 0; i < x.size(); ++i) r[i] = std::abs(y[i] - a - b * (x[i] - kXc) / 1000);
        std::vector<double> rs = r; std::nth_element(rs.begin(), rs.begin() + rs.size() / 2, rs.end()); double s = std::max(1.4826 * rs[rs.size() / 2], 1e-4), c = 1.345 * s;
        for (size_t i = 0; i < x.size(); ++i) w[i] = r[i] <= c ? 1.0 : c / r[i];
    }
    return {a, b};
}

// 열 띠별 에지 위치: 8px 폭 평균 세로 프로파일 → 3점 미분 → 극성 맞춘 최대 → 포물선 부화소
static std::vector<double> strip_edges(const cv::Mat& g, const Seg& s, int width, std::vector<double>* xs) {
    std::vector<double> out;
    for (int x = s.x0; x + width <= s.x1; x += width) {
        int H = s.y1 - s.y0; std::vector<double> p(H, 0);
        for (int y = 0; y < H; ++y) { auto r = g.ptr<unsigned char>(s.y0 + y); for (int k = 0; k < width; ++k) p[y] += r[x + k]; p[y] /= width; }
        std::vector<double> q(H, 0); for (int y = 1; y < H; ++y) q[y] = p[y] - p[y - 1];   // 미분 원본
        std::vector<double> d(H, 0); for (int y = 2; y < H - 2; ++y) d[y] = s.pol * (q[y - 1] * 0.25 + q[y] * 0.5 + q[y + 1] * 0.25);
        int m = 3; for (int y = 3; y < H - 3; ++y) if (d[y] > d[m]) m = y;
        double den = d[m - 1] - 2 * d[m] + d[m + 1], off = std::abs(den) > 1e-9 ? 0.5 * (d[m - 1] - d[m + 1]) / den : 0;
        out.push_back(m - 0.5 + off);   // q[y] 는 y-1 과 y 사이
        if (xs) xs->push_back(x + width / 2.0);
    }
    return out;
}

struct Bars { double d[5]; double ms[5]; };   // 방법별 기울기 차(위−아래), 시간

struct Ref { std::vector<double> prof4[192]; std::vector<cv::Rect> box4; std::vector<double> bx4; std::vector<double> prof[48]; std::vector<cv::Rect> box; std::vector<double> bx; std::vector<double> edge[4], ex[4]; cv::Mat band[4]; };

static Bars measure_bars(const cv::Mat& g, const Ref& R) {
    Bars b{};
    // S0, S1
    auto t0 = cv::getTickCount();
    std::vector<double> sh(48); for (int k = 0; k < 48; ++k) sh[k] = match_line(R.prof[k], line_profile(g, R.box[k])).shift;
    double t_blocks = (cv::getTickCount() - t0) * 1000.0 / cv::getTickFrequency();
    for (int h = 0; h < 2; ++h) { double sl[2];
        for (int bar = 0; bar < 2; ++bar) { std::vector<double> x(R.bx.begin() + bar * 24, R.bx.begin() + bar * 24 + 24), y(sh.begin() + bar * 24, sh.begin() + bar * 24 + 24); sl[bar] = fit_line(x, y, h == 1).second; }
        b.d[h] = sl[0] - sl[1]; b.ms[h] = t_blocks; }
    // S2
    t0 = cv::getTickCount(); double sl[2];
    for (int bar = 0; bar < 2; ++bar) { std::vector<double> x, y;
        for (int s = bar * 2; s < bar * 2 + 2; ++s) { std::vector<double> xs; auto e = strip_edges(g, kSeg[s], 8, &xs); for (size_t i = 0; i < e.size(); ++i) { x.push_back(xs[i]); y.push_back(e[i] - R.edge[s][i]); } }
        sl[bar] = fit_line(x, y, true).second; }
    b.d[2] = sl[0] - sl[1]; b.ms[2] = (cv::getTickCount() - t0) * 1000.0 / cv::getTickFrequency();
    // S3: ECC 세로 이동 (구간마다), 기울기 = (오른쪽 - 왼쪽) / 중심 거리
    t0 = cv::getTickCount(); double ty[4];
    for (int s = 0; s < 4; ++s) {
        cv::Rect r(kSeg[s].x0, kSeg[s].y0 - 6, kSeg[s].x1 - kSeg[s].x0, kSeg[s].y1 - kSeg[s].y0 + 12);
        cv::Mat cur; g(r).convertTo(cur, CV_32F);
        cv::Mat w = cv::Mat::eye(2, 3, CV_32F); w.at<float>(1, 2) = 6;   // band 는 위아래 6줄 안쪽
        try { semes::ecc5::findTransformECCWithMask(R.band[s], cur, cv::Mat(R.band[s].size(), CV_8U, cv::Scalar(255)), cv::Mat(cur.size(), CV_8U, cv::Scalar(255)), w, cv::MOTION_TRANSLATION, cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 100, 1e-6), 5); ty[s] = w.at<float>(1, 2) - 6; }
        catch (...) { ty[s] = NAN; }
    }
    auto slope = [&](int l, int r) { double xl = (kSeg[l].x0 + kSeg[l].x1) / 2.0, xr = (kSeg[r].x0 + kSeg[r].x1) / 2.0; return (ty[r] - ty[l]) / ((xr - xl) / 1000); };
    b.d[3] = slope(0, 1) - slope(2, 3); b.ms[3] = (cv::getTickCount() - t0) * 1000.0 / cv::getTickFrequency();
    // S4
    t0 = cv::getTickCount();
    for (int bar = 0; bar < 2; ++bar) { std::vector<double> x, y;
        for (int k = bar * 96; k < bar * 96 + 96; ++k) { auto f = match_line(R.prof4[k], line_profile(g, R.box4[k])); if (!f.valid) continue; x.push_back(R.bx4[k]); y.push_back(f.shift); }
        sl[bar] = fit_line(x, y, true).second; }
    b.d[4] = sl[0] - sl[1]; b.ms[4] = (cv::getTickCount() - t0) * 1000.0 / cv::getTickFrequency();
    return b;
}

static cv::Mat fft_shift(const cv::Mat& g, double dx, double dy) {
    cv::Mat f; g.convertTo(f, CV_32F); cv::Mat F; cv::dft(f, F, cv::DFT_COMPLEX_OUTPUT);
    for (int v = 0; v < F.rows; ++v) { double fy = (v <= F.rows / 2 ? v : v - F.rows) / double(F.rows); auto* row = F.ptr<cv::Vec2f>(v);
        for (int u = 0; u < F.cols; ++u) { double fx = (u <= F.cols / 2 ? u : u - F.cols) / double(F.cols); double ph = -2 * CV_PI * (fx * dx + fy * dy), c = std::cos(ph), s = std::sin(ph); auto z = row[u];
            row[u] = cv::Vec2f(float(z[0] * c - z[1] * s), float(z[0] * s + z[1] * c)); } }
    cv::Mat out; cv::idft(F, out, cv::DFT_REAL_OUTPUT | cv::DFT_SCALE); cv::Mat u8; out.convertTo(u8, CV_8U); return u8;
}

int main(int argc, char** argv) {
    const fs::path repo = fs::u8path(REPO_DIR), res = fs::u8path(EXPERIMENT_DIR) / "results";
    fs::create_directories(res);
    const fs::path assets = repo / fs::u8path(u8"experiments/hye/20261028_1258_final_Y_T_joint/assets");
    Observer obs(assets);
    cv::Mat center = cv::imread((assets / "center.bmp").u8string(), cv::IMREAD_GRAYSCALE);
    Ref R;
    { cv::FileStorage p((assets / "line_protocol.json").u8string(), cv::FileStorage::READ);
      for (auto it = p["blocks"].begin(); it != p["blocks"].end(); ++it) { auto b = (*it)["read_box"]; cv::Rect r(int(b[0]), int(b[1]), int(b[2]) - int(b[0]), int(b[3]) - int(b[1])); R.box.push_back(r); R.bx.push_back(r.x + r.width / 2.0); } }
    for (int k = 0; k < 48; ++k) R.prof[k] = line_profile(center, R.box[k]);
    for (int k = 0; k < 48; ++k) for (int q = 0; q < 4; ++q) { cv::Rect r(R.box[k].x + q * 16, R.box[k].y, 16, R.box[k].height); R.box4.push_back(r); R.bx4.push_back(r.x + 8.0); R.prof4[k * 4 + q] = line_profile(center, r); }
    for (int s = 0; s < 4; ++s) { R.edge[s] = strip_edges(center, kSeg[s], 8, &R.ex[s]); center(cv::Rect(kSeg[s].x0, kSeg[s].y0, kSeg[s].x1 - kSeg[s].x0, kSeg[s].y1 - kSeg[s].y0)).convertTo(R.band[s], CV_32F); }
    std::vector<Shot> shots;
    { std::ifstream in(repo / fs::u8path(u8"experiments/hye/20261102_124309_A0_분할측정고정_계층분할과점측정캐시/결과/cache/shots.csv")); std::string l; std::getline(in, l);
      while (std::getline(in, l)) { auto r = split(l); shots.push_back({std::stoi(r[0]), r[1], r[2], r[6], std::stod(r[3]), std::stod(r[4]), std::stoi(r[7])}); } }
    const int N = int(shots.size());
    auto path_of = [&](const Shot& s) { return repo / "data" / "clean" / fs::u8path(s.set) / fs::u8path(s.rel); };
    const char* names[5] = {"S0 블록 + 최소제곱 (지금)", "S1 블록 + Huber", "S2 열 에지 8px + Huber", "S3 구간 ECC 이동", "S4 16px 띠 192개 + Huber"};
    std::string mode = argc > 1 ? argv[1] : "all";
    // QR 평균 가로 이동 s (A19 캐시, A0 순서)
    std::vector<double> sx(N, NAN);
    { std::vector<double> raw(size_t(N) * 102); std::ifstream in(repo / fs::u8path(u8"experiments/hye/20261104_093558_A19_QR선측정_안정성점검/결과/cache/obs102.f64"), std::ios::binary); in.read(reinterpret_cast<char*>(raw.data()), raw.size() * 8);
      std::vector<double> c0(102); for (int j = 0; j < 102; ++j) c0[j] = raw[j];  // idx0 = 1014 center
      for (int i = 0; i < N; ++i) { double s = 0; for (int j = 0; j < 54; j += 2) s += (raw[size_t(i) * 102 + j] - c0[j]) / 27; sx[i] = s; } }

    if (mode == "all" || mode == "measure") {
        std::vector<Bars> B(N); std::atomic<int> next{0};
        auto work = [&]() { for (int i; (i = next++) < N;) { cv::Mat g = cv::imread(path_of(shots[i]).u8string(), cv::IMREAD_GRAYSCALE); B[i] = measure_bars(g, R); } };
        std::vector<std::thread> th; for (int k = 0; k < 6; ++k) th.emplace_back(work); for (auto& t : th) t.join();
        std::ofstream o(res / "bars_all.csv"); o << "idx,set,rel,cond,Y,T,group,s"; for (int m = 0; m < 5; ++m) o << ",d" << m; for (int m = 0; m < 5; ++m) o << ",ms" << m; o << "\n";
        for (int i = 0; i < N; ++i) { auto& s = shots[i]; o << s.idx << "," << s.set << "," << s.rel << "," << s.cond << "," << s.y << "," << s.t << "," << s.group << "," << sx[i]; for (int m = 0; m < 5; ++m) o << "," << B[i].d[m]; for (int m = 0; m < 5; ++m) o << "," << B[i].ms[m]; o << "\n"; }
        std::cout << "measure done\n";
    }
    if (mode == "all" || mode == "perturb") {
        std::vector<int> pick; for (int i = 0; i < N; i += 19) pick.push_back(i);
        cv::RNG rng(7);
        struct P { std::string name; std::function<cv::Mat(const cv::Mat&)> f; };
        std::vector<P> ps = {
            {"없음 (FFT 왕복)", [](const cv::Mat& g) { return fft_shift(g, 0, 0); }},
            {"이동 x+0.25", [](const cv::Mat& g) { return fft_shift(g, 0.25, 0); }}, {"이동 x+0.50", [](const cv::Mat& g) { return fft_shift(g, 0.5, 0); }},
            {"이동 y+0.25", [](const cv::Mat& g) { return fft_shift(g, 0, 0.25); }}, {"이동 y+0.50", [](const cv::Mat& g) { return fft_shift(g, 0, 0.5); }},
            {"밝기 x0.8", [](const cv::Mat& g) { cv::Mat o; g.convertTo(o, CV_8U, 0.8); return o; }}, {"밝기 x1.2 -20", [](const cv::Mat& g) { cv::Mat o; g.convertTo(o, CV_8U, 1.2, -20); return o; }},
            {"흐림 σ0.7", [](const cv::Mat& g) { cv::Mat o; cv::GaussianBlur(g, o, {0, 0}, 0.7); return o; }}, {"흐림 σ1.2", [](const cv::Mat& g) { cv::Mat o; cv::GaussianBlur(g, o, {0, 0}, 1.2); return o; }},
            {"잡음 σ2", [&](const cv::Mat& g) { cv::Mat n(g.size(), CV_32F); rng.fill(n, cv::RNG::NORMAL, 0, 2); cv::Mat f; g.convertTo(f, CV_32F); f += n; cv::Mat o; f.convertTo(o, CV_8U); return o; }},
            {"잡음 σ4", [&](const cv::Mat& g) { cv::Mat n(g.size(), CV_32F); rng.fill(n, cv::RNG::NORMAL, 0, 4); cv::Mat f; g.convertTo(f, CV_32F); f += n; cv::Mat o; f.convertTo(o, CV_8U); return o; }},
            {"조명 기울기 ±10%", [](const cv::Mat& g) { cv::Mat f; g.convertTo(f, CV_32F); for (int y = 0; y < f.rows; ++y) { auto* r = f.ptr<float>(y); for (int x = 0; x < f.cols; ++x) r[x] *= float(0.9 + 0.2 * x / f.cols); } cv::Mat o; f.convertTo(o, CV_8U); return o; }},
        };
        std::ofstream o(res / "bars_perturb.csv"); o << "idx,교란"; for (int m = 0; m < 5; ++m) o << ",d" << m; o << "\n";
        std::mutex mu; std::atomic<int> next{0};
        auto work = [&]() { for (int k; (k = next++) < int(pick.size());) { int i = pick[k]; cv::Mat g = cv::imread(path_of(shots[i]).u8string(), cv::IMREAD_GRAYSCALE);
            for (auto& p : ps) { auto b = measure_bars(p.f(g), R); std::lock_guard<std::mutex> lk(mu); o << i << "," << p.name; for (int m = 0; m < 5; ++m) o << "," << b.d[m]; o << "\n"; } } };
        std::vector<std::thread> th; for (int k = 0; k < 6; ++k) th.emplace_back(work); for (auto& t : th) t.join();
        std::cout << "perturb done\n";
    }

    if (mode == "all" || mode == "eval") {
        // bars_all.csv / bars_perturb.csv 를 읽어 방법별로 평가한다.
        struct Row { std::string set, cond; double y, t, s, d[5], ms[5]; int group; };
        std::vector<Row> rows;
        { std::ifstream in(res / "bars_all.csv"); std::string l; std::getline(in, l);
          while (std::getline(in, l)) { auto r = split(l); Row x; x.set = r[1]; x.cond = r[3]; x.y = std::stod(r[4]); x.t = std::stod(r[5]); x.group = std::stoi(r[6]); x.s = std::stod(r[7]);
            for (int m = 0; m < 5; ++m) { x.d[m] = std::stod(r[8 + m]); x.ms[m] = std::stod(r[13 + m]); } rows.push_back(x); } }
        std::vector<int> inr; for (int i = 0; i < int(rows.size()); ++i) if (std::abs(rows[i].t) <= 1.001) inr.push_back(i);
        auto fit = [&](const std::vector<int>& ids, int m) { cv::Mat X(int(ids.size()), 3, CV_64F), Y(int(ids.size()), 2, CV_64F), W;
            for (int k = 0; k < int(ids.size()); ++k) { auto& x = rows[ids[k]]; X.at<double>(k, 0) = 1; X.at<double>(k, 1) = x.s; X.at<double>(k, 2) = x.d[m]; Y.at<double>(k, 0) = x.y; Y.at<double>(k, 1) = x.t; }
            cv::solve(X, Y, W, cv::DECOMP_SVD); return W; };
        auto predT = [&](const cv::Mat& W, const Row& x, int m) { return W.at<double>(0, 1) + W.at<double>(1, 1) * x.s + W.at<double>(2, 1) * x.d[m]; };
        auto folds = [&](int k, unsigned seed) { std::map<std::string, std::vector<std::pair<std::pair<double, double>, int>>> by; std::map<std::pair<std::string, int>, int> seen;
            for (int i : inr) { auto& x = rows[i]; if (x.cond == "center") continue; if (!seen.count({x.set, x.group})) { seen[{x.set, x.group}] = 1; by[x.set].push_back({{x.y, x.t}, x.group}); } }
            std::map<std::pair<std::string, int>, int> f;
            for (auto& [st, v] : by) { std::sort(v.begin(), v.end());
                for (size_t b = 0; b < v.size(); b += k) { std::vector<int> perm(k); std::iota(perm.begin(), perm.end(), 0);
                    if (seed) { unsigned xx = seed * 2654435761u + unsigned(b) * 40503u; for (int j = k - 1; j > 0; --j) { xx = xx * 1103515245u + 12345u; std::swap(perm[j], perm[(xx >> 16) % (j + 1)]); } }
                    for (size_t i = b; i < std::min(v.size(), b + size_t(k)); ++i) f[{st, v[i].second}] = perm[i - b]; } }
            return f; };
        std::ofstream o(res / "scores.csv");
        o << "방법,시간 ms 중앙,1014만 학습→1020,→1029,→1029 T 최대,날짜층화 5-fold 10분할,그때 T 최대,목표 근처 |T|<=0.2 (1014만→다른 날),그때 T 최대\n";
        for (int m = 0; m < 5; ++m) {
            std::vector<int> tr; for (int i : inr) if (rows[i].set.find("1014") != std::string::npos) tr.push_back(i);
            auto W = fit(tr, m);
            auto sc = [&](std::function<bool(const Row&)> sel, int& p, int& n, double& mx) { p = n = 0; mx = 0; for (int i : inr) { auto& x = rows[i]; if (!sel(x)) continue; double e = std::abs(predT(W, x, m) - x.t); ++n; p += e <= 0.01 + 1e-12; mx = std::max(mx, e); } };
            int p20, n20, p29, n29, pn, nn; double m20, m29, mn;
            sc([](const Row& x) { return x.set.find("1020") != std::string::npos && x.cond != "center"; }, p20, n20, m20);
            sc([](const Row& x) { return x.set.find("1029") != std::string::npos; }, p29, n29, m29);
            sc([](const Row& x) { return x.set.find("1014") == std::string::npos && x.cond != "center" && std::abs(x.t) <= 0.2 + 1e-9; }, pn, nn, mn);
            double lo = 101, hi = -1, mx = 0;
            for (unsigned sd = 0; sd < 10; ++sd) { auto f = folds(5, sd); int p = 0, n = 0;
                for (int fo = 0; fo < 5; ++fo) { std::vector<int> trn; for (int i : inr) { auto it = f.find({rows[i].set, rows[i].group}); if (it == f.end() || it->second != fo) trn.push_back(i); }
                    auto Wf = fit(trn, m);
                    for (int i : inr) { auto it = f.find({rows[i].set, rows[i].group}); if (it == f.end() || it->second != fo) continue; double e = std::abs(predT(Wf, rows[i], m) - rows[i].t); ++n; p += e <= 0.01 + 1e-12; mx = std::max(mx, e); } }
                lo = std::min(lo, 100.0 * p / n); hi = std::max(hi, 100.0 * p / n); }
            std::vector<double> ms; for (auto& x : rows) ms.push_back(x.ms[m]); std::nth_element(ms.begin(), ms.begin() + ms.size() / 2, ms.end());
            char buf[512]; snprintf(buf, sizeof buf, "%s,%.1f,%d/%d,%d/%d,%.4f,%.1f~%.1f%%,%.4f,%d/%d,%.4f\n", names[m], ms[ms.size() / 2], p20, n20, p29, n29, m29, lo, hi, mx, pn, nn, mn); o << buf;
        }
        // 교란: 같은 사진의 "없음" 대비 d 변화를 T 로 환산 (1014 T 단독 사진에서 구한 d 감도)
        std::map<std::string, std::vector<std::array<double, 5>>> dd; std::map<int, std::array<double, 5>> base;
        std::vector<std::tuple<int, std::string, std::array<double, 5>>> pr;
        { std::ifstream in(res / "bars_perturb.csv"); std::string l; std::getline(in, l);
          while (std::getline(in, l)) { auto r = split(l); std::array<double, 5> d; for (int m = 0; m < 5; ++m) d[m] = std::stod(r[2 + m]); pr.push_back({std::stoi(r[0]), r[1], d}); if (r[1].rfind("없음", 0) == 0) base[std::stoi(r[0])] = d; } }
        double sens[5];
        for (int m = 0; m < 5; ++m) { double sxx = 0, sxy = 0, mx = 0, my = 0; int n = 0; for (auto& x : rows) if (x.set.find("1014") != std::string::npos && x.y == 0 && std::abs(x.t) <= 1) { mx += x.t; my += x.d[m]; ++n; } mx /= n; my /= n;
            for (auto& x : rows) if (x.set.find("1014") != std::string::npos && x.y == 0 && std::abs(x.t) <= 1) { sxx += (x.t - mx) * (x.t - mx); sxy += (x.t - mx) * (x.d[m] - my); } sens[m] = sxy / sxx; }
        std::ofstream q(res / "perturb_T.csv"); q << "교란"; for (int m = 0; m < 5; ++m) q << "," << names[m] << " |ΔT| 평균," << names[m] << " |ΔT| 최대"; q << "\n";
        std::map<std::string, std::array<std::vector<double>, 5>> agg; std::vector<std::string> order;
        for (auto& [i, nm, d] : pr) { if (nm.rfind("없음", 0) == 0 || !base.count(i)) continue; if (!agg.count(nm)) order.push_back(nm); for (int m = 0; m < 5; ++m) agg[nm][m].push_back(std::abs((d[m] - base[i][m]) / sens[m])); }
        for (auto& nm : order) { q << nm; for (int m = 0; m < 5; ++m) { auto& v = agg[nm][m]; double s = 0, mx = 0; for (double e : v) { s += e; mx = std::max(mx, e); } q << "," << s / v.size() << "," << mx; } q << "\n"; }
        q << "감도 d per T"; for (int m = 0; m < 5; ++m) q << "," << sens[m] << ","; q << "\n";
        std::cout << "eval done\n";
    }
}
