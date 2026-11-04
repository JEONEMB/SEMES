// joint v2 학습·평가 도구. 런타임 모듈(semes_joint)과 따로 빌드한다.
//  measure  <assets> <data_root> <list.csv> <features.csv>   사진마다 v2 특징 (s, d) 와 측정 진단
//  fit      <assets> <features.csv> <model.json> [세트,...]  |T|<=1 사진으로 W 를 최소제곱 적합
//  evaluate <features.csv> <out_dir>                          세 가지 분할로 |eT|<=0.01 평가
//  check    <assets> <data_root> <features.csv> <out.csv>    공개 joint() 를 돌려 도구 계산과 같은지, 시간
#include "alignment.hpp"
#include "observer_internal.hpp"
#include <opencv2/opencv.hpp>
#include <atomic>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <thread>
namespace fs = std::filesystem;
using namespace alignment_detail;

struct Row { std::string set, rel, cond; int group = -1; double y = 0, t = 0, s = NAN, d = NAN, qr = NAN, line = NAN, ms = NAN; std::string error; };
static std::vector<std::string> split(const std::string& s) { std::vector<std::string> r; std::string c; std::stringstream ss(s); while (std::getline(ss, c, ',')) r.push_back(c); return r; }
static std::vector<Row> read_rows(const fs::path& p, bool features) {
    std::ifstream in(p); if (!in) throw std::runtime_error("cannot read " + p.u8string());
    std::string l; std::getline(in, l); std::vector<Row> out;
    while (std::getline(in, l)) { auto r = split(l); Row x; x.set = r[0]; x.rel = r[1]; x.y = std::stod(r[2]); x.t = std::stod(r[3]); x.group = std::stoi(r[4]); x.cond = r[5];
        if (features && r.size() > 10) { x.s = std::stod(r[6]); x.d = std::stod(r[7]); x.qr = std::stod(r[8]); x.line = std::stod(r[9]); x.ms = std::stod(r[10]); if (r.size() > 11) x.error = r[11]; }
        out.push_back(x); }
    return out;
}
static bool in_range(const Row& x) { return std::abs(x.t) <= 1.001 && x.error.empty() && std::isfinite(x.s); }
static cv::Mat fit(const std::vector<const Row*>& rows) {
    cv::Mat X(int(rows.size()), 3, CV_64F), Y(int(rows.size()), 2, CV_64F), W;
    for (int k = 0; k < int(rows.size()); ++k) { X.at<double>(k, 0) = 1; X.at<double>(k, 1) = rows[k]->s; X.at<double>(k, 2) = rows[k]->d; Y.at<double>(k, 0) = rows[k]->y; Y.at<double>(k, 1) = rows[k]->t; }
    cv::solve(X, Y, W, cv::DECOMP_SVD); return W;
}
static cv::Point2d pred(const cv::Mat& W, const Row& x) { return {W.at<double>(0, 0) + W.at<double>(1, 0) * x.s + W.at<double>(2, 0) * x.d, W.at<double>(0, 1) + W.at<double>(1, 1) * x.s + W.at<double>(2, 1) * x.d}; }

int main(int argc, char** argv) {
    try {
        if (argc < 2) throw std::runtime_error("usage: joint_v2_tool measure|fit|evaluate|check ...");
        std::string mode = argv[1];
        if (mode == "measure" && argc >= 6) {
            Observer obs{fs::u8path(argv[2])}; auto bx = obs.block_centers();
            auto rows = read_rows(fs::u8path(argv[4]), false); const fs::path root = fs::u8path(argv[3]);
            std::atomic<int> next{0};
            auto work = [&]() { for (int i; (i = next++) < int(rows.size());) { auto& x = rows[i];
                try { cv::Mat g = read_gray(root / fs::u8path(x.set) / fs::u8path(x.rel)); auto t0 = cv::getTickCount(); auto o = obs.measure(g); auto f = v2_features(o.values, bx);
                      x.ms = (cv::getTickCount() - t0) * 1000.0 / cv::getTickFrequency(); x.s = f[0]; x.d = f[1]; x.qr = o.qr_correlation; x.line = o.line_correlation; }
                catch (const std::exception& e) { x.error = e.what(); } } };
            unsigned n = std::max(1u, std::min(6u, std::thread::hardware_concurrency())); std::vector<std::thread> th; for (unsigned k = 0; k < n; ++k) th.emplace_back(work); for (auto& t : th) t.join();
            std::ofstream o(fs::u8path(argv[5])); o.precision(10); o << "set,rel,Y,T,group,condition,s,d,qr_corr,line_corr,ms,error\n";
            int fail = 0; for (auto& x : rows) { o << x.set << "," << x.rel << "," << x.y << "," << x.t << "," << x.group << "," << x.cond << "," << x.s << "," << x.d << "," << x.qr << "," << x.line << "," << x.ms << "," << x.error << "\n"; fail += !x.error.empty(); }
            std::cout << "measured " << rows.size() << ", failures " << fail << "\n"; return 0;
        }
        if (mode == "fit" && argc >= 5) {
            Observer obs{fs::u8path(argv[2])}; auto bx = obs.block_centers();
            auto rows = read_rows(fs::u8path(argv[3]), true);
            std::vector<std::string> sets; if (argc > 5) sets = split(argv[5]);
            std::vector<const Row*> tr; for (auto& x : rows) if (in_range(x) && (sets.empty() || std::any_of(sets.begin(), sets.end(), [&](const std::string& s) { return x.set.find(s) != std::string::npos; }))) tr.push_back(&x);
            auto W = fit(tr);
            double s0 = 1e9, s1 = -1e9, d0 = 1e9, d1 = -1e9; for (auto* x : tr) { s0 = std::min(s0, x->s); s1 = std::max(s1, x->s); d0 = std::min(d0, x->d); d1 = std::max(d1, x->d); }
            cv::FileStorage f(argv[4], cv::FileStorage::WRITE | cv::FileStorage::FORMAT_JSON);
            f << "model" << "{" << "form" << "v2_qr_mean_shift_bar_slope_difference"
              << "definition" << "(y,t) = [1, s, d] * W; s = mean horizontal shift of 27 QR check points (px); d = upper bar slope - lower bar slope (px per 1000 px, least squares over 24 blocks each)"
              << "W" << "[" << "[" << W.at<double>(0, 0) << W.at<double>(0, 1) << "]" << "[" << W.at<double>(1, 0) << W.at<double>(1, 1) << "]" << "[" << W.at<double>(2, 0) << W.at<double>(2, 1) << "]" << "]"
              << "block_x" << "["; for (double v : bx) f << v; f << "]"
              << "training_range" << "{" << "s_min" << s0 << "s_max" << s1 << "d_min" << d0 << "d_max" << d1 << "}"
              << "training_images" << int(tr.size()) << "training_sets" << (sets.empty() ? std::string("all") : std::string(argv[5])) << "training_filter" << "|T| <= 1, measurement valid" << "}";
            std::cout << "fit on " << tr.size() << " images\n" << W << "\n"; return 0;
        }
        if (mode == "evaluate" && argc >= 4) {
            auto rows = read_rows(fs::u8path(argv[2]), true); const fs::path out = fs::u8path(argv[3]); fs::create_directories(out);
            std::vector<const Row*> all; for (auto& x : rows) if (in_range(x)) all.push_back(&x);
            auto has = [](const Row* x, const char* s) { return x->set.find(s) != std::string::npos; };
            std::ofstream sm(out / "summary.csv"); sm << "protocol,test,n,pass_T_0.01,pass_rate,mean_abs_eT,max_abs_eT,mean_abs_eY\n";
            std::ofstream pr(out / "predictions.csv"); pr.precision(8); pr << "protocol,set,rel,Y,T,pred_Y,pred_T,eT\n";
            auto report = [&](const std::string& proto, const std::string& test, const std::vector<std::pair<const Row*, cv::Point2d>>& v) {
                int n = 0, p = 0; double se = 0, mx = 0, sy = 0;
                for (auto& [x, q] : v) { double e = std::abs(q.y - x->t); ++n; p += e <= 0.01 + 1e-12; se += e; mx = std::max(mx, e); sy += std::abs(q.x - x->y); pr << proto << "," << x->set << "," << x->rel << "," << x->y << "," << x->t << "," << q.x << "," << q.y << "," << q.y - x->t << "\n"; }
                char b[256]; snprintf(b, sizeof b, "%s,%s,%d,%d,%.1f%%,%.4f,%.4f,%.4f\n", proto.c_str(), test.c_str(), n, p, 100.0 * p / std::max(1, n), se / std::max(1, n), mx, sy / std::max(1, n)); sm << b; };
            auto holdout = [&](const std::string& proto, std::vector<const char*> train, std::vector<const char*> tests) {
                std::vector<const Row*> tr; for (auto* x : all) for (auto s : train) if (has(x, s)) tr.push_back(x);
                auto W = fit(tr);
                for (auto t : tests) { std::vector<std::pair<const Row*, cv::Point2d>> v; for (auto* x : all) if (has(x, t) && x->cond != "center") v.push_back({x, pred(W, *x)}); report(proto, t, v);
                    std::vector<std::pair<const Row*, cv::Point2d>> nv; for (auto& e : v) if (std::abs(e.first->t) <= 0.2 + 1e-9) nv.push_back(e); report(proto, std::string(t) + " |T|<=0.2", nv); } };
            holdout("train 1014+1020 -> 1029 (v1 과 같은 미사용 날짜)", {"1014", "1020"}, {"1029"});
            holdout("train 1014 -> other dates", {"1014"}, {"1020", "1029"});
            // 날짜 층화 5-fold, 10분할: 세트마다 같은 (Y,T) 묶음을 크기순 정렬 후 블록 안에서 fold 배정
            double lo = 101, hi = -1, mx = 0;
            for (unsigned seed = 0; seed < 10; ++seed) {
                std::map<std::string, std::vector<std::pair<std::pair<double, double>, int>>> by; std::map<std::pair<std::string, int>, int> seen, fold;
                for (auto* x : all) if (x->cond != "center" && !seen.count({x->set, x->group})) { seen[{x->set, x->group}] = 1; by[x->set].push_back({{x->y, x->t}, x->group}); }
                for (auto& [st, v] : by) { std::sort(v.begin(), v.end());
                    for (size_t b = 0; b < v.size(); b += 5) { std::vector<int> perm(5); std::iota(perm.begin(), perm.end(), 0);
                        if (seed) { unsigned xx = seed * 2654435761u + unsigned(b) * 40503u; for (int j = 4; j > 0; --j) { xx = xx * 1103515245u + 12345u; std::swap(perm[j], perm[(xx >> 16) % (j + 1)]); } }
                        for (size_t i = b; i < std::min(v.size(), b + 5); ++i) fold[{st, v[i].second}] = perm[i - b]; } }
                std::vector<std::pair<const Row*, cv::Point2d>> v;
                for (int f = 0; f < 5; ++f) { std::vector<const Row*> tr; for (auto* x : all) { auto it = fold.find({x->set, x->group}); if (it == fold.end() || it->second != f) tr.push_back(x); }
                    auto W = fit(tr); for (auto* x : all) { auto it = fold.find({x->set, x->group}); if (it != fold.end() && it->second == f) v.push_back({x, pred(W, *x)}); } }
                int p = 0; for (auto& [x, q] : v) { double e = std::abs(q.y - x->t); p += e <= 0.01 + 1e-12; mx = std::max(mx, e); }
                lo = std::min(lo, 100.0 * p / v.size()); hi = std::max(hi, 100.0 * p / v.size());
                if (seed == 0) report("date-stratified 5-fold seed0", "all", v);
            }
            char b[256]; snprintf(b, sizeof b, "date-stratified 5-fold x10 seeds,all,,,%.1f~%.1f%%,,%.4f,\n", lo, hi, mx); sm << b;
            std::cout << "evaluate done\n"; return 0;
        }
        if (mode == "check" && argc >= 6) {
            initialize_joint(argv[2]);
            JointModel m; m.load(fs::u8path(argv[2]) / "joint_v2_model.json");
            auto rows = read_rows(fs::u8path(argv[4]), true); const fs::path root = fs::u8path(argv[3]);
            std::ofstream o(fs::u8path(argv[5])); o.precision(10); o << "set,rel,Y,T,joint_y,joint_t,tool_t,diff_t,outside,ms\n";
            double worst = 0; int n = 0;
            for (auto& x : rows) { if (!x.error.empty()) continue;
                cv::Mat g = read_gray(root / fs::u8path(x.set) / fs::u8path(x.rel)); auto t0 = cv::getTickCount(); auto q = joint(g); double ms = (cv::getTickCount() - t0) * 1000.0 / cv::getTickFrequency();
                auto r = pred(m.W, x); double diff = q.t - r.y; worst = std::max(worst, std::abs(diff)); ++n;
                o << x.set << "," << x.rel << "," << x.y << "," << x.t << "," << q.y << "," << q.t << "," << r.y << "," << diff << "," << alignment_diagnostics().outside_training_range << "," << ms << "\n"; }
            std::cout << "checked " << n << ", max |joint() - tool| = " << worst << "\n"; return 0;
        }
        throw std::runtime_error("bad arguments");
    } catch (const std::exception& e) { std::cerr << e.what() << "\n"; return 1; }
}
