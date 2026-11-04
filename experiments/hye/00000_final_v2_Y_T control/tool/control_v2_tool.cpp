// control v2 학습·평가 도구. 런타임 모듈(semes_control)과 따로 빌드한다.
//  measure  <assets> <data_root> <list.csv> <features.csv>         사진마다 s(QR536 평균 가로 이동), d(바 기울기 차)
//  fit      <assets> <features.csv> <model.json> [theta]          y0 = [1,s] (T=0 사진), t^ = [1,s,d] (|T|<=1), k = y0 잔차의 T 기울기
//  evaluate <features.csv> <out_dir> [theta] [v1.csv]             분할별 |eY|<=0.045, 남은 T 영향, 반복 전폭
//  check    <assets> <data_root> <features.csv> <out.csv>         공개 control() 이 도구 계산과 같은지, 시간
#include "alignment.hpp"
#include "control_internal.hpp"
#include <opencv2/opencv.hpp>
#include <atomic>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <thread>
namespace fs = std::filesystem;
using namespace legacy_control;

struct Row { std::string set, rel, cond; int group = -1, fold4 = -1; double y = 0, t = 0, s = NAN, d = NAN, ms = NAN; std::string error; };
static std::vector<std::string> split(const std::string& s) { std::vector<std::string> r; std::string c; std::stringstream ss(s); while (std::getline(ss, c, ',')) r.push_back(c); return r; }
static std::vector<Row> read_rows(const fs::path& p) {
    std::ifstream in(p); if (!in) throw std::runtime_error("cannot read " + p.u8string());
    std::string l; std::getline(in, l); bool feat = l.find(",s,") != std::string::npos; std::vector<Row> out;
    while (std::getline(in, l)) { auto r = split(l); Row x; x.set = r[0]; x.rel = r[1]; x.y = std::stod(r[2]); x.t = std::stod(r[3]); x.group = std::stoi(r[4]); x.cond = r[5]; x.fold4 = std::stoi(r[6]);
        if (feat) { x.s = std::stod(r[7]); x.d = std::stod(r[8]); x.ms = std::stod(r[9]); if (r.size() > 10) x.error = r[10]; } out.push_back(x); }
    return out;
}
static bool usable(const Row& x) { return std::abs(x.t) <= 1.001 && x.error.empty() && std::isfinite(x.s) && std::isfinite(x.d); }
static bool has(const Row& x, const char* k) { return x.set.find(k) != std::string::npos; }

struct Fit { double f0, f1, c0, c1, c2, k, theta, s0, s1, d0, d1; int n_y0, n_t;
    double y(const Row& x) const { double th = c0 + c1 * x.s + c2 * x.d; double g = std::abs(th) <= theta ? 0 : th - (th > 0 ? theta : -theta); return f0 + f1 * x.s - k * g; }
    double t(const Row& x) const { return c0 + c1 * x.s + c2 * x.d; } };
static Fit fit(const std::vector<const Row*>& tr, double theta) {
    Fit f{}; f.theta = theta;
    cv::Mat A(0, 2, CV_64F), b(0, 1, CV_64F), B(0, 3, CV_64F), c(0, 1, CV_64F);
    for (auto* x : tr) { if (std::abs(x->t) < 1e-9) { cv::Mat r = (cv::Mat_<double>(1, 2) << 1, x->s); A.push_back(r); b.push_back(x->y); }
        cv::Mat r = (cv::Mat_<double>(1, 3) << 1, x->s, x->d); B.push_back(r); c.push_back(x->t); }
    cv::Mat w, v; cv::solve(A, b, w, cv::DECOMP_SVD); cv::solve(B, c, v, cv::DECOMP_SVD);
    f.f0 = w.at<double>(0); f.f1 = w.at<double>(1); f.c0 = v.at<double>(0); f.c1 = v.at<double>(1); f.c2 = v.at<double>(2); f.n_y0 = A.rows; f.n_t = B.rows;
    double sxx = 0, sxy = 0; for (auto* x : tr) { double e = f.f0 + f.f1 * x->s - x->y; sxx += x->t * x->t; sxy += x->t * e; } f.k = sxy / sxx;
    f.s0 = f.d0 = 1e9; f.s1 = f.d1 = -1e9; for (auto* x : tr) { f.s0 = std::min(f.s0, x->s); f.s1 = std::max(f.s1, x->s); f.d0 = std::min(f.d0, x->d); f.d1 = std::max(f.d1, x->d); }
    return f;
}

int main(int argc, char** argv) {
    try {
        if (argc < 2) throw std::runtime_error("usage: control_v2_tool measure|fit|evaluate|check ...");
        std::string mode = argv[1];
        if (mode == "measure" && argc >= 6) {
            const fs::path assets = fs::u8path(argv[2]), root = fs::u8path(argv[3]);
            Tracker tracker(assets); Bars bars(assets, read_gray(assets / "center.bmp"));
            auto rows = read_rows(fs::u8path(argv[4])); std::atomic<int> next{0};
            auto work = [&]() { for (int i; (i = next++) < int(rows.size());) { auto& x = rows[i];
                try { auto g = checked_gray(read_gray(root / fs::u8path(x.set) / fs::u8path(x.rel))); auto t0 = cv::getTickCount();
                      auto tr = tracker.measure(g); x.s = tracker.mean_shift(tr); x.d = bars.slope_difference(g); x.ms = (cv::getTickCount() - t0) * 1000.0 / cv::getTickFrequency(); }
                catch (const std::exception& e) { x.error = e.what(); } } };
            std::vector<std::thread> th; for (int k = 0; k < 6; ++k) th.emplace_back(work); for (auto& t : th) t.join();
            std::ofstream o(fs::u8path(argv[5])); o.precision(10); o << "set,rel,Y,T,group,condition,fold4,s,d,ms,error\n"; int fail = 0;
            for (auto& x : rows) { o << x.set << "," << x.rel << "," << x.y << "," << x.t << "," << x.group << "," << x.cond << "," << x.fold4 << "," << x.s << "," << x.d << "," << x.ms << "," << x.error << "\n"; fail += !x.error.empty(); }
            std::cout << "measured " << rows.size() << ", failures " << fail << "\n"; return 0;
        }
        if (mode == "fit" && argc >= 5) {
            Bars bars(fs::u8path(argv[2]), read_gray(fs::u8path(argv[2]) / "center.bmp"));
            auto rows = read_rows(fs::u8path(argv[3])); double theta = argc > 5 ? std::stod(argv[5]) : 0.0075;
            std::vector<const Row*> tr; for (auto& x : rows) if (usable(x)) tr.push_back(&x);
            auto f = fit(tr, theta);
            cv::FileStorage o(argv[4], cv::FileStorage::WRITE | cv::FileStorage::FORMAT_JSON);
            o << "model" << "{" << "form" << "v2_precise_y_with_gated_residual_t"
              << "definition" << "y = y0[0] + y0[1]*s - k*soft(t_hat, theta); t_hat = t_hat[0] + t_hat[1]*s + t_hat[2]*d; soft(v,theta) = 0 if |v|<=theta else v - sign(v)*theta. s = mean horizontal shift of valid fixed QR536 points (px, center self-measure subtracted); d = upper bar slope - lower bar slope (px per 1000 px)"
              << "y0" << "[" << f.f0 << f.f1 << "]" << "t_hat" << "[" << f.c0 << f.c1 << f.c2 << "]" << "k" << f.k << "theta" << f.theta
              << "block_x" << "["; for (double v : bars.centers()) o << v; o << "]"
              << "training_range" << "{" << "s_min" << f.s0 << "s_max" << f.s1 << "d_min" << f.d0 << "d_max" << f.d1 << "}"
              << "training_images_y0" << f.n_y0 << "training_images_t_hat" << f.n_t << "training_filter" << "y0: T == 0; t_hat and k: |T| <= 1; all three sets" << "}";
            std::cout << "fit y0 n=" << f.n_y0 << " t_hat n=" << f.n_t << " k=" << f.k << "\n"; return 0;
        }
        if (mode == "evaluate" && argc >= 4) {
            auto rows = read_rows(fs::u8path(argv[2])); const fs::path out = fs::u8path(argv[3]); fs::create_directories(out);
            double theta = argc > 4 ? std::stod(argv[4]) : 0.0075;
            std::map<std::string, double> v1;   // set|rel -> v1 control Y (선택)
            if (argc > 5) { std::ifstream in(fs::u8path(argv[5])); std::string l; std::getline(in, l); while (std::getline(in, l)) { auto r = split(l); v1[r[0] + "|" + r[1]] = std::stod(r[2]); } }
            std::vector<const Row*> all; for (auto& x : rows) if (usable(x)) all.push_back(&x);
            std::ofstream sm(out / "summary.csv"); sm << "model,protocol,test,n,pass_Y_0.045,rate,mean_abs_eY,max_abs_eY\n";
            std::ofstream pr(out / "predictions.csv"); pr.precision(8); pr << "model,protocol,set,rel,Y,T,pred_Y,pred_T,eY\n";
            using Pred = std::function<double(const Row&)>;
            auto report = [&](const std::string& model, const std::string& proto, const std::string& test, std::function<bool(const Row&)> sel, Pred py, Pred pt) {
                int n = 0, p = 0; double se = 0, mx = 0;
                for (auto* x : all) { if (!sel(*x)) continue; double yy = py(*x); if (!std::isfinite(yy)) continue; double e = std::abs(yy - x->y); ++n; p += e <= 0.045 + 1e-12; se += e; mx = std::max(mx, e);
                    pr << model << "," << proto << "," << x->set << "," << x->rel << "," << x->y << "," << x->t << "," << yy << "," << (pt ? pt(*x) : NAN) << "," << yy - x->y << "\n"; }
                char b[400]; snprintf(b, sizeof b, "%s,%s,%s,%d,%d,%.1f%%,%.4f,%.4f\n", model.c_str(), proto.c_str(), test.c_str(), n, p, 100.0 * p / std::max(1, n), se / std::max(1, n), mx); sm << b; };
            auto y_only = [](const Row& x) { return has(x, "1014") && x.cond == "Y"; };
            auto small = [](double tm) { return std::function<bool(const Row&)>([tm](const Row& x) { return !has(x, "1014") && x.cond != "center" && std::abs(x.t) <= tm + 1e-9; }); };
            // 1) 1014 4-fold (v1 검증과 같은 fold), Y 단독 80장
            std::map<int, Fit> f4; for (int f = 0; f < 4; ++f) { std::vector<const Row*> tr; for (auto* x : all) if (has(*x, "1014") && x->fold4 != f) tr.push_back(x); f4[f] = fit(tr, theta); }
            report("v2", "1014 4-fold", "1014 Y 단독", y_only, [&](const Row& x) { return x.fold4 >= 0 ? f4[x.fold4].y(x) : NAN; }, nullptr);
            // 2) 1014 만 학습 -> 다른 날짜 (v1 도 1014 만 학습)
            std::vector<const Row*> t14; for (auto* x : all) if (has(*x, "1014")) t14.push_back(x); auto F14 = fit(t14, theta);
            for (double tm : {0.01, 0.03, 0.1}) {
                std::string test = "1020+1029 |T|<=" + std::to_string(tm).substr(0, 4);
                report("v2", "1014만 학습", test, small(tm), [&](const Row& x) { return F14.y(x); }, [&](const Row& x) { return F14.t(x); });
                if (!v1.empty()) report("v1", "1014만 학습", test, small(tm), [&](const Row& x) { auto it = v1.find(x.set + "|" + x.rel); return it == v1.end() ? NAN : it->second; }, nullptr);
            }
            if (!v1.empty()) report("v1", "전수(1014) 모델", "1014 Y 단독 (재대입)", y_only, [&](const Row& x) { auto it = v1.find(x.set + "|" + x.rel); return it == v1.end() ? NAN : it->second; }, nullptr);
            // 3) 날짜 층화 5-fold 10분할, |T|<=0.1 (v1 은 1014 만 학습했으므로 같은 값)
            double lo = 101, hi = -1, mx = 0;
            for (unsigned seed = 0; seed < 10; ++seed) {
                std::map<std::string, std::vector<std::pair<std::pair<double, double>, int>>> by; std::set<std::pair<std::string, int>> seen; std::map<std::pair<std::string, int>, int> fold;
                for (auto* x : all) if (x->cond != "center" && !seen.count({x->set, x->group})) { seen.insert({x->set, x->group}); by[x->set].push_back({{x->y, x->t}, x->group}); }
                for (auto& [st, v] : by) { std::sort(v.begin(), v.end()); for (size_t b = 0; b < v.size(); b += 5) { std::vector<int> perm(5); std::iota(perm.begin(), perm.end(), 0);
                    if (seed) { unsigned xx = seed * 2654435761u + unsigned(b) * 40503u; for (int j = 4; j > 0; --j) { xx = xx * 1103515245u + 12345u; std::swap(perm[j], perm[(xx >> 16) % (j + 1)]); } }
                    for (size_t i = b; i < std::min(v.size(), b + 5); ++i) fold[{st, v[i].second}] = perm[i - b]; } }
                std::map<int, Fit> ff; for (int f = 0; f < 5; ++f) { std::vector<const Row*> tr; for (auto* x : all) { auto it = fold.find({x->set, x->group}); if (it == fold.end() || it->second != f) tr.push_back(x); } ff[f] = fit(tr, theta); }
                int n = 0, p = 0; for (auto* x : all) { auto it = fold.find({x->set, x->group}); if (it == fold.end() || std::abs(x->t) > 0.1 + 1e-9) continue; double e = std::abs(ff[it->second].y(*x) - x->y); ++n; p += e <= 0.045 + 1e-12; mx = std::max(mx, e); }
                lo = std::min(lo, 100.0 * p / n); hi = std::max(hi, 100.0 * p / n);
            }
            { char b[256]; snprintf(b, sizeof b, "v2,날짜층화 5-fold x10,|T|<=0.1,,,%.1f~%.1f%%,,%.4f\n", lo, hi, mx); sm << b; }
            // 4) 남은 T 가 Y 로 새는 기울기 (1029: Y 0.05 고정, T 0.04~0.82), 5) 같은 자세 반복 전폭 (1014만 학습, |T|<=0.1)
            std::ofstream rb(out / "robustness.csv"); rb << "model,residual_T_slope_dY_dT_1029,repeat_groups,repeat_range_median,repeat_range_max,groups_range_le_0.045\n";
            auto robust = [&](const std::string& name, Pred py) {
                double sx = 0, sy = 0, sxx = 0, sxy = 0; int n = 0; for (auto* x : all) if (has(*x, "1029")) { double yy = py(*x); if (!std::isfinite(yy)) continue; double e = yy - x->y; sx += x->t; sy += e; sxx += x->t * x->t; sxy += x->t * e; ++n; }
                double slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
                std::map<std::pair<std::string, int>, std::vector<double>> g; for (auto* x : all) if (small(0.1)(*x)) { double yy = py(*x); if (std::isfinite(yy)) g[{x->set, x->group}].push_back(yy); }
                std::vector<double> r; int ok = 0; for (auto& [k, v] : g) if (v.size() >= 2) { double w = *std::max_element(v.begin(), v.end()) - *std::min_element(v.begin(), v.end()); r.push_back(w); ok += w <= 0.045; }
                std::sort(r.begin(), r.end()); rb << name << "," << slope << "," << r.size() << "," << r[r.size() / 2] << "," << r.back() << "," << ok << "\n"; };
            robust("v2", [&](const Row& x) { return F14.y(x); });
            if (!v1.empty()) robust("v1", [&](const Row& x) { auto it = v1.find(x.set + "|" + x.rel); return it == v1.end() ? NAN : it->second; });
            std::cout << "evaluate done\n"; return 0;
        }
        if (mode == "check" && argc >= 6) {
            initialize_control(argv[2]); ModelV2 m(fs::u8path(argv[2]) / "control_v2_model.json");
            auto rows = read_rows(fs::u8path(argv[4])); const fs::path root = fs::u8path(argv[3]);
            std::ofstream o(fs::u8path(argv[5])); o.precision(10); o << "set,rel,Y,T,control_y,control_t,tool_y,diff_y,precondition,outside,ms\n"; double worst = 0; int n = 0;
            for (auto& x : rows) { if (!x.error.empty()) continue; cv::Mat g = read_gray(root / fs::u8path(x.set) / fs::u8path(x.rel));
                auto t0 = cv::getTickCount(); auto q = control(g); double ms = (cv::getTickCount() - t0) * 1000.0 / cv::getTickFrequency();
                auto e = m.predict(x.s, x.d); double diff = q.y - e.y; worst = std::max(worst, std::abs(diff)); ++n; auto dg = alignment_diagnostics();
                o << x.set << "," << x.rel << "," << x.y << "," << x.t << "," << q.y << "," << q.t << "," << e.y << "," << diff << "," << dg.control_precondition_met << "," << dg.outside_training_range << "," << ms << "\n"; }
            std::cout << "checked " << n << ", max |control() - tool| = " << worst << "\n"; return 0;
        }
        throw std::runtime_error("bad arguments");
    } catch (const std::exception& e) { std::cerr << e.what() << "\n"; return 1; }
}
