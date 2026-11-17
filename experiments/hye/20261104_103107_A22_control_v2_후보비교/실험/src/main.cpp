// A22: control_v2 후보 비교. control(2차 Y)은 T≈0 전제에서 Y 를 낸다. 남은 T 가 Y 로 새는 것(T 1 = Y 5.6 mm 상당)이 약점이다.
//  입력: A0 캐시(control 측정기: NCC→LK15→스플라인 GN, 고정 536점), A21 바 기울기 차 d(지금 블록 정합), A10 의 v1 control 출력.
//  후보: [s], [s, d], [P12, d], [s, d, s·d], [P12]  (s = 536점 가로 이동 평균, P12 = control 의 위치 2차 다항식 12값)
//  판정: 한 장 |eY| <= 0.045 (ALGORITHM_RULES 2026-11-04). 보조: 같은 자세 반복의 Y 전폭 <= 0.045.
#include "cache.hpp"
#include "model.hpp"
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <functional>
using namespace a0;

static std::vector<double> P12(const Cache& c, int i) {
    cv::Mat X(0, 6, CV_64F), Y(0, 2, CV_64F);
    for (int p = 0; p < kPoints; ++p) { if (!c.points[p].selected || !c.v(i)[p]) continue; double u = (c.points[p].x - 1906) / 486, v = (c.points[p].y - 1070) / 122;
        cv::Mat r = (cv::Mat_<double>(1, 6) << 1, u, v, u * u, u * v, v * v); X.push_back(r); auto d = c.dz(i, p); cv::Mat y = (cv::Mat_<double>(1, 2) << d.x, d.y); Y.push_back(y); }
    cv::Mat B; cv::solve(X, Y, B, cv::DECOMP_SVD); std::vector<double> f; for (int k = 0; k < 6; ++k) f.push_back(B.at<double>(k, 0)); for (int k = 0; k < 6; ++k) f.push_back(B.at<double>(k, 1)); return f;
}
static double smean(const Cache& c, int i) { double s = 0; int n = 0; for (int p = 0; p < kPoints; ++p) if (c.points[p].selected && c.v(i)[p]) { s += c.dz(i, p).x; ++n; } return s / n; }
static std::vector<std::string> sp(const std::string& s) { std::vector<std::string> r; std::string t; std::stringstream ss(s); while (std::getline(ss, t, ',')) r.push_back(t); return r; }

int main() {
    const fs::path res = fs::u8path(EXPERIMENT_DIR) / "results"; fs::create_directories(res);
    const fs::path repo = fs::u8path(REPO_DIR);
    Cache c = load(cache_dir()); const int N = c.n();
    std::vector<double> d(N, NAN), v1y(N, NAN);
    { std::ifstream in(repo / fs::u8path(u8"experiments/hye/20261104_094732_A21_바측정_강건화비교/결과/results/bars_all.csv")); std::string l; std::getline(in, l); while (std::getline(in, l)) { auto r = sp(l); d[std::stoi(r[0])] = std::stod(r[8]); } }
    { std::ifstream in(repo / fs::u8path(u8"experiments/hye/20261102_133623_A10_전달모듈대조_joint와control실측/결과/results/per_shot.csv")); std::string l; std::getline(in, l); while (std::getline(in, l)) { auto r = sp(l); v1y[std::stoi(r[0])] = std::stod(r[8]); } }
    std::vector<std::vector<double>> p12(N); std::vector<double> s(N);
    for (auto& x : c.shots) if (x.measured) { p12[x.idx] = P12(c, x.idx); s[x.idx] = smean(c, x.idx); }
    struct Cand { std::string name; std::function<std::vector<double>(int)> f; };
    std::vector<Cand> cs = {
        {"C1 [s]", [&](int i) { return std::vector<double>{s[i]}; }},
        {"C2 [s, d]", [&](int i) { return std::vector<double>{s[i], d[i]}; }},
        {"C3 [P12, d]", [&](int i) { auto f = p12[i]; f.push_back(d[i]); return f; }},
        {"C4 [s, d, s·d]", [&](int i) { return std::vector<double>{s[i], d[i], s[i] * d[i] / 100}; }},
        {"C5 [P12]", [&](int i) { return p12[i]; }},
    };
    auto inr = [&](const Shot& x) { return x.measured && std::abs(x.t) <= 1.001; };
    auto has = [](const Shot& x, const char* k) { return x.set.find(k) != std::string::npos; };
    auto fitpred = [&](const Cand& cd, std::function<bool(const Shot&)> tr, double lam) { std::vector<std::vector<double>> X; std::vector<cv::Point2d> Y;
        for (auto& x : c.shots) if (inr(x) && tr(x)) { X.push_back(cd.f(x.idx)); Y.push_back({x.y, x.t}); } LinearModel m; m.fit(X, Y, lam); return m; };
    // 평가 묶음
    auto sel_small = [&](const Shot& x, double tmax) { return !has(x, "1014") && x.condition != "center" && std::abs(x.t) <= tmax + 1e-9; };
    Table t; t.head = {"모델", "1014 Y 단독 80장 (4-fold OOF) 통과", "최대", "1014만 학습→1020·1029 |T|<=0.03 통과", "최대", "→ |T|<=0.1", "최대", "날짜층화 5-fold |T|<=0.1 (10분할 최소~최대)", "최대", "남은 T 기울기 dY/dT (1029)", "반복 묶음 Y 전폭 중앙 / 최대 (|T|<=0.1, 1014만 학습)", "전폭<=0.045 묶음"};
    auto row = [&](const std::string& name, std::function<double(const Shot&, int)> py) {   // py(shot, protocol) -> Y 예측. protocol 0: 1014 4-fold, 1: 1014만 학습, 2..: 날짜층화 seed
        auto score = [&](std::function<bool(const Shot&)> sel, int proto, int& p, int& n, double& mx) { p = n = 0; mx = 0; for (auto& x : c.shots) if (inr(x) && sel(x)) { double e = std::abs(py(x, proto) - x.y); ++n; p += e <= 0.045 + 1e-12; mx = std::max(mx, e); } };
        int p1, n1, p2, n2, p3, n3; double m1, m2, m3;
        score([&](const Shot& x) { return has(x, "1014") && x.condition == "Y"; }, 0, p1, n1, m1);
        score([&](const Shot& x) { return sel_small(x, 0.03); }, 1, p2, n2, m2);
        score([&](const Shot& x) { return sel_small(x, 0.1); }, 1, p3, n3, m3);
        double lo = 101, hi = -1, mx = 0;
        for (int sd = 0; sd < 10; ++sd) { int p = 0, n = 0; double m = 0; score([&](const Shot& x) { return x.condition != "center" && std::abs(x.t) <= 0.1 + 1e-9; }, 2 + sd, p, n, m); lo = std::min(lo, 100.0 * p / n); hi = std::max(hi, 100.0 * p / n); mx = std::max(mx, m); }
        // 남은 T 기울기: 1029 (Y=0.05 고정, T 0.04~0.82) 에서 eY 를 T 로 회귀
        double sx = 0, sy = 0, sxx = 0, sxy = 0; int nn = 0; for (auto& x : c.shots) if (inr(x) && has(x, "1029")) { double e = py(x, 1) - x.y; sx += x.t; sy += e; sxx += x.t * x.t; sxy += x.t * e; ++nn; }
        double slope = (nn * sxy - sx * sy) / (nn * sxx - sx * sx);
        std::map<int, std::vector<double>> g; for (auto& x : c.shots) if (inr(x) && sel_small(x, 0.1)) g[x.group].push_back(py(x, 1));
        std::vector<double> ranges; int ok = 0; for (auto& [k, v] : g) if (v.size() >= 2) { double r = *std::max_element(v.begin(), v.end()) - *std::min_element(v.begin(), v.end()); ranges.push_back(r); ok += r <= 0.045; }
        double rmax = 0; for (double r : ranges) rmax = std::max(rmax, r);
        char b[1024]; snprintf(b, sizeof b, "%d/%d", p1, n1); std::string a1 = b; snprintf(b, sizeof b, "%d/%d", p2, n2); std::string a2 = b; snprintf(b, sizeof b, "%d/%d", p3, n3); std::string a3 = b;
        t.add({name, a1, num(m1, 4), a2, num(m2, 4), a3, num(m3, 4), num(lo, 1) + "~" + num(hi, 1) + "%", num(mx, 4), num(slope, 3), num(median(ranges), 4) + " / " + num(rmax, 4), std::to_string(ok) + "/" + std::to_string(ranges.size())});
    };
    // v1: 1014 4-fold OOF 는 검증 기록(python raw_oof4)이 있으나 Y 단독 판정만 필요하므로, 전수 모델 출력(A10)을 1014 밖 평가에 쓴다(v1 은 1014 만 학습).
    std::map<std::string, double> v1oof;
    { std::ifstream in(fs::u8path(REPO_DIR) / fs::u8path(u8"experiments/hye/20261028_1327_final_Y_T control/validation/python_predictions.csv")); std::string l; std::getline(in, l);
      while (std::getline(in, l)) { auto r = sp(l); if (r[12] != "raw_oof4") continue; double pred = std::stod(r[5]); v1oof[r[0]] = r[4] == "Y" ? pred : 0.0; } }
    row("v1 control (전달 모델, Y-or-T)", [&](const Shot& x, int proto) { if (proto == 0) { auto k = x.rel.substr(x.rel.find('/') + 1); auto it = v1oof.find(k); return it == v1oof.end() ? NAN : it->second; } return v1y[x.idx]; });
    for (auto& cd : cs) for (double lam : {0.0, 0.01}) {
        std::map<int, LinearModel> fold4; for (int f = 0; f < 4; ++f) fold4[f] = fitpred(cd, [&](const Shot& x) { return has(x, "1014") && x.fold_mixed != f; }, lam);
        LinearModel only14 = fitpred(cd, [&](const Shot& x) { return has(x, "1014"); }, lam);
        // 날짜층화 5-fold 10분할 (A17 과 같은 배정 규칙)
        std::map<int, std::map<int, LinearModel>> strat; std::map<int, std::vector<int>> fold_of;
        for (int sd = 0; sd < 10; ++sd) {
            std::map<std::string, std::vector<std::pair<std::pair<double, double>, int>>> by; std::map<int, int> gf; std::set<int> seen;
            for (auto& x : c.shots) if (inr(x) && x.condition != "center" && !seen.count(x.group)) { seen.insert(x.group); by[x.set].push_back({{x.y, x.t}, x.group}); }
            for (auto& [st, v] : by) { std::sort(v.begin(), v.end()); for (size_t b = 0; b < v.size(); b += 5) { std::vector<int> perm(5); std::iota(perm.begin(), perm.end(), 0);
                if (sd) { unsigned xx = unsigned(sd) * 2654435761u + unsigned(b) * 40503u; for (int j = 4; j > 0; --j) { xx = xx * 1103515245u + 12345u; std::swap(perm[j], perm[(xx >> 16) % (j + 1)]); } }
                for (size_t i = b; i < std::min(v.size(), b + 5); ++i) gf[v[i].second] = perm[i - b]; } }
            fold_of[sd].assign(N, -1); for (auto& x : c.shots) if (gf.count(x.group) && x.condition != "center") fold_of[sd][x.idx] = gf[x.group];
            for (int f = 0; f < 5; ++f) strat[sd][f] = fitpred(cd, [&](const Shot& x) { return fold_of[sd][x.idx] != f; }, lam);
        }
        row(cd.name + (lam > 0 ? " ridge 0.01" : ""), [&](const Shot& x, int proto) {
            auto f = cd.f(x.idx);
            if (proto == 0) return x.fold_mixed >= 0 ? fold4[x.fold_mixed].predict(f).x : NAN;
            if (proto == 1) return only14.predict(f).x;
            int sd = proto - 2; int fo = fold_of[sd][x.idx]; return fo < 0 ? NAN : strat[sd][fo].predict(f).x; });
    }
    // 2부: 운용 영역(남은 T 작음)에 맞춘 학습. 학습 자료를 |T| <= tau 로 제한하면 d 계수가 필요한 만큼만 남는다.
    for (double tau : {0.0, 0.03, 0.05, 0.1, 0.2})
    for (int fs = 0; fs < 2; ++fs) {
        Cand cd = fs == 0 ? cs[1] : cs[2];
        auto use = [&](const Shot& x) { return std::abs(x.t) <= tau + 1e-9; };
        std::map<int, LinearModel> fold4; for (int f = 0; f < 4; ++f) fold4[f] = fitpred(cd, [&](const Shot& x) { return has(x, "1014") && x.fold_mixed != f && use(x); }, 0.0);
        LinearModel only14 = fitpred(cd, [&](const Shot& x) { return has(x, "1014") && use(x); }, 0.0);
        std::map<int, std::map<int, LinearModel>> strat; std::map<int, std::vector<int>> fold_of;
        for (int sd = 0; sd < 10; ++sd) {
            std::map<std::string, std::vector<std::pair<std::pair<double, double>, int>>> by; std::map<int, int> gf; std::set<int> seen;
            for (auto& x : c.shots) if (inr(x) && x.condition != "center" && !seen.count(x.group)) { seen.insert(x.group); by[x.set].push_back({{x.y, x.t}, x.group}); }
            for (auto& [st, v] : by) { std::sort(v.begin(), v.end()); for (size_t b = 0; b < v.size(); b += 5) { std::vector<int> perm(5); std::iota(perm.begin(), perm.end(), 0);
                if (sd) { unsigned xx = unsigned(sd) * 2654435761u + unsigned(b) * 40503u; for (int j = 4; j > 0; --j) { xx = xx * 1103515245u + 12345u; std::swap(perm[j], perm[(xx >> 16) % (j + 1)]); } }
                for (size_t i = b; i < std::min(v.size(), b + 5); ++i) gf[v[i].second] = perm[i - b]; } }
            fold_of[sd].assign(N, -1); for (auto& x : c.shots) if (gf.count(x.group) && x.condition != "center") fold_of[sd][x.idx] = gf[x.group];
            for (int f = 0; f < 5; ++f) strat[sd][f] = fitpred(cd, [&](const Shot& x) { return fold_of[sd][x.idx] != f && use(x); }, 0.0);
        }
        row(cd.name + " 학습 |T|<=" + num(tau, 2), [&](const Shot& x, int proto) {
            auto f = cd.f(x.idx);
            if (proto == 0) return x.fold_mixed >= 0 ? fold4[x.fold_mixed].predict(f).x : NAN;
            if (proto == 1) return only14.predict(f).x;
            int sd = proto - 2; int fo = fold_of[sd][x.idx]; return fo < 0 ? NAN : strat[sd][fo].predict(f).x; });
    }
    // 3부: 문턱 보정. Y = f0(s) - k * soft(T^, theta). f0 는 T=0 자료로 맞춘 s 의 직선(정밀 Y),
    //  T^ 는 [s, d] 를 모든 범위 자료로 맞춘 T, k 는 f0 의 잔차를 T 로 회귀한 기울기(학습 자료 안에서). soft 는 |T^|<=theta 면 0.
    auto soft = [](double v, double th) { return std::abs(v) <= th ? 0.0 : v - (v > 0 ? th : -th); };
    for (double th : {0.0, 0.003, 0.005, 0.0075, 0.01}) {
        auto build = [&](std::function<bool(const Shot&)> tr) {
            Cand c1 = cs[0], c2 = cs[1];
            LinearModel f0 = fitpred(c1, [&](const Shot& x) { return tr(x) && std::abs(x.t) < 1e-9; }, 0.0);
            LinearModel tt = fitpred(c2, tr, 0.0);
            double sxx = 0, sxy = 0; for (auto& x : c.shots) if (inr(x) && tr(x)) { double e = f0.predict(c1.f(x.idx)).x - x.y; sxx += x.t * x.t; sxy += x.t * e; }
            double k = sxy / sxx;
            return std::function<double(int)>([=](int i) { double Th = tt.predict(c2.f(i)).y; return f0.predict(c1.f(i)).x - k * soft(Th, th); });
        };
        std::map<int, std::function<double(int)>> fold4; for (int f = 0; f < 4; ++f) fold4[f] = build([&](const Shot& x) { return has(x, "1014") && x.fold_mixed != f; });
        auto only14 = build([&](const Shot& x) { return has(x, "1014"); });
        std::map<int, std::map<int, std::function<double(int)>>> strat; std::map<int, std::vector<int>> fold_of;
        for (int sd = 0; sd < 10; ++sd) {
            std::map<std::string, std::vector<std::pair<std::pair<double, double>, int>>> by; std::map<int, int> gf; std::set<int> seen;
            for (auto& x : c.shots) if (inr(x) && x.condition != "center" && !seen.count(x.group)) { seen.insert(x.group); by[x.set].push_back({{x.y, x.t}, x.group}); }
            for (auto& [st, v] : by) { std::sort(v.begin(), v.end()); for (size_t b = 0; b < v.size(); b += 5) { std::vector<int> perm(5); std::iota(perm.begin(), perm.end(), 0);
                if (sd) { unsigned xx = unsigned(sd) * 2654435761u + unsigned(b) * 40503u; for (int j = 4; j > 0; --j) { xx = xx * 1103515245u + 12345u; std::swap(perm[j], perm[(xx >> 16) % (j + 1)]); } }
                for (size_t i = b; i < std::min(v.size(), b + 5); ++i) gf[v[i].second] = perm[i - b]; } }
            fold_of[sd].assign(N, -1); for (auto& x : c.shots) if (gf.count(x.group) && x.condition != "center") fold_of[sd][x.idx] = gf[x.group];
            for (int f = 0; f < 5; ++f) strat[sd][f] = build([&](const Shot& x) { return fold_of[sd][x.idx] != f; });
        }
        row("H 문턱 보정 theta=" + num(th, 4), [&](const Shot& x, int proto) {
            if (proto == 0) return x.fold_mixed >= 0 ? fold4[x.fold_mixed](x.idx) : NAN;
            if (proto == 1) return only14(x.idx);
            int sd = proto - 2; int fo = fold_of[sd][x.idx]; return fo < 0 ? NAN : strat[sd][fo](x.idx); });
    }
    t.csv(res / "scores.csv");
    std::cout << "done\n";
}
