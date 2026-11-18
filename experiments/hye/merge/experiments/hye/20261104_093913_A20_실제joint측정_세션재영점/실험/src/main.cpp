// A20: 실제 joint 측정값(QR ECC 54 + 가로선 48, A19 에서 575장 추출)으로
//  (1) 세션 센터 사진 기준 재영점(그 세션의 center 측정값을 뺀다)의 효과,
//  (2) 날짜 층화 5-fold(10분할) 와 "1014 로만 학습 → 1020·1029 평가" 두 분할에서 |eT| <= 0.01 통과를 본다.
//  세션 기준: 1014 = center.bmp, 1020 = 기준 위치/center1.bmp, 1029 = 자기 center 가 없어 1020 center1.
//  center 사진은 기준으로만 쓰고 평가에서 뺀다. T+2.00(학습 범위 밖)도 평가에서 뺀다.
#include "cache.hpp"
#include "model.hpp"
#include <iostream>
#include <map>
#include <numeric>
using namespace a0;

static std::vector<int> folds(const Cache& c, int k, unsigned seed) {
    std::map<std::string, std::vector<std::pair<std::pair<double, double>, int>>> by_set;
    std::map<int, std::pair<double, double>> gyt; std::map<int, std::string> gset;
    for (auto& s : c.shots) if (s.measured && s.condition != "center") { gyt[s.group] = {s.y, s.t}; gset[s.group] = s.set; }
    for (auto& [g, yt] : gyt) by_set[gset[g]].push_back({yt, g});
    std::map<int, int> gf;
    for (auto& [set, v] : by_set) {
        std::sort(v.begin(), v.end());
        for (size_t b = 0; b < v.size(); b += k) {
            std::vector<int> perm(k); std::iota(perm.begin(), perm.end(), 0);
            if (seed) { unsigned x = seed * 2654435761u + unsigned(b) * 40503u; for (int j = k - 1; j > 0; --j) { x = x * 1103515245u + 12345u; std::swap(perm[j], perm[(x >> 16) % (j + 1)]); } }
            for (size_t i = b; i < std::min(v.size(), b + size_t(k)); ++i) gf[v[i].second] = perm[i - b];
        }
    }
    std::vector<int> f(c.n(), -1);
    for (auto& s : c.shots) if (s.measured && s.condition != "center") f[s.idx] = gf[s.group];
    return f;
}

int main() {
    const fs::path results = fs::u8path(EXPERIMENT_DIR) / "results";
    fs::create_directories(results);
    Cache c = load(cache_dir());
    const int N = c.n();
    std::vector<double> raw(size_t(N) * 102);
    { std::ifstream in(fs::u8path(REPO_DIR) / fs::u8path(u8"experiments/hye/20261104_093558_A19_QR선측정_안정성점검/결과/cache/obs102.f64"), std::ios::binary);
      in.read(reinterpret_cast<char*>(raw.data()), raw.size() * sizeof(double)); if (!in) throw std::runtime_error("obs102 cache"); }
    auto obs = [&](int i) { return std::vector<double>(raw.begin() + size_t(i) * 102, raw.begin() + size_t(i) * 102 + 102); };
    int c14 = -1, c20 = -1; std::vector<int> c20all;
    for (auto& s : c.shots) { if (s.rel == "center.bmp" && s.set.find("1014") != std::string::npos) c14 = s.idx; if (s.rel.find("center1.bmp") != std::string::npos) c20 = s.idx; if (s.condition == "center" && s.set.find("1020") != std::string::npos) c20all.push_back(s.idx); }
    std::vector<double> c20m(102, 0); for (int i : c20all) { auto o = obs(i); for (int j = 0; j < 102; ++j) c20m[j] += o[j] / c20all.size(); }
    auto ref = [&](const Shot& s, int mode) -> std::vector<double> {
        if (mode == 0 || s.set.find("1014") != std::string::npos) return obs(c14);   // mode0: 모든 세션이 1014 center 하나 = 지금 joint
        return mode == 1 ? obs(c20) : c20m;                                            // mode1: center1, mode2: center1~3 평균
    };
    static const double kLineX[48] = {432,496,560,624,688,752,816,880,944,1008,1072,1136,2732,2796,2860,2924,2988,3052,3116,3180,3244,3308,3372,3436,
                                      432,496,560,624,688,752,816,880,944,1008,1072,1136,2632,2696,2760,2824,2888,2952,3016,3080,3144,3208,3272,3336};
    // 저차원 물리 특징: s = QR 평균 가로 이동, QR 평균 세로 이동, 위·아래 선의 (절편, 기울기). 허용 관측(QR + 위·아래 바)만 쓴다.
    auto compact = [&](const std::vector<double>& x, int kind) {
        double sx = 0, sy = 0; for (int i = 0; i < 54; i += 2) { sx += x[i] / 27; sy += x[i + 1] / 27; }
        std::vector<double> f = {sx};
        for (int g = 0; g < 2; ++g) { double a = 0, b = 0, xx = 0, xy = 0, mx = 0, my = 0; int n = 24;
            for (int k = g * 24; k < g * 24 + 24; ++k) { mx += (kLineX[k] - 1934) / 1000.0 / n; my += x[54 + k] / n; }
            for (int k = g * 24; k < g * 24 + 24; ++k) { double u = (kLineX[k] - 1934) / 1000.0 - mx; xx += u * u; xy += u * (x[54 + k] - my); }
            b = xy / xx; a = my - b * mx; if (kind >= 1) f.push_back(a); f.push_back(b); }
        if (kind >= 2) f.push_back(sy);
        if (kind >= 3) { size_t m = f.size(); for (size_t i = 0; i < m; ++i) f.push_back(f[i] * sx / 100.0); }   // s 와의 곱 (Θ 1차식 계수가 이동량에 따라 바뀌는 효과)
        return f;
    };
    auto fmap = [](const std::vector<double>& x, bool inter) {   // joint feature_map 과 같은 상호작용 항
        if (!inter) return x;
        std::vector<double> z = x; double h = 0; for (int i = 0; i < 54; i += 2) h += x[i]; h /= 2700.;
        for (int i = 0; i < 102; ++i) z.push_back(x[i] * h); return z;
    };
    Table t; t.head = {"기준", "특징", "분할", "평가", "장수", "T 통과 |eT|<=0.01", "최소(10분할)", "T 평균", "T 최대", "Y 평균 (참고)"};
    Table bad; bad.head = {"기준", "특징", "분할", "사진", "Y", "T", "eY", "eT"};
    const char* refname[] = {"1014 center 하나 (지금 joint)", "세션 center1", "세션 center1~3 평균"};
    const char* fnames[] = {"102", "204 (joint 상호작용)", "3: s + 선 기울기 2", "5: s + 선 (절편, 기울기) 2", "6: + QR 세로", "12: 6 + s 곱"};
    for (int mode = 0; mode < 3; ++mode) for (int inter = 0; inter < 6; ++inter) {
        std::vector<std::vector<double>> F(N);
        for (auto& s : c.shots) if (s.measured) { auto o = obs(s.idx), r = ref(s, mode); for (int j = 0; j < 102; ++j) o[j] -= r[j]; F[s.idx] = inter < 2 ? fmap(o, inter) : compact(o, inter - 2); }
        std::string fname = fnames[inter];
        auto eval_set = [&](const std::string& split, std::function<int(const Shot&)> foldf, int nf, const std::vector<std::string>& sets, bool collect, double& pass_out) {
            auto o = run_oof(c, F, foldf, nf, 0.01);
            for (auto& d : sets) {
                std::vector<double> et, ey;
                for (size_t k = 0; k < o.shot.size(); ++k) { auto& s = c.shots[o.shot[k]]; if (s.condition == "center" || s.t > 1.001) continue; if (d != "all" && s.set.find(d) == std::string::npos) continue;
                    et.push_back(std::abs(o.pred[k].y - o.truth[k].y)); ey.push_back(std::abs(o.pred[k].x - o.truth[k].x));
                    if (collect && et.back() > 0.01) bad.add({refname[mode], fname, split, s.name(), num(s.y, 3), num(s.t, 3), num(o.pred[k].x - o.truth[k].x, 4), num(o.pred[k].y - o.truth[k].y, 4)}); }
                if (et.empty()) continue;
                int p = 0; for (double v : et) p += v <= 0.01 + 1e-12;
                if (d == "all") pass_out = 100.0 * p / et.size();
                if (collect) t.add({refname[mode], fname, split, d, std::to_string(et.size()), std::to_string(p) + "/" + std::to_string(et.size()) + " (" + num(100.0 * p / et.size(), 1) + "%)", "", num(mean(et), 4), num(max_abs(et), 4), num(mean(ey), 4)});
            }
        };
        // 날짜 층화 5-fold, seed 0 상세 + 10분할 최소
        std::vector<double> ps;
        for (unsigned sd = 0; sd < 10; ++sd) { auto f = folds(c, 5, sd); double pa = 0; eval_set("날짜층화 5-fold", [&](const Shot& s) { return f[s.idx]; }, 5, {"all", "20261014", "20261020", "20261029"}, sd == 0, pa); ps.push_back(pa); }
        t.add({refname[mode], fname, "날짜층화 5-fold", "10분할 전체 범위", "", num(*std::min_element(ps.begin(), ps.end()), 1) + "% ~ " + num(*std::max_element(ps.begin(), ps.end()), 1) + "%", "", "", "", ""});
        // 1014 로만 학습 → 1020, 1029 평가 (fold 0 = 시험, 1014 는 학습에만)
        double dummy = 0;
        eval_set("1014로만 학습", [&](const Shot& s) { return s.set.find("1014") != std::string::npos ? -1 : 0; }, 1, {"20261020", "20261029"}, true, dummy);
    }
    // 2부: 기울기 기반 저차원 모델 다듬기 (재영점 없음 = 지금 joint 와 같은 1014 center 하나)
    {
        auto slopes = [&](const std::vector<double>& x, int seg) {  // seg=2: 위·아래 전체, seg=4: 위왼·위오·아래왼·아래오
            std::vector<double> b; int per = 48 / seg;
            for (int g = 0; g < seg; ++g) { double mx = 0, my = 0, xx = 0, xy = 0;
                for (int k = g * per; k < g * per + per; ++k) { mx += (kLineX[k] - 1934) / 1000.0 / per; my += x[54 + k] / per; }
                for (int k = g * per; k < g * per + per; ++k) { double u = (kLineX[k] - 1934) / 1000.0 - mx; xx += u * u; xy += u * (x[54 + k] - my); }
                b.push_back(xy / xx); }
            return b; };
        auto sx_of = [](const std::vector<double>& x) { double s = 0; for (int i = 0; i < 54; i += 2) s += x[i] / 27; return s; };
        struct V { std::string name; std::function<std::vector<double>(const std::vector<double>&)> f; };
        std::vector<V> vs = {
            {"A s + 기울기(위, 아래)", [&](const std::vector<double>& x) { auto b = slopes(x, 2); return std::vector<double>{sx_of(x), b[0], b[1]}; }},
            {"B s + 기울기 차(위-아래)", [&](const std::vector<double>& x) { auto b = slopes(x, 2); return std::vector<double>{sx_of(x), b[0] - b[1]}; }},
            {"C s + 구간 기울기 4", [&](const std::vector<double>& x) { auto b = slopes(x, 4); return std::vector<double>{sx_of(x), b[0], b[1], b[2], b[3]}; }},
            {"D A + 2차 항", [&](const std::vector<double>& x) { auto b = slopes(x, 2); double s = sx_of(x) / 100; std::vector<double> z = {s, b[0], b[1]}, f = z; for (size_t i = 0; i < 3; ++i) for (size_t j = i; j < 3; ++j) f.push_back(z[i] * z[j]); return f; }},
            {"E A + s 곱", [&](const std::vector<double>& x) { auto b = slopes(x, 2); double s = sx_of(x); return std::vector<double>{s, b[0], b[1], s * b[0] / 100, s * b[1] / 100}; }},
            {"F QR 9점 가로 + 기울기(위, 아래)", [&](const std::vector<double>& x) { auto b = slopes(x, 2); std::vector<double> f; for (int i = 0; i < 54; i += 2) f.push_back(x[i]); f.push_back(b[0]); f.push_back(b[1]); return f; }},
            {"G QR 27점 가로·세로 54 + 기울기 2", [&](const std::vector<double>& x) { auto b = slopes(x, 2); std::vector<double> f(x.begin(), x.begin() + 54); f.push_back(b[0]); f.push_back(b[1]); return f; }},
        };
        Table nr; nr.head = {"특징", "분할", "범위", "장수", "T 통과", "T 최대", "실패 사진"};
        Table u; u.head = {"특징", "차원", "날짜층화 5-fold 전체 (10분할 범위)", "그때 1020", "그때 T 최대", "1014만 학습→1020", "→1020 T 최대", "1014만 학습→1029", "→1029 T 최대", "→1029 Y 평균"};
        for (auto& v : vs) for (double lam : {0.0, 0.01}) {
            std::vector<std::vector<double>> F(N);
            for (auto& s : c.shots) if (s.measured) { auto o = obs(s.idx), r = obs(c14); for (int j = 0; j < 102; ++j) o[j] -= r[j]; F[s.idx] = v.f(o); }
            auto score = [&](const Oof& o, const std::string& d, double& mx, double& my) { int n = 0, p = 0; mx = 0; my = 0;
                for (size_t k = 0; k < o.shot.size(); ++k) { auto& s = c.shots[o.shot[k]]; if (s.condition == "center" || s.t > 1.001) continue; if (d != "all" && s.set.find(d) == std::string::npos) continue;
                    double e = std::abs(o.pred[k].y - o.truth[k].y); ++n; p += e <= 0.01 + 1e-12; mx = std::max(mx, e); my += std::abs(o.pred[k].x - o.truth[k].x); }
                my /= std::max(1, n); return std::make_pair(p, n); };
            double lo = 101, hi = -1, m20 = 0, mall = 0; std::pair<int, int> p20;
            for (unsigned sd = 0; sd < 10; ++sd) { auto f = folds(c, 5, sd); auto o = run_oof(c, F, [&](const Shot& s) { return f[s.idx]; }, 5, lam);
                double a, b; auto pa = score(o, "all", a, b); double r = 100.0 * pa.first / pa.second; lo = std::min(lo, r); hi = std::max(hi, r); mall = std::max(mall, a);
                if (sd == 0) p20 = score(o, "20261020", m20, b); }
            auto o2 = run_oof(c, F, [&](const Shot& s) { return s.set.find("1014") != std::string::npos ? -1 : 0; }, 1, lam);
            double x20, y20, x29, y29; auto q20 = score(o2, "20261020", x20, y20), q29 = score(o2, "20261029", x29, y29);
            if (lam == 0 && (v.name[0] == 'A' || v.name[0] == 'B')) {
                auto near = [&](const Oof& o, const std::string& split, double tmax) { int n = 0, p = 0; double mx = 0; std::string fails;
                    for (size_t k = 0; k < o.shot.size(); ++k) { auto& s = c.shots[o.shot[k]]; if (s.condition == "center" || std::abs(s.t) > tmax + 1e-9) continue;
                        double e = std::abs(o.pred[k].y - o.truth[k].y); ++n; p += e <= 0.01 + 1e-12; mx = std::max(mx, e); if (e > 0.01) fails += s.name() + " (" + num(e, 4) + ") "; }
                    nr.add({v.name, split, "|T| <= " + num(tmax, 2), std::to_string(n), std::to_string(p) + "/" + std::to_string(n), num(mx, 4), fails}); };
                auto f0 = folds(c, 5, 0); auto o1 = run_oof(c, F, [&](const Shot& s) { return f0[s.idx]; }, 5, 0);
                for (double tm : {0.1, 0.2, 0.5, 1.0}) { near(o1, "날짜층화 5-fold (seed0)", tm); near(o2, "1014만 학습 → 1020+1029", tm); }
            }
            u.add({v.name + (lam > 0 ? " (ridge 0.01)" : ""), std::to_string(F[c14].size()), num(lo, 1) + "~" + num(hi, 1) + "%", std::to_string(p20.first) + "/" + std::to_string(p20.second), num(mall, 4),
                   std::to_string(q20.first) + "/" + std::to_string(q20.second), num(x20, 4), std::to_string(q29.first) + "/" + std::to_string(q29.second), num(x29, 4), num(y29, 4)});
        }
        u.csv(results / "compact_models.csv"); nr.csv(results / "near_target.csv");
    }
    t.csv(results / "scores.csv"); bad.csv(results / "t_outliers.csv");
    std::cout << "done\n";
}
