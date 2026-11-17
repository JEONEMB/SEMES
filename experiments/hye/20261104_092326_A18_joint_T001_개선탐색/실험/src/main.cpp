// A18: joint T 를 날짜 층화 5-fold 에서 |eT| <= 0.01 안으로 넣기 위한 탐색.
//  A17 의 남은 오차는 Y<->T 누출 (eY ~ -5 eT). 선은 T 에 민감하고 Y 에 둔하므로 선 정보를 잡음 적게, 물리적으로 묶어 쓰는 변형들을 비교한다.
//  모든 변형은 같은 10가지 분할(seed 0..9)에서 OOF 로 평가한다. 시험 정답은 학습/선택에 쓰지 않는다.
#include "cache.hpp"
#include "model.hpp"
#include <iostream>
#include <map>
#include <numeric>

using namespace a0;

static const double kLineX[48] = {432,496,560,624,688,752,816,880,944,1008,1072,1136,2732,2796,2860,2924,2988,3052,3116,3180,3244,3308,3372,3436,
                                  432,496,560,624,688,752,816,880,944,1008,1072,1136,2632,2696,2760,2824,2888,2952,3016,3080,3144,3208,3272,3336};

static std::vector<int> folds(const Cache& c, int k, unsigned seed) {
    std::map<std::string, std::vector<std::pair<std::pair<double, double>, int>>> by_set;
    std::map<int, std::pair<double, double>> gyt; std::map<int, std::string> gset;
    for (auto& s : c.shots) if (s.measured && s.fold_mixed >= 0) { gyt[s.group] = {s.y, s.t}; gset[s.group] = s.set; }
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
    for (auto& s : c.shots) if (s.measured && s.fold_mixed >= 0) f[s.idx] = gf[s.group];
    return f;
}

// 선 구간(a: 위왼, 위오, 아래왼, 아래오)마다 이동량 = a + b*(x - 중심)/1000 를 유효 블록으로 맞춘다.
static std::vector<double> line_fit(const Cache& c, int i, bool split) {
    std::vector<double> out;
    int segs = split ? 4 : 2, per = 48 / segs;
    for (int g = 0; g < segs; ++g) {
        double sx = 0, sy = 0, sxx = 0, sxy = 0; int n = 0;
        for (int b = g * per; b < (g + 1) * per; ++b) {
            if (!c.line_ok(i)[b]) continue;
            double x = (kLineX[b] - 1934.0) / 1000.0, y = c.line(i)[b * 2];
            sx += x; sy += y; sxx += x * x; sxy += x * y; ++n;
        }
        double den = n * sxx - sx * sx;
        double bb = (n > 1 && std::abs(den) > 1e-12) ? (n * sxy - sx * sy) / den : 0, aa = n ? (sy - bb * sx) / n : 0;
        out.push_back(aa); out.push_back(bb);
    }
    return out;
}

// 강건 학습: 1차 적합 후 학습 잔차가 큰 사진(T 잔차 > k*MAD)을 빼고 다시 맞춘다. 반복 묶음은 평균 특징으로 묶어 학습할 수도 있다.
static Oof run_oof_robust(const Cache& c, const std::vector<std::vector<double>>& feat, const std::vector<int>& fold, int nf, double lam, double kmad, bool group_mean) {
    Oof out;
    for (int f = 0; f < nf; ++f) {
        std::vector<std::vector<double>> X; std::vector<cv::Point2d> Y;
        std::map<int, std::pair<std::vector<double>, int>> gacc; std::map<int, cv::Point2d> gy;
        for (auto& s : c.shots) {
            if (!s.measured || fold[s.idx] == f) continue;
            if (group_mean && s.group >= 0) {
                auto& a = gacc[s.group]; if (a.first.empty()) a.first.assign(feat[s.idx].size(), 0);
                for (size_t j = 0; j < a.first.size(); ++j) a.first[j] += feat[s.idx][j];
                ++a.second; gy[s.group] = {s.y, s.t};
            } else { X.push_back(feat[s.idx]); Y.push_back({s.y, s.t}); }
        }
        for (auto& [g, a] : gacc) { for (auto& v : a.first) v /= a.second; X.push_back(a.first); Y.push_back(gy[g]); }
        LinearModel m; m.fit(X, Y, lam);
        if (kmad > 0) {
            std::vector<double> r(X.size()); for (size_t i = 0; i < X.size(); ++i) r[i] = std::abs(m.predict(X[i]).y - Y[i].y);
            double md = median(r) * 1.4826;
            std::vector<std::vector<double>> X2; std::vector<cv::Point2d> Y2;
            for (size_t i = 0; i < X.size(); ++i) if (r[i] <= kmad * md) { X2.push_back(X[i]); Y2.push_back(Y[i]); }
            m.fit(X2, Y2, lam);
        }
        for (auto& s : c.shots) {
            if (!s.measured || fold[s.idx] != f) continue;
            out.shot.push_back(s.idx); out.truth.push_back({s.y, s.t}); out.pred.push_back(m.predict(feat[s.idx])); out.fold.push_back(f);
        }
    }
    return out;
}

// 촬영 구간(trial)별 영점 오프셋을 학습에서 함께 추정한다. 1014 는 S1..S6, 1020/1029 는 날짜 하나를 한 구간으로 본다.
// 모델: 명령 T = f(x) + o_trial. 예측은 f(x) (구간 정보 없이 사진만). 보정 라벨 = 명령 - o_trial (o 는 학습 fold 에서만 추정).
static std::vector<int> trial_of(const Cache& c) {
    std::map<std::string, int> tr;
    std::ifstream in(fs::u8path(REPO_DIR) / fs::u8path(u8"experiments/hye/20261022_1750_trial원점_절편외삽역산/결과/per_shot_zero_estimates.csv"));
    std::string line; std::getline(in, line);
    while (std::getline(in, line)) { auto r = split_csv(line); tr[r[1]] = r[0][1] - '1'; }
    std::vector<int> t(c.n(), -1);
    for (auto& s : c.shots) {
        if (s.set.find("20261014") != std::string::npos) { auto it = tr.find(s.rel); t[s.idx] = it == tr.end() ? -1 : it->second; }
        else if (s.set.find("20261020") != std::string::npos) t[s.idx] = 6;
        else t[s.idx] = 7;
    }
    return t;
}
static const int kTrials = 8;

struct Eval { double pass_mean, pass_min, max_worst, max_mean, ypass_mean; int fails_worst; };

int main() {
    const fs::path results = fs::u8path(EXPERIMENT_DIR) / "results";
    fs::create_directories(results);
    Cache c = load(cache_dir());
    const int N = c.n();
    std::vector<std::vector<double>> qr(N), qrx(N), l4(N), l8(N), raw(N);
    std::vector<double> H(N, 0);
    for (auto& s : c.shots) if (s.measured) {
        int i = s.idx;
        for (int q = 0; q < 3; ++q) { auto a = qr_affine(c, i, q); for (int j = 0; j < 6; ++j) qr[i].push_back(a[j]); for (int j = 0; j < 3; ++j) qrx[i].push_back(a[j]); H[i] += a[0] / 3; }
        l4[i] = line_fit(c, i, false); l8[i] = line_fit(c, i, true);
        raw[i] = features(c, i, true);
    }
    auto cat = [](std::vector<double> a, const std::vector<double>& b) { a.insert(a.end(), b.begin(), b.end()); return a; };
    auto quad = [&](int i, const std::vector<double>& base) {
        // H 와 선 기울기·절편의 2차 항 (회전 투영의 비선형, Y x T 상호작용)
        std::vector<double> f = base; double h = H[i] / 100.0;
        std::vector<double> z = {h, l4[i][0], l4[i][1], l4[i][2], l4[i][3]};
        for (size_t a = 0; a < z.size(); ++a) for (size_t b = a; b < z.size(); ++b) f.push_back(z[a] * z[b]);
        return f;
    };
    std::map<std::string, std::vector<std::vector<double>>> F;
    for (int i = 0; i < N; ++i) if (c.shots[i].measured) {
        F["V0 66d (기준)"].push_back(raw[i]);
        F["V1 QR18+선4"].push_back(cat(qr[i], l4[i]));
        F["V2 QR18+선8"].push_back(cat(qr[i], l8[i]));
        F["V3 QR18+선8+2차"].push_back(quad(i, cat(qr[i], l8[i])));
        F["V4 66d+2차"].push_back(quad(i, raw[i]));
        F["V5 QRx9+선8"].push_back(cat(qrx[i], l8[i]));
        F["V6 QRx9+선8+2차"].push_back(quad(i, cat(qrx[i], l8[i])));
    } else for (auto& kv : F) kv.second.push_back({});
    // map 이 처음 비어 있을 때 측정 안 된 사진 자리 맞추기
    for (auto& kv : F) kv.second.resize(N);
    for (int i = 0; i < N; ++i) if (!c.shots[i].measured) for (auto& kv : F) kv.second[i].clear();

    std::vector<double> lambdas = {0, 1e-3, 1e-2, 1e-1};
    Table t; t.head = {"변형", "ridge", "차원", "T 통과 평균 (10분할)", "T 통과 최소", "실패 장수 최대", "T 최대 평균", "T 최대 최악", "Y≤0.045 평균 (참고)"};
    Table per; per.head = {"변형", "ridge", "seed", "사진", "Y", "T", "eY", "eT"};
    for (auto& [name, feat] : F) for (double lam : lambdas) {
        std::vector<double> pass, mx, yp; int fw = 0;
        for (unsigned sd = 0; sd < 10; ++sd) {
            auto f = folds(c, 5, sd);
            auto o = run_oof(c, feat, [&](const Shot& s) { return f[s.idx]; }, 5, lam);
            int n = 0, p = 0, py = 0; double m = 0;
            for (size_t k = 0; k < o.shot.size(); ++k) {
                if (o.truth[k].y > 1.001) continue;
                double et = std::abs(o.pred[k].y - o.truth[k].y), ey = std::abs(o.pred[k].x - o.truth[k].x);
                ++n; p += et <= 0.01 + 1e-12; py += ey <= 0.045 + 1e-12; m = std::max(m, et);
                if (et > 0.01 && sd == 0) per.add({name, num(lam, 3), std::to_string(sd), c.shots[o.shot[k]].name(), num(o.truth[k].x, 3), num(o.truth[k].y, 3),
                                                    num(o.pred[k].x - o.truth[k].x, 4), num(o.pred[k].y - o.truth[k].y, 4)});
            }
            pass.push_back(100.0 * p / n); mx.push_back(m); yp.push_back(100.0 * py / n); fw = std::max(fw, n - p);
        }
        t.add({name, num(lam, 3), std::to_string(feat[0].size()), num(mean(pass), 2) + "%", num(*std::min_element(pass.begin(), pass.end()), 2) + "%",
               std::to_string(fw), num(mean(mx), 4), num(*std::max_element(mx.begin(), mx.end()), 4), num(mean(yp), 1) + "%"});
        std::cout << name << " " << lam << " " << mean(pass) << "\n";
    }
    // 오차 분해: 같은 자세 반복 묶음(1020, 1029)에서 묶음 평균 오차(편향)와 묶음 안 흩어짐(잡음)
    {
        auto f = folds(c, 5, 0);
        auto o = run_oof(c, F["V0 66d (기준)"], [&](const Shot& s) { return f[s.idx]; }, 5, 0.01);
        std::map<int, std::vector<double>> g;
        std::map<int, std::string> gn;
        for (size_t k = 0; k < o.shot.size(); ++k) { auto& s = c.shots[o.shot[k]]; if (o.truth[k].y > 1.001) continue; g[s.group].push_back(o.pred[k].y - o.truth[k].y); gn[s.group] = s.name(); }
        Table d; d.head = {"묶음", "장수", "eT 1", "eT 2", "eT 3", "묶음 평균", "묶음 안 SD", "3장 평균 |e|"};
        std::vector<double> bias, within, avg3;
        for (auto& [gid, v] : g) if (v.size() >= 3) {
            double m = mean(v), sdv = sd(v); bias.push_back(m); within.push_back(sdv); avg3.push_back(std::abs(m));
            d.add({gn[gid], std::to_string(v.size()), num(v[0], 4), num(v[1], 4), num(v[2], 4), num(m, 4), num(sdv, 4), num(std::abs(m), 4)});
        }
        d.csv(results / "repeat_decomposition.csv");
        Table pr; pr.head = {"shot", "set", "Y", "T", "predY", "predT", "eY", "eT"};
        for (size_t k = 0; k < o.shot.size(); ++k) { auto& s = c.shots[o.shot[k]]; pr.add({std::to_string(s.idx), s.set.substr(6), num(o.truth[k].x, 3), num(o.truth[k].y, 3), num(o.pred[k].x, 5), num(o.pred[k].y, 5), num(o.pred[k].x - o.truth[k].x, 5), num(o.pred[k].y - o.truth[k].y, 5)}); }
        pr.csv(results / "oof_v0_ridge001_seed0.csv");
        int p1 = 0, n1 = 0, p3 = 0;
        for (auto& [gid, v] : g) if (v.size() >= 3) { for (double e : v) { ++n1; p1 += std::abs(e) <= 0.01; } p3 += std::abs(mean(v)) <= 0.01; }
        std::cout << "반복묶음 " << bias.size() << "개: 묶음평균 SD " << sd(bias) << " max|평균| " << max_abs(bias) << ", 묶음안 SD 중앙 " << median(within)
                  << " 최대 " << max_abs(within) << " | 1장 통과 " << p1 << "/" << n1 << ", 3장 평균 통과 " << p3 << "/" << bias.size() << "\n";
    }
    // 강건 학습과 턴당 N장 평균
    {
        Table r; r.head = {"학습", "평가", "T 통과 평균 (10분할)", "T 통과 최소", "실패 최대", "T 최대 최악"};
        struct Cfg { std::string name; double k; bool gm; };
        std::vector<Cfg> cfgs = {{"기준 (ridge 0.01)", 0, false}, {"잔차 3MAD 제외 후 재적합", 3, false}, {"잔차 2.5MAD 제외", 2.5, false}, {"반복 묶음 평균 특징으로 학습", 0, true}, {"묶음 평균 + 3MAD 제외", 3, true}};
        for (auto& cf : cfgs) for (int mode = 0; mode < 2; ++mode) {
            std::vector<double> pass; int fw = 0; double mw = 0;
            for (unsigned sd = 0; sd < 10; ++sd) {
                auto f = folds(c, 5, sd);
                auto o = run_oof_robust(c, F["V0 66d (기준)"], f, 5, 0.01, cf.k, cf.gm);
                int n = 0, p = 0; double m = 0;
                if (mode == 0) {
                    for (size_t k = 0; k < o.shot.size(); ++k) { if (o.truth[k].y > 1.001) continue; double e = std::abs(o.pred[k].y - o.truth[k].y); ++n; p += e <= 0.01 + 1e-12; m = std::max(m, e); }
                } else {  // 반복 묶음(같은 자세 3장)이 있는 사진은 묶음 평균 예측으로 판정. 1014 단발 사진은 1장 그대로.
                    std::map<int, std::vector<double>> g;
                    for (size_t k = 0; k < o.shot.size(); ++k) { if (o.truth[k].y > 1.001) continue; g[c.shots[o.shot[k]].group].push_back(o.pred[k].y - o.truth[k].y); }
                    for (auto& [gid, v] : g) { double e = std::abs(mean(v)); ++n; p += e <= 0.01 + 1e-12; m = std::max(m, e); }
                }
                pass.push_back(100.0 * p / n); fw = std::max(fw, n - p); mw = std::max(mw, m);
            }
            r.add({cf.name, mode ? "같은 자세 반복 평균 (묶음 단위)" : "1장", num(mean(pass), 2) + "%", num(*std::min_element(pass.begin(), pass.end()), 2) + "%", std::to_string(fw), num(mw, 4)});
        }
        r.csv(results / "robust_and_average.csv");
    }
    {
        auto tri = trial_of(c);
        int miss = 0; for (auto& s : c.shots) if (s.measured && tri[s.idx] < 0) ++miss;
        std::cout << "구간 없는 사진 " << miss << "\n";
        Table r; r.head = {"학습", "평가 라벨", "단위", "T 통과 평균 (10분할)", "T 통과 최소", "실패 최대", "T 최대 최악", "구간 오프셋 T (seed0 fold0)"};
        for (int gmode = 0; gmode < 2; ++gmode) for (int lab = 0; lab < 2; ++lab) {
            std::vector<double> pass; int fw = 0; double mw = 0; std::string offs;
            for (unsigned sd = 0; sd < 10; ++sd) {
                auto f = folds(c, 5, sd);
                std::map<int, std::vector<double>> g;
                int n = 0, p = 0; double m = 0;
                for (int fo = 0; fo < 5; ++fo) {
                    std::vector<std::vector<double>> X; std::vector<cv::Point2d> Y;
                    for (auto& s : c.shots) {
                        if (!s.measured || f[s.idx] == fo) continue;
                        auto x = F["V0 66d (기준)"][s.idx];
                        for (int k = 0; k < kTrials; ++k) x.push_back(tri[s.idx] == k ? 1.0 : 0.0);
                        X.push_back(x); Y.push_back({s.y, s.t});
                    }
                    LinearModel mdl; mdl.fit(X, Y, 0.01);
                    // f(x): 구간 열을 '평균 구간'(가중 평균)이 아니라 0 으로 둔다 = 기준(center 가 속한) 영점
                    std::vector<double> o(kTrials);
                    for (int k = 0; k < kTrials; ++k) { int col = 66 + k; o[k] = mdl.W.at<double>(col, 1) * (1.0 - mdl.mu[col]) / mdl.sd_[col] - mdl.W.at<double>(col, 1) * (0.0 - mdl.mu[col]) / mdl.sd_[col]; }
                    if (sd == 0 && fo == 0) for (int k = 0; k < kTrials; ++k) offs += num(o[k], 4) + (k + 1 < kTrials ? " " : "");
                    for (auto& s : c.shots) {
                        if (!s.measured || f[s.idx] != fo || s.t > 1.001) continue;
                        auto x = F["V0 66d (기준)"][s.idx];
                        for (int k = 0; k < kTrials; ++k) x.push_back(0.0);
                        double pred = mdl.predict(x).y;
                        double truth = s.t - (lab && tri[s.idx] >= 0 ? o[tri[s.idx]] : 0.0);
                        double e = pred - truth;
                        if (gmode) g[s.group].push_back(e);
                        else { ++n; p += std::abs(e) <= 0.01 + 1e-12; m = std::max(m, std::abs(e)); }
                    }
                }
                if (gmode) for (auto& [gid, v] : g) { double e = std::abs(mean(v)); ++n; p += e <= 0.01 + 1e-12; m = std::max(m, e); }
                pass.push_back(100.0 * p / n); fw = std::max(fw, n - p); mw = std::max(mw, m);
            }
            r.add({"구간 오프셋 함께 추정", lab ? "명령 - 구간 오프셋 (보정 라벨)" : "명령값 그대로", gmode ? "같은 자세 묶음 평균" : "1장",
                   num(mean(pass), 2) + "%", num(*std::min_element(pass.begin(), pass.end()), 2) + "%", std::to_string(fw), num(mw, 4), offs});
        }
        r.csv(results / "trial_offset.csv");
    }
    // 판정: 서로 다른 관측으로 낸 T 두 개가 어긋나면 재촬영. 기준값은 학습 fold 의 OOF 불일치 분포(내부 5-fold)에서 정한다.
    {
        std::vector<std::vector<double>> fl(N), fq(N);
        for (int i = 0; i < N; ++i) if (c.shots[i].measured) {
            fl[i].assign(raw[i].begin() + 18, raw[i].end()); fl[i].push_back(H[i]);        // 선 48 + 판 가로 이동
            fq[i] = qr[i];                                                                 // QR 18
        }
        Table r; r.head = {"판정 규칙", "기준 분위", "T 통과 (받아들인 사진) 평균", "최소", "받아들인 사진 중 실패 최대", "받은 사진 T 최대 최악", "재촬영 비율 평균", "튀는 사진(>0.01) 잡은 비율"};
        for (int rule = 0; rule < 2; ++rule) for (double qq : {0.90, 0.95, 0.98}) {
            std::vector<double> pass, rej, caught; int fw = 0; double mw = 0;
            for (unsigned sd = 0; sd < 10; ++sd) {
                auto f = folds(c, 5, sd);
                int n = 0, p = 0, nr = 0, nbad = 0, nbadc = 0; double m = 0;
                for (int fo = 0; fo < 5; ++fo) {
                    auto fit = [&](const std::vector<std::vector<double>>& Fx, std::function<bool(const Shot&)> use) {
                        std::vector<std::vector<double>> X; std::vector<cv::Point2d> Y;
                        for (auto& s : c.shots) if (s.measured && use(s)) { X.push_back(Fx[s.idx]); Y.push_back({s.y, s.t}); }
                        LinearModel mm; mm.fit(X, Y, 0.01); return mm; };
                    auto trn = [&](const Shot& s) { return f[s.idx] != fo; };
                    LinearModel full = fit(F["V0 66d (기준)"], trn), alt = fit(rule == 0 ? fl : fq, trn);
                    // 기준값: 학습 fold 안에서 내부 4-fold OOF 불일치의 분위수 (시험 정답 안 씀)
                    std::vector<double> dis;
                    for (int in = 0; in < 5; ++in) if (in != fo) {
                        auto trn2 = [&](const Shot& s) { return f[s.idx] != fo && f[s.idx] != in; };
                        LinearModel a = fit(F["V0 66d (기준)"], trn2), b = fit(rule == 0 ? fl : fq, trn2);
                        for (auto& s : c.shots) if (s.measured && f[s.idx] == in) dis.push_back(std::abs(a.predict(F["V0 66d (기준)"][s.idx]).y - b.predict((rule == 0 ? fl : fq)[s.idx]).y));
                    }
                    double thr = quantile(dis, qq);
                    for (auto& s : c.shots) {
                        if (!s.measured || f[s.idx] != fo || s.t > 1.001) continue;
                        double pt = full.predict(F["V0 66d (기준)"][s.idx]).y, pa = alt.predict((rule == 0 ? fl : fq)[s.idx]).y;
                        double e = std::abs(pt - s.t); bool bad = e > 0.01; nbad += bad;
                        if (std::abs(pt - pa) > thr) { ++nr; nbadc += bad; continue; }
                        ++n; p += e <= 0.01 + 1e-12; m = std::max(m, e);
                    }
                }
                pass.push_back(100.0 * p / n); rej.push_back(100.0 * nr / (n + nr)); caught.push_back(nbad ? 100.0 * nbadc / nbad : 100.0);
                fw = std::max(fw, n - p); mw = std::max(mw, m);
            }
            r.add({rule == 0 ? "전체 T vs 선만 T" : "전체 T vs QR만 T", num(qq, 2), num(mean(pass), 2) + "%", num(*std::min_element(pass.begin(), pass.end()), 2) + "%",
                   std::to_string(fw), num(mw, 4), num(mean(rej), 1) + "%", num(mean(caught), 1) + "%"});
        }
        r.csv(results / "retake_rule.csv");
    }
    t.csv(results / "scores.csv");
    per.csv(results / "t_outliers_seed0.csv");
}
