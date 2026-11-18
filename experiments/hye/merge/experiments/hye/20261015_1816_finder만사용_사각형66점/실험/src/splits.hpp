// 원래 gls_comparison_20261014/splits.json 을 만든 분할 규칙 (zero_experiment.splits, gls_comparison_20261014/compare.py protocols)
//   interleaved: 축마다 촬영 순서로 줄 세워 번갈아 2분할
//   stratified5: sklearn StratifiedKFold(5, shuffle=True, random_state=20261014), 구간으로 층화
//   blocked_within_session5: 구간마다 값 순서로 5덩어리
//   repeated5_seed0..19: StratifiedKFold(5, shuffle=True, random_state=7000+seed)
//   session_out_missing_origin: 구간 하나씩 빼기
// StratifiedKFold 의 섞기는 numpy RandomState(MT19937, init_genrand) + random_interval 이라 std::mt19937 로 그대로 재현된다.
#pragma once
#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <random>
#include <string>
#include <vector>
#include "spread.hpp"

namespace sp {

inline double cx_nan() { return std::numeric_limits<double>::quiet_NaN(); }

struct Fold {
    std::string protocol, fold;
    std::vector<int> train, test;
};

// numpy legacy random_interval
inline uint32_t random_interval(std::mt19937& g, uint32_t max) {
    if (max == 0) return 0;
    uint32_t mask = max;
    mask |= mask >> 1; mask |= mask >> 2; mask |= mask >> 4; mask |= mask >> 8; mask |= mask >> 16;
    uint32_t v;
    while ((v = (uint32_t)g() & mask) > max) {}
    return v;
}
// sklearn StratifiedKFold(n, shuffle=True, random_state=seed).split(ids, y) 의 test 위치 (ids 안의 위치)
inline std::vector<std::vector<int>> stratified(const std::vector<std::string>& y, int n_splits, uint32_t seed) {
    std::mt19937 g(seed);
    std::vector<std::string> cls = y;
    std::sort(cls.begin(), cls.end());
    cls.erase(std::unique(cls.begin(), cls.end()), cls.end());
    // 첫 등장 순서로 번호
    std::vector<int> first(cls.size());
    for (size_t c = 0; c < cls.size(); ++c) first[c] = (int)(std::find(y.begin(), y.end(), cls[c]) - y.begin());
    std::vector<int> order(cls.size());
    for (size_t c = 0; c < cls.size(); ++c) order[c] = (int)c;
    std::sort(order.begin(), order.end(), [&](int a, int b) { return first[a] < first[b]; });
    std::vector<int> perm(cls.size());
    for (size_t r = 0; r < order.size(); ++r) perm[order[r]] = (int)r;
    std::vector<int> enc(y.size());
    for (size_t i = 0; i < y.size(); ++i) enc[i] = perm[std::find(cls.begin(), cls.end(), y[i]) - cls.begin()];
    std::vector<int> yo = enc;
    std::sort(yo.begin(), yo.end());
    int K = (int)cls.size();
    std::vector<std::vector<int>> alloc(n_splits, std::vector<int>(K, 0));
    for (int i = 0; i < n_splits; ++i)
        for (size_t j = i; j < yo.size(); j += n_splits) ++alloc[i][yo[j]];
    std::vector<int> tf(y.size(), -1);
    for (int k = 0; k < K; ++k) {
        std::vector<int> f;
        for (int i = 0; i < n_splits; ++i) for (int r = 0; r < alloc[i][k]; ++r) f.push_back(i);
        for (int i = (int)f.size() - 1; i >= 1; --i) std::swap(f[i], f[random_interval(g, (uint32_t)i)]);
        size_t q = 0;
        for (size_t i = 0; i < y.size(); ++i) if (enc[i] == k) tf[i] = f[q++];
    }
    std::vector<std::vector<int>> out(n_splits);
    for (size_t i = 0; i < y.size(); ++i) out[tf[i]].push_back((int)i);
    return out;
}

// m: 원래 순서(center, Theta 이름순, Y 이름순). 촬영 순서는 data/clean 에 없어서 원래 manifest.csv mtime 순서를 규칙으로 적었다:
//   T: +0.01 → +1.00, -0.01 → -1.00 / Y: Y1+ 0.10→2.00, Y1- -0.10→-2.00, Y2+ 0.05→1.95, Y2- -1.95→-0.05
inline double capture_rank(const cx::Item& it) {
    double a = std::abs(it.value);
    if (it.axis == 'T') return (it.value > 0 ? 0 : 1000) + a;
    if (it.session == "Y1+") return a;
    if (it.session == "Y1-") return 100 + a;
    if (it.session == "Y2+") return 200 + a;
    return 300 - a;  // Y2-
}

inline std::vector<Fold> folds(const std::vector<cx::Item>& m) {
    std::vector<int> ids;
    for (int i = 0; i < (int)m.size(); ++i) if (m[i].axis != 'C') ids.push_back(i);
    auto make = [&](const std::string& p, const std::string& f, std::vector<int> te) {
        std::sort(te.begin(), te.end());
        Fold o{p, f, {}, te};
        for (int i : ids) if (!std::binary_search(te.begin(), te.end(), i)) o.train.push_back(i);
        return o;
    };
    std::vector<Fold> out;
    // interleaved
    for (int k = 0; k < 2; ++k) {
        std::vector<int> te;
        for (char a : {'Y', 'T'}) {
            std::vector<int> ix;
            for (int i = 0; i < (int)m.size(); ++i) if (m[i].axis == a) ix.push_back(i);
            std::stable_sort(ix.begin(), ix.end(), [&](int p, int q) { return capture_rank(m[p]) < capture_rank(m[q]); });
            for (size_t j = k; j < ix.size(); j += 2) te.push_back(ix[j]);
        }
        out.push_back(make("interleaved", "f" + std::to_string(k), te));
    }
    std::vector<std::string> y;
    for (int i : ids) y.push_back(m[i].session);
    auto strat = [&](const std::string& p, uint32_t seed, bool fprefix) {
        auto f = stratified(y, 5, seed);
        for (int k = 0; k < 5; ++k) {
            std::vector<int> te;
            for (int j : f[k]) te.push_back(ids[j]);
            out.push_back(make(p, (fprefix ? "f" : "") + std::to_string(k), te));
        }
    };
    strat("stratified5", 20261014u, true);
    // blocked_within_session5: 구간 첫 등장 순서
    std::vector<std::string> ses;
    for (int i : ids) if (std::find(ses.begin(), ses.end(), m[i].session) == ses.end()) ses.push_back(m[i].session);
    for (int k = 0; k < 5; ++k) {
        std::vector<int> te;
        for (auto& s : ses) {
            std::vector<int> ix;
            for (int i = 0; i < (int)m.size(); ++i) if (m[i].session == s) ix.push_back(i);
            std::stable_sort(ix.begin(), ix.end(), [&](int p, int q) { return m[p].value < m[q].value; });
            size_t n = ix.size(), base = n / 5, extra = n % 5, start = 0;
            for (int j = 0; j < k; ++j) start += base + (j < (int)extra);
            size_t len = base + (k < (int)extra);
            for (size_t j = start; j < start + len; ++j) te.push_back(ix[j]);
        }
        out.push_back(make("blocked_within_session5", std::to_string(k), te));
    }
    for (int s = 0; s < 20; ++s) strat("repeated5_seed" + std::to_string(s), 7000u + s, false);
    // session_out
    std::vector<std::string> ss = ses;
    std::sort(ss.begin(), ss.end());
    for (auto& s : ss) {
        std::vector<int> te;
        for (int i = 0; i < (int)m.size(); ++i) if (m[i].session == s) te.push_back(i);
        out.push_back(make("session_out_missing_origin", s, te));
    }
    return out;
}

// 원래 splits.json 과 맞추는 서명: Σ_fold Σ_test (위치+1) x (fold 번호+1)
inline long long signature(const std::vector<Fold>& f, const std::string& protocol) {
    long long s = 0;
    int k = 0;
    for (auto& x : f)
        if (x.protocol == protocol) {
            ++k;
            for (int i : x.test) s += (long long)(i + 1) * k;
        }
    return s;
}

// zero_experiment.score
struct PredRow {
    int i;
    char axis, pred_axis;
    double value, pred, cost;
    std::string session, branch, method, protocol, fold;
};
struct Score {
    int n = 0;
    double axis_sign_accuracy = 0, strict_pass09 = 0, vector_Y_max = 0, vector_T_max = 0;
    double Y_axis_sign_accuracy = 0, T_axis_sign_accuracy = 0, Y_matched_MAE = 0, T_matched_MAE = 0;
    int Y_n = 0, T_n = 0, Y_wrong_axis = 0, T_wrong_axis = 0;
};
inline int sgn(double v) { return (v > 0) - (v < 0); }
inline Score score(const std::vector<PredRow>& d) {
    Score s;
    s.n = (int)d.size();
    int ok_sign = 0, pass = 0;
    std::map<char, int> n, oks, wa;
    std::map<char, double> mae;
    std::map<char, int> good;
    for (auto& r : d) {
        bool ok = r.axis == r.pred_axis, sign = sgn(r.value) == sgn(r.pred);
        double err = std::abs(r.value - r.pred);
        ok_sign += ok && sign;
        pass += ok && sign && err <= 0.09;
        double ty = r.axis == 'Y' ? r.value : 0, tt = r.axis == 'T' ? r.value : 0;
        double py = r.pred_axis == 'Y' ? r.pred : 0, pt = r.pred_axis == 'T' ? r.pred : 0;
        s.vector_Y_max = std::max(s.vector_Y_max, std::abs(ty - py));
        s.vector_T_max = std::max(s.vector_T_max, std::abs(tt - pt));
        ++n[r.axis];
        oks[r.axis] += ok && sign;
        if (ok) { mae[r.axis] += err; ++good[r.axis]; }
        else ++wa[r.axis];
    }
    s.axis_sign_accuracy = (double)ok_sign / s.n;
    s.strict_pass09 = (double)pass / s.n;
    s.Y_n = n['Y']; s.T_n = n['T'];
    s.Y_axis_sign_accuracy = n['Y'] ? (double)oks['Y'] / n['Y'] : cx_nan();
    s.T_axis_sign_accuracy = n['T'] ? (double)oks['T'] / n['T'] : cx_nan();
    s.Y_matched_MAE = good['Y'] ? mae['Y'] / good['Y'] : cx_nan();
    s.T_matched_MAE = good['T'] ? mae['T'] / good['T'] : cx_nan();
    s.Y_wrong_axis = wa['Y']; s.T_wrong_axis = wa['T'];
    return s;
}

}  // namespace sp
