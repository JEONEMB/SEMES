// 원점 후보 여러 개 + Y/T 동시 추정 (joint_center_630_20261022/model.py JointOrigins, local_model.py LocalOrigins)
// 그리고 분할에 쓰인 난수: numpy default_rng(PCG64, SeedSequence) permutation, sklearn StratifiedGroupKFold.
#pragma once
#include <opencv2/core.hpp>
#include <intrin.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <set>
#include <string>
#include <vector>
#include "gls.hpp"

namespace g09 {

// ---- numpy SeedSequence + PCG64 ----
struct PCG64 {
    uint64_t sh = 0, sl = 0, ih = 0, il = 0;  // state, inc (128 bit)
    bool has32 = false;
    uint32_t u32 = 0;
    static constexpr uint64_t MH = 0x2360ED051FC65DA4ULL, ML = 0x4385DF649FCCF645ULL;
    void step() {
        uint64_t hi;
        uint64_t lo = _umul128(sl, ML, &hi);
        hi += sh * ML + sl * MH;
        uint64_t nl = lo + il;
        hi += ih + (nl < lo);
        sh = hi;
        sl = nl;
    }
    uint64_t next64() {
        step();
        uint64_t v = sh ^ sl;
        unsigned rot = (unsigned)(sh >> 58);
        return (v >> rot) | (v << ((64 - rot) & 63));
    }
    uint32_t next32() {
        if (has32) { has32 = false; return u32; }
        uint64_t n = next64();
        has32 = true;
        u32 = (uint32_t)(n >> 32);
        return (uint32_t)(n & 0xffffffffULL);
    }
    explicit PCG64(uint32_t seed) {
        // SeedSequence(seed).generate_state(4, uint64)
        const uint32_t INIT_A = 0x43b0d7e5, MULT_A = 0x931e8875, INIT_B = 0x8b51f9dd, MULT_B = 0x58f38ded, ML_ = 0xca01f9dd, MR_ = 0x4973f715;
        uint32_t hc = INIT_A;
        auto hashmix = [&](uint32_t v) { v ^= hc; hc *= MULT_A; v *= hc; v ^= v >> 16; return v; };
        auto mix = [&](uint32_t x, uint32_t y) { uint32_t r = ML_ * x - MR_ * y; r ^= r >> 16; return r; };
        uint32_t pool[4];
        for (int i = 0; i < 4; ++i) pool[i] = hashmix(i == 0 ? seed : 0);
        for (int s = 0; s < 4; ++s)
            for (int d = 0; d < 4; ++d)
                if (s != d) pool[d] = mix(pool[d], hashmix(pool[s]));
        uint32_t st[8];
        uint32_t hb = INIT_B;
        for (int i = 0; i < 8; ++i) {
            uint32_t v = pool[i % 4];
            v ^= hb;
            hb *= MULT_B;
            v *= hb;
            v ^= v >> 16;
            st[i] = v;
        }
        uint64_t w[4];
        for (int i = 0; i < 4; ++i) w[i] = (uint64_t)st[2 * i] | ((uint64_t)st[2 * i + 1] << 32);
        // pcg_setseq_128_srandom_r(initstate = (w0<<64)|w1, initseq = (w2<<64)|w3)
        sh = 0; sl = 0;
        ih = (w[2] << 1) | (w[3] >> 63);
        il = (w[3] << 1) | 1u;
        step();
        uint64_t nl = sl + w[1];
        sh = sh + w[0] + (nl < sl);
        sl = nl;
        step();
    }
    uint64_t interval(uint64_t mx) {
        if (mx == 0) return 0;
        uint64_t mask = mx;
        mask |= mask >> 1; mask |= mask >> 2; mask |= mask >> 4; mask |= mask >> 8; mask |= mask >> 16; mask |= mask >> 32;
        uint64_t v;
        if (mx <= 0xffffffffULL) { while ((v = (next32() & mask)) > mx) {} }
        else { while ((v = (next64() & mask)) > mx) {} }
        return v;
    }
    template <class T>
    std::vector<T> permutation(std::vector<T> x) {
        for (size_t i = x.size(); i-- > 1;) std::swap(x[i], x[(size_t)interval(i)]);
        return x;
    }
};

// ---- sklearn StratifiedGroupKFold(shuffle=True) : 반환은 시험 fold 번호 ----
inline std::vector<int> stratified_group_kfold(const std::vector<std::string>& y, const std::vector<std::string>& groups, int n_splits, uint32_t seed) {
    const int n = (int)y.size();
    std::vector<std::string> cls(y), gs(groups);
    std::sort(cls.begin(), cls.end());
    cls.erase(std::unique(cls.begin(), cls.end()), cls.end());
    std::sort(gs.begin(), gs.end());
    gs.erase(std::unique(gs.begin(), gs.end()), gs.end());
    const int nc = (int)cls.size(), ng = (int)gs.size();
    std::vector<int> yi(n), gi(n);
    std::vector<double> ycnt(nc, 0);
    for (int i = 0; i < n; ++i) {
        yi[i] = (int)(std::lower_bound(cls.begin(), cls.end(), y[i]) - cls.begin());
        gi[i] = (int)(std::lower_bound(gs.begin(), gs.end(), groups[i]) - gs.begin());
        ycnt[yi[i]] += 1;
    }
    std::vector<std::vector<double>> cpg(ng, std::vector<double>(nc, 0));
    for (int i = 0; i < n; ++i) cpg[gi[i]][yi[i]] += 1;
    NpRandomState rng(seed);
    for (size_t i = cpg.size(); i-- > 1;) {
        size_t j = (size_t)rng.interval(i);
        if (i != j) std::swap(cpg[i], cpg[j]);
    }
    auto npstd = [](const std::vector<double>& v) {
        double s = 0;
        for (double x : v) s += x;
        double m = s / v.size(), q = 0;
        for (double x : v) q += (x - m) * (x - m);
        return std::sqrt(q / v.size());
    };
    std::vector<double> negstd(ng);
    for (int g = 0; g < ng; ++g) negstd[g] = -npstd(cpg[g]);
    std::vector<int> order(ng);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return negstd[a] < negstd[b]; });
    std::vector<std::vector<double>> cpf(n_splits, std::vector<double>(nc, 0));
    std::vector<int> group_fold(ng, -1);
    for (int g : order) {
        int best = -1;
        double min_eval = std::numeric_limits<double>::infinity(), min_samples = std::numeric_limits<double>::infinity();
        for (int f = 0; f < n_splits; ++f) {
            for (int c = 0; c < nc; ++c) cpf[f][c] += cpg[g][c];
            std::vector<double> stdc(nc);
            for (int c = 0; c < nc; ++c) {
                std::vector<double> col(n_splits);
                for (int k = 0; k < n_splits; ++k) col[k] = cpf[k][c] / ycnt[c];
                stdc[c] = npstd(col);
            }
            for (int c = 0; c < nc; ++c) cpf[f][c] -= cpg[g][c];
            double s = 0;
            for (double v : stdc) s += v;
            double eval = s / nc;
            double samples = 0;
            for (int c = 0; c < nc; ++c) samples += cpf[f][c];
            bool close = std::abs(eval - min_eval) <= 1e-8 + 1e-5 * std::abs(min_eval);
            if (eval < min_eval || (close && samples < min_samples)) { min_eval = eval; min_samples = samples; best = f; }
        }
        for (int c = 0; c < nc; ++c) cpf[best][c] += cpg[g][c];
        group_fold[g] = best;  // 섞인 행 번호 g 를 그룹 번호로 쓴다 (sklearn 그대로)
    }
    std::vector<int> fold(n);
    for (int i = 0; i < n; ++i) fold[i] = group_fold[gi[i]];
    return fold;
}

// ---- JointOrigins / LocalOrigins ----
inline std::vector<double> phi(double y, double t, int degree) {
    if (degree == 1) return {y, t};
    return {y, t, y * y, y * t, t * t};
}

struct JointConfig {
    std::string feature = "P12";
    int degree = 2;
    double shrink = .5;
    std::string origins = "learned_actual";
    bool local = false;
    std::string label() const {
        char b[128];
        std::snprintf(b, sizeof b, "%s d%d s%.1f %s%s", feature.c_str(), degree, shrink, origins.c_str(), local ? " local" : "");
        return b;
    }
};

struct JointOut {
    std::vector<std::array<double, 2>> YT, alt;
    std::vector<int> center;
    std::vector<double> cost, second, margin;
};

struct JointModel {
    JointConfig cfg;
    double scale[2] = {1, 1}, lower[2] = {0, 0}, upper[2] = {0, 0};
    cv::Mat A;                          // P x d
    std::vector<cv::Mat> As;            // 원점별 A (local)
    cv::Mat centers;                    // k x d
    std::vector<std::string> ids;
    bool diag = false;
    cv::Mat Wv;                         // 1 x d (diag) 또는 d x d
    cv::Mat cw;                         // k x d

    // X: N x d, truth: N x 2
    void fit(const cv::Mat& X, const std::vector<std::array<double, 2>>& truth, const std::vector<std::string>& groups, const std::vector<std::string>& scopes,
             const std::vector<int>& indices, const std::vector<std::string>& relative) {
        const int d = X.cols, deg = cfg.degree, P = deg == 1 ? 2 : 5;
        for (int a = 0; a < 2; ++a) {
            double mx = 0, lo = 1e300, hi = -1e300;
            for (int i : indices) mx = std::max(mx, std::abs(truth[i][a]));
            scale[a] = std::max(mx, .05);
            for (int i : indices) { lo = std::min(lo, truth[i][a] / scale[a]); hi = std::max(hi, truth[i][a] / scale[a]); }
            lower[a] = lo - .25;
            upper[a] = hi + .25;
        }
        std::vector<int> moving;
        for (int i : indices) if (scopes[i] != "center") moving.push_back(i);
        std::vector<std::string> uniq;
        for (int i : moving) uniq.push_back(groups[i]);
        std::sort(uniq.begin(), uniq.end());
        uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
        const bool fixed = cfg.origins == "fixed";
        const int nG = fixed ? 0 : (int)uniq.size(), nc = nG + P, n = (int)moving.size();
        cv::Mat D = cv::Mat::zeros(n, nc, CV_64F), Y(n, d, CV_64F);
        for (int r = 0; r < n; ++r) {
            int i = moving[r];
            if (!fixed) D.at<double>(r, (int)(std::lower_bound(uniq.begin(), uniq.end(), groups[i]) - uniq.begin())) = 1;
            auto f = phi(truth[i][0] / scale[0], truth[i][1] / scale[1], deg);
            for (int j = 0; j < P; ++j) D.at<double>(r, nG + j) = f[j];
            X.row(i).copyTo(Y.row(r));
        }
        cv::Mat coef;
        cv::solve(D.t() * D + cv::Mat::eye(nc, nc, CV_64F) * 1e-6, D.t() * Y, coef, cv::DECOMP_LU);
        A = coef.rowRange(nG, nc).clone();
        std::vector<cv::Mat> rows;
        ids.clear();
        rows.push_back(cv::Mat::zeros(1, d, CV_64F));
        ids.push_back("reference:fixed_center");
        if (cfg.origins == "learned" || cfg.origins == "learned_actual")
            for (int g = 0; g < nG; ++g) { rows.push_back(coef.row(g).clone()); ids.push_back("learned:" + uniq[g]); }
        if (cfg.origins == "actual" || cfg.origins == "learned_actual")
            for (int i : indices) {
                const std::string& rel = relative[i];
                const std::string tail = "20261014\\center.bmp";
                bool ref = rel.size() >= tail.size() && rel.compare(rel.size() - tail.size(), tail.size(), tail) == 0;
                if (scopes[i] == "center" && !ref) { rows.push_back(X.row(i).clone()); ids.push_back("actual:" + rel); }
            }
        cv::vconcat(rows, centers);
        cv::Mat R = Y - D * coef;
        const double dof = std::max(n - nc, 1);
        std::vector<double> sd(d);
        for (int j = 0; j < d; ++j) {
            double s = 0;
            for (int r = 0; r < n; ++r) s += R.at<double>(r, j) * R.at<double>(r, j);
            sd[j] = std::sqrt(std::max(s / dof, 1e-12));
        }
        if (cfg.shrink == 1.) {
            diag = true;
            Wv.create(1, d, CV_64F);
            for (int j = 0; j < d; ++j) Wv.at<double>(0, j) = 1 / sd[j];
        } else {
            diag = false;
            cv::Mat S = R.t() * R / dof, C(d, d, CV_64F);
            for (int i = 0; i < d; ++i)
                for (int j = 0; j < d; ++j) C.at<double>(i, j) = (1 - cfg.shrink) * S.at<double>(i, j) / sd[i] / sd[j] + (i == j ? cfg.shrink : 0.);
            cv::Mat Ci;
            cv::invert(C, Ci, cv::DECOMP_LU);
            Wv = cholesky_lower(Ci).t();
            for (int i = 0; i < d; ++i)
                for (int j = 0; j < d; ++j) Wv.at<double>(i, j) /= sd[j];
        }
        As.assign(centers.rows, A.clone());
        if (cfg.local) fit_local(X, truth, groups, scopes, indices);
        cw = whiten(centers);
    }
    cv::Mat whiten(const cv::Mat& M) const {
        if (diag) {
            cv::Mat o = M.clone();
            for (int r = 0; r < o.rows; ++r)
                for (int j = 0; j < o.cols; ++j) o.at<double>(r, j) *= Wv.at<double>(0, j);
            return o;
        }
        return M * Wv.t();
    }
    void fit_local(const cv::Mat& X, const std::vector<std::array<double, 2>>& truth, const std::vector<std::string>& groups,
                   const std::vector<std::string>& scopes, const std::vector<int>& indices) {
        const int deg = cfg.degree, P = deg == 1 ? 2 : 5, d = X.cols;
        std::set<std::string> gset;
        for (int i : indices) gset.insert(groups[i]);
        std::vector<std::string> gnames;
        std::vector<std::pair<cv::Mat, cv::Mat>> local;  // (b, A)
        for (const auto& group : gset) {
            std::vector<int> ii;
            for (int i : indices) if (groups[i] == group && scopes[i] != "center") ii.push_back(i);
            if (ii.size() < 6) continue;
            std::vector<int> cols;
            if (group == "date20261020") for (int j = 0; j < P; ++j) cols.push_back(j);
            else if (group[0] == 'T') { cols.push_back(1); if (deg == 2) cols.push_back(4); }
            else if (group[0] == 'Y') { cols.push_back(0); if (deg == 2) cols.push_back(2); }
            else continue;
            const int n = (int)ii.size(), nc = 1 + (int)cols.size();
            cv::Mat Dm = cv::Mat::zeros(n, nc, CV_64F), T(n, d, CV_64F);
            for (int r = 0; r < n; ++r) {
                int i = ii[r];
                auto f = phi(truth[i][0] / scale[0], truth[i][1] / scale[1], deg);
                Dm.at<double>(r, 0) = 1;
                for (size_t c = 0; c < cols.size(); ++c) Dm.at<double>(r, 1 + (int)c) = f[cols[c]];
                for (int j = 0; j < d; ++j) {
                    double v = X.at<double>(i, j);
                    for (int u = 0; u < P; ++u)
                        if (std::find(cols.begin(), cols.end(), u) == cols.end()) v -= f[u] * A.at<double>(u, j);
                    T.at<double>(r, j) = v;
                }
            }
            cv::Mat coef;
            cv::solve(Dm.t() * Dm + cv::Mat::eye(nc, nc, CV_64F) * 1e-6, Dm.t() * T, coef, cv::DECOMP_LU);
            cv::Mat An = A.clone();
            for (size_t c = 0; c < cols.size(); ++c) coef.row(1 + (int)c).copyTo(An.row(cols[c]));
            gnames.push_back(group);
            local.push_back({coef.row(0).clone(), An});
        }
        for (int k = 0; k < centers.rows; ++k) {
            const std::string& name = ids[k];
            std::string group;
            if (name.rfind("learned:", 0) == 0) group = name.substr(8);
            else if (name.rfind("actual:20261020", 0) == 0) group = "date20261020";
            auto it = std::find(gnames.begin(), gnames.end(), group);
            if (group.empty() || it == gnames.end()) continue;
            size_t g = it - gnames.begin();
            As[k] = local[g].second.clone();
            if (name.rfind("learned:", 0) == 0) local[g].first.copyTo(centers.row(k));
        }
    }
    // 원점 k 하나에서 (Y,T) 풀기
    void solve_one(const double* z, const cv::Mat& Aw, const cv::Mat& lp, double q[2], double& cost) const {
        const int d = Aw.cols, deg = cfg.degree;
        for (int a = 0; a < 2; ++a) {
            double s = 0;
            for (int j = 0; j < d; ++j) s += z[j] * lp.at<double>(j, a);
            q[a] = std::min(std::max(s, lower[a]), upper[a]);
        }
        std::vector<double> r(d), j0(d), j1(d);
        for (int it = 0; it < 16; ++it) {
            auto f = phi(q[0], q[1], deg);
            double a = 0, b = 0, c = 0, g0 = 0, g1 = 0;
            for (int j = 0; j < d; ++j) {
                double pred = 0;
                for (size_t u = 0; u < f.size(); ++u) pred += f[u] * Aw.at<double>((int)u, j);
                if (deg == 1) { j0[j] = Aw.at<double>(0, j); j1[j] = Aw.at<double>(1, j); }
                else {
                    j0[j] = Aw.at<double>(0, j) + 2 * q[0] * Aw.at<double>(2, j) + q[1] * Aw.at<double>(3, j);
                    j1[j] = Aw.at<double>(1, j) + q[0] * Aw.at<double>(3, j) + 2 * q[1] * Aw.at<double>(4, j);
                }
                r[j] = z[j] - pred;
                a += j0[j] * j0[j]; b += j0[j] * j1[j]; c += j1[j] * j1[j]; g0 += j0[j] * r[j]; g1 += j1[j] * r[j];
            }
            double damp = 1e-8 * (a + c + 1);
            a += damp; c += damp;
            double det = std::max(a * c - b * b, 1e-20);
            double s0 = (c * g0 - b * g1) / det, s1 = (a * g1 - b * g0) / det;
            s0 = std::min(std::max(s0, -.2), .2);
            s1 = std::min(std::max(s1, -.2), .2);
            q[0] = std::min(std::max(q[0] + s0, lower[0]), upper[0]);
            q[1] = std::min(std::max(q[1] + s1, lower[1]), upper[1]);
        }
        auto f = phi(q[0], q[1], deg);
        cost = 0;
        for (int j = 0; j < d; ++j) {
            double pred = 0;
            for (size_t u = 0; u < f.size(); ++u) pred += f[u] * Aw.at<double>((int)u, j);
            cost += (z[j] - pred) * (z[j] - pred);
        }
    }
    JointOut predict(const cv::Mat& X, const std::vector<int>& rows, int kmax = -1) const {
        const int k = kmax > 0 ? std::min(kmax, centers.rows) : centers.rows, d = X.cols;
        std::vector<cv::Mat> Aw(k), lp(k);
        for (int o = 0; o < k; ++o) {
            Aw[o] = whiten(As[o]);
            cv::invert(Aw[o].rowRange(0, 2), lp[o], cv::DECOMP_SVD);
        }
        JointOut out;
        std::vector<double> xz(d), z(d);
        for (int i : rows) {
            cv::Mat xr = X.row(i);
            cv::Mat xw = whiten(xr);
            std::vector<double> cost(k);
            std::vector<std::array<double, 2>> Q(k);
            for (int o = 0; o < k; ++o) {
                for (int j = 0; j < d; ++j) z[j] = xw.at<double>(0, j) - cw.at<double>(o, j);
                double q[2];
                solve_one(z.data(), Aw[o], lp[o], q, cost[o]);
                Q[o] = {q[0] * scale[0], q[1] * scale[1]};
            }
            std::vector<int> ord(k);
            std::iota(ord.begin(), ord.end(), 0);
            std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return cost[a] < cost[b]; });
            int best = ord[0], sec = k > 1 ? ord[1] : best;
            out.YT.push_back(Q[best]);
            out.alt.push_back(Q[sec]);
            out.center.push_back(best);
            out.cost.push_back(cost[best]);
            out.second.push_back(cost[sec]);
            out.margin.push_back((cost[sec] - cost[best]) / (cost[sec] + cost[best] + 1e-12));
        }
        return out;
    }
};

}  // namespace g09
