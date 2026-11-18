// 전체점 Y/T 회귀와 검증 분할 (원본: joint_yt_generalization_20261022/forward_model.py, joint_model.py, evaluate.py)
// - ForwardGLS(degree=1): 가중 최소제곱 [1, Y/scale, T/scale] -> 좌표, 잔차 공분산을 shrink 로 줄여 GLS 역투영
// - JointRegressor: global(한 번 맞춤) / local(시험 사진의 평균 수평 이동 h 근처에 가중 0.01+exp(-0.5((h_i-h)/width)^2))
// - sklearn StratifiedGroupKFold(shuffle, numpy RandomState 시드)를 그대로 옮겼다(MT19937 + legacy shuffle).
#pragma once
#include <opencv2/core.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <numeric>
#include <set>
#include <string>
#include <vector>

namespace ytm {

using Mat = std::vector<double>;  // 행 우선

struct Config {
    std::string family, feature;  // global/local, raw1260/raw_shift
    double shrink = 0.1, width = 0;
    std::string str() const {
        char b[128];
        if (family == "local") std::snprintf(b, sizeof b, "local %s s%.3g w%.0f", feature.c_str(), shrink, width);
        else std::snprintf(b, sizeof b, "global %s s%.3g", feature.c_str(), shrink);
        return b;
    }
    bool operator==(const Config& o) const { return family == o.family && feature == o.feature && shrink == o.shrink && width == o.width; }
};

// D: 사진 i 의 630x2 변위 (dx,dy 교차). h = 평균 dx
struct Data {
    int n = 0;
    std::vector<std::vector<double>> D;  // n x 1260
    std::vector<double> h;               // 평균 dx (px)
    std::vector<double> y, t;
};

inline std::vector<double> feature_row(const Data& d, int i, const std::string& kind) {
    const auto& r = d.D[i];
    if (kind == "raw1260") return r;
    std::vector<double> o(r);
    double hh = d.h[i] / 100;
    o.reserve(r.size() * 2);
    for (double v : r) o.push_back(v * hh);
    return o;
}

class ForwardGLS {
public:
    double shrink = 0.1;
    std::vector<double> origin, proj;  // proj: p x 2
    double scale[2]{1, 1}, initial[4]{};
    int p = 0;

    void fit(const std::vector<const std::vector<double>*>& X, const std::vector<double>& Y, const std::vector<double>& T, std::vector<double> w) {
        const int n = (int)X.size();
        p = (int)X[0]->size();
        if (w.empty()) w.assign(n, 1.0);
        double wm = 0;
        for (double v : w) wm += v;
        wm /= n;
        std::vector<double> sw(n);
        for (int i = 0; i < n; ++i) { w[i] /= wm; sw[i] = std::sqrt(w[i]); }
        double my = 0, mt = 0;
        for (int i = 0; i < n; ++i) { my = std::max(my, std::abs(Y[i])); mt = std::max(mt, std::abs(T[i])); }
        scale[0] = std::max(my, .05);
        scale[1] = std::max(mt, .05);
        // A = [1, Y/s, T/s] * sw, lstsq(rcond 1e-10) via SVD
        cv::Mat A(n, 3, CV_64F);
        for (int i = 0; i < n; ++i) {
            A.at<double>(i, 0) = sw[i];
            A.at<double>(i, 1) = Y[i] / scale[0] * sw[i];
            A.at<double>(i, 2) = T[i] / scale[1] * sw[i];
        }
        cv::Mat s, u, vt;
        cv::SVD::compute(A, s, u, vt);
        double cut = 1e-10 * s.at<double>(0);
        // pinv = V diag(1/s) U^T  (3 x n)
        cv::Mat pinv = cv::Mat::zeros(3, n, CV_64F);
        for (int k = 0; k < 3; ++k) {
            double sk = s.at<double>(k);
            if (sk <= cut) continue;
            for (int r = 0; r < 3; ++r)
                for (int i = 0; i < n; ++i) pinv.at<double>(r, i) += vt.at<double>(k, r) * u.at<double>(i, k) / sk;
        }
        // coef (3 x p) = pinv * (X*sw)
        std::vector<double> coef(3 * p, 0.0);
        for (int i = 0; i < n; ++i) {
            const double* x = X[i]->data();
            double c0 = pinv.at<double>(0, i) * sw[i], c1 = pinv.at<double>(1, i) * sw[i], c2 = pinv.at<double>(2, i) * sw[i];
            for (int j = 0; j < p; ++j) { coef[j] += c0 * x[j]; coef[p + j] += c1 * x[j]; coef[2 * p + j] += c2 * x[j]; }
        }
        origin.assign(coef.begin(), coef.begin() + p);
        // r = (X - A coef) * sw ; std = max(sqrt(mean r^2), 1e-6)
        std::vector<double> R((size_t)n * p), sd(p, 0.0);
        for (int i = 0; i < n; ++i) {
            double a1 = Y[i] / scale[0], a2 = T[i] / scale[1];
            const double* x = X[i]->data();
            double* rr = &R[(size_t)i * p];
            for (int j = 0; j < p; ++j) {
                double v = (x[j] - (coef[j] + a1 * coef[p + j] + a2 * coef[2 * p + j])) * sw[i];
                rr[j] = v;
                sd[j] += v * v;
            }
        }
        for (int j = 0; j < p; ++j) sd[j] = std::max(std::sqrt(sd[j] / n), 1e-6);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < p; ++j) R[(size_t)i * p + j] /= sd[j];
        // V = (C/std)^T : p x 2
        std::vector<double> V(2 * (size_t)p);
        for (int j = 0; j < p; ++j) { V[j * 2] = coef[p + j] / sd[j]; V[j * 2 + 1] = coef[2 * p + j] / sd[j]; }
        const double a = shrink, b = (1 - a) / n;
        // K = a I + b R R^T (n x n), RV = R V (n x 2)
        cv::Mat K(n, n, CV_64F), RV(n, 2, CV_64F);
        for (int i = 0; i < n; ++i) {
            const double* ri = &R[(size_t)i * p];
            for (int k = i; k < n; ++k) {
                const double* rk = &R[(size_t)k * p];
                double sum = 0;
                for (int j = 0; j < p; ++j) sum += ri[j] * rk[j];
                K.at<double>(i, k) = K.at<double>(k, i) = b * sum + (i == k ? a : 0);
            }
            double s0 = 0, s1 = 0;
            for (int j = 0; j < p; ++j) { s0 += ri[j] * V[j * 2]; s1 += ri[j] * V[j * 2 + 1]; }
            RV.at<double>(i, 0) = s0;
            RV.at<double>(i, 1) = s1;
        }
        cv::Mat Z;
        cv::solve(K, RV, Z, cv::DECOMP_CHOLESKY);
        // Q = (V - b R^T Z)/a ; proj = Q / std
        proj.assign(2 * (size_t)p, 0.0);
        for (int j = 0; j < p; ++j) {
            double q0 = 0, q1 = 0;
            for (int i = 0; i < n; ++i) { q0 += R[(size_t)i * p + j] * Z.at<double>(i, 0); q1 += R[(size_t)i * p + j] * Z.at<double>(i, 1); }
            proj[j * 2] = (V[j * 2] - b * q0) / a / sd[j];
            proj[j * 2 + 1] = (V[j * 2 + 1] - b * q1) / a / sd[j];
        }
        // gram = C @ proj (2x2), 대칭화, initial = pinv(gram)
        double g[4] = {0, 0, 0, 0};
        for (int j = 0; j < p; ++j) {
            g[0] += coef[p + j] * proj[j * 2];
            g[1] += coef[p + j] * proj[j * 2 + 1];
            g[2] += coef[2 * p + j] * proj[j * 2];
            g[3] += coef[2 * p + j] * proj[j * 2 + 1];
        }
        double off = (g[1] + g[2]) / 2;
        cv::Mat G = (cv::Mat_<double>(2, 2) << g[0], off, off, g[3]), Gi;
        cv::invert(G, Gi, cv::DECOMP_SVD);
        for (int k = 0; k < 4; ++k) initial[k] = Gi.at<double>(k / 2, k % 2);
    }
    void predict(const std::vector<double>& x, double& py, double& pt) const {
        double h0 = 0, h1 = 0;
        for (int j = 0; j < p; ++j) { double d = x[j] - origin[j]; h0 += d * proj[j * 2]; h1 += d * proj[j * 2 + 1]; }
        double q0 = h0 * initial[0] + h1 * initial[2], q1 = h0 * initial[1] + h1 * initial[3];
        py = q0 * scale[0];
        pt = q1 * scale[1];
    }
};

// 학습 tr 로 맞추고 te 를 예측. local 은 시험 사진마다 가중 모델을 새로 맞춘다.
inline void fit_predict(const Data& d, const Config& c, const std::vector<int>& tr, const std::vector<int>& te, std::vector<double>& py,
                        std::vector<double>& pt) {
    std::vector<std::vector<double>> Xtr(tr.size());
    std::vector<const std::vector<double>*> Xp(tr.size());
    std::vector<double> Y(tr.size()), T(tr.size());
    for (size_t k = 0; k < tr.size(); ++k) {
        Xtr[k] = feature_row(d, tr[k], c.feature);
        Xp[k] = &Xtr[k];
        Y[k] = d.y[tr[k]];
        T[k] = d.t[tr[k]];
    }
    py.assign(te.size(), 0);
    pt.assign(te.size(), 0);
    if (c.family == "global") {
        ForwardGLS m;
        m.shrink = c.shrink;
        m.fit(Xp, Y, T, {});
        for (size_t k = 0; k < te.size(); ++k) m.predict(feature_row(d, te[k], c.feature), py[k], pt[k]);
        return;
    }
#pragma omp parallel for schedule(dynamic)
    for (int k = 0; k < (int)te.size(); ++k) {
        double h = d.h[te[k]];
        std::vector<double> w(tr.size());
        for (size_t i = 0; i < tr.size(); ++i) {
            double z = (d.h[tr[i]] - h) / c.width;
            w[i] = .01 + std::exp(-.5 * z * z);
        }
        ForwardGLS m;
        m.shrink = c.shrink;
        m.fit(Xp, Y, T, w);
        m.predict(feature_row(d, te[k], c.feature), py[k], pt[k]);
    }
}

// ---- numpy RandomState(seed) MT19937 + legacy shuffle ----
class MT19937 {
    uint32_t mt[624];
    int mti = 625;
public:
    explicit MT19937(uint32_t s) {
        mt[0] = s;
        for (mti = 1; mti < 624; ++mti) mt[mti] = 1812433253u * (mt[mti - 1] ^ (mt[mti - 1] >> 30)) + (uint32_t)mti;
    }
    uint32_t next() {
        static const uint32_t mag01[2] = {0x0u, 0x9908b0dfu};
        uint32_t y;
        if (mti >= 624) {
            int kk;
            for (kk = 0; kk < 624 - 397; ++kk) {
                y = (mt[kk] & 0x80000000u) | (mt[kk + 1] & 0x7fffffffu);
                mt[kk] = mt[kk + 397] ^ (y >> 1) ^ mag01[y & 1];
            }
            for (; kk < 623; ++kk) {
                y = (mt[kk] & 0x80000000u) | (mt[kk + 1] & 0x7fffffffu);
                mt[kk] = mt[kk + (397 - 624)] ^ (y >> 1) ^ mag01[y & 1];
            }
            y = (mt[623] & 0x80000000u) | (mt[0] & 0x7fffffffu);
            mt[623] = mt[396] ^ (y >> 1) ^ mag01[y & 1];
            mti = 0;
        }
        y = mt[mti++];
        y ^= (y >> 11);
        y ^= (y << 7) & 0x9d2c5680u;
        y ^= (y << 15) & 0xefc60000u;
        y ^= (y >> 18);
        return y;
    }
    uint32_t interval(uint32_t max) {  // numpy random_interval (max < 2^32)
        if (max == 0) return 0;
        uint32_t mask = max;
        mask |= mask >> 1; mask |= mask >> 2; mask |= mask >> 4; mask |= mask >> 8; mask |= mask >> 16;
        uint32_t v;
        while ((v = (next() & mask)) > max) {}
        return v;
    }
};

// sklearn StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(X, y, groups) 의 시험 위치들
inline std::vector<std::vector<int>> sgkf_test(const std::vector<std::string>& y, const std::vector<std::string>& groups, int n_splits, uint32_t seed) {
    const int n = (int)y.size();
    std::vector<std::string> uy(y.begin(), y.end()), ug(groups.begin(), groups.end());
    std::sort(uy.begin(), uy.end());
    uy.erase(std::unique(uy.begin(), uy.end()), uy.end());
    std::sort(ug.begin(), ug.end());
    ug.erase(std::unique(ug.begin(), ug.end()), ug.end());
    const int C = (int)uy.size(), G = (int)ug.size();
    std::vector<int> yinv(n), ginv(n);
    std::vector<double> ycnt(C, 0);
    for (int i = 0; i < n; ++i) {
        yinv[i] = (int)(std::lower_bound(uy.begin(), uy.end(), y[i]) - uy.begin());
        ginv[i] = (int)(std::lower_bound(ug.begin(), ug.end(), groups[i]) - ug.begin());
        ycnt[yinv[i]] += 1;
    }
    std::vector<std::vector<double>> ypg(G, std::vector<double>(C, 0));
    for (int i = 0; i < n; ++i) ypg[ginv[i]][yinv[i]] += 1;
    MT19937 rng(seed);
    for (int i = G - 1; i >= 1; --i) {
        int j = (int)rng.interval((uint32_t)i);
        if (i != j) std::swap(ypg[i], ypg[j]);
    }
    auto stdv = [](const std::vector<double>& v) {
        double m = 0;
        for (double x : v) m += x;
        m /= v.size();
        double s = 0;
        for (double x : v) s += (x - m) * (x - m);
        return std::sqrt(s / v.size());
    };
    std::vector<double> key(G);
    for (int g = 0; g < G; ++g) key[g] = -stdv(ypg[g]);
    std::vector<int> order(G);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return key[a] < key[b]; });
    std::vector<std::vector<double>> ypf(n_splits, std::vector<double>(C, 0));
    std::vector<std::set<int>> gpf(n_splits);
    for (int gi : order) {
        const auto& gc = ypg[gi];
        int best = -1;
        double min_eval = INFINITY, min_samples = INFINITY;
        for (int f = 0; f < n_splits; ++f) {
            for (int c = 0; c < C; ++c) ypf[f][c] += gc[c];
            double ev = 0;
            for (int c = 0; c < C; ++c) {
                std::vector<double> col(n_splits);
                for (int k = 0; k < n_splits; ++k) col[k] = ypf[k][c] / ycnt[c];
                ev += stdv(col);
            }
            ev /= C;
            for (int c = 0; c < C; ++c) ypf[f][c] -= gc[c];
            double samples = 0;
            for (int c = 0; c < C; ++c) samples += ypf[f][c];
            bool close = std::isfinite(min_eval) && std::abs(ev - min_eval) <= (1e-8 + 1e-5 * std::abs(min_eval));  // np.isclose
            if (ev < min_eval || (close && samples < min_samples)) { min_eval = ev; min_samples = samples; best = f; }
        }
        for (int c = 0; c < C; ++c) ypf[best][c] += gc[c];
        gpf[best].insert(gi);
    }
    std::vector<std::vector<int>> out(n_splits);
    for (int f = 0; f < n_splits; ++f)
        for (int i = 0; i < n; ++i)
            if (gpf[f].count(ginv[i])) out[f].push_back(i);
    return out;
}

struct Metric {
    int n = 0, passed = 0;
    double ymae = 0, ymax = 0, tmae = 0, tmax = 0;
};
inline Metric metric(const std::vector<double>& y, const std::vector<double>& py, const std::vector<double>& t, const std::vector<double>& pt) {
    Metric m;
    m.n = (int)y.size();
    for (int i = 0; i < m.n; ++i) {
        double ey = std::abs(py[i] - y[i]), et = std::abs(pt[i] - t[i]);
        m.passed += (ey <= .09 + 1e-12 && et <= .09 + 1e-12);
        m.ymae += ey; m.tmae += et;
        m.ymax = std::max(m.ymax, ey); m.tmax = std::max(m.tmax, et);
    }
    if (m.n) { m.ymae /= m.n; m.tmae /= m.n; }
    return m;
}

}  // namespace ytm
