// Y/T 이차곡선 GLS 판정기와 4-fold 분할.
//   zero_experiment.py manifest/fit/predict, multiref_4fold_20261021/experiment.py fit/enrich/stats,
//   sklearn StratifiedKFold(shuffle=True) 를 numpy RandomState(MT19937) 난수열 그대로 옮겼다.
#pragma once
#include <opencv2/core.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace g09 {

// ---- numpy RandomState.shuffle (legacy, MT19937 init_genrand) ----
struct NpRandomState {
    std::mt19937 mt;
    explicit NpRandomState(uint32_t seed) : mt(seed) {}
    uint64_t interval(uint64_t mx) {
        if (mx == 0) return 0;
        uint64_t mask = mx;
        mask |= mask >> 1; mask |= mask >> 2; mask |= mask >> 4; mask |= mask >> 8; mask |= mask >> 16; mask |= mask >> 32;
        uint64_t v;
        while ((v = ((uint64_t)mt() & mask)) > mx) {}
        return v;
    }
    template <class T>
    void shuffle(std::vector<T>& x) {
        for (size_t i = x.size() - 1; i >= 1; --i) {
            size_t j = (size_t)interval(i);
            std::swap(x[i], x[j]);
        }
    }
};

// sklearn StratifiedKFold(n_splits, shuffle=True, random_state).  y: 클래스 문자열. 반환: 시험 fold 번호
inline std::vector<int> stratified_kfold(const std::vector<std::string>& y, int n_splits, uint32_t seed) {
    const int n = (int)y.size();
    std::vector<std::string> cls(y);
    std::sort(cls.begin(), cls.end());
    cls.erase(std::unique(cls.begin(), cls.end()), cls.end());
    // 등장 순서로 다시 번호
    std::vector<int> first(cls.size(), n);
    for (int i = 0; i < n; ++i) {
        int c = (int)(std::lower_bound(cls.begin(), cls.end(), y[i]) - cls.begin());
        first[c] = std::min(first[c], i);
    }
    std::vector<int> order(cls.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return first[a] < first[b]; });
    std::vector<int> perm(cls.size());
    for (size_t r = 0; r < order.size(); ++r) perm[order[r]] = (int)r;
    std::vector<int> enc(n);
    for (int i = 0; i < n; ++i) enc[i] = perm[std::lower_bound(cls.begin(), cls.end(), y[i]) - cls.begin()];
    const int nc = (int)cls.size();
    std::vector<int> sorted(enc);
    std::sort(sorted.begin(), sorted.end());
    std::vector<std::vector<int>> alloc(n_splits, std::vector<int>(nc, 0));
    for (int i = 0; i < n_splits; ++i)
        for (int j = i; j < n; j += n_splits) alloc[i][sorted[j]]++;
    NpRandomState rng(seed);
    std::vector<int> test(n, -1);
    for (int k = 0; k < nc; ++k) {
        std::vector<int> f;
        for (int s = 0; s < n_splits; ++s) for (int c = 0; c < alloc[s][k]; ++c) f.push_back(s);
        rng.shuffle(f);
        size_t q = 0;
        for (int i = 0; i < n; ++i) if (enc[i] == k) test[i] = f[q++];
    }
    return test;
}

// ---- 사진 목록 (zero_experiment.manifest) ----
struct Item {
    std::string file;   // T+0.01.bmp
    char axis = 'C';    // C/Y/T
    double value = 0;
    std::string session;  // C, T+, T-, Y1+, Y1-, Y2+, Y2-
};

inline Item make_item(const std::string& name) {
    Item it;
    it.file = name;
    if (name == "center.bmp") { it.axis = 'C'; it.session = "C"; return it; }
    it.axis = name[0];
    it.value = std::stod(name.substr(1, name.size() - 5));
    std::string s = it.axis == 'T' ? "T" : ((std::llabs(std::llround(it.value / .05)) % 2) ? "Y2" : "Y1");
    it.session = s + (it.value > 0 ? "+" : "-");
    return it;
}

// ---- 모델 ----
using Vec = std::vector<double>;
struct Branch {
    std::string name;
    char axis;
    Vec base, s, c;  // 12
};
struct Model {
    std::vector<Branch> branches;  // 삽입 순서 (Y 먼저, 세션 정렬 순)
    cv::Mat W;                     // d x d
};

// np.linalg.lstsq(A, Z): 최소 노름 최소제곱
inline cv::Mat lstsq(const cv::Mat& A, const cv::Mat& Z) {
    cv::Mat x;
    cv::solve(A, Z, x, cv::DECOMP_SVD);
    return x;
}

// 하삼각 촐레스키 (numpy cholesky)
inline cv::Mat cholesky_lower(const cv::Mat& S) {
    const int n = S.rows;
    cv::Mat L = cv::Mat::zeros(n, n, CV_64F);
    for (int j = 0; j < n; ++j) {
        double s = S.at<double>(j, j);
        for (int k = 0; k < j; ++k) s -= L.at<double>(j, k) * L.at<double>(j, k);
        L.at<double>(j, j) = std::sqrt(s);
        for (int i = j + 1; i < n; ++i) {
            double t = S.at<double>(i, j);
            for (int k = 0; k < j; ++k) t -= L.at<double>(i, k) * L.at<double>(j, k);
            L.at<double>(i, j) = t / L.at<double>(j, j);
        }
    }
    return L;
}

// 잔차 R(행) → W = chol(inv(C)).T / std   (C = 상관행렬을 shrink 만큼 단위행렬 쪽으로)
inline cv::Mat whitening(const cv::Mat& R, double dof, double shrink) {
    const int d = R.cols;
    cv::Mat S = R.t() * R / dof;
    std::vector<double> sd(d);
    for (int i = 0; i < d; ++i) sd[i] = std::sqrt(std::max(S.at<double>(i, i), 1e-14));
    cv::Mat C(d, d, CV_64F);
    for (int i = 0; i < d; ++i)
        for (int j = 0; j < d; ++j) C.at<double>(i, j) = S.at<double>(i, j) / sd[i] / sd[j];
    C = (1 - shrink) * C + shrink * cv::Mat::eye(d, d, CV_64F);
    cv::Mat Ci;
    cv::invert(C, Ci, cv::DECOMP_LU);
    cv::Mat W = cholesky_lower(Ci).t();
    for (int i = 0; i < d; ++i)
        for (int j = 0; j < d; ++j) W.at<double>(i, j) /= sd[j];
    return W;
}

// zero_experiment.fit 의 곡선 부분 (mode: "center" 또는 "inferred_session")
inline Model fit_branches(const std::vector<Item>& m, const std::vector<Vec>& X, const std::vector<int>& tr, const std::string& mode, bool quad,
                          cv::Mat* residual_out = nullptr, int* rank_out = nullptr) {
    Model M;
    const int d = (int)X[0].size();
    const bool use_sessions = mode == "inferred_session";
    std::vector<cv::Mat> residuals;
    int prank = 0;
    for (char a : {'Y', 'T'}) {
        std::vector<int> ii;
        for (int i : tr) if (m[i].axis == a) ii.push_back(i);
        std::vector<std::string> sessions;
        for (int i : ii) sessions.push_back(m[i].session);
        std::sort(sessions.begin(), sessions.end());
        sessions.erase(std::unique(sessions.begin(), sessions.end()), sessions.end());
        const int nb = quad ? 2 : 1;
        const int cols = nb + (use_sessions ? (int)sessions.size() : 0);
        cv::Mat A = cv::Mat::zeros((int)ii.size(), cols, CV_64F), Z((int)ii.size(), d, CV_64F);
        for (int r = 0; r < (int)ii.size(); ++r) {
            double v = m[ii[r]].value;
            A.at<double>(r, 0) = v;
            if (quad) A.at<double>(r, 1) = v * v;
            if (use_sessions) {
                int j = (int)(std::find(sessions.begin(), sessions.end(), m[ii[r]].session) - sessions.begin());
                A.at<double>(r, nb + j) = 1;
            }
            for (int c = 0; c < d; ++c) Z.at<double>(r, c) = X[ii[r]][c];
        }
        cv::Mat co = lstsq(A, Z);
        prank += cols;
        residuals.push_back(Z - A * co);
        Vec s(d), c(d, 0.0);
        for (int k = 0; k < d; ++k) { s[k] = co.at<double>(0, k); if (quad) c[k] = co.at<double>(1, k); }
        if (use_sessions) {
            for (int j = 0; j < (int)sessions.size(); ++j) {
                Vec b(d);
                for (int k = 0; k < d; ++k) b[k] = co.at<double>(nb + j, k);
                M.branches.push_back({sessions[j], a, b, s, c});
            }
        } else {
            M.branches.push_back({std::string(1, a), a, Vec(d, 0.0), s, c});
        }
    }
    if (residual_out) cv::vconcat(residuals, *residual_out);
    if (rank_out) *rank_out = prank;
    return M;
}

// multiref_4fold fit: 가지 맞춤 후 가지별 잔차로 공분산, dof = n-16, 축소 shrink
inline Model fit_multiref(const std::vector<Item>& m, const std::vector<Vec>& X, const std::vector<int>& tr, const std::string& mode, bool quad, double shrink) {
    Model M = fit_branches(m, X, tr, mode, quad);
    const int d = (int)X[0].size();
    cv::Mat R((int)tr.size(), d, CV_64F);
    for (int r = 0; r < (int)tr.size(); ++r) {
        int i = tr[r];
        const std::string key = mode == "inferred_session" ? m[i].session : std::string(1, m[i].axis);
        const Branch* b = nullptr;
        for (auto& x : M.branches) if (x.name == key) b = &x;
        double v = m[i].value;
        for (int k = 0; k < d; ++k) R.at<double>(r, k) = X[i][k] - b->base[k] - b->s[k] * v - b->c[k] * v * v;
    }
    M.W = whitening(R, std::max((int)tr.size() - 16, 1), shrink);
    return M;
}

// zero_experiment.fit 원래 W (dof = n - 계수 수, shrink .001)
inline Model fit_zero(const std::vector<Item>& m, const std::vector<Vec>& X, const std::vector<int>& tr, const std::string& mode, bool quad) {
    cv::Mat R;
    int rank = 0;
    Model M = fit_branches(m, X, tr, mode, quad, &R, &rank);
    M.W = whitening(R, std::max(R.rows - rank, 1), .001);
    return M;
}

struct Pred {
    int index;
    char pred_axis;
    double pred, cost;
    std::string branch;
    std::vector<double> costs;  // 가지별 비용
    std::vector<double> values; // 가지별 추정
};

inline Pred predict_one(const Model& M, const Vec& x, const std::string& mode) {
    const int d = (int)x.size();
    Pred p{};
    p.cost = std::numeric_limits<double>::infinity();
    int best = -1;
    for (size_t bi = 0; bi < M.branches.size(); ++bi) {
        const Branch& b = M.branches[bi];
        Vec z(d, 0.0), sw(d, 0.0), cw(d, 0.0);
        for (int r = 0; r < d; ++r) {
            const double* w = M.W.ptr<double>(r);
            double a = 0, s = 0, c = 0;
            for (int k = 0; k < d; ++k) { a += (x[k] - b.base[k]) * w[k]; s += b.s[k] * w[k]; c += b.c[k] * w[k]; }
            z[r] = a; sw[r] = s; cw[r] = c;
        }
        double zs = 0, ss = 0;
        for (int r = 0; r < d; ++r) { zs += z[r] * sw[r]; ss += sw[r] * sw[r]; }
        double v = zs / ss;
        for (int it = 0; it < 8; ++it) {
            double num = 0, den = 0;
            for (int r = 0; r < d; ++r) {
                double J = sw[r] + 2 * v * cw[r];
                double res = z[r] - v * sw[r] - v * v * cw[r];
                num += J * res;
                den += J * J;
            }
            v += num / std::max(den, 1e-20);
        }
        double cost = 0;
        for (int r = 0; r < d; ++r) { double res = z[r] - v * sw[r] - v * v * cw[r]; cost += res * res; }
        if (mode == "inferred_session") {
            double sg = b.name.back() == '+' ? 1 : -1;
            if (!(v * sg >= 0)) cost = std::numeric_limits<double>::infinity();
        }
        p.costs.push_back(cost);
        p.values.push_back(v);
        if (best < 0 || cost < p.cost) { best = (int)bi; p.cost = cost; }
    }
    if (best < 0) best = 0;
    p.pred_axis = M.branches[best].axis;
    p.pred = p.values[best];
    p.branch = M.branches[best].name;
    p.cost = p.costs[best];
    return p;
}

// enrich
struct Row {
    int index;
    std::string file;
    char axis;
    double value;
    std::string session;
    char pred_axis;
    double pred;
    std::string branch;
    double cost;
    bool correct;
    double eY, eT;
    bool pass;
    int fold = -1;
};

inline double sgn(double v) { return (v > 0) - (v < 0); }

inline Row enrich(const std::vector<Item>& m, int i, const Pred& p) {
    Row r;
    r.index = i;
    r.file = m[i].file;
    r.axis = m[i].axis;
    r.value = m[i].value;
    r.session = m[i].session;
    r.pred_axis = p.pred_axis;
    r.pred = p.pred;
    r.branch = p.branch;
    r.cost = p.cost;
    r.correct = r.axis == r.pred_axis && sgn(r.value) == sgn(r.pred);
    double tY = r.axis == 'Y' ? r.value : 0, tT = r.axis == 'T' ? r.value : 0;
    double pY = r.pred_axis == 'Y' ? r.pred : 0, pT = r.pred_axis == 'T' ? r.pred : 0;
    r.eY = std::abs(tY - pY);
    r.eT = std::abs(tT - pT);
    r.pass = r.correct && r.eY <= .09 && r.eT <= .09;
    return r;
}

struct Stats {
    int n = 0, Y_correct = 0, Y_n = 0, T_correct = 0, T_n = 0, failures = 0;
    double Y_max = 0, T_max = 0;
};
inline Stats stats(const std::vector<Row>& rows) {
    Stats s;
    s.Y_max = -1; s.T_max = -1;
    for (auto& r : rows) {
        s.n++;
        if (r.axis == 'Y') { s.Y_n++; s.Y_correct += r.correct; }
        if (r.axis == 'T') { s.T_n++; s.T_correct += r.correct; }
        s.failures += !r.pass;
        s.Y_max = std::max(s.Y_max, r.eY);
        s.T_max = std::max(s.T_max, r.eT);
    }
    return s;
}

}  // namespace g09
