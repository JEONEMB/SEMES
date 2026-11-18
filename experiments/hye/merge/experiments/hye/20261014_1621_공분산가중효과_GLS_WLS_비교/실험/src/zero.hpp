// codex zero_experiment.py (fit / predict / score / splits) 와 gls_comparison compare.py 의 공분산 함수 C++ 이식.
// 자료: 281장 (0 = center), 축 Y/T, 시행(session) 6개의 0점 분기, 2차 응답 곡선, 부호 제약 분기 추론.
#pragma once
#include <opencv2/core.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <vector>
#include "kfold.hpp"

namespace zx {

struct Row {
    std::string file, session;
    char axis;  // 'C', 'Y', 'T'
    double value;
};

struct Branch {
    std::string name;
    char axis;
    cv::Mat base, s, c;  // 1 x p
};
struct Model {
    std::vector<Branch> br;
    cv::Mat W;  // p x p (Z = X W^T)
};

inline cv::Mat lstsq(const cv::Mat& A, const cv::Mat& B) {
    cv::Mat X;
    cv::solve(A, B, X, cv::DECOMP_SVD);
    return X;
}
inline int matrix_rank(const cv::Mat& A) {
    cv::Mat w;
    cv::SVD::compute(A, w, cv::SVD::NO_UV);
    double mx = 0;
    for (int i = 0; i < w.rows; ++i) mx = std::max(mx, w.at<double>(i));
    double tol = mx * std::max(A.rows, A.cols) * std::numeric_limits<double>::epsilon();
    int r = 0;
    for (int i = 0; i < w.rows; ++i) r += w.at<double>(i) > tol;
    return r;
}
inline cv::Mat cholesky_lower(const cv::Mat& S) {  // S = L L^T
    const int n = S.rows;
    cv::Mat L = cv::Mat::zeros(n, n, CV_64F);
    for (int j = 0; j < n; ++j) {
        double s = S.at<double>(j, j);
        const double* Lj = L.ptr<double>(j);
        for (int k = 0; k < j; ++k) s -= Lj[k] * Lj[k];
        double d = std::sqrt(s);
        L.at<double>(j, j) = d;
        for (int i = j + 1; i < n; ++i) {
            double t = S.at<double>(i, j);
            const double* Li = L.ptr<double>(i);
            for (int k = 0; k < j; ++k) t -= Li[k] * Lj[k];
            L.at<double>(i, j) = t / d;
        }
    }
    return L;
}
inline cv::Mat inv(const cv::Mat& A) {
    cv::Mat B;
    cv::invert(A, B, cv::DECOMP_LU);
    return B;
}

// 축마다 [v, v^2, 시행 원-핫] 최소제곱. branches 는 축 Y 다음 T, 각 축 안 시행 이름 사전순
inline Model fit_curves(const std::vector<Row>& m, const cv::Mat& X, const std::vector<int>& tr, cv::Mat* R = nullptr, int* rank = nullptr) {
    Model out;
    cv::Mat res;
    int rk = 0;
    for (char a : {'Y', 'T'}) {
        std::vector<int> ii;
        for (int i : tr) if (m[i].axis == a) ii.push_back(i);
        std::set<std::string> gs;
        for (int i : ii) gs.insert(m[i].session);
        std::vector<std::string> groups(gs.begin(), gs.end());
        cv::Mat A((int)ii.size(), 2 + (int)groups.size(), CV_64F, cv::Scalar(0)), Xa;
        for (int r = 0; r < A.rows; ++r) {
            double v = m[ii[r]].value;
            A.at<double>(r, 0) = v;
            A.at<double>(r, 1) = v * v;
            for (size_t g = 0; g < groups.size(); ++g) A.at<double>(r, 2 + (int)g) = m[ii[r]].session == groups[g];
            Xa.push_back(X.row(ii[r]));
        }
        cv::Mat co = lstsq(A, Xa);
        res.push_back(cv::Mat(Xa - A * co));
        rk += matrix_rank(A);
        for (size_t g = 0; g < groups.size(); ++g)
            out.br.push_back({groups[g], a, co.row(2 + (int)g).clone(), co.row(0).clone(), co.row(1).clone()});
    }
    if (R) *R = res;
    if (rank) *rank = rk;
    return out;
}

// sklearn LedoitWolf(assume_centered=True): 공분산과 축소 계수
inline cv::Mat ledoit_wolf(const cv::Mat& X, double* shrink_out = nullptr) {
    const int n = X.rows, p = X.cols;
    cv::Mat emp = X.t() * X / n;
    double tr = cv::trace(emp)[0], mu = tr / p;
    cv::Mat X2 = X.mul(X);
    double beta_ = 0;
    for (int k = 0; k < n; ++k) {
        double s = cv::sum(X2.row(k))[0];
        beta_ += s * s;
    }
    cv::Mat XtX = X.t() * X;
    double delta_ = cv::sum(XtX.mul(XtX))[0] / ((double)n * n);
    double beta = 1.0 / ((double)p * n) * (beta_ / n - delta_);
    double delta = (delta_ - 2.0 * mu * tr + p * mu * mu) / p;
    beta = std::min(beta, delta);
    double sh = beta == 0 ? 0 : beta / delta;
    if (shrink_out) *shrink_out = sh;
    cv::Mat C = (1 - sh) * emp;
    for (int i = 0; i < p; ++i) C.at<double>(i, i) += sh * mu;
    return C;
}

// compare.py covariance: identity / diagonal / legacy / ledoit_wolf
inline cv::Mat covariance(const cv::Mat& R, int rank, const std::string& kind, double* alpha = nullptr) {
    const int p = R.cols;
    cv::Mat S = R.t() * R / std::max(R.rows - rank, 1);
    std::vector<double> sd(p);
    for (int j = 0; j < p; ++j) sd[j] = std::sqrt(std::max(S.at<double>(j, j), 1e-14));
    if (alpha) *alpha = 1;
    if (kind == "identity") return cv::Mat::eye(p, p, CV_64F);
    if (kind == "diagonal") {
        cv::Mat D = cv::Mat::zeros(p, p, CV_64F);
        for (int j = 0; j < p; ++j) D.at<double>(j, j) = sd[j] * sd[j];
        return D;
    }
    cv::Mat C(p, p, CV_64F);
    if (kind == "legacy") {
        for (int i = 0; i < p; ++i)
            for (int j = 0; j < p; ++j) C.at<double>(i, j) = S.at<double>(i, j) / sd[i] / sd[j] * .999 + (i == j ? .001 : 0);
        if (alpha) *alpha = .001;
    } else {
        cv::Mat Z(R.size(), CV_64F);
        for (int i = 0; i < R.rows; ++i)
            for (int j = 0; j < p; ++j) Z.at<double>(i, j) = R.at<double>(i, j) / sd[j];
        C = ledoit_wolf(Z, alpha);
    }
    for (int i = 0; i < p; ++i)
        for (int j = 0; j < p; ++j) C.at<double>(i, j) *= sd[i] * sd[j];
    return C;
}

// zero_experiment.fit(mode='inferred_session', quad=True) 의 거리: chol(inv(C)).T / std
inline cv::Mat legacy_W(const cv::Mat& R, int rank) {
    const int p = R.cols;
    cv::Mat S = R.t() * R / std::max(R.rows - rank, 1);
    std::vector<double> sd(p);
    for (int j = 0; j < p; ++j) sd[j] = std::sqrt(std::max(S.at<double>(j, j), 1e-14));
    cv::Mat C(p, p, CV_64F);
    for (int i = 0; i < p; ++i)
        for (int j = 0; j < p; ++j) C.at<double>(i, j) = S.at<double>(i, j) / sd[i] / sd[j] * .999 + (i == j ? .001 : 0);
    cv::Mat W = cholesky_lower(inv(C)).t();
    for (int i = 0; i < p; ++i)
        for (int j = 0; j < p; ++j) W.at<double>(i, j) /= sd[j];
    return W;
}
inline cv::Mat precision_root(const cv::Mat& S) { return cholesky_lower(inv(S)).t(); }

struct Pred {
    int i;
    char axis, pred_axis;
    double value, pred, cost;
    std::string branch;
    double eY() const { return std::abs((pred_axis == 'Y' ? pred : 0) - (axis == 'Y' ? value : 0)); }
    double eT() const { return std::abs((pred_axis == 'T' ? pred : 0) - (axis == 'T' ? value : 0)); }
    static int sgn(double v) { return (v > 0) - (v < 0); }
    bool correct() const { return axis == pred_axis && sgn(value) == sgn(pred); }
    bool pass() const { return correct() && eY() <= .09 && eT() <= .09; }
};

// zero_experiment.predict(mode='inferred_session'): 분기마다 2차 곡선 GN 8회, 부호가 분기와 다르면 비용 무한대
inline std::vector<Pred> predict(const std::vector<Row>& m, const cv::Mat& X, const std::vector<int>& te, const Model& md) {
    const int p = X.cols;
    struct BW { cv::Mat sw, cw, base; double sgn; };
    std::vector<BW> bw;
    for (auto& b : md.br) bw.push_back({b.s * md.W.t(), b.c * md.W.t(), b.base, b.name.back() == '+' ? 1.0 : -1.0});
    std::vector<Pred> out;
    for (int i : te) {
        double best = 0;
        int bj = -1;
        double bv = 0;
        for (size_t j = 0; j < bw.size(); ++j) {
            cv::Mat z = (X.row(i) - bw[j].base) * md.W.t();
            const double *Z = z.ptr<double>(), *S = bw[j].sw.ptr<double>(), *C = bw[j].cw.ptr<double>();
            double ss = 0, zs = 0;
            for (int k = 0; k < p; ++k) { zs += Z[k] * S[k]; ss += S[k] * S[k]; }
            double v = zs / ss;
            for (int it = 0; it < 8; ++it) {
                double jr = 0, jj = 0;
                for (int k = 0; k < p; ++k) {
                    double J = S[k] + 2 * v * C[k], r = Z[k] - v * S[k] - v * v * C[k];
                    jr += J * r;
                    jj += J * J;
                }
                v += jr / std::max(jj, 1e-20);
            }
            double cost = 0;
            for (int k = 0; k < p; ++k) { double r = Z[k] - v * S[k] - v * v * C[k]; cost += r * r; }
            if (!(v * bw[j].sgn >= 0)) cost = std::numeric_limits<double>::infinity();
            if (bj < 0 || cost < best) { best = cost; bj = (int)j; bv = v; }
        }
        out.push_back({i, m[i].axis, md.br[bj].axis, m[i].value, bv, best, md.br[bj].name});
    }
    return out;
}

struct Score {
    int n = 0, correct = 0, failed = 0, Yc = 0, Tc = 0;
    double Ymax = 0, Tmax = 0;
};
inline Score score(const std::vector<Pred>& p) {
    Score s;
    s.n = (int)p.size();
    for (auto& r : p) {
        s.correct += r.correct();
        s.failed += !r.pass();
        (r.axis == 'Y' ? s.Yc : s.Tc) += r.correct();
        s.Ymax = std::max(s.Ymax, r.eY());
        s.Tmax = std::max(s.Tmax, r.eT());
    }
    return s;
}

struct Fold {
    std::string name;
    std::vector<int> tr, te;
};
inline std::vector<int> noncenter(const std::vector<Row>& m) {
    std::vector<int> ids;
    for (int i = 0; i < (int)m.size(); ++i) if (m[i].axis != 'C') ids.push_back(i);
    return ids;
}
inline std::vector<std::string> session_labels(const std::vector<Row>& m) {
    std::vector<std::string> s;
    for (auto& r : m) s.push_back(r.session);
    return s;
}
inline std::vector<Fold> stratified(const std::vector<Row>& m, int k, uint32_t seed) {
    std::vector<Fold> out;
    auto sp = kf::split(noncenter(m), session_labels(m), k, seed);
    for (size_t f = 0; f < sp.size(); ++f) out.push_back({std::to_string(f), sp[f].first, sp[f].second});
    return out;
}
inline std::vector<Fold> session_out(const std::vector<Row>& m) {
    std::set<std::string> us;
    for (auto& r : m) if (r.axis != 'C') us.insert(r.session);
    std::vector<Fold> out;
    for (auto& s : us) {
        Fold f{s, {}, {}};
        for (int i = 0; i < (int)m.size(); ++i)
            if (m[i].axis != 'C') (m[i].session == s ? f.te : f.tr).push_back(i);
        out.push_back(f);
    }
    return out;
}
// compare.py blocked_within_session5: 시행마다 값 순서로 5등분해 k 번째 묶음을 시험
inline std::vector<Fold> blocked5(const std::vector<Row>& m) {
    std::vector<std::string> order;
    for (auto& r : m)
        if (r.axis != 'C' && std::find(order.begin(), order.end(), r.session) == order.end()) order.push_back(r.session);
    std::vector<Fold> out;
    for (int k = 0; k < 5; ++k) {
        std::vector<int> te;
        for (auto& s : order) {
            std::vector<int> ii;
            for (int i = 0; i < (int)m.size(); ++i) if (m[i].session == s) ii.push_back(i);
            std::stable_sort(ii.begin(), ii.end(), [&](int a, int b) { return m[a].value < m[b].value; });
            // np.array_split: 앞쪽 묶음이 하나씩 더 크다
            int n = (int)ii.size(), q = n / 5, r = n % 5, start = 0;
            for (int j = 0; j < k; ++j) start += q + (j < r);
            int len = q + (k < r);
            te.insert(te.end(), ii.begin() + start, ii.begin() + start + len);
        }
        std::sort(te.begin(), te.end());
        Fold f{std::to_string(k), {}, te};
        for (int i : noncenter(m)) if (!std::binary_search(te.begin(), te.end(), i)) f.tr.push_back(i);
        out.push_back(f);
    }
    return out;
}

}  // namespace zx
