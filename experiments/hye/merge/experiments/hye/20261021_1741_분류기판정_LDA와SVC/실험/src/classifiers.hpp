// sklearn StandardScaler, LinearDiscriminantAnalysis(solver='lsqr', shrinkage=float), SVC(rbf, C, gamma) 를 수식대로 옮겼다.
// SVC 는 libsvm Solver(2차 정보 작업집합 선택 WSS3, 축소 shrinking, tol 1e-3, Q 를 float 로 저장)와 일대일 투표를 그대로 따른다.
#pragma once
#include <opencv2/core.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace clf {

using Mat = std::vector<std::vector<double>>;

struct Scaler {
    std::vector<double> mu, sd;
    void fit(const Mat& X) {
        size_t n = X.size(), d = X[0].size();
        mu.assign(d, 0); sd.assign(d, 0);
        for (auto& r : X) for (size_t k = 0; k < d; ++k) mu[k] += r[k];
        for (auto& v : mu) v /= n;
        for (auto& r : X) for (size_t k = 0; k < d; ++k) sd[k] += (r[k] - mu[k]) * (r[k] - mu[k]);
        for (auto& v : sd) { v = std::sqrt(v / n); if (v < 10 * std::numeric_limits<double>::epsilon()) v = 1; }
    }
    Mat transform(const Mat& X) const {
        Mat o = X;
        for (auto& r : o) for (size_t k = 0; k < r.size(); ++k) r[k] = (r[k] - mu[k]) / sd[k];
        return o;
    }
};

inline std::vector<std::string> classes_of(const std::vector<std::string>& y) {
    auto c = y;
    std::sort(c.begin(), c.end());
    c.erase(std::unique(c.begin(), c.end()), c.end());
    return c;
}

// ---------------------------------------------------------------- LDA (lsqr + 고정 축소)
struct LDA {
    std::vector<std::string> classes;
    cv::Mat coef, intercept;  // K x d, K x 1
    void fit(const Mat& X, const std::vector<std::string>& y, double shrink) {
        classes = classes_of(y);
        int K = (int)classes.size(), d = (int)X[0].size(), n = (int)X.size();
        cv::Mat means(K, d, CV_64F, cv::Scalar(0)), cov(d, d, CV_64F, cv::Scalar(0));
        std::vector<double> prior(K, 0);
        for (int c = 0; c < K; ++c) {
            std::vector<int> idx;
            for (int i = 0; i < n; ++i) if (y[i] == classes[c]) idx.push_back(i);
            prior[c] = (double)idx.size() / n;
            for (int i : idx) for (int k = 0; k < d; ++k) means.at<double>(c, k) += X[i][k];
            for (int k = 0; k < d; ++k) means.at<double>(c, k) /= idx.size();
            cv::Mat emp(d, d, CV_64F, cv::Scalar(0));
            for (int i : idx)
                for (int a = 0; a < d; ++a)
                    for (int b = 0; b < d; ++b) emp.at<double>(a, b) += (X[i][a] - means.at<double>(c, a)) * (X[i][b] - means.at<double>(c, b));
            emp /= (double)idx.size();
            double mu = cv::trace(emp)[0] / d;
            cv::Mat sh = (1 - shrink) * emp;
            for (int a = 0; a < d; ++a) sh.at<double>(a, a) += shrink * mu;
            cov += prior[c] * sh;
        }
        cv::Mat sol;
        cv::solve(cov, means.t(), sol, cv::DECOMP_SVD);
        coef = sol.t();
        intercept = cv::Mat(K, 1, CV_64F);
        for (int c = 0; c < K; ++c) intercept.at<double>(c) = -0.5 * means.row(c).dot(coef.row(c)) + std::log(prior[c]);
    }
    std::string predict(const std::vector<double>& x) const {
        int K = coef.rows;
        std::vector<double> s(K);
        for (int c = 0; c < K; ++c) { double t = intercept.at<double>(c); for (int k = 0; k < coef.cols; ++k) t += coef.at<double>(c, k) * x[k]; s[c] = t; }
        if (K == 2) return (s[1] - s[0]) > 0 ? classes[1] : classes[0];
        int b = 0;
        for (int c = 1; c < K; ++c) if (s[c] > s[b]) b = c;
        return classes[b];
    }
};

// ---------------------------------------------------------------- libsvm C-SVC (RBF)
struct BinarySVM {
    std::vector<int> sv;        // 학습 표본 위치
    std::vector<double> coef;   // alpha_i * y_i
    double rho = 0;
};

class Solver {
   public:
    // Q(i,j) = y_i y_j K(i,j) (float), QD = 1
    int l;
    std::vector<signed char> y;
    std::vector<double> G, G_bar, alpha, p, QD;
    std::vector<char> status;  // 0 lower, 1 upper, 2 free
    std::vector<int> active_set;
    int active_size;
    double C, eps;
    bool unshrink = false;
    std::vector<std::vector<float>> Q;  // l x l, 현재 순서
    const double TAU = 1e-12, INF = std::numeric_limits<double>::infinity();

    bool is_upper(int i) const { return status[i] == 1; }
    bool is_lower(int i) const { return status[i] == 0; }
    bool is_free(int i) const { return status[i] == 2; }
    void upd(int i) { status[i] = alpha[i] >= C ? 1 : (alpha[i] <= 0 ? 0 : 2); }
    void swap_index(int i, int j) {
        std::swap(y[i], y[j]); std::swap(G[i], G[j]); std::swap(status[i], status[j]); std::swap(alpha[i], alpha[j]);
        std::swap(p[i], p[j]); std::swap(active_set[i], active_set[j]); std::swap(G_bar[i], G_bar[j]); std::swap(QD[i], QD[j]);
        std::swap(Q[i], Q[j]);
        for (auto& row : Q) std::swap(row[i], row[j]);
    }
    void reconstruct_gradient() {
        if (active_size == l) return;
        int nr_free = 0;
        for (int j = active_size; j < l; ++j) G[j] = G_bar[j] + p[j];
        for (int j = 0; j < active_size; ++j) if (is_free(j)) nr_free++;
        if ((long long)nr_free * l > 2LL * active_size * (l - active_size)) {
            for (int i = active_size; i < l; ++i)
                for (int j = 0; j < active_size; ++j) if (is_free(j)) G[i] += alpha[j] * Q[i][j];
        } else {
            for (int i = 0; i < active_size; ++i)
                if (is_free(i)) { double a = alpha[i]; for (int j = active_size; j < l; ++j) G[j] += a * Q[i][j]; }
        }
    }
    int select_working_set(int& oi, int& oj) {
        double Gmax = -INF, Gmax2 = -INF;
        int Gmax_idx = -1, Gmin_idx = -1;
        double obj_diff_min = INF;
        for (int t = 0; t < active_size; ++t) {
            if (y[t] == +1) { if (!is_upper(t) && -G[t] >= Gmax) { Gmax = -G[t]; Gmax_idx = t; } }
            else { if (!is_lower(t) && G[t] >= Gmax) { Gmax = G[t]; Gmax_idx = t; } }
        }
        int i = Gmax_idx;
        for (int j = 0; j < active_size; ++j) {
            if (y[j] == +1) {
                if (!is_lower(j)) {
                    double gd = Gmax + G[j];
                    if (G[j] >= Gmax2) Gmax2 = G[j];
                    if (gd > 0) {
                        double qc = QD[i] + QD[j] - 2.0 * y[i] * Q[i][j];
                        double od = qc > 0 ? -(gd * gd) / qc : -(gd * gd) / TAU;
                        if (od <= obj_diff_min) { Gmin_idx = j; obj_diff_min = od; }
                    }
                }
            } else {
                if (!is_upper(j)) {
                    double gd = Gmax - G[j];
                    if (-G[j] >= Gmax2) Gmax2 = -G[j];
                    if (gd > 0) {
                        double qc = QD[i] + QD[j] + 2.0 * y[i] * Q[i][j];
                        double od = qc > 0 ? -(gd * gd) / qc : -(gd * gd) / TAU;
                        if (od <= obj_diff_min) { Gmin_idx = j; obj_diff_min = od; }
                    }
                }
            }
        }
        if (Gmax + Gmax2 < eps || Gmin_idx == -1) return 1;
        oi = Gmax_idx; oj = Gmin_idx;
        return 0;
    }
    bool be_shrunk(int i, double g1, double g2) const {
        if (is_upper(i)) return y[i] == +1 ? -G[i] > g1 : -G[i] > g2;
        if (is_lower(i)) return y[i] == +1 ? G[i] > g2 : G[i] > g1;
        return false;
    }
    void do_shrinking() {
        double g1 = -INF, g2 = -INF;
        for (int i = 0; i < active_size; ++i) {
            if (y[i] == +1) { if (!is_upper(i)) g1 = std::max(g1, -G[i]); if (!is_lower(i)) g2 = std::max(g2, G[i]); }
            else { if (!is_upper(i)) g2 = std::max(g2, -G[i]); if (!is_lower(i)) g1 = std::max(g1, G[i]); }
        }
        if (!unshrink && g1 + g2 <= eps * 10) { unshrink = true; reconstruct_gradient(); active_size = l; }
        for (int i = 0; i < active_size; ++i)
            if (be_shrunk(i, g1, g2)) {
                active_size--;
                while (active_size > i) {
                    if (!be_shrunk(active_size, g1, g2)) { swap_index(i, active_size); break; }
                    active_size--;
                }
            }
    }
    double solve(std::vector<double>& out_alpha) {
        active_size = l;
        status.assign(l, 0);
        for (int i = 0; i < l; ++i) upd(i);
        active_set.resize(l);
        for (int i = 0; i < l; ++i) active_set[i] = i;
        G = p; G_bar.assign(l, 0);
        for (int i = 0; i < l; ++i)
            if (!is_lower(i)) {
                for (int j = 0; j < l; ++j) G[j] += alpha[i] * Q[i][j];
                if (is_upper(i)) for (int j = 0; j < l; ++j) G_bar[j] += C * Q[i][j];
            }
        long long iter = 0, max_iter = std::max(10000000LL, (long long)100 * l);
        int counter = std::min(l, 1000) + 1;
        while (iter < max_iter) {
            if (--counter == 0) { counter = std::min(l, 1000); do_shrinking(); }
            int i, j;
            if (select_working_set(i, j) != 0) {
                reconstruct_gradient();
                active_size = l;
                if (select_working_set(i, j) != 0) break;
                else counter = 1;
            }
            ++iter;
            double Ci = C, Cj = C, oai = alpha[i], oaj = alpha[j];
            if (y[i] != y[j]) {
                double qc = QD[i] + QD[j] + 2 * Q[i][j];
                if (qc <= 0) qc = TAU;
                double delta = (-G[i] - G[j]) / qc, diff = alpha[i] - alpha[j];
                alpha[i] += delta; alpha[j] += delta;
                if (diff > 0) { if (alpha[j] < 0) { alpha[j] = 0; alpha[i] = diff; } }
                else { if (alpha[i] < 0) { alpha[i] = 0; alpha[j] = -diff; } }
                if (diff > Ci - Cj) { if (alpha[i] > Ci) { alpha[i] = Ci; alpha[j] = Ci - diff; } }
                else { if (alpha[j] > Cj) { alpha[j] = Cj; alpha[i] = Cj + diff; } }
            } else {
                double qc = QD[i] + QD[j] - 2 * Q[i][j];
                if (qc <= 0) qc = TAU;
                double delta = (G[i] - G[j]) / qc, sum = alpha[i] + alpha[j];
                alpha[i] -= delta; alpha[j] += delta;
                if (sum > Ci) { if (alpha[i] > Ci) { alpha[i] = Ci; alpha[j] = sum - Ci; } }
                else { if (alpha[j] < 0) { alpha[j] = 0; alpha[i] = sum; } }
                if (sum > Cj) { if (alpha[j] > Cj) { alpha[j] = Cj; alpha[i] = sum - Cj; } }
                else { if (alpha[i] < 0) { alpha[i] = 0; alpha[j] = sum; } }
            }
            double dai = alpha[i] - oai, daj = alpha[j] - oaj;
            for (int k = 0; k < active_size; ++k) G[k] += Q[i][k] * dai + Q[j][k] * daj;
            bool ui = is_upper(i), uj = is_upper(j);
            upd(i); upd(j);
            if (ui != is_upper(i)) { if (ui) for (int k = 0; k < l; ++k) G_bar[k] -= Ci * Q[i][k]; else for (int k = 0; k < l; ++k) G_bar[k] += Ci * Q[i][k]; }
            if (uj != is_upper(j)) { if (uj) for (int k = 0; k < l; ++k) G_bar[k] -= Cj * Q[j][k]; else for (int k = 0; k < l; ++k) G_bar[k] += Cj * Q[j][k]; }
        }
        if (iter >= max_iter && active_size < l) { reconstruct_gradient(); active_size = l; }
        // rho
        double r;
        {
            int nr_free = 0;
            double ub = INF, lb = -INF, sum_free = 0;
            for (int i = 0; i < active_size; ++i) {
                double yG = y[i] * G[i];
                if (is_upper(i)) { if (y[i] == -1) ub = std::min(ub, yG); else lb = std::max(lb, yG); }
                else if (is_lower(i)) { if (y[i] == +1) ub = std::min(ub, yG); else lb = std::max(lb, yG); }
                else { ++nr_free; sum_free += yG; }
            }
            r = nr_free > 0 ? sum_free / nr_free : (ub + lb) / 2;
        }
        out_alpha.assign(l, 0);
        for (int i = 0; i < l; ++i) out_alpha[active_set[i]] = alpha[i];
        return r;
    }
};

inline double rbf_train(const std::vector<double>& a, const std::vector<double>& b, double sa, double sb, double gamma) {
    double d = 0;
    for (size_t k = 0; k < a.size(); ++k) d += a[k] * b[k];
    return std::exp(-gamma * (sa + sb - 2 * d));
}
inline double rbf_pred(const std::vector<double>& a, const std::vector<double>& b, double gamma) {
    double s = 0;
    for (size_t k = 0; k < a.size(); ++k) { double d = a[k] - b[k]; s += d * d; }
    return std::exp(-gamma * s);
}

struct SVC {
    double C = 1, gamma = .1;
    std::vector<std::string> classes;
    std::vector<std::vector<double>> X;
    std::vector<std::vector<int>> pair_idx;  // 쌍별 학습 표본 위치
    std::vector<std::vector<double>> pair_coef;
    std::vector<double> pair_rho;
    void fit(const Mat& Xin, const std::vector<std::string>& y) {
        X = Xin;
        classes = classes_of(y);
        int K = (int)classes.size();
        std::vector<std::vector<int>> by(K);
        for (int i = 0; i < (int)y.size(); ++i) by[std::lower_bound(classes.begin(), classes.end(), y[i]) - classes.begin()].push_back(i);
        std::vector<double> sq(X.size());
        for (size_t i = 0; i < X.size(); ++i) { double s = 0; for (double v : X[i]) s += v * v; sq[i] = s; }
        pair_idx.clear(); pair_coef.clear(); pair_rho.clear();
        for (int a = 0; a < K; ++a)
            for (int b = a + 1; b < K; ++b) {
                std::vector<int> idx = by[a];
                idx.insert(idx.end(), by[b].begin(), by[b].end());
                int l = (int)idx.size();
                Solver s;
                s.l = l; s.C = C; s.eps = 1e-3;
                s.y.resize(l);
                for (int i = 0; i < l; ++i) s.y[i] = i < (int)by[a].size() ? 1 : -1;
                s.p.assign(l, -1.0);
                s.alpha.assign(l, 0.0);
                s.QD.assign(l, 1.0);
                for (int i = 0; i < l; ++i) s.QD[i] = rbf_train(X[idx[i]], X[idx[i]], sq[idx[i]], sq[idx[i]], gamma);
                s.Q.assign(l, std::vector<float>(l));
                for (int i = 0; i < l; ++i)
                    for (int j = 0; j < l; ++j) s.Q[i][j] = (float)(s.y[i] * s.y[j] * rbf_train(X[idx[i]], X[idx[j]], sq[idx[i]], sq[idx[j]], gamma));
                std::vector<signed char> y0 = s.y;
                std::vector<double> al;
                double rho = s.solve(al);
                std::vector<int> keep;
                std::vector<double> co;
                for (int i = 0; i < l; ++i) if (std::abs(al[i]) > 0) { keep.push_back(idx[i]); co.push_back(al[i] * y0[i]); }
                pair_idx.push_back(keep); pair_coef.push_back(co); pair_rho.push_back(rho);
            }
    }
    std::string predict(const std::vector<double>& x) const {
        int K = (int)classes.size();
        std::vector<int> vote(K, 0);
        int p = 0;
        for (int a = 0; a < K; ++a)
            for (int b = a + 1; b < K; ++b, ++p) {
                double s = 0;
                for (size_t t = 0; t < pair_idx[p].size(); ++t) s += pair_coef[p][t] * rbf_pred(X[pair_idx[p][t]], x, gamma);
                s -= pair_rho[p];
                if (s > 0) ++vote[a]; else ++vote[b];
            }
        int best = 0;
        for (int c = 1; c < K; ++c) if (vote[c] > vote[best]) best = c;
        return classes[best];
    }
};

}  // namespace clf
