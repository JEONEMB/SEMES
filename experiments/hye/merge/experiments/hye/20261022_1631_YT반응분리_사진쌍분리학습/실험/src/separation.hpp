// Y/T 반응 분리 학습 후보 (원본: joint_yt_disentangle_20261022/learned_model.py, geometric/geometric_models.py(RepeatGLS),
// paired_model.py(PairedRidge, graph_operator), conditional/conditional_model.py(ConditionalDecoder, ScalarGLS),
// neural_model.py(PairResidualNet), joint_center_630_20261022/features.py(FeatureMaps P12))
#pragma once
#include <opencv2/core.hpp>
#include <cmath>
#include <map>
#include <random>
#include <set>
#include <string>
#include <tuple>
#include <vector>
#include "corners630.hpp"
#include "ytmodel.hpp"
#include "nprng.hpp"

namespace sep {

using Row = std::vector<double>;

// 학습 사진 메타데이터 (fit 에만 쓴다)
struct Meta {
    std::vector<std::string> group;  // cv_group
    std::vector<int> date;           // 20261014 / 20261020
    std::vector<std::string> strat;  // old / new_center / new_mixed
};

// ---- 작은 행렬 도우미 (행 우선 cv::Mat 사용) ----
inline cv::Mat rows_to_mat(const std::vector<Row>& X) {
    cv::Mat m((int)X.size(), (int)X[0].size(), CV_64F);
    for (int i = 0; i < m.rows; ++i) std::copy(X[i].begin(), X[i].end(), m.ptr<double>(i));
    return m;
}
// numpy lstsq(A, B, rcond): SVD 최소 노름 해
inline cv::Mat lstsq(const cv::Mat& A, const cv::Mat& B, double rcond) {
    cv::Mat s, u, vt;
    cv::SVD::compute(A, s, u, vt);
    double cut = rcond * s.at<double>(0);
    cv::Mat inv = cv::Mat::zeros(s.rows, s.rows, CV_64F);
    for (int k = 0; k < s.rows; ++k)
        if (s.at<double>(k) > cut) inv.at<double>(k, k) = 1 / s.at<double>(k);
    return vt.t() * inv * (u.t() * B);
}
inline double eps_rcond(int m, int n) { return std::max(m, n) * 2.220446049250313e-16; }  // rcond=None
// K = Z Z^T (n x n), 결정적 순서
inline cv::Mat gram(const cv::Mat& Z) {
    cv::Mat K(Z.rows, Z.rows, CV_64F);
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < Z.rows; ++i) {
        const double* a = Z.ptr<double>(i);
        for (int k = i; k < Z.rows; ++k) {
            const double* b = Z.ptr<double>(k);
            double s = 0;
            for (int j = 0; j < Z.cols; ++j) s += a[j] * b[j];
            K.at<double>(i, k) = K.at<double>(k, i) = s;
        }
    }
    return K;
}
// (V - b Z^T solve(aI + b ZZ^T, Z V)) / a
inline cv::Mat shrunk_solve(const cv::Mat& Z, const cv::Mat& V, double a, double b) {
    cv::Mat K = gram(Z) * b;
    for (int i = 0; i < K.rows; ++i) K.at<double>(i, i) += a;
    cv::Mat ZV = Z * V, S;
    cv::solve(K, ZV, S, cv::DECOMP_LU);
    return (V - b * (Z.t() * S)) / a;
}

// ---- 특징 ----
// P12: 630점 변위의 전역 2차 다항 계수 (6 x 2)
struct P12Map {
    cv::Mat pinv;  // 6 x 630
    P12Map() {
        cv::Mat A(630, 6, CV_64F);
        for (int k = 0; k < 630; ++k) {
            double u = (c630::kCorners[k].px - 1906) / 486, v = (c630::kCorners[k].py - 1070) / 122;
            double r[6] = {1, u, v, u * u, u * v, v * v};
            for (int j = 0; j < 6; ++j) A.at<double>(k, j) = r[j];
        }
        cv::invert(A, pinv, cv::DECOMP_SVD);
    }
    Row operator()(const Row& D) const {  // D: 1260 (dx,dy 교차) -> 12 (a0x,a0y,a1x,...)
        Row o(12, 0.0);
        for (int a = 0; a < 6; ++a)
            for (int k = 0; k < 630; ++k) {
                double w = pinv.at<double>(a, k);
                o[a * 2] += w * D[2 * k];
                o[a * 2 + 1] += w * D[2 * k + 1];
            }
        return o;
    }
};
inline Row raw_shift(const Row& D) {
    double h = 0;
    for (int k = 0; k < 630; ++k) h += D[2 * k];
    h = h / 630 / 100;
    Row o(D);
    for (double v : D) o.push_back(v * h);
    return o;
}

// ---- ForwardGLS 1차 (행렬 입력) ----
struct FGLS {
    ytm::ForwardGLS m;
    void fit(const std::vector<Row>& X, const std::vector<double>& y, const std::vector<double>& t, double shrink) {
        std::vector<const std::vector<double>*> p(X.size());
        for (size_t i = 0; i < X.size(); ++i) p[i] = &X[i];
        m.shrink = shrink;
        m.fit(p, y, t, {});
    }
    void predict(const Row& x, double& py, double& pt) const { m.predict(x, py, pt); }
};

// ---- RepeatGLS ----
struct RepeatGLS {
    double shrink = .1, mix = 0;
    std::string mode;
    Row origin;
    cv::Mat beta;  // p x 2
    void fit(const std::vector<Row>& X, const std::vector<double>& y, const std::vector<double>& t, const Meta& md) {
        const int n = (int)X.size(), p = (int)X[0].size();
        std::vector<std::string> names;
        for (int i = 0; i < n; ++i) if (md.date[i] == 20261020) names.push_back(md.group[i]);
        std::sort(names.begin(), names.end());
        names.erase(std::unique(names.begin(), names.end()), names.end());
        const int G = (int)names.size();
        cv::Mat gx = cv::Mat::zeros(G, p, CV_64F), gy = cv::Mat::zeros(G, 2, CV_64F);
        std::vector<std::vector<int>> members(G);
        for (int i = 0; i < n; ++i)
            if (md.date[i] == 20261020) members[std::lower_bound(names.begin(), names.end(), md.group[i]) - names.begin()].push_back(i);
        for (int g = 0; g < G; ++g) {
            for (int i : members[g]) {
                for (int j = 0; j < p; ++j) gx.at<double>(g, j) += X[i][j];
                gy.at<double>(g, 0) += y[i];
                gy.at<double>(g, 1) += t[i];
            }
            gx.row(g) /= (double)members[g].size();
            gy.row(g) /= (double)members[g].size();
        }
        cv::Mat coef;  // 3 x p
        if (mode == "new_condition_means") {
            cv::Mat A(G, 3, CV_64F);
            for (int g = 0; g < G; ++g) { A.at<double>(g, 0) = 1; A.at<double>(g, 1) = gy.at<double>(g, 0); A.at<double>(g, 2) = gy.at<double>(g, 1); }
            coef = lstsq(A, gx, eps_rcond(G, 3));
        } else {  // axis_pairs
            cv::Mat slopes(2, p, CV_64F);
            for (int axis = 0; axis < 2; ++axis) {
                cv::Mat acc = cv::Mat::zeros(1, p, CV_64F);
                double dd = 0;
                int cnt = 0;
                for (int i = 0; i < G; ++i)
                    for (int j = 0; j < i; ++j)
                        if (std::abs(gy.at<double>(i, 1 - axis) - gy.at<double>(j, 1 - axis)) < 1e-9 &&
                            std::abs(gy.at<double>(i, axis) - gy.at<double>(j, axis)) > 1e-8) {
                            double dy = gy.at<double>(i, axis) - gy.at<double>(j, axis);
                            acc += dy * (gx.row(i) - gx.row(j));
                            dd += dy * dy;
                            ++cnt;
                        }
                if (cnt < 2) throw std::runtime_error("axis_pairs: 짝 부족");
                cv::Mat s = acc / dd;
                s.copyTo(slopes.row(axis));
            }
            cv::Mat org = cv::Mat::zeros(1, p, CV_64F);
            for (int g = 0; g < G; ++g) org += gx.row(g) - gy.at<double>(g, 0) * slopes.row(0) - gy.at<double>(g, 1) * slopes.row(1);
            org /= G;
            coef = cv::Mat(3, p, CV_64F);
            org.copyTo(coef.row(0));
            slopes.copyTo(coef.rowRange(1, 3));
        }
        origin.assign(coef.ptr<double>(0), coef.ptr<double>(0) + p);
        cv::Mat slope = coef.rowRange(1, 3);
        // 모델 잔차 / sqrt(n), 반복 잔차 / sqrt(dof)
        int dof = 0, nrep = 0;
        for (auto& mem : members) if (mem.size() > 1) { dof += (int)mem.size() - 1; nrep += (int)mem.size(); }
        cv::Mat R = cv::Mat::zeros(n + nrep, p, CV_64F);
        double f1 = std::sqrt(1 - mix) / std::sqrt((double)n), f2 = std::sqrt(mix) / std::sqrt((double)std::max(dof, 1));
        for (int i = 0; i < n; ++i) {
            double* r = R.ptr<double>(i);
            for (int j = 0; j < p; ++j)
                r[j] = (X[i][j] - (coef.at<double>(0, j) + y[i] * coef.at<double>(1, j) + t[i] * coef.at<double>(2, j))) * f1;
        }
        int row = n;
        for (int g = 0; g < G; ++g) {
            if (members[g].size() < 2) continue;
            for (int i : members[g]) {
                double* r = R.ptr<double>(row++);
                for (int j = 0; j < p; ++j) r[j] = (X[i][j] - gx.at<double>(g, j)) * f2;
            }
        }
        cv::Mat sd(1, p, CV_64F);
        for (int j = 0; j < p; ++j) {
            double s = 0;
            for (int i = 0; i < R.rows; ++i) s += R.at<double>(i, j) * R.at<double>(i, j);
            sd.at<double>(j) = std::max(std::sqrt(s), 1e-6);
        }
        cv::Mat Z(R.size(), CV_64F), Vt(2, p, CV_64F);
        for (int i = 0; i < R.rows; ++i)
            for (int j = 0; j < p; ++j) Z.at<double>(i, j) = R.at<double>(i, j) / sd.at<double>(j);
        for (int a = 0; a < 2; ++a)
            for (int j = 0; j < p; ++j) Vt.at<double>(a, j) = slope.at<double>(a, j) / sd.at<double>(j);
        cv::Mat Q = shrunk_solve(Z, Vt.t(), shrink, 1 - shrink);
        cv::Mat M = Vt * Q, Mi;
        cv::invert(M, Mi, cv::DECOMP_SVD);
        beta = Q * Mi;
        for (int j = 0; j < p; ++j) beta.row(j) /= sd.at<double>(j);
    }
    void predict(const Row& x, double& py, double& pt) const {
        py = pt = 0;
        for (int j = 0; j < beta.rows; ++j) { double d = x[j] - origin[j]; py += d * beta.at<double>(j, 0); pt += d * beta.at<double>(j, 1); }
    }
};

// ---- graph_operator / PairedRidge ----
inline cv::Mat graph_operator(const std::vector<double>& y, const std::vector<double>& t, const std::vector<int>& dates, const std::string& kind) {
    const int n = (int)y.size();
    cv::Mat L = cv::Mat::zeros(n, n, CV_64F);
    std::map<std::tuple<int, double, double>, std::vector<int>> groups;
    for (int i = 0; i < n; ++i) {
        if (kind == "repeat") groups[{dates[i], y[i], t[i]}].push_back(i);
        else if (kind == "sameT") groups[{dates[i], t[i], 0.0}].push_back(i);
        else groups[{dates[i], y[i], 0.0}].push_back(i);
    }
    int count = 0;
    for (auto& [k, ix] : groups) {
        if (ix.size() < 2) continue;
        double m = (double)ix.size();
        for (int a : ix)
            for (int b : ix) L.at<double>(a, b) += (a == b ? 1.0 : 0.0) - 1.0 / m;
        count += (int)ix.size() - 1;
    }
    return L * ((double)n / std::max(count, 1));
}

struct PairedRidge {
    double alpha = 1, pair_w = 1, rep_w = 10;
    Row mean, sd;
    cv::Mat beta;  // p x 2
    double origins[2]{};
    void fit(const std::vector<Row>& X, const std::vector<double>& y, const std::vector<double>& t, const std::vector<int>& dates) {
        const int n = (int)X.size(), p = (int)X[0].size();
        mean.assign(p, 0); sd.assign(p, 0);
        for (auto& r : X) for (int j = 0; j < p; ++j) mean[j] += r[j];
        for (auto& v : mean) v /= n;
        for (auto& r : X) for (int j = 0; j < p; ++j) sd[j] += (r[j] - mean[j]) * (r[j] - mean[j]);
        for (auto& v : sd) v = std::max(std::sqrt(v / n), 1e-6);
        cv::Mat Z(n, p, CV_64F);
        for (int i = 0; i < n; ++i) for (int j = 0; j < p; ++j) Z.at<double>(i, j) = (X[i][j] - mean[j]) / sd[j];
        cv::Mat Lr = graph_operator(y, t, dates, "repeat"), K = Z * Z.t(), I = cv::Mat::eye(n, n, CV_64F);
        beta = cv::Mat(p, 2, CV_64F);
        for (int out = 0; out < 2; ++out) {
            cv::Mat La = graph_operator(y, t, dates, out == 0 ? "sameT" : "sameY");
            cv::Mat W = I + pair_w * La + rep_w * Lr;
            cv::Mat yv(n, 1, CV_64F);
            for (int i = 0; i < n; ++i) yv.at<double>(i) = out == 0 ? y[i] : t[i];
            cv::Mat Wy = (I + pair_w * La) * yv;
            double org = cv::mean(yv)[0];
            cv::Mat dual;
            cv::solve(W * K + alpha * I, Wy - org, dual, cv::DECOMP_LU);
            cv::Mat c = Z.t() * dual;
            c.copyTo(beta.col(out));
            origins[out] = org;
        }
    }
    void predict(const Row& x, double& py, double& pt) const {
        py = origins[0]; pt = origins[1];
        for (int j = 0; j < beta.rows; ++j) { double z = (x[j] - mean[j]) / sd[j]; py += z * beta.at<double>(j, 0); pt += z * beta.at<double>(j, 1); }
    }
};

// ---- ScalarGLS / ConditionalDecoder ----
struct ScalarGLS {
    double shrink = .5;
    Row origin, beta;
    void fit(const std::vector<Row>& X, const std::vector<double>& y) {
        const int n = (int)X.size(), p = (int)X[0].size();
        cv::Mat A(n, 2, CV_64F), Xm = rows_to_mat(X);
        for (int i = 0; i < n; ++i) { A.at<double>(i, 0) = 1; A.at<double>(i, 1) = y[i]; }
        cv::Mat co = lstsq(A, Xm, 1e-10);
        origin.assign(co.ptr<double>(0), co.ptr<double>(0) + p);
        cv::Mat r = Xm - A * co;
        Row sd(p);
        for (int j = 0; j < p; ++j) {
            double s = 0;
            for (int i = 0; i < n; ++i) s += r.at<double>(i, j) * r.at<double>(i, j);
            sd[j] = std::max(std::sqrt(s / n), 1e-6);
        }
        cv::Mat R(n, p, CV_64F), v(p, 1, CV_64F);
        for (int i = 0; i < n; ++i) for (int j = 0; j < p; ++j) R.at<double>(i, j) = r.at<double>(i, j) / sd[j];
        for (int j = 0; j < p; ++j) v.at<double>(j) = co.at<double>(1, j) / sd[j];
        cv::Mat q = shrunk_solve(R, v, shrink, (1 - shrink) / n);
        double vq = v.dot(q);
        beta.resize(p);
        for (int j = 0; j < p; ++j) beta[j] = q.at<double>(j) / vq / sd[j];
    }
    double predict(const Row& x) const {
        double s = 0;
        for (size_t j = 0; j < beta.size(); ++j) s += (x[j] - origin[j]) * beta[j];
        return s;
    }
};

struct Conditional {
    double base_shrink = .1, shrink = .5;
    FGLS base;
    cv::Mat co;  // 6 x 12
    ScalarGLS head;
    // X: P12 특징, Xb: raw_shift 특징
    void fit(const std::vector<Row>& X, const std::vector<Row>& Xb, const std::vector<double>& y, const std::vector<double>& t, const Meta& md) {
        const int n = (int)X.size();
        auto tests = ytm::sgkf_test(md.strat, md.group, 3, 20261023);
        std::vector<double> py(n), pt(n);
        for (auto& te : tests) {
            std::set<int> ts(te.begin(), te.end());
            std::vector<Row> Xt;
            std::vector<double> yt, tt;
            for (int i = 0; i < n; ++i) if (!ts.count(i)) { Xt.push_back(Xb[i]); yt.push_back(y[i]); tt.push_back(t[i]); }
            FGLS m;
            m.fit(Xt, yt, tt, base_shrink);
            for (int i : te) m.predict(Xb[i], py[i], pt[i]);
        }
        base.fit(Xb, y, t, base_shrink);
        cv::Mat A(n, 6, CV_64F);
        for (int i = 0; i < n; ++i) {
            double r[6] = {1, y[i], t[i], y[i] * y[i], y[i] * t[i], t[i] * t[i]};
            for (int j = 0; j < 6; ++j) A.at<double>(i, j) = r[j];
        }
        co = lstsq(A, rows_to_mat(X), 1e-10);
        std::vector<Row> xx(n);
        for (int i = 0; i < n; ++i) xx[i] = transform(X[i], pt[i]);
        head.shrink = shrink;
        head.fit(xx, y);
    }
    Row transform(const Row& x, double t) const {  // remove_quadratic
        Row z(x);
        for (size_t j = 0; j < z.size(); ++j) z[j] = x[j] - t * co.at<double>(2, (int)j) - t * t * co.at<double>(5, (int)j);
        return z;
    }
    void predict(const Row& x, const Row& xb, double& py, double& pt) const {
        base.predict(xb, py, pt);
        py = head.predict(transform(x, pt));
    }
};

// ---- PairResidualNet (torch float32 학습을 double 로 옮김. 초기 가중치는 원래 시드의 torch 값) ----
struct PairNet {
    int comps = 64;
    double pair_w = 1;
    FGLS base;
    Row mean, sd;
    cv::Mat pmean, comp, ev;  // PCA
    Row zscale;
    cv::Mat W1, b1, W2, b2;
    Row pca_z(const Row& x) const {
        const int p = (int)x.size(), k = comp.rows;
        Row z(k, 0.0);
        for (int c = 0; c < k; ++c) {
            double s = 0;
            for (int j = 0; j < p; ++j) s += ((x[j] - mean[j]) / sd[j] - pmean.at<double>(j)) * comp.at<double>(c, j);
            z[c] = s / std::sqrt(ev.at<double>(c));
        }
        return z;
    }
    void fit(const std::vector<Row>& X, const std::vector<double>& y, const std::vector<double>& t, const std::vector<int>& dates) {
        const int n = (int)X.size(), p = (int)X[0].size();
        base.fit(X, y, t, .01);
        std::vector<double> by(n), bt(n);
        for (int i = 0; i < n; ++i) base.predict(X[i], by[i], bt[i]);
        mean.assign(p, 0); sd.assign(p, 0);
        for (auto& r : X) for (int j = 0; j < p; ++j) mean[j] += r[j];
        for (auto& v : mean) v /= n;
        for (auto& r : X) for (int j = 0; j < p; ++j) sd[j] += (r[j] - mean[j]) * (r[j] - mean[j]);
        for (auto& v : sd) v = std::max(std::sqrt(v / n), 1e-6);
        cv::Mat S(n, p, CV_64F);
        for (int i = 0; i < n; ++i) for (int j = 0; j < p; ++j) S.at<double>(i, j) = (X[i][j] - mean[j]) / sd[j];
        cv::reduce(S, pmean, 0, cv::REDUCE_AVG);
        pmean = pmean.t();
        cv::Mat C(n, p, CV_64F);
        for (int i = 0; i < n; ++i) for (int j = 0; j < p; ++j) C.at<double>(i, j) = S.at<double>(i, j) - pmean.at<double>(j);
        // PCA: C C^T 고유분해 -> 주성분
        const int k = std::min(comps, n - 2);
        cv::Mat K = gram(C), evals, evecs;
        cv::eigen(K, evals, evecs);
        comp = cv::Mat(k, p, CV_64F);
        ev = cv::Mat(k, 1, CV_64F);
        for (int c = 0; c < k; ++c) {
            double lam = evals.at<double>(c);
            cv::Mat v = evecs.row(c) * C / std::sqrt(lam);
            // sklearn svd_flip(u_based_decision=False): 각 주성분에서 절댓값 최대 성분을 양수로
            cv::Point mx;
            cv::minMaxLoc(cv::abs(v), nullptr, nullptr, nullptr, &mx);
            if (v.at<double>(mx) < 0) v = -v;
            v.copyTo(comp.row(c));
            ev.at<double>(c) = lam / (n - 1);
        }
        const int din = k + 2;
        std::vector<Row> Z(n);
        for (int i = 0; i < n; ++i) { Z[i] = pca_z(X[i]); Z[i].push_back(by[i]); Z[i].push_back(bt[i]); }
        zscale.assign(din, 0);
        for (int j = 0; j < din; ++j) {
            double m = 0, s = 0;
            for (auto& z : Z) m += z[j];
            m /= n;
            for (auto& z : Z) s += (z[j] - m) * (z[j] - m);
            zscale[j] = std::max(std::sqrt(s / n), 1e-5);
        }
        cv::Mat Zm(n, din, CV_64F);
        for (int i = 0; i < n; ++i) for (int j = 0; j < din; ++j) Zm.at<double>(i, j) = Z[i][j] / zscale[j];
        // 초기 가중치: torch.manual_seed(20261022) 의 nn.Linear(66,32) 값 (nprng.hpp)
        if (din != 66) throw std::runtime_error("PairNet: 입력 차원 66 이 아님");
        W1 = cv::Mat(32, din, CV_64F); b1 = cv::Mat(1, 32, CV_64F);
        for (int i = 0; i < 32; ++i) for (int j = 0; j < din; ++j) W1.at<double>(i, j) = nprng::kW1[i][j];
        for (int i = 0; i < 32; ++i) b1.at<double>(i) = nprng::kB1[i];
        W2 = cv::Mat::zeros(2, 32, CV_64F); b2 = cv::Mat::zeros(1, 2, CV_64F);
        cv::Mat Ly = graph_operator(y, t, dates, "sameT"), Lt = graph_operator(y, t, dates, "sameY");
        cv::Mat base_m(n, 2, CV_64F), target(n, 2, CV_64F), wt(n, 1, CV_64F);
        for (int i = 0; i < n; ++i) {
            base_m.at<double>(i, 0) = by[i]; base_m.at<double>(i, 1) = bt[i];
            target.at<double>(i, 0) = y[i]; target.at<double>(i, 1) = t[i];
            wt.at<double>(i) = dates[i] == 20261020 ? 3. : 1.;
        }
        std::vector<cv::Mat*> P{&W1, &b1, &W2, &b2};
        std::vector<cv::Mat> m1, m2;
        for (auto* q : P) { m1.push_back(cv::Mat::zeros(q->size(), CV_64F)); m2.push_back(cv::Mat::zeros(q->size(), CV_64F)); }
        const double lr = .002, wd = .05, be1 = .9, be2 = .999, eps = 1e-8;
        for (int ep = 1; ep <= 800; ++ep) {
            cv::Mat pre = Zm * W1.t() + cv::repeat(b1, n, 1), H;
            H = pre.clone();
            for (int i = 0; i < H.rows; ++i) for (int j = 0; j < H.cols; ++j) H.at<double>(i, j) = std::tanh(pre.at<double>(i, j));
            cv::Mat outm = H * W2.t() + cv::repeat(b2, n, 1);
            cv::Mat delta = .2 * outm, res = base_m + delta - target;
            cv::Mat gres(n, 2, CV_64F);
            cv::Mat r0 = res.col(0).clone(), r1 = res.col(1).clone();
            cv::Mat L0 = Ly * r0, L1 = Lt * r1;
            for (int i = 0; i < n; ++i)
                for (int c = 0; c < 2; ++c) {
                    double g = 2 * wt.at<double>(i) * res.at<double>(i, c) / (2.0 * n);
                    g += pair_w * 2 * (c == 0 ? L0.at<double>(i) : L1.at<double>(i)) / (2.0 * n);
                    g += .05 * 2 * delta.at<double>(i, c) / (2.0 * n);
                    gres.at<double>(i, c) = .2 * g;
                }
            cv::Mat gW2 = gres.t() * H, gb2;
            cv::reduce(gres, gb2, 0, cv::REDUCE_SUM);
            cv::Mat gH = gres * W2, gpre(n, 32, CV_64F);
            for (int i = 0; i < n; ++i) for (int j = 0; j < 32; ++j) gpre.at<double>(i, j) = gH.at<double>(i, j) * (1 - H.at<double>(i, j) * H.at<double>(i, j));
            cv::Mat gW1 = gpre.t() * Zm, gb1;
            cv::reduce(gpre, gb1, 0, cv::REDUCE_SUM);
            std::vector<cv::Mat> G{gW1, gb1, gW2, gb2};
            double bc1 = 1 - std::pow(be1, ep), bc2 = 1 - std::pow(be2, ep);
            for (size_t q = 0; q < P.size(); ++q) {
                cv::Mat& w = *P[q];
                w *= (1 - lr * wd);
                m1[q] = be1 * m1[q] + (1 - be1) * G[q];
                m2[q] = be2 * m2[q] + (1 - be2) * G[q].mul(G[q]);
                cv::Mat den;
                cv::sqrt(m2[q], den);
                den = den / std::sqrt(bc2) + eps;
                w -= (lr / bc1) * m1[q] / den;
            }
        }
    }
    void predict(const Row& x, double& py, double& pt) const {
        double by, bt;
        base.predict(x, by, bt);
        Row z = pca_z(x);
        z.push_back(by); z.push_back(bt);
        for (size_t j = 0; j < z.size(); ++j) z[j] /= zscale[j];
        double o[2] = {b2.at<double>(0), b2.at<double>(1)};
        for (int h = 0; h < 32; ++h) {
            double s = b1.at<double>(h);
            for (size_t j = 0; j < z.size(); ++j) s += W1.at<double>(h, (int)j) * z[j];
            s = std::tanh(s);
            o[0] += W2.at<double>(0, h) * s;
            o[1] += W2.at<double>(1, h) * s;
        }
        py = by + .2 * o[0];
        pt = bt + .2 * o[1];
    }
};

}  // namespace sep
