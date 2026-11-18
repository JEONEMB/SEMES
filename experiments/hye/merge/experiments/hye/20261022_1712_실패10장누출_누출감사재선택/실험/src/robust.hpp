// 실패 10장 감사 후 추가 후보 (원본: joint_yt_erroraudit_20261022/robust/robust_model.py(RobustBaggedForward), phase_model.py(PhaseNuisance))
#pragma once
#include <opencv2/core.hpp>
#include <algorithm>
#include <cmath>
#include <map>
#include <random>
#include <string>
#include <tuple>
#include <vector>
#include "separation.hpp"

namespace rob {

using sep::Row;

inline double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// gls_decoder(slope 2xp, residual nxp, weights, shrink) -> beta p x 2
inline cv::Mat gls_decoder(const cv::Mat& slope, const cv::Mat& residual, std::vector<double> w, double shrink) {
    const int n = residual.rows, p = residual.cols;
    double s = 0;
    for (double v : w) s += v;
    for (auto& v : w) v /= s;
    cv::Mat R(n, p, CV_64F);
    for (int i = 0; i < n; ++i) {
        double sw = std::sqrt(w[i]);
        for (int j = 0; j < p; ++j) R.at<double>(i, j) = residual.at<double>(i, j) * sw;
    }
    std::vector<double> sd(p);
    for (int j = 0; j < p; ++j) {
        double q = 0;
        for (int i = 0; i < n; ++i) q += R.at<double>(i, j) * R.at<double>(i, j);
        sd[j] = std::max(std::sqrt(q), 1e-6);
    }
    cv::Mat Z(n, p, CV_64F), Vt(2, p, CV_64F);
    for (int i = 0; i < n; ++i) for (int j = 0; j < p; ++j) Z.at<double>(i, j) = R.at<double>(i, j) / sd[j];
    for (int a = 0; a < 2; ++a) for (int j = 0; j < p; ++j) Vt.at<double>(a, j) = slope.at<double>(a, j) / sd[j];
    cv::Mat Q = sep::shrunk_solve(Z, Vt.t(), shrink, 1 - shrink), Mi;
    cv::invert(Vt * Q, Mi, cv::DECOMP_SVD);
    cv::Mat beta = Q * Mi;
    for (int j = 0; j < p; ++j) beta.row(j) /= sd[j];
    return beta;
}

struct RobustCfg {
    std::string fit_mode;  // new_condition_means / all_condition_means
    double shrink = .1;
    double huber = 0;      // 0 = None
    int bags = 1;
    std::string str() const {
        return std::string(fit_mode == "new_condition_means" ? "new_means" : "all_means") + " s" + hye::fmt(shrink, 2) + (huber > 0 ? " huber1.5" : "") +
               (bags > 1 ? " bags12" : "");
    }
};

// RobustBaggedForward. bags>1 의 지수분포 가중은 원래 시드 numpy default_rng(20261022).exponential 난수열(nprng.hpp)을 그대로 쓴다.
struct RobustBagged {
    RobustCfg c;
    cv::Mat beta;      // p x 2
    double bias[2]{};
    void fit(const std::vector<Row>& X, const std::vector<double>& y, const std::vector<double>& t, const std::vector<int>& dates) {
        const int n = (int)X.size(), p = (int)X[0].size();
        std::vector<std::tuple<int, double, double>> keys(n);
        for (int i = 0; i < n; ++i) keys[i] = {dates[i], y[i], t[i]};
        auto names = keys;
        std::sort(names.begin(), names.end());
        names.erase(std::unique(names.begin(), names.end()), names.end());
        const int ng = (int)names.size();
        std::vector<int> gi(n), counts(ng, 0);
        for (int i = 0; i < n; ++i) { gi[i] = (int)(std::lower_bound(names.begin(), names.end(), keys[i]) - names.begin()); counts[gi[i]]++; }
        cv::Mat gx = cv::Mat::zeros(ng, p, CV_64F), A(ng, 3, CV_64F), AX(n, 3, CV_64F);
        for (int i = 0; i < n; ++i) for (int j = 0; j < p; ++j) gx.at<double>(gi[i], j) += X[i][j];
        std::vector<int> gd(ng);
        for (int g = 0; g < ng; ++g) {
            gx.row(g) /= (double)counts[g];
            A.at<double>(g, 0) = 1; A.at<double>(g, 1) = std::get<1>(names[g]); A.at<double>(g, 2) = std::get<2>(names[g]);
            gd[g] = std::get<0>(names[g]);
        }
        for (int i = 0; i < n; ++i) { AX.at<double>(i, 0) = 1; AX.at<double>(i, 1) = y[i]; AX.at<double>(i, 2) = t[i]; }
        std::vector<double> selected(ng, 1.0), domain(ng);
        if (c.fit_mode == "new_condition_means")
            for (int g = 0; g < ng; ++g) selected[g] = gd[g] == 20261020;
        for (int g = 0; g < ng; ++g) {
            int cnt = 0;
            for (int h = 0; h < ng; ++h) cnt += gd[h] == gd[g];
            domain[g] = 1.0 / cnt;
        }
        beta = cv::Mat::zeros(p, 2, CV_64F);
        bias[0] = bias[1] = 0;
        for (int bag = 0; bag < c.bags; ++bag) {
            std::vector<double> boot(ng, 1.0);
            if (c.bags > 1)
                for (int g = 0; g < ng; ++g) boot[g] = std::max(nprng::kExp[bag * ng + g], 0.05);
            std::vector<double> base(ng), fw(ng), robust(ng, 1.0);
            for (int g = 0; g < ng; ++g) { base[g] = domain[g] * boot[g]; fw[g] = base[g] * selected[g]; }
            cv::Mat coef;
            int iters = c.huber > 0 ? 5 : 1;
            for (int it = 0; it < iters; ++it) {
                std::vector<double> w(ng);
                cv::Mat Aw(ng, 3, CV_64F), Gw(ng, p, CV_64F);
                for (int g = 0; g < ng; ++g) {
                    w[g] = fw[g] * robust[g];
                    double sw = std::sqrt(w[g]);
                    for (int k = 0; k < 3; ++k) Aw.at<double>(g, k) = A.at<double>(g, k) * sw;
                    for (int j = 0; j < p; ++j) Gw.at<double>(g, j) = gx.at<double>(g, j) * sw;
                }
                coef = sep::lstsq(Aw, Gw, sep::eps_rcond(ng, 3));
                cv::Mat rr = gx - A * coef;
                double wsum = 0;
                for (double v : w) wsum += v;
                std::vector<double> scale(p);
                for (int j = 0; j < p; ++j) {
                    double s = 0;
                    for (int g = 0; g < ng; ++g) s += w[g] * rr.at<double>(g, j) * rr.at<double>(g, j);
                    scale[j] = std::max(std::sqrt(s / wsum), 1e-6);
                }
                std::vector<double> dist(ng), dsel;
                for (int g = 0; g < ng; ++g) {
                    double s = 0;
                    for (int j = 0; j < p; ++j) { double z = rr.at<double>(g, j) / scale[j]; s += z * z; }
                    dist[g] = std::sqrt(s / p);
                    if (selected[g] > 0) dsel.push_back(dist[g]);
                }
                double cutoff = (c.huber > 0 ? c.huber : 1.0) * std::max(median(dsel), 1e-9);
                for (int g = 0; g < ng; ++g) robust[g] = std::max(std::min(1.0, cutoff / std::max(dist[g], 1e-9)), 0.05);
            }
            std::vector<double> rw(n);
            for (int i = 0; i < n; ++i) rw[i] = base[gi[i]] / counts[gi[i]];
            cv::Mat Xm = sep::rows_to_mat(X);
            cv::Mat residual = Xm - AX * coef;
            if (c.huber > 0) {
                double rs = 0;
                for (double v : rw) rs += v;
                std::vector<double> scale(p);
                for (int j = 0; j < p; ++j) {
                    double s = 0;
                    for (int i = 0; i < n; ++i) s += rw[i] * residual.at<double>(i, j) * residual.at<double>(i, j);
                    scale[j] = std::max(std::sqrt(s / rs), 1e-6);
                }
                std::vector<double> dist(n);
                for (int i = 0; i < n; ++i) {
                    double s = 0;
                    for (int j = 0; j < p; ++j) { double z = residual.at<double>(i, j) / scale[j]; s += z * z; }
                    dist[i] = std::sqrt(s / p);
                }
                double cutoff = c.huber * std::max(median(dist), 1e-9);
                for (int i = 0; i < n; ++i) rw[i] *= std::max(std::min(1.0, cutoff / std::max(dist[i], 1e-9)), 0.05);
            }
            cv::Mat b = gls_decoder(coef.rowRange(1, 3), residual, rw, c.shrink);
            beta += b;
            cv::Mat bb = -coef.row(0) * b;
            bias[0] += bb.at<double>(0);
            bias[1] += bb.at<double>(1);
        }
        beta /= c.bags;
        bias[0] /= c.bags;
        bias[1] /= c.bags;
    }
    void predict(const Row& x, double& py, double& pt) const {
        py = bias[0]; pt = bias[1];
        for (int j = 0; j < beta.rows; ++j) { py += x[j] * beta.at<double>(j, 0); pt += x[j] * beta.at<double>(j, 1); }
    }
};

// PhaseNuisance: 같은 조건 반복 사진에서 평균 이동의 부화소 위상(sin, cos)에 따른 모양 변화를 빼고 RepeatGLS(.01, mix 0, cond_means)
struct PhaseCfg {
    bool enabled = false;
    std::string phase = "x";
    double alpha = 1;
    std::string str() const { return enabled ? "phase " + phase + " a" + hye::fmt(alpha, 2) : "phase off"; }
};
struct Phase {
    PhaseCfg c;
    cv::Mat coef;  // k x 1260
    sep::RepeatGLS model;
    std::vector<double> ph(const Row& D) const {
        double mx = 0, my = 0;
        for (int k = 0; k < 630; ++k) { mx += D[2 * k]; my += D[2 * k + 1]; }
        mx /= 630; my /= 630;
        const double tp = 2 * M_PI;
        if (c.phase == "x") return {std::sin(tp * mx), std::cos(tp * mx) - 1};
        return {std::sin(tp * mx), std::sin(tp * my), std::cos(tp * mx) - 1, std::cos(tp * my) - 1};
    }
    Row transform(const Row& D) const {
        if (!c.enabled) return D;
        auto p = ph(D);
        Row o(D);
        for (size_t a = 0; a < p.size(); ++a)
            for (int j = 0; j < 1260; ++j) o[j] -= p[a] * coef.at<double>((int)a, j);
        return o;
    }
    void fit(const std::vector<Row>& D, const std::vector<double>& y, const std::vector<double>& t, const sep::Meta& md) {
        const int n = (int)D.size();
        if (c.enabled) {
            std::vector<std::string> names;
            for (int i = 0; i < n; ++i) if (md.date[i] == 20261020) names.push_back(md.group[i]);
            std::sort(names.begin(), names.end());
            names.erase(std::unique(names.begin(), names.end()), names.end());
            std::vector<std::vector<double>> pp;
            std::vector<Row> ss;
            for (auto& g : names) {
                std::vector<int> ix;
                for (int i = 0; i < n; ++i) if (md.date[i] == 20261020 && md.group[i] == g) ix.push_back(i);
                if (ix.size() < 2) continue;
                std::vector<std::vector<double>> P;
                std::vector<Row> S;
                for (int i : ix) {
                    P.push_back(ph(D[i]));
                    double mx = 0, my = 0;
                    for (int k = 0; k < 630; ++k) { mx += D[i][2 * k]; my += D[i][2 * k + 1]; }
                    mx /= 630; my /= 630;
                    Row s(1260);
                    for (int k = 0; k < 630; ++k) { s[2 * k] = D[i][2 * k] - mx; s[2 * k + 1] = D[i][2 * k + 1] - my; }
                    S.push_back(s);
                }
                const size_t kk = P[0].size();
                std::vector<double> pm(kk, 0);
                Row sm(1260, 0);
                for (size_t q = 0; q < ix.size(); ++q) {
                    for (size_t a = 0; a < kk; ++a) pm[a] += P[q][a] / ix.size();
                    for (int j = 0; j < 1260; ++j) sm[j] += S[q][j] / ix.size();
                }
                for (size_t q = 0; q < ix.size(); ++q) {
                    std::vector<double> a(kk);
                    for (size_t r = 0; r < kk; ++r) a[r] = P[q][r] - pm[r];
                    Row b(1260);
                    for (int j = 0; j < 1260; ++j) b[j] = S[q][j] - sm[j];
                    pp.push_back(a);
                    ss.push_back(b);
                }
            }
            const int m = std::max((int)pp.size(), 1), kk = (int)pp[0].size();
            cv::Mat PP(kk, kk, CV_64F, cv::Scalar(0)), PS(kk, 1260, CV_64F, cv::Scalar(0));
            for (size_t q = 0; q < pp.size(); ++q)
                for (int a = 0; a < kk; ++a) {
                    for (int b = 0; b < kk; ++b) PP.at<double>(a, b) += pp[q][a] * pp[q][b];
                    for (int j = 0; j < 1260; ++j) PS.at<double>(a, j) += pp[q][a] * ss[q][j];
                }
            PP /= m;
            PS /= m;
            for (int a = 0; a < kk; ++a) PP.at<double>(a, a) += c.alpha;
            cv::solve(PP, PS, coef, cv::DECOMP_LU);
            for (int a = 0; a < kk; ++a) {  // 점 평균 이동은 그대로 둔다
                double mx = 0, my = 0;
                for (int k = 0; k < 630; ++k) { mx += coef.at<double>(a, 2 * k); my += coef.at<double>(a, 2 * k + 1); }
                mx /= 630; my /= 630;
                for (int k = 0; k < 630; ++k) { coef.at<double>(a, 2 * k) -= mx; coef.at<double>(a, 2 * k + 1) -= my; }
            }
        }
        std::vector<Row> X(n);
        for (int i = 0; i < n; ++i) X[i] = sep::raw_shift(transform(D[i]));
        model.shrink = .01; model.mix = 0; model.mode = "new_condition_means";
        model.fit(X, y, t, md);
    }
    void predict(const Row& D, double& py, double& pt) const { model.predict(sep::raw_shift(transform(D)), py, pt); }
};

}  // namespace rob
