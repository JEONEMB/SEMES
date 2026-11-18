// v7b 축별 다항 모델 + 전체 공분산 화이트닝 (원본: analysis_v7b/lib.py Model, twostage.py TwoStage)
#pragma once
#include <opencv2/core.hpp>
#include <cmath>
#include <string>
#include <vector>

namespace v7b {

enum Ax { AY = 0, AT = 1, AC = 2 };

inline cv::Mat lstsq(const cv::Mat& A, const cv::Mat& B) {  // numpy lstsq (최소 노름)
    cv::Mat X;
    cv::solve(A, B, X, cv::DECOMP_SVD);
    return X;
}

// 상삼각 W 로 W^T W = inv(C) (numpy cholesky(inv(C)).T 와 같은 이차형식)
inline cv::Mat whiten(const cv::Mat& C) {
    cv::Mat Q = C.inv(cv::DECOMP_SVD);
    int n = Q.rows;
    cv::Mat Lm = cv::Mat::zeros(n, n, CV_64F);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j <= i; ++j) {
            double s = Q.at<double>(i, j);
            for (int k = 0; k < j; ++k) s -= Lm.at<double>(i, k) * Lm.at<double>(j, k);
            Lm.at<double>(i, j) = i == j ? std::sqrt(s) : s / Lm.at<double>(j, j);
        }
    return Lm.t();
}

struct Model {
    int deg = 2;
    bool free_icpt = true;
    double locY = -1, locT = -1;  // 0 근처 학습 범위 (음수 = 전부)
    cv::Mat U;                    // 12 x 2 방해 무늬
    cv::Mat W;                    // 12 x 12
    std::vector<cv::Mat> P[2];    // 축별 [s, c] (각 1 x 12)
    cv::Mat b[2];

    void fit(const std::vector<std::vector<double>>& Fe, const std::vector<double>& lab, const std::vector<int>& ax) {
        std::vector<cv::Mat> res;
        for (int a = 0; a < 2; ++a) {
            std::vector<int> idx;
            for (size_t i = 0; i < Fe.size(); ++i) {
                if (!(ax[i] == a || ax[i] == AC)) continue;
                double lim = a == AY ? locY : locT;
                if (lim >= 0 && !(std::abs(lab[i]) <= lim)) continue;
                idx.push_back((int)i);
            }
            int nc = deg + (free_icpt ? 1 : 0);
            cv::Mat A((int)idx.size(), nc, CV_64F), Y((int)idx.size(), 12, CV_64F);
            for (int r = 0; r < A.rows; ++r) {
                double v = lab[idx[r]];
                for (int k = 0; k < deg; ++k) A.at<double>(r, k) = std::pow(v, k + 1);
                if (free_icpt) A.at<double>(r, nc - 1) = 1;
                for (int j = 0; j < 12; ++j) Y.at<double>(r, j) = Fe[idx[r]][j];
            }
            cv::Mat co = lstsq(A, Y);
            P[a].clear();
            for (int k = 0; k < deg; ++k) P[a].push_back(co.row(k).clone());
            b[a] = free_icpt ? co.row(nc - 1).clone() : cv::Mat::zeros(1, 12, CV_64F);
            cv::Mat R = Y - A * co;
            for (int r = 0; r < A.rows; ++r) if (ax[idx[r]] != AC) res.push_back(R.row(r));
        }
        cv::Mat R;
        cv::vconcat(res, R);
        int dof = std::max(R.rows - 6, 1);
        cv::Mat C = R.t() * R / dof + 1e-10 * cv::Mat::eye(12, 12, CV_64F);
        W = whiten(C);
    }
    cv::Mat fv(int a, double v) const {
        cv::Mat f = b[a].clone();
        for (size_t k = 0; k < P[a].size(); ++k) f += P[a][k] * std::pow(v, (double)(k + 1));
        return f.t();
    }
    cv::Mat dv(int a, double v) const {
        cv::Mat f = cv::Mat::zeros(1, 12, CV_64F);
        for (size_t k = 0; k < P[a].size(); ++k) f += P[a][k] * ((k + 1) * std::pow(v, (double)k));
        return f.t();
    }
    // 반환 (값, 비용)
    std::pair<double, double> solve(const cv::Mat& f /*12x1*/, int a, int iters = 8) const {
        double v = 0;
        for (int it = 0; it < iters; ++it) {
            cv::Mat A;
            cv::hconcat(dv(a, v), U, A);
            cv::Mat p = lstsq(W * A, W * (f - fv(a, v)));
            v += p.at<double>(0);
        }
        cv::Mat z = f - fv(a, v);
        cv::Mat u = lstsq(W * U, W * z);
        z = z - U * u;
        cv::Mat r = W * z;
        return {v, r.dot(r)};
    }
    double llr(const cv::Mat& f) const { return std::log(solve(f, AY).second / solve(f, AT).second); }
};

// 2단계: 축 판정 = 0 근처 1차 모델(절편 0), 값 = 전체 2차 모델
struct TwoStage {
    Model A, V;
    TwoStage(const cv::Mat& U, bool value_free) {
        A.deg = 1; A.free_icpt = false; A.locY = 0.6; A.locT = 0.12; A.U = U;
        V.deg = 2; V.free_icpt = value_free; V.U = U;
    }
    void fit(const std::vector<std::vector<double>>& Fe, const std::vector<double>& lab, const std::vector<int>& ax) {
        A.fit(Fe, lab, ax);
        V.fit(Fe, lab, ax);
    }
    struct Pred { double vY, vT, llr; };
    Pred predict(const std::vector<double>& fe) const {
        cv::Mat f(12, 1, CV_64F);
        for (int j = 0; j < 12; ++j) f.at<double>(j) = fe[j];
        return {V.solve(f, AY).first, V.solve(f, AT).first, A.llr(f)};
    }
};

}  // namespace v7b
