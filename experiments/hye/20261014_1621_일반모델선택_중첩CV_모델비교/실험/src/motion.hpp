// codex modeling/experiment.py MotionModel 의 C++ 이식 (unified.py 후보에 쓰는 부분: p12, qr18, nuisance, shrink, grid)
#pragma once
#include <opencv2/core.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <vector>

namespace mm {

struct Cand {
    std::string meas, id, kind = "p12", nuisance = "none";
    int degree = 2;
    double shrink = 0.1;
    bool grid = false;
};

// 한 측정 방식의 자료: 사진별 점 이동 (P x 2), 점 좌표·영역
struct Data {
    std::vector<cv::Mat> D;  // n 개, P x 2 CV_64F (center 이동을 뺀 값)
    std::vector<double> px, py;
    std::vector<int> region;
};

inline cv::Mat pinv(const cv::Mat& A) {
    cv::Mat P;
    cv::invert(A, P, cv::DECOMP_SVD);
    return P;
}
inline cv::Mat lstsq(const cv::Mat& A, const cv::Mat& B) {
    cv::Mat X;
    cv::solve(A, B, X, cv::DECOMP_SVD);
    return X;
}
// x=(X-1900)/400, y=(Y-1070)/120, [1, x, y, x^2, xy, y^2] (degree 2) / [1, x, y] (degree 1)
inline cv::Mat spatial_basis(const std::vector<double>& X, const std::vector<double>& Y, const std::vector<int>& ix, int degree) {
    cv::Mat B((int)ix.size(), degree == 1 ? 3 : 6, CV_64F);
    for (int r = 0; r < B.rows; ++r) {
        double x = (X[ix[r]] - 1900) / 400, y = (Y[ix[r]] - 1070) / 120;
        double v[6] = {1, x, y, x * x, x * y, y * y};
        for (int c = 0; c < B.cols; ++c) B.at<double>(r, c) = v[c];
    }
    return B;
}

struct Pred {
    char axis;
    double value, costY, costT, hypY, hypT;
};

struct Model {
    Cand c;
    std::vector<int> all;
    cv::Mat N, NtNinv;           // nuisance (2P x m)
    cv::Mat Binv;                // p12
    std::vector<std::vector<int>> reg_idx;
    std::vector<cv::Mat> reg_pinv;  // qr18
    cv::Mat coY, coT, W, cowY, cowT;

    cv::Mat transform(const cv::Mat& D0) const {
        cv::Mat D = D0.clone();
        const int P = D.rows;
        if (!N.empty()) {
            cv::Mat z = D.reshape(1, 2 * P);  // 점 우선 (x0,y0,x1,y1,...)
            cv::Mat coef = NtNinv * (N.t() * z);
            z = z - N * coef;
            D = z.reshape(1, P).clone();
        }
        if (c.kind == "p12") {
            cv::Mat co = Binv * D;  // 6 x 2
            return co.reshape(1, 1).clone();
        }
        cv::Mat out(1, 0, CV_64F);
        for (size_t r = 0; r < reg_idx.size(); ++r) {
            cv::Mat d((int)reg_idx[r].size(), 2, CV_64F);
            for (int i = 0; i < d.rows; ++i) D.row(reg_idx[r][i]).copyTo(d.row(i));
            cv::Mat co = reg_pinv[r] * d;
            cv::hconcat(out, co.reshape(1, 1), out);
        }
        return out;
    }
    cv::Mat transform_all(const Data& d, const std::vector<int>& rows) const {
        cv::Mat X;
        for (int i : rows) X.push_back(transform(d.D[i]));
        return X;
    }
    static cv::Mat design(const std::vector<double>& v, int degree) {
        cv::Mat A((int)v.size(), degree + 1, CV_64F);
        for (int i = 0; i < A.rows; ++i)
            for (int k = 0; k <= degree; ++k) A.at<double>(i, k) = std::pow(v[i], k);
        return A;
    }

    void fit(const Data& d, const std::vector<int>& tr, const std::vector<char>& axes, const std::vector<double>& values) {
        const int P = (int)d.px.size();
        all.resize(P);
        for (int i = 0; i < P; ++i) all[i] = i;
        // nuisance: dy = 모든 점 같은 y 이동, rot = 점 좌표 회전
        std::vector<cv::Mat> cols;
        if (c.nuisance == "dy" || c.nuisance == "both") {
            cv::Mat v(2 * P, 1, CV_64F);
            for (int i = 0; i < P; ++i) { v.at<double>(2 * i) = 0; v.at<double>(2 * i + 1) = 1; }
            cols.push_back(v);
        }
        if (c.nuisance == "rot" || c.nuisance == "both") {
            cv::Mat v(2 * P, 1, CV_64F);
            for (int i = 0; i < P; ++i) { v.at<double>(2 * i) = -(d.py[i] - 1070) / 400; v.at<double>(2 * i + 1) = (d.px[i] - 1900) / 400; }
            cols.push_back(v);
        }
        if (!cols.empty()) {
            cv::hconcat(cols, N);
            NtNinv = (N.t() * N).inv(cv::DECOMP_SVD);
        }
        if (c.kind == "p12") Binv = pinv(spatial_basis(d.px, d.py, all, 2));
        else {
            for (int r = 0; r < 3; ++r) {
                std::vector<int> ix;
                for (int i = 0; i < P; ++i) if (d.region[i] == r) ix.push_back(i);
                if (ix.empty()) continue;
                reg_idx.push_back(ix);
                reg_pinv.push_back(pinv(spatial_basis(d.px, d.py, ix, 1)));
            }
        }
        cv::Mat X = transform_all(d, tr);
        // fit_maps: 축마다 [1, v, v^2] 최소제곱
        cv::Mat R;
        for (char a : {'Y', 'T'}) {
            std::vector<double> v;
            cv::Mat Xa;
            for (size_t k = 0; k < tr.size(); ++k)
                if (axes[tr[k]] == a) { v.push_back(values[tr[k]]); Xa.push_back(X.row((int)k)); }
            cv::Mat A = design(v, c.degree);
            cv::Mat co = lstsq(A, Xa);
            (a == 'Y' ? coY : coT) = co;
            R.push_back(cv::Mat(Xa - A * co));
        }
        const int p = X.cols, n = R.rows;
        std::vector<double> sd(p);
        for (int j = 0; j < p; ++j) {
            double m = 0, s = 0;
            for (int i = 0; i < n; ++i) m += R.at<double>(i, j);
            m /= n;
            for (int i = 0; i < n; ++i) s += (R.at<double>(i, j) - m) * (R.at<double>(i, j) - m);
            sd[j] = std::max(std::sqrt(s / n), 1e-4);
        }
        W = cv::Mat::zeros(p, p, CV_64F);
        if (c.shrink < 1 && p < 100) {
            cv::Mat Z(n, p, CV_64F);
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < p; ++j) Z.at<double>(i, j) = R.at<double>(i, j) / sd[j];
            cv::Mat mean;
            cv::reduce(Z, mean, 0, cv::REDUCE_AVG);
            for (int i = 0; i < n; ++i) Z.row(i) -= mean;
            cv::Mat cov = Z.t() * Z / (n - 1);
            cov = (1 - c.shrink) * cov + c.shrink * cv::Mat::eye(p, p, CV_64F);
            cv::Mat ev, evec;
            cv::eigen(cov, ev, evec);  // evec 행 = 고유벡터
            for (int i = 0; i < p; ++i)
                for (int j = 0; j < p; ++j) W.at<double>(i, j) = evec.at<double>(j, i) / std::sqrt(std::max(ev.at<double>(j), 1e-5)) / sd[i];
        } else
            for (int j = 0; j < p; ++j) W.at<double>(j, j) = 1 / sd[j];
        cowY = coY * W;
        cowT = coT * W;
    }

    Pred predict_one(const cv::Mat& D) const {
        cv::Mat Xw = transform(D) * W;
        const int p = Xw.cols;
        const double* x = Xw.ptr<double>();
        double t2[2], c2[2];
        for (int ai = 0; ai < 2; ++ai) {
            const cv::Mat& b = ai == 0 ? cowY : cowT;
            const int L = b.rows;
            const double *b0 = b.ptr<double>(0), *b1 = b.ptr<double>(1), *bb2 = L > 2 ? b.ptr<double>(2) : nullptr;
            double num = 0, den = 0;
            for (int j = 0; j < p; ++j) { num += (x[j] - b0[j]) * b1[j]; den += b1[j] * b1[j]; }
            double t = num / den;
            const double lim = ai == 0 ? 3 : 1.5;
            if (L > 2)
                for (int it = 0; it < 12; ++it) {
                    double sn = 0, sd = 0;
                    for (int j = 0; j < p; ++j) {
                        double pr = b0[j] + t * b1[j] + t * t * bb2[j], der = b1[j] + 2 * t * bb2[j];
                        sn += (x[j] - pr) * der;
                        sd += der * der;
                    }
                    t = std::min(std::max(t + sn / std::max(sd, 1e-12), -lim), lim);
                }
            auto cost_at = [&](double tt) {
                double s = 0;
                for (int j = 0; j < p; ++j) {
                    double pr = b0[j] + tt * b1[j] + (L > 2 ? tt * tt * bb2[j] : 0);
                    s += (x[j] - pr) * (x[j] - pr);
                }
                return s;
            };
            double cost = cost_at(t);
            if (c.grid) {
                double step = ai == 0 ? 0.05 : 0.01, base = std::nearbyint(t / step), bc = 0, bt = 0;
                for (int k = -1; k <= 1; ++k) {
                    double tt = step * (base + k), cc = cost_at(tt);
                    if (k == -1 || cc < bc) { bc = cc; bt = tt; }
                }
                t = bt;
                cost = bc;
            }
            t2[ai] = t;
            c2[ai] = cost;
        }
        Pred r;
        r.axis = c2[0] < c2[1] ? 'Y' : 'T';
        r.value = r.axis == 'Y' ? t2[0] : t2[1];
        r.costY = c2[0]; r.costT = c2[1]; r.hypY = t2[0]; r.hypT = t2[1];
        return r;
    }
};

// records / metrics / objective
struct Rec {
    int i;
    char axis, pred_axis;
    double value, pred;
    double eY() const { return std::abs((pred_axis == 'Y' ? pred : 0) - (axis == 'Y' ? value : 0)); }
    double eT() const { return std::abs((pred_axis == 'T' ? pred : 0) - (axis == 'T' ? value : 0)); }
    static int sgn(double v) { return (v > 0) - (v < 0); }
    bool axis_ok() const { return axis == pred_axis; }
    bool joint() const { return axis_ok() && sgn(value) == sgn(pred); }
    bool strict() const { return joint() && eY() <= 0.09 && eT() <= 0.09; }
};
struct Metrics {
    int n = 0, axis_err = 0, joint_err = 0, strict_fail = 0;
    double maxY = 0, maxT = 0;
    int Yn = 0, Tn = 0, Yok = 0, Tok = 0;
    double Ymax = 0, Tmax = 0, Ymae = 0, Tmae = 0;
    double obj3 = 0, obj4 = 0;
};
inline Metrics metrics(const std::vector<Rec>& r) {
    Metrics m;
    m.n = (int)r.size();
    double sY = 0, sT = 0;
    for (auto& x : r) {
        m.axis_err += !x.axis_ok();
        m.joint_err += !x.joint();
        m.strict_fail += !x.strict();
        m.maxY = std::max(m.maxY, x.eY());
        m.maxT = std::max(m.maxT, x.eT());
        m.obj3 = std::max(m.obj3, std::max(x.eY(), x.eT()));
        m.obj4 += x.eY() * x.eY() + x.eT() * x.eT();
        double se = std::abs(x.pred - x.value);
        if (x.axis == 'Y') { m.Yn++; if (x.axis_ok()) { m.Yok++; m.Ymax = std::max(m.Ymax, se); sY += se; } }
        else { m.Tn++; if (x.axis_ok()) { m.Tok++; m.Tmax = std::max(m.Tmax, se); sT += se; } }
    }
    m.obj4 /= std::max(1, m.n);
    m.Ymae = m.Yok ? sY / m.Yok : 0;
    m.Tmae = m.Tok ? sT / m.Tok : 0;
    return m;
}
// 사전 순 목표: 축·부호 오류, 허용오차 실패, 최대 벡터 오차, 평균 제곱 오차
inline bool better(const Metrics& a, const Metrics& b) {
    if (a.joint_err != b.joint_err) return a.joint_err < b.joint_err;
    if (a.strict_fail != b.strict_fail) return a.strict_fail < b.strict_fail;
    if (a.obj3 != b.obj3) return a.obj3 < b.obj3;
    return a.obj4 < b.obj4;
}

}  // namespace mm
