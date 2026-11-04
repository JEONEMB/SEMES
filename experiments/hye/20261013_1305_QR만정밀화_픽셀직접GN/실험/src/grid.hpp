// 특징 조합 알고리즘 C2~C5 (analysis_v3/v1_data/grid.py), 20261012_1524 실험과 같은 구현
#pragma once
#include <opencv2/core.hpp>
#include <cmath>
#include <vector>
namespace grid {
using M = cv::Mat;
inline M lstsq(const M& A, const M& B) { M X; cv::solve(A, B, X, cv::DECOMP_SVD); return X; }
inline M rows_of(const M& F, const std::vector<int>& r) {
    M o((int)r.size(), F.cols, CV_64F);
    for (size_t i = 0; i < r.size(); ++i) F.row(r[i]).copyTo(o.row((int)i));
    return o;
}
inline M cols_of(const M& F, const std::vector<int>& c) {
    M o(F.rows, (int)c.size(), CV_64F);
    for (size_t j = 0; j < c.size(); ++j) F.col(c[j]).copyTo(o.col((int)j));
    return o;
}
inline M ones_col(const M& A) {
    M o;
    cv::hconcat(A, M::ones(A.rows, 1, CV_64F), o);
    return o;
}
inline M std_pop(const M& F) {
    M mu, sd(1, F.cols, CV_64F);
    cv::reduce(F, mu, 0, cv::REDUCE_AVG);
    for (int j = 0; j < F.cols; ++j) {
        double s = 0;
        for (int i = 0; i < F.rows; ++i) s += std::pow(F.at<double>(i, j) - mu.at<double>(j), 2);
        sd.at<double>(j) = std::sqrt(s / F.rows);
    }
    M o;
    cv::vconcat(mu, sd, o);
    return o;
}
inline M standardize(const M& F, const M& ms) {
    M o = F.clone();
    for (int i = 0; i < o.rows; ++i)
        for (int j = 0; j < o.cols; ++j) o.at<double>(i, j) = (o.at<double>(i, j) - ms.at<double>(0, j)) / (ms.at<double>(1, j) + 1e-12);
    return o;
}
inline M ridge_fit(const M& A1, const M& y, double al) {
    M G = A1.t() * A1;
    for (int j = 0; j + 1 < G.cols; ++j) G.at<double>(j, j) += al;
    M W;
    cv::solve(G, A1.t() * y, W, cv::DECOMP_LU);
    return W;
}
inline M predict(int alg, const M& F, const M& X, const std::vector<int>& tr, const std::vector<int>& te, double* alpha = nullptr) {
    M Ftr = rows_of(F, tr), Fte = rows_of(F, te), Xtr = rows_of(X, tr);
    if (alg == 2 || alg == 3) {
        M ms = std_pop(Ftr);
        M A = ones_col(standardize(Ftr, ms)), B = ones_col(standardize(Fte, ms));
        if (alg == 2) {
            if (A.cols > A.rows * 0.8) return M();
            return B * lstsq(A, Xtr);
        }
        double best = 1e300, ba = 0;
        for (double al : {1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0}) {
            double s = 0;
            int c = 0;
            for (int p = 0; p < 2; ++p) {
                std::vector<int> in, ou;
                for (int i = 0; i < A.rows; ++i) (i % 2 == p ? ou : in).push_back(i);
                M e = rows_of(A, ou) * ridge_fit(rows_of(A, in), rows_of(Xtr, in), al) - rows_of(Xtr, ou);
                for (int i = 0; i < e.rows; ++i) { s += std::pow(e.at<double>(i, 0) * 80, 2) + std::pow(e.at<double>(i, 1) * 14.3, 2); c += 2; }
            }
            double sc = std::sqrt(s / c);
            if (sc < best) { best = sc; ba = al; }
        }
        if (alpha) *alpha = ba;
        return B * ridge_fit(A, Xtr, ba);
    }
    if (F.cols < 2) return M();
    M A = ones_col(Xtr), Bc = lstsq(A, Ftr);
    M J = Bc.rowRange(0, 2).t(), b = Bc.row(2), R = Ftr - A * Bc;
    M Y = Fte - cv::repeat(b, Fte.rows, 1);
    if (alg == 4) {
        M ms = std_pop(R);
        M Jn = J.clone(), Yn = Y.clone();
        for (int j = 0; j < J.rows; ++j) { double sd = ms.at<double>(1, j) + 1e-12; Jn.row(j) /= sd; Yn.col(j) /= sd; }
        return lstsq(Jn, Yn.t()).t();
    }
    M mu;
    cv::reduce(R, mu, 0, cv::REDUCE_AVG);
    M Rc = R - cv::repeat(mu, R.rows, 1);
    M S = Rc.t() * Rc / (R.rows - 1);
    double jit = cv::trace(S)[0] / S.rows * 1e-2;
    for (int j = 0; j < S.rows; ++j) S.at<double>(j, j) += jit;
    M Si = S.inv(cv::DECOMP_LU), Mp;
    cv::solve(J.t() * Si * J, J.t() * Si, Mp, cv::DECOMP_LU);
    return Y * Mp.t();
}

}  // namespace grid
