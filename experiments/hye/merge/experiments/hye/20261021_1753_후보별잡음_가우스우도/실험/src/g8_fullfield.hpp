// 630점 이동 전체(1260차원) 모델: 축별 [v, v^2] + 시행 절편, 잔차 표준화 후 SVD 축소 백색화.
// 원본: codex_experiments_20261014/fullfield_4fold_20261021/run.py fit (hybridfield 도 같은 fit 을 쓴다)
#pragma once
#include "g8_model.hpp"

namespace g8 {

struct FullModel {
    Branches br;  // 백색화된 b, s, c
    std::vector<double> sd;
    cv::Mat V;    // r x d
    std::vector<double> adjust;
    double alpha = 1;
    // whiten(x) = z/sqrt(alpha) + ((z V^T) * adjust) V, z = x/std
    cv::Mat whiten(const cv::Mat& X) const {
        cv::Mat z = X.clone();
        for (int j = 0; j < z.cols; ++j) z.col(j) /= sd[j];
        cv::Mat p = z * V.t();
        for (int j = 0; j < p.cols; ++j) p.col(j) *= adjust[j];
        return z / std::sqrt(alpha) + p * V;
    }
};
inline FullModel full_fit(const Manifest& m, const cv::Mat& X, const std::vector<int>& tr, bool quad, double alpha) {
    FullModel F;
    F.alpha = alpha;
    std::vector<cv::Mat> res;
    for (char a : {'Y', 'T'}) {
        std::vector<int> ii;
        for (int i : tr) if (m.it[i].axis == a) ii.push_back(i);
        std::vector<std::string> ses;
        for (int i : ii) ses.push_back(m.it[i].session);
        std::sort(ses.begin(), ses.end());
        ses.erase(std::unique(ses.begin(), ses.end()), ses.end());
        const int nb = quad ? 2 : 1;
        cv::Mat A = cv::Mat::zeros((int)ii.size(), nb + (int)ses.size(), CV_64F);
        for (int r = 0; r < (int)ii.size(); ++r) {
            double v = m.it[ii[r]].value;
            A.at<double>(r, 0) = v;
            if (quad) A.at<double>(r, 1) = v * v;
            A.at<double>(r, nb + (int)(std::find(ses.begin(), ses.end(), m.it[ii[r]].session) - ses.begin())) = 1;
        }
        cv::Mat Z = rows(X, ii), co = lstsq(A, Z);
        res.push_back(Z - A * co);
        for (int j = 0; j < (int)ses.size(); ++j)
            F.br.push_back({ses[j], a, co.row(nb + j).clone(), co.row(0).clone(), quad ? co.row(1).clone() : cv::Mat::zeros(1, X.cols, CV_64F)});
    }
    cv::Mat R;
    cv::vconcat(res, R);
    F.sd.resize(R.cols);
    for (int j = 0; j < R.cols; ++j) {
        double s = 0;
        for (int i = 0; i < R.rows; ++i) s += R.at<double>(i, j) * R.at<double>(i, j);
        F.sd[j] = std::sqrt(std::max(s / R.rows, 1e-10));
    }
    for (int j = 0; j < R.cols; ++j) R.col(j) /= F.sd[j] * std::sqrt((double)R.rows);
    // R = U S V^T. R R^T 의 고유분해로 같은 S, V 를 얻는다 (n << 1260)
    cv::Mat G = R * R.t(), ev, U;
    cv::eigen(G, ev, U);  // 내림차순, 행 = 고유벡터
    const int n = R.rows;
    std::vector<double> s(n);
    for (int i = 0; i < n; ++i) s[i] = std::sqrt(std::max(ev.at<double>(i), 0.0));
    double smax = s[0];
    F.V = cv::Mat::zeros(n, R.cols, CV_64F);
    for (int i = 0; i < n; ++i) {
        if (s[i] > smax * 1e-9) F.V.row(i) = U.row(i) * R / s[i];
        F.adjust.push_back(1 / std::sqrt(alpha + (1 - alpha) * s[i] * s[i]) - 1 / std::sqrt(alpha));
    }
    for (auto& b : F.br) { b.b = F.whiten(b.b); b.s = F.whiten(b.s); b.c = F.whiten(b.c); }
    return F;
}
inline std::vector<Pred> full_predict(const Manifest& m, const cv::Mat& X, const std::vector<int>& te, const FullModel& F) {
    cv::Mat Xw = F.whiten(rows(X, te));
    std::vector<std::string> keys;
    std::vector<char> axes;
    std::vector<std::vector<double>> V, C;
    for (auto& b : F.br) {
        std::vector<double> v, c;
        branch_solve(Xw - cv::repeat(b.b, Xw.rows, 1), b.s, b.c, v, c);
        for (size_t t = 0; t < c.size(); ++t) if (!sign_ok(b.key, v[t])) c[t] = std::numeric_limits<double>::infinity();
        keys.push_back(b.key);
        axes.push_back(b.axis);
        V.push_back(v);
        C.push_back(c);
    }
    return choose(m, te, keys, axes, V, C);
}

}  // namespace g8
