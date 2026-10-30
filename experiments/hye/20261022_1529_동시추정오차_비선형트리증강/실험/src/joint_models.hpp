// 동시 추정 모델 두 개 (원본: joint_yt_improved_20261022/engine.py, fullfield.py)
//   1. ExtraTrees(80그루, min_samples_leaf 1, max_features 1.0) + 'noise' 증강
//      sklearn ExtraTreesRegressor 를 수식대로 옮겼다: 부트스트랩 없음, 마디마다 특징을 무작위 순서로 모두 보고
//      특징마다 [최소, 최대) 균일 임계값 하나, 다출력 MSE 감소가 가장 큰 분할, 불순도 0 또는 1장이면 잎.
//      StandardScaler 는 특징별 선형 변환이라 균일 임계값 분할의 분포를 바꾸지 않아 생략했다.
//      난수열(sklearn 트리 내부 난수, numpy default_rng 정규분포)은 C++ 에서 같게 만들 수 없어 고정 시드 mt19937 을 쓴다.
//   2. 좌표 선형 GLS (gls_full): 1260 좌표에 [1, Y, T] 순방향 직선, 잔차 표준화 공분산을 a I + b R'R 로 줄이고
//      Woodbury 로 역행렬 없이 (Y, T) 선형 추정기를 만든다.
#pragma once
#include <opencv2/core.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

namespace jm {

using Mat = cv::Mat;

inline Mat lstsq(const Mat& A, const Mat& B) { Mat X; cv::solve(A, B, X, cv::DECOMP_SVD); return X; }

// ---- 난수 (플랫폼과 무관하게 같은 값) ----
struct Rng {
    std::mt19937_64 g;
    explicit Rng(uint64_t s) : g(s) {}
    double uniform() { return (g() >> 11) * (1.0 / 9007199254740992.0); }  // [0,1)
    uint32_t below(uint32_t n) { return (uint32_t)(uniform() * n); }
    double normal() {  // Box-Muller
        double u1 = uniform(), u2 = uniform();
        if (u1 < 1e-300) u1 = 1e-300;
        return std::sqrt(-2 * std::log(u1)) * std::cos(2 * M_PI * u2);
    }
};

// ---- ExtraTrees ----
struct Node { int feat = -1; double thr = 0; int left = -1, right = -1; double v[2] = {0, 0}; };
struct Tree {
    std::vector<Node> nodes;
    void fit(const Mat& X, const Mat& y, Rng& rng) {
        std::vector<int> idx(X.rows);
        std::iota(idx.begin(), idx.end(), 0);
        nodes.clear();
        nodes.reserve(2 * X.rows);
        build(X, y, idx, 0, (int)idx.size(), rng);
    }
    int build(const Mat& X, const Mat& y, std::vector<int>& idx, int a, int b, Rng& rng) {
        const int id = (int)nodes.size();
        nodes.push_back(Node());
        const int n = b - a, K = y.cols;
        double s[2] = {0, 0}, ss[2] = {0, 0};
        for (int i = a; i < b; ++i)
            for (int k = 0; k < K; ++k) { double v = y.at<double>(idx[i], k); s[k] += v; ss[k] += v * v; }
        double imp = 0;
        for (int k = 0; k < K; ++k) { nodes[id].v[k] = s[k] / n; imp += ss[k] / n - (s[k] / n) * (s[k] / n); }
        imp /= K;
        if (n < 2 || imp <= 2.220446049250313e-16) return id;
        // 특징 무작위 순서
        std::vector<int> feats(X.cols);
        std::iota(feats.begin(), feats.end(), 0);
        for (int i = X.cols - 1; i > 0; --i) std::swap(feats[i], feats[rng.below(i + 1)]);
        int bf = -1;
        double bthr = 0, bscore = -1e300;
        for (int f : feats) {
            double mn = 1e300, mx = -1e300;
            for (int i = a; i < b; ++i) { double v = X.at<double>(idx[i], f); mn = std::min(mn, v); mx = std::max(mx, v); }
            if (mx <= mn + 1e-7) continue;  // 상수 특징
            double thr = mn + rng.uniform() * (mx - mn);
            if (thr >= mx) thr = mn;
            double sl[2] = {0, 0}, sr[2] = {0, 0};
            int nl = 0;
            for (int i = a; i < b; ++i) {
                bool left = X.at<double>(idx[i], f) <= thr;
                nl += left;
                for (int k = 0; k < K; ++k) (left ? sl : sr)[k] += y.at<double>(idx[i], k);
            }
            const int nr = n - nl;
            if (nl < 1 || nr < 1) continue;
            double score = 0;
            for (int k = 0; k < K; ++k) score += sl[k] * sl[k] / nl + sr[k] * sr[k] / nr;
            if (score > bscore) { bscore = score; bf = f; bthr = thr; }
        }
        if (bf < 0) return id;
        int m = std::partition(idx.begin() + a, idx.begin() + b, [&](int i) { return X.at<double>(i, bf) <= bthr; }) - idx.begin();
        nodes[id].feat = bf;
        nodes[id].thr = bthr;
        int l = build(X, y, idx, a, m, rng);
        int r = build(X, y, idx, m, b, rng);
        nodes[id].left = l;
        nodes[id].right = r;
        return id;
    }
    void predict(const double* x, double* out) const {
        int i = 0;
        while (nodes[i].feat >= 0) i = x[nodes[i].feat] <= nodes[i].thr ? nodes[i].left : nodes[i].right;
        out[0] = nodes[i].v[0];
        out[1] = nodes[i].v[1];
    }
};

struct Forest {
    std::vector<Tree> trees;
    uint64_t seed = 20261022;
    int n_trees = 80;
    bool noise = true;
    // engine.Stabilized(policy='noise'): 잔차([1, Y, T, Y², T²] 최소제곱) 표준편차 x 0.5 정규 잡음 3벌 + 원본
    void fit(const Mat& X, const Mat& y) {
        Rng rng(seed);
        Mat xx = X.clone(), yy = y.clone();
        if (noise) {
            const int n = X.rows;
            Mat A(n, 5, CV_64F);
            for (int i = 0; i < n; ++i) {
                double Y = y.at<double>(i, 0), T = y.at<double>(i, 1);
                double r[5] = {1, Y, T, Y * Y, T * T};
                for (int c = 0; c < 5; ++c) A.at<double>(i, c) = r[c];
            }
            Mat R = X - A * lstsq(A, X);
            std::vector<double> sc(X.cols);
            for (int c = 0; c < X.cols; ++c) {
                double m = cv::mean(R.col(c))[0], s = 0;
                for (int i = 0; i < n; ++i) s += (R.at<double>(i, c) - m) * (R.at<double>(i, c) - m);
                sc[c] = std::sqrt(s / n);
            }
            for (int rep = 0; rep < 3; ++rep) {
                Mat z = X.clone();
                for (int i = 0; i < n; ++i)
                    for (int c = 0; c < X.cols; ++c) z.at<double>(i, c) += rng.normal() * sc[c] * 0.5;
                xx.push_back(z);
                yy.push_back(y);
            }
        }
        trees.assign(n_trees, Tree());
        for (int t = 0; t < n_trees; ++t) {
            Rng tr(seed * 1000003ULL + 7919ULL * (t + 1));
            trees[t].fit(xx, yy, tr);
        }
    }
    Mat predict(const Mat& X) const {
        Mat out = Mat::zeros(X.rows, 2, CV_64F);
        for (int i = 0; i < X.rows; ++i) {
            double acc[2] = {0, 0}, o[2];
            for (auto& t : trees) { t.predict(X.ptr<double>(i), o); acc[0] += o[0]; acc[1] += o[1]; }
            out.at<double>(i, 0) = acc[0] / trees.size();
            out.at<double>(i, 1) = acc[1] / trees.size();
        }
        return out;
    }
};

// ---- 좌표 선형 GLS (fullfield.JointLinear family='gls_full') ----
struct GlsFull {
    double alpha = 0.1;
    Mat origin, beta;  // 1 x p, p x 2
    void fit(const Mat& X, const Mat& y) {
        const int n = X.rows, p = X.cols;
        Mat A(n, 3, CV_64F);
        for (int i = 0; i < n; ++i) { A.at<double>(i, 0) = 1; A.at<double>(i, 1) = y.at<double>(i, 0); A.at<double>(i, 2) = y.at<double>(i, 1); }
        Mat co = lstsq(A, X), r = X - A * co;
        std::vector<double> sd(p);
        for (int j = 0; j < p; ++j) { Mat c = r.col(j); sd[j] = std::max(std::sqrt(c.dot(c) / n), 1e-7); }
        Mat rr = r.clone(), s(p, 2, CV_64F), cs(2, p, CV_64F);
        for (int j = 0; j < p; ++j) {
            rr.col(j) /= sd[j];
            for (int k = 0; k < 2; ++k) { cs.at<double>(k, j) = co.at<double>(k + 1, j) / sd[j]; s.at<double>(j, k) = cs.at<double>(k, j); }
        }
        const double a = alpha, b = (1 - alpha) / n;
        Mat M = a * Mat::eye(n, n, CV_64F) + b * rr * rr.t(), Z;
        cv::solve(M, rr * s, Z, cv::DECOMP_LU);
        Mat Q = (s - b * rr.t() * Z) / a;
        Mat beta0 = Q * (cs * Q).inv();
        beta = beta0.clone();
        for (int j = 0; j < p; ++j) beta.row(j) /= sd[j];
        origin = co.row(0).clone();
    }
    Mat predict(const Mat& X) const {
        Mat D = X.clone();
        for (int i = 0; i < X.rows; ++i) D.row(i) -= origin;
        return D * beta;
    }
};

}  // namespace jm
