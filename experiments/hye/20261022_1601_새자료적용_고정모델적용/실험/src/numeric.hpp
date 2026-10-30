// 작은 선형대수·분할 도구 (numpy·sklearn 동작을 수식대로 옮긴 것)
#pragma once
#include <opencv2/core.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace num {

using Mat = cv::Mat;  // CV_64F

inline Mat zeros(int r, int c) { return Mat::zeros(r, c, CV_64F); }
// np.linalg.lstsq(A, B) (A 열 완전 계수 가정, SVD)
inline Mat lstsq(const Mat& A, const Mat& B) {
    Mat X;
    cv::solve(A, B, X, cv::DECOMP_SVD);
    return X;
}
inline Mat rows(const Mat& M, const std::vector<int>& idx) {
    Mat o(0, M.cols, CV_64F);
    for (int i : idx) o.push_back(M.row(i));
    return o;
}
// 하삼각 L, A = L L^T
inline Mat cholesky(const Mat& A) {
    const int n = A.rows;
    Mat L = zeros(n, n);
    for (int j = 0; j < n; ++j) {
        double s = A.at<double>(j, j);
        for (int k = 0; k < j; ++k) s -= L.at<double>(j, k) * L.at<double>(j, k);
        L.at<double>(j, j) = std::sqrt(s);
        for (int i = j + 1; i < n; ++i) {
            double t = A.at<double>(i, j);
            for (int k = 0; k < j; ++k) t -= L.at<double>(i, k) * L.at<double>(j, k);
            L.at<double>(i, j) = t / L.at<double>(j, j);
        }
    }
    return L;
}
inline double cond2(const Mat& A) {  // np.linalg.cond (2-norm)
    Mat w;
    cv::SVD::compute(A, w);
    return w.at<double>(0) / w.at<double>(w.rows - 1);
}

// ---- numpy RandomState(seed) 의 shuffle (MT19937, legacy random_interval) ----
struct NpRandomState {
    std::mt19937 mt;
    explicit NpRandomState(uint32_t seed) : mt(seed) {}
    uint32_t interval(uint32_t max) {
        if (max == 0) return 0;
        uint32_t mask = max;
        mask |= mask >> 1; mask |= mask >> 2; mask |= mask >> 4; mask |= mask >> 8; mask |= mask >> 16;
        uint32_t v;
        while ((v = (uint32_t)mt() & mask) > max) {}
        return v;
    }
    template <class T>
    void shuffle(std::vector<T>& a) {
        for (size_t i = a.size() - 1; i >= 1; --i) {
            size_t j = interval((uint32_t)i);
            std::swap(a[i], a[j]);
        }
    }
};

// sklearn StratifiedKFold(n_splits, shuffle=True, random_state=seed)._make_test_folds
inline std::vector<int> stratified_test_folds(const std::vector<std::string>& y, int n_splits, uint32_t seed) {
    NpRandomState rng(seed);
    // 클래스 = 처음 나타난 순서
    std::vector<std::string> order;
    std::map<std::string, int> code;
    {
        std::map<std::string, int> first;
        for (int i = 0; i < (int)y.size(); ++i) if (!first.count(y[i])) first[y[i]] = i;
        std::vector<std::pair<int, std::string>> v;
        for (auto& [k, i] : first) v.push_back({i, k});
        std::sort(v.begin(), v.end());
        for (size_t c = 0; c < v.size(); ++c) code[v[c].second] = (int)c;
    }
    const int nc = (int)code.size();
    std::vector<int> enc(y.size());
    for (size_t i = 0; i < y.size(); ++i) enc[i] = code[y[i]];
    std::vector<int> sorted = enc;
    std::sort(sorted.begin(), sorted.end());
    std::vector<std::vector<int>> alloc(n_splits, std::vector<int>(nc, 0));
    for (int i = 0; i < n_splits; ++i)
        for (size_t k = i; k < sorted.size(); k += n_splits) alloc[i][sorted[k]]++;
    std::vector<int> test(y.size(), -1);
    for (int k = 0; k < nc; ++k) {
        std::vector<int> folds;
        for (int i = 0; i < n_splits; ++i) for (int r = 0; r < alloc[i][k]; ++r) folds.push_back(i);
        rng.shuffle(folds);
        size_t p = 0;
        for (size_t i = 0; i < y.size(); ++i) if (enc[i] == k) test[i] = folds[p++];
    }
    return test;
}

inline uint64_t fnv1a(const std::string& s) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (unsigned char c : s) { h ^= c; h *= 0x100000001b3ULL; }
    return h;
}

}  // namespace num
