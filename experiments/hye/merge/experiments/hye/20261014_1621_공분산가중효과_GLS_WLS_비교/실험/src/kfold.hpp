// sklearn StratifiedKFold(shuffle=True, random_state=seed) 와 같은 분할.
// numpy RandomState(seed) = MT19937 init_genrand, shuffle = 뒤에서부터 random_interval(마스크 거절 표본).
#pragma once
#include <algorithm>
#include <cstdint>
#include <map>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace kf {

inline uint32_t random_interval(std::mt19937& rng, uint32_t max) {
    if (max == 0) return 0;
    uint32_t mask = max;
    mask |= mask >> 1; mask |= mask >> 2; mask |= mask >> 4; mask |= mask >> 8; mask |= mask >> 16;
    uint32_t v;
    while ((v = (uint32_t)rng() & mask) > max) {}
    return v;
}

// 표본 순서의 라벨 y -> 표본별 시험 분할 번호
inline std::vector<int> test_folds(const std::vector<std::string>& y, int n_splits, uint32_t seed) {
    std::map<std::string, int> first;  // 처음 나온 순서로 부호화
    std::vector<int> enc(y.size());
    for (size_t i = 0; i < y.size(); ++i) {
        auto it = first.find(y[i]);
        if (it == first.end()) it = first.emplace(y[i], (int)first.size()).first;
        enc[i] = it->second;
    }
    // sklearn: 사전순 고유값 -> 첫 등장 위치 순서로 다시 번호. 위의 첫 등장 순 번호와 같다.
    int nc = (int)first.size();
    std::vector<int> order = enc;
    std::sort(order.begin(), order.end());
    std::vector<std::vector<int>> alloc(n_splits, std::vector<int>(nc, 0));
    for (int i = 0; i < n_splits; ++i)
        for (size_t j = i; j < order.size(); j += n_splits) alloc[i][order[j]]++;
    std::mt19937 rng(seed);
    std::vector<int> folds(y.size());
    for (int k = 0; k < nc; ++k) {
        std::vector<int> f;
        for (int i = 0; i < n_splits; ++i) f.insert(f.end(), alloc[i][k], i);
        for (int i = (int)f.size() - 1; i >= 1; --i) std::swap(f[i], f[random_interval(rng, (uint32_t)i)]);
        size_t p = 0;
        for (size_t i = 0; i < y.size(); ++i)
            if (enc[i] == k) folds[i] = f[p++];
    }
    return folds;
}

using Split = std::pair<std::vector<int>, std::vector<int>>;  // 학습, 시험 (원래 번호, 오름차순)

// indices 의 부분집합을 labels[indices] 로 층화해 나눈다 (codex split_folds)
inline std::vector<Split> split(const std::vector<int>& indices, const std::vector<std::string>& labels, int n, uint32_t seed) {
    std::vector<std::string> y;
    for (int i : indices) y.push_back(labels[i]);
    auto tf = test_folds(y, n, seed);
    std::vector<Split> out(n);
    for (int f = 0; f < n; ++f)
        for (size_t i = 0; i < indices.size(); ++i) (tf[i] == f ? out[f].second : out[f].first).push_back(indices[i]);
    return out;
}

}  // namespace kf
