// A17: 날짜 층화 5-fold (각 날짜 80% 학습 / 20% 시험) 로 joint 대리 모델을 다시 평가한다.
//  사용자 요청 2026-11-04. 기준: joint 는 |eT| <= 0.01 (T 1틱), Y 는 참고로 |eY| <= 0.045.
//  fold 배정: 세트마다 같은 (Y,T) 묶음을 (Y,T) 오름차순으로 정렬해 0..k-1 을 돌려 준다 (A0 계층 배정과 같은 규칙, k만 다름).
//  반복 3장은 같은 묶음이라 같은 fold 에 들어간다. center 는 학습에만 쓴다.
#include "cache.hpp"
#include "model.hpp"
#include <iostream>
#include <map>

using namespace a0;

static std::vector<int> stratified_folds(const Cache& c, int k, unsigned seed = 0) {
    std::map<std::string, std::vector<std::pair<std::pair<double, double>, int>>> by_set;
    std::map<int, std::pair<double, double>> gyt;
    for (auto& s : c.shots) if (s.measured && s.fold_mixed >= 0) gyt[s.group] = {s.y, s.t};
    std::map<int, std::string> gset;
    for (auto& s : c.shots) if (s.measured && s.fold_mixed >= 0) gset[s.group] = s.set;
    for (auto& [g, yt] : gyt) by_set[gset[g]].push_back({yt, g});
    std::map<int, int> gfold;
    for (auto& [set, v] : by_set) {
        std::sort(v.begin(), v.end());
        // seed 0: 정렬 순서대로 돌려 배정. seed>0: k개씩 끊은 블록 안에서 fold 순서를 결정적으로 섞는다 (크기 층화는 유지).
        for (size_t b = 0; b < v.size(); b += k) {
            std::vector<int> perm(k);
            for (int j = 0; j < k; ++j) perm[j] = j;
            if (seed) { unsigned x = seed * 2654435761u + unsigned(b) * 40503u; for (int j = k - 1; j > 0; --j) { x = x * 1103515245u + 12345u; std::swap(perm[j], perm[(x >> 16) % (j + 1)]); } }
            for (size_t i = b; i < std::min(v.size(), b + size_t(k)); ++i) gfold[v[i].second] = perm[i - b];
        }
    }
    std::vector<int> f(c.n(), -1);
    for (auto& s : c.shots) if (s.measured && s.fold_mixed >= 0) f[s.idx] = gfold[s.group];
    return f;
}

struct Res { std::string name; Oof o; };

int main() {
    const fs::path results = fs::u8path(EXPERIMENT_DIR) / "results";
    fs::create_directories(results);
    Cache c = load(cache_dir());
    std::vector<std::vector<double>> f66(c.n()), f132(c.n());
    for (auto& s : c.shots) if (s.measured) {
        f66[s.idx] = features(c, s.idx, true);
        // joint 처럼 평균 QR 가로 이동과의 상호작용 항을 더한다 (QR 아핀 상수항 3개의 평균 = 판 가로 이동)
        double h = 0;
        for (int q = 0; q < 3; ++q) h += qr_affine(c, s.idx, q)[0];
        h /= 3.0 * 100.0;
        f132[s.idx] = f66[s.idx];
        for (double v : f66[s.idx]) f132[s.idx].push_back(v * h);
    }
    auto f5 = stratified_folds(c, 5), f4 = stratified_folds(c, 4);
    auto by5 = [&](const Shot& s) { return f5[s.idx]; };
    auto by4 = [&](const Shot& s) { return f4[s.idx]; };
    auto mixed = [&](const Shot& s) { return s.fold_mixed; };
    std::vector<Res> runs = {
        {"기존 혼합 4-fold (A3 재현), 66d OLS", run_oof(c, f66, mixed, 4, 0)},
        {"날짜 층화 4-fold, 66d OLS", run_oof(c, f66, by4, 4, 0)},
        {"날짜 층화 5-fold, 66d OLS", run_oof(c, f66, by5, 5, 0)},
        {"날짜 층화 5-fold, 66d+상호작용 132d ridge 0.01", run_oof(c, f132, by5, 5, 0.01)},
        {"날짜 층화 5-fold, 66d+상호작용 132d ridge 1", run_oof(c, f132, by5, 5, 1.0)},
    };
    Table t; t.head = {"평가", "날짜", "장수", "T 평균", "T 최대", "T p95", "|eT|<=0.01", "Y 평균", "Y 최대", "|eY|<=0.045 (참고)"};
    Table out; out.head = {"평가", "사진", "Y 정답", "T 정답", "Y 예측", "T 예측", "eY", "eT", "eY/eT"};
    for (auto& r : runs) {
        for (std::string d : {"all", "20261014", "20261020", "20261029"}) {
            std::vector<double> et, ey;
            for (size_t i = 0; i < r.o.shot.size(); ++i) {
                auto& s = c.shots[r.o.shot[i]];
                if (r.o.truth[i].y > 1.001) continue;  // T+2.00 범위 밖 제외 (A3 와 같음)
                if (d != "all" && s.set.find(d) == std::string::npos) continue;
                et.push_back(std::abs(r.o.pred[i].y - r.o.truth[i].y));
                ey.push_back(std::abs(r.o.pred[i].x - r.o.truth[i].x));
            }
            if (et.empty()) continue;
            int pt = 0, py = 0; for (double v : et) pt += v <= 0.01 + 1e-12; for (double v : ey) py += v <= 0.045 + 1e-12;
            t.add({r.name, d, std::to_string(et.size()), num(mean(et), 4), num(max_abs(et), 4), num(quantile(et, .95), 4),
                   std::to_string(pt) + "/" + std::to_string(et.size()) + " (" + num(100.0 * pt / et.size(), 1) + "%)",
                   num(mean(ey), 4), num(max_abs(ey), 4), num(100.0 * py / ey.size(), 1) + "%"});
        }
        for (size_t i = 0; i < r.o.shot.size(); ++i) {
            if (r.o.truth[i].y > 1.001) continue;
            double et = r.o.pred[i].y - r.o.truth[i].y, ey = r.o.pred[i].x - r.o.truth[i].x;
            if (std::abs(et) > 0.01) out.add({r.name, c.shots[r.o.shot[i]].name(), num(r.o.truth[i].x, 3), num(r.o.truth[i].y, 3),
                num(r.o.pred[i].x, 4), num(r.o.pred[i].y, 4), num(ey, 4), num(et, 4), num(ey / et, 2)});
        }
    }
    // 반복: 나누는 방식만 바꾼 5-fold 를 seed 1..9 로 더 돌려 흔들림 폭을 본다
    Table rep; rep.head = {"seed", "전체 |eT|<=0.01", "1020 |eT|<=0.01", "전체 T 최대", "1020 T 최대"};
    for (unsigned sd = 0; sd < 10; ++sd) {
        auto fs_ = stratified_folds(c, 5, sd);
        auto o = run_oof(c, f66, [&](const Shot& s) { return fs_[s.idx]; }, 5, 0);
        int na = 0, pa = 0, n20 = 0, p20 = 0; double ma = 0, m20 = 0;
        for (size_t i = 0; i < o.shot.size(); ++i) {
            if (o.truth[i].y > 1.001) continue;
            double e = std::abs(o.pred[i].y - o.truth[i].y);
            ++na; pa += e <= 0.01 + 1e-12; ma = std::max(ma, e);
            if (c.shots[o.shot[i]].set.find("20261020") != std::string::npos) { ++n20; p20 += e <= 0.01 + 1e-12; m20 = std::max(m20, e); }
        }
        rep.add({std::to_string(sd), std::to_string(pa) + "/" + std::to_string(na) + " (" + num(100.0 * pa / na, 1) + "%)",
                 std::to_string(p20) + "/" + std::to_string(n20) + " (" + num(100.0 * p20 / n20, 1) + "%)", num(ma, 4), num(m20, 4)});
    }
    rep.csv(results / "repeat_seeds.csv");
    t.csv(results / "scores.csv");
    out.csv(results / "t_outliers.csv");
    Table fa; fa.head = {"shot", "set", "rel", "Y", "T", "fold5"};
    for (auto& s : c.shots) if (s.measured) fa.add({std::to_string(s.idx), s.set, s.rel, num(s.y, 3), num(s.t, 3), std::to_string(f5[s.idx])});
    fa.csv(results / "folds5.csv");
    std::cout << "done\n";
}
