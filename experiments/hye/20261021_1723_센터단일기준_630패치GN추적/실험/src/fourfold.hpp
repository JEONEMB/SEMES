// multiref_4fold_20261021/experiment.py 의 평가 틀: 측정 x {center, inferred_session} x {선형, 이차} x 축소 {.001,.05,.2,1}
// 바깥 StratifiedKFold(4, rs 20261021), 안쪽 StratifiedKFold(3, rs 100+fold) 로 설정을 골라 바깥 시험에 쓴다(중첩 선택). 재대입 280도 함께.
#pragma once
#include <tuple>
#include "yt630.hpp"

namespace yt {

struct Cfg { int feat; bool sess; bool quad; double shrink; };
struct FoldEval {
    std::vector<std::string> fnames;
    std::vector<Cfg> cfgs;
    std::vector<std::vector<Row>> oof, resub;   // [cfg]
    std::vector<Row> nested;
    std::vector<int> selected;                  // fold 별 고른 설정
    std::vector<std::vector<std::tuple<int, double, int>>> inner_rank;
    std::vector<std::vector<int>> fold_test;     // 바깥 시험 사진 인덱스
    std::vector<std::vector<Stats>> fold_stats;  // [cfg][fold]
};

inline std::string cfg_name(int k) { char b[8]; std::snprintf(b, sizeof b, "c%02d", k); return b; }
inline std::string cfg_desc(const FoldEval& E, int k) {
    const Cfg& c = E.cfgs[k];
    char sh[16]; std::snprintf(sh, sizeof sh, "%g", c.shrink);
    return E.fnames[c.feat] + " / " + (c.sess ? "session" : "center") + " / " + (c.quad ? "quad" : "linear") + " / " + sh;
}

inline std::vector<int> outer_ids(const std::vector<Img>& m) {
    std::vector<int> ids;
    for (int i = 0; i < (int)m.size(); ++i) if (m[i].axis != 'C') ids.push_back(i);
    return ids;
}
inline std::vector<std::pair<std::vector<int>, std::vector<int>>> outer_folds(const std::vector<Img>& m) {
    auto ids = outer_ids(m);
    std::vector<std::string> y;
    for (int i : ids) y.push_back(m[i].session);
    auto f = stratified_kfold(y, 4, 20261021);
    std::vector<std::pair<std::vector<int>, std::vector<int>>> out;
    for (auto& p : f) {
        std::vector<int> a, b;
        for (int j : p.first) a.push_back(ids[j]);
        for (int j : p.second) b.push_back(ids[j]);
        out.push_back({a, b});
    }
    return out;
}
inline std::vector<std::pair<std::vector<int>, std::vector<int>>> inner_folds(const std::vector<Img>& m, const std::vector<int>& tr, int f) {
    std::vector<std::string> y;
    for (int i : tr) y.push_back(m[i].session);
    auto in = stratified_kfold(y, 3, 100 + f);
    std::vector<std::pair<std::vector<int>, std::vector<int>>> out;
    for (auto& p : in) {
        std::vector<int> a, b;
        for (int j : p.first) a.push_back(tr[j]);
        for (int j : p.second) b.push_back(tr[j]);
        out.push_back({a, b});
    }
    return out;
}

inline FoldEval four_fold(const std::vector<Img>& m, const std::vector<std::string>& fnames, const std::vector<Feat>& F) {
    FoldEval E;
    E.fnames = fnames;
    for (int f = 0; f < (int)F.size(); ++f)
        for (bool s : {false, true})
            for (bool q : {false, true})
                for (double sh : {.001, .05, .2, 1.}) E.cfgs.push_back({f, s, q, sh});
    const int C = (int)E.cfgs.size();
    E.oof.assign(C, {});
    E.resub.assign(C, {});
    E.fold_stats.assign(C, {});
    auto folds = outer_folds(m);
    std::vector<std::vector<std::vector<Row>>> per(C, std::vector<std::vector<Row>>(folds.size()));
    for (int f = 0; f < (int)folds.size(); ++f) {
        const auto& tr = folds[f].first;
        const auto& te = folds[f].second;
        E.fold_test.push_back(te);
        auto inner = inner_folds(m, tr, f);
        std::vector<std::tuple<int, double, int>> rank(C);
#pragma omp parallel for schedule(dynamic) num_threads(4)
        for (int k = 0; k < C; ++k) {
            const Cfg& c = E.cfgs[k];
            std::vector<Row> ip;
            for (auto& in : inner) {
                Model md = fit(m, F[c.feat], in.first, c.sess, c.quad, c.shrink);
                auto r = enrich(m, predict(md, F[c.feat], in.second, c.sess));
                ip.insert(ip.end(), r.begin(), r.end());
            }
            auto s = stats(ip);
            rank[k] = std::make_tuple(s.fail, std::max(s.Ymax, s.Tmax), k);
            Model md = fit(m, F[c.feat], tr, c.sess, c.quad, c.shrink);
            per[k][f] = enrich(m, predict(md, F[c.feat], te, c.sess));
        }
        int chosen = std::get<2>(*std::min_element(rank.begin(), rank.end()));
        E.selected.push_back(chosen);
        auto sr = rank;
        std::sort(sr.begin(), sr.end());
        E.inner_rank.push_back(sr);
        E.nested.insert(E.nested.end(), per[chosen][f].begin(), per[chosen][f].end());
    }
    auto ids = outer_ids(m);
#pragma omp parallel for schedule(dynamic) num_threads(4)
    for (int k = 0; k < C; ++k) {
        for (int f = 0; f < (int)folds.size(); ++f) {
            E.oof[k].insert(E.oof[k].end(), per[k][f].begin(), per[k][f].end());
            E.fold_stats[k].push_back(stats(per[k][f]));
        }
        const Cfg& c = E.cfgs[k];
        Model md = fit(m, F[c.feat], ids, c.sess, c.quad, c.shrink);
        E.resub[k] = enrich(m, predict(md, F[c.feat], ids, c.sess));
    }
    return E;
}

}  // namespace yt
