// P12 요약과 전체 630점 변위장 혼합: P12 GLS 비용에 fullfield 백색화 잔차를 가중으로 더한 비용, 30설정 4-fold.
// 원본: codex_experiments_20261014/hybridfield_4fold_20261021/run_hybrid.py (저장소 실험/run_hybrid.py 와 같음)
// 측정·판정 공통부: 20261021_1753_후보별잡음_가우스우도/실험/src (fullfield fit 은 g8_fullfield.hpp)
#include "g8_fullfield.hpp"
#include "g8_report.hpp"
#include <iostream>

using namespace g8;

namespace {
struct Cfg { std::string feature; double alpha, beta; };
}

static int run();
int main() {
    try {
        return run();
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}

static int run() {
    omp_set_num_threads(4);
    hye::begin_results();
    auto shots = hye::load_1014();
    auto M = measure_all(shots, {{"gn17", 17, 0.8, 0, 6}, {"gn31", 31, 0.8, 0, 6}}, false);
    const auto& m = M.m;
    std::map<std::string, cv::Mat> FF, X12;
    for (auto k : {"gn17", "gn31"}) {
        auto D = rel_D(M.tr[k]);
        FF[k] = full_field(D);
        X12[k] = spatial(M.pts, D);
    }
    std::vector<Cfg> cfgs;
    for (auto f : {"gn17", "gn31"})
        for (double a : {0.2, 0.5, 1.0})
            for (double b : {0.03, 0.1, 0.3, 1.0, 3.0}) cfgs.push_back({f, a, b});
    auto pf = [&](int k, const std::vector<int>& tr, const std::vector<int>& te) {
        const auto& c = cfgs[k];
        FullModel full = full_fit(m, FF.at(c.feature), tr, true, c.alpha);
        cv::Mat XX = full.whiten(rows(FF.at(c.feature), te));
        GLSModel g = mfit(m, X12.at(c.feature), tr, true, 0.2);
        cv::Mat PP = rows(X12.at(c.feature), te) * g.W.t();
        const double w = std::sqrt(c.beta * 12 / 1260);
        std::vector<std::string> keys;
        std::vector<char> axes;
        std::vector<std::vector<double>> V, C;
        for (auto& b : g.br) {
            auto& f = find_branch(full.br, b.key);
            cv::Mat z, s, cc;
            cv::hconcat(PP - cv::repeat(b.b * g.W.t(), PP.rows, 1), (XX - cv::repeat(f.b, XX.rows, 1)) * w, z);
            cv::hconcat(b.s * g.W.t(), f.s * w, s);
            cv::hconcat(b.c * g.W.t(), f.c * w, cc);
            std::vector<double> v, cost;
            branch_solve(z, s, cc, v, cost);
            for (size_t t = 0; t < cost.size(); ++t) if (!sign_ok(b.key, v[t])) cost[t] = std::numeric_limits<double>::infinity();
            keys.push_back(b.key);
            axes.push_back(b.axis);
            V.push_back(v);
            C.push_back(cost);
        }
        return choose(m, te, keys, axes, V, C);
    };
    Nested N = run_nested(m, (int)cfgs.size(), pf);

    Report rep;
    rep.E.title = "P12 요약과 전체 630점 혼합 비용 4-fold";
    rep.E.when = "10-21 17:58";
    rep.E.problem = "P12 요약과 전체 630점 변위장을 함께 쓰면 각각보다 나아지는지 확인해야 했다.";
    rep.E.solution = "P12 GLS 비용(공분산 축소 0.2)에 fullfield 백색화 잔차를 beta x 12/1260 가중으로 더한 혼합 비용 30설정을 4-fold로 평가했다.";
    rep.E.prefix = "h";
    rep.E.source = "codex_experiments_20261014/hybridfield_4fold_20261021";
    rep.E.cfg_head = {"측정", "fullfield 축소 alpha", "fullfield 가중 beta"};
    for (auto& c : cfgs) {
        rep.E.cfg_cols.push_back({c.feature, hye::fmt(c.alpha, 1), hye::fmt(c.beta, 2)});
        rep.E.cfg_desc.push_back(c.feature + "/" + hye::fmt(c.alpha, 1) + "/" + hye::fmt(c.beta, 2));
    }
    rep.shots = &shots;
    rep.M = &M;
    rep.N = &N;
    rep.method_note =
        "측정 2종: gn17 = GN(창 17, 흐림 0.8), gn31 = GN(창 31). 같은 측정에서 P12(12계수)와 1260차원 변위장을 함께 만든다.\n"
        "후보마다 비용 = |W_P12 (x12 - 곡선)|^2 + beta x 12/1260 x |fullfield 백색화 잔차|^2 이고, 두 부분의 곡선 [v, v^2]를 하나의 v로 같이 푼다. P12 쪽은 multiref fit(이차, 축소 0.2), fullfield 쪽은 #57 fit(이차, 축소 alpha)이다.";
    Stats ns = stats(N.nested);
    int resub0 = 0, oofsum = 0;
    for (int k = 0; k < (int)cfgs.size(); ++k) { resub0 += stats(N.resub[k]).fail == 0; oofsum += stats(N.oof[k]).fail; }
    // 원래 값: hybridfield_4fold_20261021/summary.csv, nested_selection.json
    auto& par = rep.par;
    par.add("내부 선택 OOF Y 정답", 79, ns.Yc, 0, 0);
    par.add("내부 선택 OOF T 정답", 199, ns.Tc, 0, 0);
    par.add("내부 선택 OOF 실패", 2, ns.fail, 0, 0);
    par.add("내부 선택 OOF Y 최대 오차 (mm)", 0.26603637172139616, ns.Ymax, 1e-4, 6);
    par.add("내부 선택 OOF T 최대 오차 (deg)", 0.05, ns.Tmax, 1e-4, 6);
    std::string ch;
    for (auto& s : N.sel) ch += (s.fold ? "," : "") + cfg_name('h', s.chosen);
    par.add_text("fold별 고른 설정", "h013,h014,h027,h027", ch, ch == "h013,h014,h027,h027");
    par.add("30설정 OOF 실패 합", 50, oofsum, 0, 0);
    par.add("재대입 실패 0 설정 수", 1, resub0, 0, 0);
    par.add("h000 OOF Y 최대 오차", 0.2661768353396073, stats(N.oof[0]).Ymax, 1e-4, 6);
    par.add("h000 재대입 Y 최대 오차", 0.2666000012580514, stats(N.resub[0]).Ymax, 1e-4, 6);
    par.add("h012 OOF 실패 (사후 최고)", 1, stats(N.oof[12]).fail, 0, 0);
    par.add("h012 OOF Y 최대 오차", 0.2660762346328593, stats(N.oof[12]).Ymax, 1e-4, 6);

    int bk = rep.best_oof();
    Stats bs = stats(N.oof[bk]);
    // beta 가 클수록 fullfield 비중이 커진다: beta 별 OOF 실패 합
    std::map<double, int> by_beta;
    for (int k = 0; k < (int)cfgs.size(); ++k) by_beta[cfgs[k].beta] += stats(N.oof[k]).fail;
    std::ostringstream ex;
    ex << "## fullfield 가중(beta)별 OOF 실패 합 (6설정씩)\n\n| beta | 실패 합 |\n|---|---|\n";
    for (auto& kv : by_beta) ex << "| " << hye::fmt(kv.first, 2) << " | " << kv.second << " |\n";
    ex << "\n";
    rep.extra_result = ex.str();
    std::ostringstream one;
    one << "혼합 비용은 P12 비용이 주인이고 fullfield 잔차는 beta x 12/1260 만큼만 더해지므로 beta가 작을 때는 P12 판정과 거의 같고(beta 0.03 실패 합 " << by_beta[0.03]
        << ") 커질수록 fullfield의 과적합이 섞여(beta 3 실패 합 " << by_beta[3.0] << "), 내부 선택 OOF는 " << ns.fail << "장(" << fail_names(m, N.nested)
        << ")으로 사후 최고 " << cfg_name('h', bk) << "(실패 " << bs.fail << ")와 비슷한 수준에 머물렀다.";
    rep.oneliner = one.str();
    rep.notported =
        "- center 630점 위치(QR 디코딩 격자 + erf 경계 모델 맞춤)는 옮기지 않고 #56 실험의 `points630.hpp` 고정 ID 목록을 쓴다.\n"
        "- 원본의 lru_cache(같은 학습 집합의 fit 재사용)는 계산 순서만 바꾸므로 옮기지 않고 매번 다시 맞춘다.";
    rep.write();
    std::cout << "parity " << par.ok << "/" << par.total << "\n";
    return 0;
}
