// 후보별 독립 곡선: 6후보가 기울기·곡률을 공유하지 않고 후보마다 절편·기울기·2차항을 따로 맞춘다, 32설정 4-fold.
// 원본: codex_experiments_20261014/branch_curves_4fold_20261021/run.py (저장소 실험/run.py 와 같음)
// 측정·판정 공통부: 20261021_1753_후보별잡음_가우스우도/실험/src
#include "g8_report.hpp"
#include <iostream>

using namespace g8;

namespace {
struct Cfg { std::string feature; bool quad; double alpha; };

GLSModel bfit(const Manifest& m, const cv::Mat& X, const std::vector<int>& tr, const Cfg& c) {
    GLSModel g;
    std::vector<std::string> ses;
    for (int i : tr) ses.push_back(m.it[i].session);
    std::sort(ses.begin(), ses.end());
    ses.erase(std::unique(ses.begin(), ses.end()), ses.end());
    std::vector<cv::Mat> res;
    int rank = 0;
    for (auto& k : ses) {
        std::vector<int> ii;
        for (int i : tr) if (m.it[i].session == k) ii.push_back(i);
        cv::Mat A((int)ii.size(), c.quad ? 3 : 2, CV_64F);
        for (int r = 0; r < A.rows; ++r) {
            double v = m.it[ii[r]].value;
            A.at<double>(r, 0) = 1;
            A.at<double>(r, 1) = v;
            if (c.quad) A.at<double>(r, 2) = v * v;
        }
        cv::Mat Z = rows(X, ii), co = lstsq(A, Z);
        rank += matrix_rank(A);
        res.push_back(Z - A * co);
        g.br.push_back({k, k[0], co.row(0).clone(), co.row(1).clone(), c.quad ? co.row(2).clone() : cv::Mat::zeros(1, X.cols, CV_64F)});
    }
    cv::Mat R;
    cv::vconcat(res, R);
    cv::Mat S = R.t() * R / (double)((int)tr.size() - rank);
    auto sd = col_std(S);
    cv::Mat C = corr(S, sd) * (1 - c.alpha) + cv::Mat::eye(S.rows, S.cols, CV_64F) * c.alpha;
    g.W = whitener(C, sd);
    return g;
}
}  // namespace

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
    auto M = measure_all(shots, {{"gn17", 17, 0.8, 0, 6}, {"gn31", 31, 0.8, 0, 6}, {"huber6", 17, 0.8, 1, 6}}, false);
    const auto& m = M.m;
    auto D17 = rel_D(M.tr["gn17"]);
    auto w = quality_weights(M.tr["gn17"]);
    std::map<std::string, cv::Mat> F{{"patch", spatial(M.pts, D17)},
                                     {"quality", spatial(M.pts, D17, &w)},
                                     {"gn31", spatial(M.pts, rel_D(M.tr["gn31"]))},
                                     {"huber6", spatial(M.pts, rel_D(M.tr["huber6"]))}};
    std::vector<Cfg> cfgs;
    for (auto f : {"patch", "quality", "gn31", "huber6"})
        for (bool q : {false, true})
            for (double a : {0.001, 0.05, 0.2, 1.0}) cfgs.push_back({f, q, a});
    auto pf = [&](int k, const std::vector<int>& tr, const std::vector<int>& te) {
        const auto& c = cfgs[k];
        return zpredict(m, F.at(c.feature), te, bfit(m, F.at(c.feature), tr, c));
    };
    Nested N = run_nested(m, (int)cfgs.size(), pf);

    Report rep;
    rep.E.title = "후보별 독립 변형 곡선 4-fold";
    rep.E.when = "10-21 17:59";
    rep.E.problem = "6후보 모델의 공유 기울기·곡률을 후보마다 따로 두면 나아지는지 확인해야 했다.";
    rep.E.solution = "후보(시행)마다 독립 절편·기울기·2차항을 맞추고 공분산 상관 축소 GLS로 고르는 32설정을 4-fold로 평가했다.";
    rep.E.prefix = "b";
    rep.E.source = "codex_experiments_20261014/branch_curves_4fold_20261021";
    rep.E.cfg_head = {"측정", "이차항", "축소 alpha"};
    for (auto& c : cfgs) {
        rep.E.cfg_cols.push_back({c.feature, c.quad ? "예" : "아니오", hye::fmt(c.alpha, 3)});
        rep.E.cfg_desc.push_back(c.feature + "/" + (c.quad ? "quad" : "linear") + "/" + hye::fmt(c.alpha, 3));
    }
    rep.shots = &shots;
    rep.M = &M;
    rep.N = &N;
    rep.method_note =
        "측정 4종: patch = GN17 P12, quality = 품질 가중 P12, gn31 = 창 31 GN P12, huber6 = Huber(delta 6) GN17 P12.\n"
        "후보 6개(T+, T-, Y1+, Y1-, Y2+, Y2-)마다 [1, v] 또는 [1, v, v^2]를 따로 최소제곱으로 맞추고(곡선 공유 없음), 잔차 공분산 S = R^T R / (n - 계수 수)를 상관으로 바꿔 alpha 만큼 단위행렬로 축소한 뒤 GLS 비용으로 고른다.";
    Stats ns = stats(N.nested);
    int resub0 = 0, oofsum = 0;
    for (int k = 0; k < (int)cfgs.size(); ++k) { resub0 += stats(N.resub[k]).fail == 0; oofsum += stats(N.oof[k]).fail; }
    // 원래 값: branch_curves_4fold_20261021/summary.csv, nested_selection.json
    auto& par = rep.par;
    par.add("내부 선택 OOF Y 정답", 79, ns.Yc, 0, 0);
    par.add("내부 선택 OOF T 정답", 198, ns.Tc, 0, 0);
    par.add("내부 선택 OOF 실패", 3, ns.fail, 0, 0);
    par.add("내부 선택 OOF Y 최대 오차 (mm)", 0.2652790889697762, ns.Ymax, 1e-4, 6);
    par.add("내부 선택 OOF T 최대 오차 (deg)", 0.05, ns.Tmax, 1e-4, 6);
    std::string ch;
    for (auto& s : N.sel) ch += (s.fold ? "," : "") + cfg_name('b', s.chosen);
    par.add_text("fold별 고른 설정", "b010,b019,b018,b000", ch, ch == "b010,b019,b018,b000");
    par.add("32설정 OOF 실패 합", 51, oofsum, 0, 0);
    par.add("재대입 실패 0 설정 수", 2, resub0, 0, 0);
    par.add("b000 OOF 실패", 2, stats(N.oof[0]).fail, 0, 0);
    par.add("b000 OOF Y 최대 오차", 0.2654538991615575, stats(N.oof[0]).Ymax, 1e-4, 6);
    par.add("b000 재대입 Y 최대 오차", 0.2661842762792198, stats(N.resub[0]).Ymax, 1e-4, 6);
    par.add("b012 OOF 실패 (사후 최고)", 1, stats(N.oof[12]).fail, 0, 0);
    par.add("b012 OOF Y 최대 오차", 0.266589410911658, stats(N.oof[12]).Ymax, 1e-4, 6);

    int bk = rep.best_oof();
    Stats bs = stats(N.oof[bk]);
    std::ostringstream one;
    one << "후보마다 곡선을 따로 두면 한 후보의 학습 사진이 바깥 fold에서 15장(Y)뿐이라 기울기·곡률이 그 15장에 맞춰지고, 시행 간 공유가 주던 정보가 사라져 실패 사진("
        << fail_names(m, N.nested) << ")이 다른 축 후보로 읽혀 내부 선택 OOF " << ns.fail << "장을 틀렸고, 사후 최고 " << cfg_name('b', bk) << "도 "
        << bs.fail << "장을 틀렸다.";
    rep.oneliner = one.str();
    rep.notported =
        "- center 630점 위치(QR 디코딩 격자 + erf 경계 모델 맞춤)는 옮기지 않고 #56 실험의 `points630.hpp` 고정 ID 목록을 쓴다.\n"
        "- 원본의 설정별 모델 json(b000_model.json 등) 저장은 옮기지 않았다.";
    rep.write();
    std::cout << "parity " << par.ok << "/" << par.total << "\n";
    return 0;
}
