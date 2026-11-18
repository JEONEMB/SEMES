// 후보별 가우스 우도: 학습 fold에서 시행 후보마다 이분산 공분산과 사전확률을 두고 -2 log 우도로 후보를 고른다, 81설정 4-fold.
// 원본: codex_experiments_20261014/likelihood_4fold_20261021/run.py (저장소 실험/run.py 와 같음)
// 측정·판정 공통부: g8_measure.hpp, g8_model.hpp, g8_report.hpp (#57~#61 이 이 폴더의 src 를 같이 쓴다)
#include "g8_report.hpp"
#include <iostream>

using namespace g8;

namespace {
struct Cfg { std::string feature, group; double alpha; std::string prior; };

struct LModel {
    Branches br;
    std::vector<cv::Mat> W;
    std::vector<double> pen;
};
LModel lfit(const Manifest& m, const cv::Mat& X, const std::vector<int>& tr, const Cfg& c) {
    LModel L;
    L.br = zfit(m, X, tr, true);
    cv::Mat R = residual_rows(m, X, tr, L.br);
    const int n = (int)tr.size(), d = X.cols;
    cv::Mat S = R.t() * R / n;
    auto sd = col_std(S);
    cv::Mat r = R.clone();
    for (int j = 0; j < d; ++j) r.col(j) /= sd[j];
    cv::Mat pooled = r.t() * r / n;
    for (auto& b : L.br) {
        std::vector<int> mask;
        for (int i = 0; i < n; ++i) {
            auto& it = m.it[tr[i]];
            bool in = c.group == "pooled" || (c.group == "axis" ? it.axis == b.axis : it.session == b.key);
            if (in) mask.push_back(i);
        }
        cv::Mat rm = rows(r, mask);
        cv::Mat local = rm.t() * rm / (double)mask.size();
        cv::Mat cov = local * (1 - c.alpha) + pooled * c.alpha;
        cv::Mat dg = cv::Mat::zeros(d, d, CV_64F);
        for (int j = 0; j < d; ++j) dg.at<double>(j, j) = pooled.at<double>(j, j);
        cov = cov * 0.95 + dg * 0.05 + cv::Mat::eye(d, d, CV_64F) * 1e-9;
        L.W.push_back(whitener(cov, sd));
        cv::Mat Lc = cholesky_lower(cov);
        double logdet = 0;
        for (int j = 0; j < d; ++j) logdet += 2 * std::log(Lc.at<double>(j, j));
        double prob;
        if (c.prior == "uniform_branch") prob = 1.0 / L.br.size();
        else if (c.prior == "equal_axis") {
            int na = 0;
            for (auto& q : L.br) na += q.axis == b.axis;
            prob = 0.5 / na;
        } else {
            int k = 0;
            for (int i : tr) k += m.it[i].session == b.key;
            prob = (double)k / n;
        }
        L.pen.push_back(logdet - 2 * std::log(prob));
    }
    return L;
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
    auto M = measure_all(shots, {{"gn17", 17, 0.8, 0, 6}, {"gn31", 31, 0.8, 0, 6}}, false);
    const auto& m = M.m;
    auto D17 = rel_D(M.tr["gn17"]);
    auto w = quality_weights(M.tr["gn17"]);
    std::map<std::string, cv::Mat> F{{"patch", spatial(M.pts, D17)}, {"quality", spatial(M.pts, D17, &w)}, {"gn31", spatial(M.pts, rel_D(M.tr["gn31"]))}};

    std::vector<Cfg> cfgs;
    for (auto f : {"patch", "quality", "gn31"})
        for (auto g : {"pooled", "axis", "session"})
            for (double a : {0.001, 0.2, 1.0})
                for (auto p : {"uniform_branch", "equal_axis", "empirical"}) cfgs.push_back({f, g, a, p});
    auto pf = [&](int k, const std::vector<int>& tr, const std::vector<int>& te) {
        const auto& c = cfgs[k];
        const cv::Mat& X = F.at(c.feature);
        LModel L = lfit(m, X, tr, c);
        GLSModel g{L.br, cv::Mat()};
        return zpredict(m, X, te, g, &L.pen, &L.W);
    };
    Nested N = run_nested(m, (int)cfgs.size(), pf);

    Report rep;
    rep.E.title = "후보별 잡음 가우스 우도 4-fold";
    rep.E.when = "10-21 17:53";
    rep.E.problem = "GLS 비용 대신 후보별 가우스 우도로 고르면 나아지는지 확인해야 했다.";
    rep.E.solution = "학습 fold에서 시행 후보마다 이분산 공분산과 사전확률을 두고 비용 = 백색화 잔차제곱 + log det 공분산 - 2 log 사전확률로 고르는 81설정을 4-fold로 평가했다.";
    rep.E.prefix = "l";
    rep.E.source = "codex_experiments_20261014/likelihood_4fold_20261021";
    rep.E.cfg_head = {"측정", "공분산 묶음", "공통 섞음", "사전확률"};
    for (auto& c : cfgs) {
        rep.E.cfg_cols.push_back({c.feature, c.group, hye::fmt(c.alpha, 3), c.prior});
        rep.E.cfg_desc.push_back(c.feature + "/" + c.group + "/" + hye::fmt(c.alpha, 3) + "/" + c.prior);
    }
    rep.shots = &shots;
    rep.M = &M;
    rep.N = &N;
    rep.method_note =
        "측정 3종: patch = GN17(창 17, 흐림 0.8) 630점 이동의 P12(위치 2차식 12계수), quality = 같은 이동을 품질 가중(1/(1+(rmse/6)^2+(FB/0.15)^2), 무효 x0.05)으로 맞춘 P12, gn31 = 창 31 GN의 P12.\n"
        "후보 6개(T+, T-, Y1+, Y1-, Y2+, Y2-)는 축별로 [v, v^2]를 공유하고 시행 절편만 따로 둔 곡선이다. 잔차를 표준화한 뒤 공분산을 pooled(전체), axis(축별), session(후보별)로 묶고 전체와 alpha 비율로 섞은 다음, 0.95:0.05로 대각을 더했다.";
    Stats ns = stats(N.nested);
    int resub0 = 0, oofsum = 0;
    for (int k = 0; k < (int)cfgs.size(); ++k) { resub0 += stats(N.resub[k]).fail == 0; oofsum += stats(N.oof[k]).fail; }
    // 원래 값: likelihood_4fold_20261021/summary.csv, nested_selection.json
    auto& par = rep.par;
    par.add("내부 선택 OOF Y 정답", 78, ns.Yc, 0, 0);
    par.add("내부 선택 OOF T 정답", 199, ns.Tc, 0, 0);
    par.add("내부 선택 OOF 실패", 3, ns.fail, 0, 0);
    par.add("내부 선택 OOF Y 최대 오차 (mm)", 0.2659817476698715, ns.Ymax, 1e-4, 6);
    par.add("내부 선택 OOF T 최대 오차 (deg)", 0.05, ns.Tmax, 1e-4, 6);
    std::string ch;
    for (auto& s : N.sel) ch += (s.fold ? "," : "") + cfg_name('l', s.chosen);
    par.add_text("fold별 고른 설정", "l012,l041,l068,l027", ch, ch == "l012,l041,l068,l027");
    par.add("81설정 OOF 실패 합", 211, oofsum, 0, 0);  // summary.csv oof4 행 합 214 에서 nested_selected 3 을 뺀 값
    par.add("재대입 실패 0 설정 수", 44, resub0, 0, 0);
    par.add("l000 OOF Y 최대 오차", 0.2662131351540125, stats(N.oof[0]).Ymax, 1e-4, 6);
    par.add("l002 재대입 Y 최대 오차", 0.0366118715455432, stats(N.resub[2]).Ymax, 1e-4, 6);
    par.add("l002 재대입 T 최대 오차", 0.0028597094124291, stats(N.resub[2]).Tmax, 1e-5, 6);

    int bk = rep.best_oof();
    Stats bs = stats(N.oof[bk]);
    double pen_gap = 0;
    {
        // 사전확률 차이가 비용에 주는 크기: empirical 에서 T 후보(학습 75장)와 Y 후보(학습 15장) 의 -2 log p 차
        pen_gap = -2 * std::log(15.0 / 210) + 2 * std::log(75.0 / 210);
    }
    std::ostringstream one;
    one << "우도는 GLS 비용에 log det 공분산과 사전확률 항만 더한 것이라 후보 사이 비용이 수 단위(경험 사전확률의 T와 Y 시행 차 " << hye::fmt(pen_gap, 2)
        << ")로만 바뀌고, 실패 사진(" << fail_names(m, N.nested) << ")은 정답 후보와의 비용 차가 최소 " << hye::fmt(min_fail_gap(N.nested), 2)
        << "로 그보다 커서 그대로 남았으며, 설정 선택이 fold마다 흔들려 내부 선택이 사후 최고(" << cfg_name('l', bk) << ", 실패 " << bs.fail << ")보다 " << (ns.fail - bs.fail) << "장 더 틀렸다.";
    rep.oneliner = one.str();
    rep.notported =
        "- center 630점 위치(QR 디코딩 격자 + erf 경계 모델 맞춤, `analysis_grid_corners_20261019`)는 옮기지 않고 `points630.hpp`에 고정 ID 목록으로 넣었다. 패치를 놓을 자리의 정의로만 쓴다.\n"
        "- 원본의 설정별 모델 json(l000_model.json 등) 저장은 옮기지 않았다.";
    rep.write();
    std::cout << "parity " << par.ok << "/" << par.total << "\n";
    return 0;
}
