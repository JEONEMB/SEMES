// 630점 이동 전체(1260차원)를 직접 모델링: 축별 곡선 + 시행 절편, 잔차 SVD 축소 백색화, 30설정 4-fold.
// 원본: codex_experiments_20261014/fullfield_4fold_20261021/run.py, verify.py (저장소 실험/run.py, verify.py 와 같음)
// 측정·판정 공통부: 20261021_1753_후보별잡음_가우스우도/실험/src
#include "g8_fullfield.hpp"
#include "g8_report.hpp"
#include <iostream>

using namespace g8;

namespace {
struct Cfg { std::string feature; bool quad; double alpha; };

// verify.py 와 같은 검산: 백색화 비용 = x^T (alpha I + (1-alpha) R^T R)^-1 x (난수는 cv::RNG 고정 시드, numpy 난수열과 다름)
double whitening_check() {
    cv::RNG rng(21);
    cv::Mat R(12, 25, CV_64F), X(4, 25, CV_64F);
    rng.fill(R, cv::RNG::NORMAL, 0, 1);
    rng.fill(X, cv::RNG::NORMAL, 0, 1);
    R /= std::sqrt(12.0);
    const double alpha = 0.2;
    cv::Mat w, u, vt;
    cv::SVD::compute(R, w, u, vt);
    cv::Mat p = X * vt.t();
    for (int j = 0; j < p.cols; ++j) p.col(j) *= 1 / std::sqrt(alpha + (1 - alpha) * w.at<double>(j) * w.at<double>(j)) - 1 / std::sqrt(alpha);
    cv::Mat Y = X / std::sqrt(alpha) + p * vt;
    cv::Mat P;
    cv::invert(cv::Mat::eye(25, 25, CV_64F) * alpha + R.t() * R * (1 - alpha), P, cv::DECOMP_LU);
    double err = 0;
    for (int i = 0; i < 4; ++i) err = std::max(err, std::abs(Y.row(i).dot(Y.row(i)) - cv::Mat(X.row(i) * P * X.row(i).t()).at<double>(0)));
    return err;
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
    std::map<std::string, cv::Mat> F{{"gn17", full_field(rel_D(M.tr["gn17"]))}, {"gn31", full_field(rel_D(M.tr["gn31"]))}, {"huber6", full_field(rel_D(M.tr["huber6"]))}};
    std::vector<Cfg> cfgs;
    for (auto f : {"gn17", "gn31", "huber6"})
        for (bool q : {false, true})
            for (double a : {0.01, 0.05, 0.2, 0.5, 1.0}) cfgs.push_back({f, q, a});
    auto pf = [&](int k, const std::vector<int>& tr, const std::vector<int>& te) {
        const auto& c = cfgs[k];
        return full_predict(m, F.at(c.feature), te, full_fit(m, F.at(c.feature), tr, c.quad, c.alpha));
    };
    Nested N = run_nested(m, (int)cfgs.size(), pf);

    // 학습 공분산이 시험 특징에 영향받지 않는지 (verify.py 두 번째 검사)
    bool leak_free;
    {
        auto folds = outer_folds(m);
        cv::Mat X = F["gn17"].clone();
        FullModel a = full_fit(m, X, folds[0].tr, false, 0.01);
        for (int i : folds[0].te) X.row(i) += 999;
        FullModel b = full_fit(m, X, folds[0].tr, false, 0.01);
        leak_free = cv::norm(a.V, b.V, cv::NORM_INF) == 0 && a.sd == b.sd && a.adjust == b.adjust;
    }
    double werr = whitening_check();

    Report rep;
    rep.E.title = "전체 630점 직접 1260차원 SVD 축소 4-fold";
    rep.E.when = "10-21 17:56";
    rep.E.problem = "P12 요약 없이 630점 변위 전체(1260차원)를 직접 모델링하면 나아지는지 확인해야 했다.";
    rep.E.solution = "1260차원 변위에 축별 곡선과 시행 절편을 맞추고 잔차 SVD 축소로 백색화해 30설정을 4-fold로 평가했다.";
    rep.E.prefix = "f";
    rep.E.source = "codex_experiments_20261014/fullfield_4fold_20261021";
    rep.E.cfg_head = {"측정", "이차항", "축소 alpha"};
    for (auto& c : cfgs) {
        rep.E.cfg_cols.push_back({c.feature, c.quad ? "예" : "아니오", hye::fmt(c.alpha, 2)});
        rep.E.cfg_desc.push_back(c.feature + "/" + (c.quad ? "quad" : "linear") + "/" + hye::fmt(c.alpha, 2));
    }
    rep.shots = &shots;
    rep.M = &M;
    rep.N = &N;
    rep.method_note =
        "측정 3종: gn17 = GN(창 17, 흐림 0.8), gn31 = GN(창 31), huber6 = GN17에 화소 Huber(delta 6) IRLS 2회. 630점 dx, dy를 그대로 1260차원 특징으로 쓴다.\n"
        "축마다 [v] 또는 [v, v^2]와 시행 절편을 최소제곱으로 맞추고, 잔차를 열마다 RMS로 나눈 R/sqrt(n)의 SVD로 공분산 alpha I + (1-alpha) R^T R 의 역제곱근을 만들어 백색화한다. "
        "SVD는 R R^T(학습 장수 크기)의 고유분해로 같은 특이값·벡터를 얻었다.";
    Stats ns = stats(N.nested);
    int resub0 = 0, oofsum = 0;
    for (int k = 0; k < (int)cfgs.size(); ++k) { resub0 += stats(N.resub[k]).fail == 0; oofsum += stats(N.oof[k]).fail; }
    std::ostringstream ex;
    ex << "## 수치 검사 (verify.py)\n\n| 검사 | 값 |\n|---|---|\n| 백색화 비용과 조밀 정밀도 행렬 비용의 최대 차 (난수 12x25, alpha 0.2) | " << std::scientific << werr << std::fixed
       << " |\n| 시험 특징 +999 해도 학습 공분산(V, std, adjust) 그대로 | " << (leak_free ? "예" : "아니오") << " |\n\n";
    rep.extra_result = ex.str();
    // 원래 값: fullfield_4fold_20261021/summary.csv, nested_selection.json
    auto& par = rep.par;
    par.add("내부 선택 OOF Y 정답", 74, ns.Yc, 0, 0);
    par.add("내부 선택 OOF T 정답", 199, ns.Tc, 0, 0);
    par.add("내부 선택 OOF 실패", 7, ns.fail, 0, 0);
    par.add("내부 선택 OOF Y 최대 오차 (mm)", 0.26583585353418815, ns.Ymax, 1e-4, 6);
    par.add("내부 선택 OOF T 최대 오차 (deg)", 0.05, ns.Tmax, 1e-4, 6);
    std::string ch;
    for (auto& s : N.sel) ch += (s.fold ? "," : "") + cfg_name('f', s.chosen);
    par.add_text("fold별 고른 설정", "f018,f012,f015,f013", ch, ch == "f018,f012,f015,f013");
    par.add("30설정 OOF 실패 합", 222, oofsum, 0, 0);
    par.add("재대입 실패 0 설정 수", 24, resub0, 0, 0);
    par.add("f000 OOF 실패", 9, stats(N.oof[0]).fail, 0, 0);
    par.add("f000 OOF T 최대 오차", 0.0501518491970043, stats(N.oof[0]).Tmax, 1e-4, 6);
    par.add("f000 재대입 Y 최대 오차", 0.0004934959121991, stats(N.resub[0]).Ymax, 2e-5, 6);
    par.add("f000 재대입 T 최대 오차", 0.000177578774775, stats(N.resub[0]).Tmax, 2e-5, 6);
    par.add("백색화 검산 오차 < 1e-9", 1, werr < 1e-9, 0, 0);

    int bk = rep.best_oof();
    Stats bs = stats(N.oof[bk]);
    std::ostringstream one;
    one << "1260차원 모델은 학습 210장이 차원보다 적어 잔차 공분산을 SVD 축소로 메워 쓰므로 학습 사진에는 거의 완벽히 맞아(재대입 실패 0 설정 " << resub0 << "/" << cfgs.size()
        << ", f000 재대입 Y 최대 " << hye::fmt(stats(N.resub[0]).Ymax, 4) << " mm) 시험 사진의 작은 변위장 차이를 축 차이로 읽었고, 그래서 OOF 실패 합이 " << oofsum
        << "(설정당 평균 " << hye::fmt((double)oofsum / cfgs.size(), 1) << ")이었고 내부 선택도 " << ns.fail << "장(" << fail_names(m, N.nested) << ")을 틀렸다.";
    rep.oneliner = one.str();
    (void)bs;
    rep.notported =
        "- center 630점 위치(QR 디코딩 격자 + erf 경계 모델 맞춤)는 옮기지 않고 #56 실험의 `points630.hpp` 고정 ID 목록을 쓴다.\n"
        "- 원본의 설정별 모델 npz(f000_model.npz 등) 저장은 옮기지 않았다.\n"
        "- verify.py 의 난수 검산은 numpy 난수열 대신 cv::RNG(21)로 같은 크기(12x25, 4x25)를 만들어 같은 식을 검사했다.";
    rep.write();
    std::cout << "parity " << par.ok << "/" << par.total << "\n";
    return 0;
}
