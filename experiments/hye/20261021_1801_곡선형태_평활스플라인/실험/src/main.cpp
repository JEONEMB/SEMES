// 후보별 평활 3차 B-스플라인 곡선: 학습 fold에서 시행마다 스플라인(내부 매듭 0/2/4, 2차 차분 벌점)을 맞추고 GLS 비용으로 고른다, 54설정 4-fold.
// 원본: codex_experiments_20261014/spline_4fold_20261021/run.py (저장소 실험/run.py 와 같음)
// scipy.interpolate.BSpline(extrapolate=True) 값·미분을 de Boor 식 그대로 옮겼다.
// 측정·판정 공통부: 20261021_1753_후보별잡음_가우스우도/실험/src
#include "g8_report.hpp"
#include <iostream>

using namespace g8;

namespace {
struct Cfg { std::string feature; int nknots; double reg, alpha; };

// scipy BSpline: 매듭 t, 계수 c (n x d), 차수 k. 구간 밖은 끝 구간 다항식으로 연장
struct Spline {
    std::vector<double> t;
    cv::Mat c;
    int k = 3;
    int interval(double x) const {
        const int n = (int)t.size() - k - 1;
        int l = k;
        while (x < t[l] && l != k) --l;
        ++l;
        while (x >= t[l] && l != n) ++l;
        return l - 1;
    }
    cv::Mat basis_row(double x, int& ell) const {  // k+1 개 값
        ell = interval(x);
        std::vector<double> h(k + 1, 0), hh(k + 1, 0);
        h[0] = 1;
        for (int j = 1; j <= k; ++j) {
            for (int q = 0; q < j; ++q) hh[q] = h[q];
            h[0] = 0;
            for (int q = 1; q <= j; ++q) {
                int ind = ell + q;
                double xb = t[ind], xa = t[ind - j];
                if (xb == xa) { h[q] = 0; continue; }
                double w = hh[q - 1] / (xb - xa);
                h[q - 1] += w * (xb - x);
                h[q] = w * (x - xa);
            }
        }
        return cv::Mat(h, true).t();
    }
    cv::Mat operator()(const std::vector<double>& xs) const {
        cv::Mat o((int)xs.size(), c.cols, CV_64F, cv::Scalar(0));
        for (int r = 0; r < (int)xs.size(); ++r) {
            int ell;
            cv::Mat h = basis_row(xs[r], ell);
            for (int i = 0; i <= k; ++i) o.row(r) += c.row(ell - k + i) * h.at<double>(i);
        }
        return o;
    }
    Spline derivative() const {
        Spline d;
        const int n = c.rows;
        cv::Mat cp = cv::Mat::zeros((int)t.size(), c.cols, CV_64F);
        c.copyTo(cp.rowRange(0, n));
        d.c = cv::Mat::zeros((int)t.size() - 2, c.cols, CV_64F);
        for (int i = 0; i < (int)t.size() - 2 - k; ++i) {
            double dt = t[i + k + 1] - t[i + 1];
            d.c.row(i) = (cp.row(i + 1) - cp.row(i)) * k / dt;
        }
        d.t.assign(t.begin() + 1, t.end() - 1);
        d.k = k - 1;
        return d;
    }
};
// 기저 행렬 (np.eye 계수의 BSpline)
cv::Mat design(const std::vector<double>& knots, const std::vector<double>& xs) {
    Spline s;
    s.t = knots;
    const int n = (int)knots.size() - 4;
    s.c = cv::Mat::eye(n, n, CV_64F);
    return s(xs);
}
struct SBranch { std::string key; char axis; double lo, scale; Spline sp; };
struct SModel { std::vector<SBranch> br; cv::Mat W; };

SModel sfit(const Manifest& m, const cv::Mat& X, const std::vector<int>& tr, const Cfg& c) {
    SModel g;
    std::vector<std::string> ses;
    for (int i : tr) ses.push_back(m.it[i].session);
    std::sort(ses.begin(), ses.end());
    ses.erase(std::unique(ses.begin(), ses.end()), ses.end());
    std::vector<cv::Mat> res;
    for (auto& k : ses) {
        std::vector<int> ii;
        for (int i : tr) if (m.it[i].session == k) ii.push_back(i);
        double lo = 1e300, hi = -1e300;
        for (int i : ii) { lo = std::min(lo, m.it[i].value); hi = std::max(hi, m.it[i].value); }
        double scale = hi - lo;
        std::vector<double> t;
        for (int i : ii) t.push_back((m.it[i].value - lo) / scale);
        std::vector<double> knots(4, 0.0);
        for (int j = 1; j <= c.nknots; ++j) knots.push_back(j * (1.0 / (c.nknots + 1)));  // np.linspace(0,1,nk+2)[1:-1] = j*step
        for (int j = 0; j < 4; ++j) knots.push_back(1.0);
        const int n = (int)knots.size() - 4;
        cv::Mat A = design(knots, t);
        cv::Mat pen = cv::Mat::zeros(n - 2, n, CV_64F);
        for (int r = 0; r < n - 2; ++r) { pen.at<double>(r, r) = 1; pen.at<double>(r, r + 1) = -2; pen.at<double>(r, r + 2) = 1; }
        cv::Mat Z = rows(X, ii), coef;
        cv::solve(A.t() * A + pen.t() * pen * c.reg + cv::Mat::eye(n, n, CV_64F) * 1e-10, A.t() * Z, coef, cv::DECOMP_LU);
        res.push_back(Z - A * coef);
        Spline sp;
        sp.t = knots;
        sp.c = coef;
        g.br.push_back({k, k[0], lo, scale, sp});
    }
    cv::Mat R;
    cv::vconcat(res, R);
    cv::Mat S = R.t() * R / (double)R.rows;
    auto sd = col_std(S);
    cv::Mat C = corr(S, sd) * (1 - c.alpha) + cv::Mat::eye(S.rows, S.cols, CV_64F) * c.alpha;
    g.W = whitener(C, sd);
    return g;
}

std::vector<Pred> spredict(const Manifest& m, const cv::Mat& Xall, const std::vector<int>& te, const SModel& g) {
    cv::Mat X = rows(Xall, te), Xw = X * g.W.t();
    std::vector<std::string> keys;
    std::vector<char> axes;
    std::vector<std::vector<double>> V, C;
    for (auto& b : g.br) {
        const double sg = b.key.back() == '+' ? 1 : -1;
        std::vector<double> grid;
        for (int j = 0; j < 81; ++j) {
            double gg = j * ((1.2 - -0.2) / 80) + -0.2;  // np.linspace(-.2, 1.2, 81) = j*step + start, 끝점은 stop
            if (j == 80) gg = 1.2;
            if ((b.lo + gg * b.scale) * sg >= 0) grid.push_back(gg);
        }
        cv::Mat fw = b.sp(grid) * g.W.t();
        Spline der = b.sp.derivative();
        std::vector<double> v(te.size()), cost(te.size());
        for (int r = 0; r < (int)te.size(); ++r) {
            int best = 0;
            double bc = std::numeric_limits<double>::infinity();
            for (int q = 0; q < fw.rows; ++q) {
                double s = 0;
                for (int j = 0; j < fw.cols; ++j) {
                    double d = Xw.at<double>(r, j) - fw.at<double>(q, j);
                    s += d * d;
                }
                if (s < bc) { bc = s; best = q; }
            }
            double t = grid[best];
            for (int it = 0; it < 12; ++it) {
                cv::Mat rr = (X.row(r) - b.sp({t})) * g.W.t(), jj = der({t}) * g.W.t();
                double step = rr.dot(jj) / std::max(jj.dot(jj), 1e-20);
                t = std::min(std::max(t + std::min(std::max(step, -0.1), 0.1), -0.2), 1.2);
            }
            v[r] = b.lo + t * b.scale;
            cv::Mat rr = (X.row(r) - b.sp({t})) * g.W.t();
            cost[r] = rr.dot(rr);
            if (v[r] * sg < 0) cost[r] = std::numeric_limits<double>::infinity();
        }
        keys.push_back(b.key);
        axes.push_back(b.axis);
        V.push_back(v);
        C.push_back(cost);
    }
    return choose(m, te, keys, axes, V, C);
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
        for (int nk : {0, 2, 4})
            for (double r : {0.01, 1.0, 100.0})
                for (double a : {0.05, 0.5}) cfgs.push_back({f, nk, r, a});
    auto pf = [&](int k, const std::vector<int>& tr, const std::vector<int>& te) {
        const auto& c = cfgs[k];
        return spredict(m, F.at(c.feature), te, sfit(m, F.at(c.feature), tr, c));
    };
    Nested N = run_nested(m, (int)cfgs.size(), pf);

    Report rep;
    rep.E.title = "후보별 평활 스플라인 곡선 4-fold";
    rep.E.when = "10-21 18:01";
    rep.E.problem = "명령값과 특징의 관계가 이차식보다 휘어 있다면 매끄러운 곡선이 판정을 개선하는지 확인해야 했다.";
    rep.E.solution = "후보마다 3차 B-스플라인(내부 매듭 0/2/4, 2차 차분 벌점)을 학습 fold에서 맞추고 격자 탐색 + GN으로 명령값을 푸는 54설정을 4-fold로 평가했다.";
    rep.E.prefix = "s";
    rep.E.source = "codex_experiments_20261014/spline_4fold_20261021";
    rep.E.cfg_head = {"측정", "내부 매듭", "평활 벌점", "축소 alpha"};
    for (auto& c : cfgs) {
        rep.E.cfg_cols.push_back({c.feature, std::to_string(c.nknots), hye::fmt(c.reg, 2), hye::fmt(c.alpha, 2)});
        rep.E.cfg_desc.push_back(c.feature + "/k" + std::to_string(c.nknots) + "/" + hye::fmt(c.reg, 2) + "/" + hye::fmt(c.alpha, 2));
    }
    rep.shots = &shots;
    rep.M = &M;
    rep.N = &N;
    rep.method_note =
        "측정 3종: patch = GN17 P12, quality = 품질 가중 P12, gn31 = 창 31 GN P12.\n"
        "후보마다 학습 명령값 범위를 [0, 1]로 바꾼 뒤 3차 B-스플라인 계수를 (A^T A + 벌점 x D2^T D2 + 1e-10 I)^-1 A^T X 로 맞춘다. 예측은 범위 -20%~+120% 81점 격자에서 GLS 비용 최소점을 찾고 GN 12회(한 번에 0.1 이하)로 다듬는다.";
    Stats ns = stats(N.nested);
    int resub0 = 0, oofsum = 0;
    for (int k = 0; k < (int)cfgs.size(); ++k) { resub0 += stats(N.resub[k]).fail == 0; oofsum += stats(N.oof[k]).fail; }
    // 매듭 수별 OOF 실패 합
    std::map<int, int> by_k;
    for (int k = 0; k < (int)cfgs.size(); ++k) by_k[cfgs[k].nknots] += stats(N.oof[k]).fail;
    std::ostringstream ex;
    ex << "## 내부 매듭 수별 OOF 실패 합 (18설정씩)\n\n| 내부 매듭 | 실패 합 |\n|---|---|\n";
    for (auto& kv : by_k) ex << "| " << kv.first << " | " << kv.second << " |\n";
    ex << "\n";
    rep.extra_result = ex.str();
    // 원래 값: spline_4fold_20261021/summary.csv, nested_selection.json
    auto& par = rep.par;
    par.add("내부 선택 OOF Y 정답", 80, ns.Yc, 0, 0);
    par.add("내부 선택 OOF T 정답", 198, ns.Tc, 0, 0);
    par.add("내부 선택 OOF 실패", 2, ns.fail, 0, 0);
    par.add("내부 선택 OOF Y 최대 오차 (mm)", 0.26549937776520305, ns.Ymax, 1e-4, 6);
    par.add("내부 선택 OOF T 최대 오차 (deg)", 0.05, ns.Tmax, 1e-4, 6);
    std::string ch;
    for (auto& s : N.sel) ch += (s.fold ? "," : "") + cfg_name('s', s.chosen);
    par.add_text("fold별 고른 설정", "s021,s005,s021,s002", ch, ch == "s021,s005,s021,s002");
    par.add("54설정 OOF 실패 합", 644, oofsum, 0, 0);
    par.add("재대입 실패 0 설정 수", 0, resub0, 0, 0);
    par.add("s000 OOF 실패", 1, stats(N.oof[0]).fail, 0, 0);
    par.add("s000 OOF Y 최대 오차", 0.266491725088265, stats(N.oof[0]).Ymax, 1e-4, 6);
    par.add("s000 재대입 Y 최대 오차", 0.2666480311105397, stats(N.resub[0]).Ymax, 1e-4, 6);
    par.add("s020 OOF 실패 (사후 최고)", 1, stats(N.oof[20]).fail, 0, 0);
    par.add("s020 OOF Y 최대 오차", 0.2654946910683919, stats(N.oof[20]).Ymax, 1e-4, 6);

    int bk = rep.best_oof();
    Stats bs = stats(N.oof[bk]);
    std::string chosen_k;
    for (auto& s : N.sel) chosen_k += (s.fold ? ", " : "") + std::to_string(cfgs[s.chosen].nknots);
    std::ostringstream one;
    one << "내부 매듭이 없으면 후보마다 3차 다항식 하나라 이차곡선과 거의 같아 OOF 실패 합이 " << by_k[0] << "(18설정)에 그쳤고, 매듭을 2개·4개로 늘리면 후보별 학습 15~75장의 잡음까지 따라가 "
        << by_k[2] << "·" << by_k[4] << "로 커졌으며, 내부 선택은 fold마다 매듭 " << chosen_k << "개 설정을 골라 " << ns.fail << "장(" << fail_names(m, N.nested)
        << ")을 틀렸고 사후 최고 " << cfg_name('s', bk) << "도 " << bs.fail << "장을 틀렸다.";
    rep.oneliner = one.str();
    rep.notported = "- center 630점 위치(QR 디코딩 격자 + erf 경계 모델 맞춤)는 옮기지 않고 #56 실험의 `points630.hpp` 고정 ID 목록을 쓴다.";
    rep.write();
    std::cout << "parity " << par.ok << "/" << par.total << "\n";
    return 0;
}
