// 동시 추정 오차 원인과 개선: 원래 P12 + ExtraTrees + 잡음 증강, 630점 좌표 선형 GLS, 반응 방향 진단
// 원본: codex_experiments_20261014/joint_yt_improved_20261022/ (refine.py = fixedfeature, coordinate_only.py + fullfield.py,
//       experiment.py 의 diagnostics, engine.py 의 Forward·Stabilized). 630점 측정은 track_corners.py 방식(struct630.hpp).
#include "joint_models.hpp"
#include "numeric.hpp"
#include "struct630.hpp"
#include <hye/dataset.hpp>
#include <hye/plot.hpp>
#include <hye/report.hpp>
#include <iostream>

using namespace hye;
using cv::Mat;

namespace {

struct Metrics { int n = 0, passed = 0; double ymae = 0, ymax = 0, tmae = 0, tmax = 0, sign = 0, inact = 0, inact_max = 0; int ypass = 0, tpass = 0; };
Metrics metrics(const Mat& p, const Mat& y) {
    Metrics m;
    m.n = p.rows;
    int na = 0, ni = 0;
    for (int i = 0; i < p.rows; ++i) {
        double eY = std::abs(p.at<double>(i, 0) - y.at<double>(i, 0)), eT = std::abs(p.at<double>(i, 1) - y.at<double>(i, 1));
        m.passed += eY <= .09 && eT <= .09;
        m.ypass += eY <= .09; m.tpass += eT <= .09;
        m.ymae += eY; m.tmae += eT; m.ymax = std::max(m.ymax, eY); m.tmax = std::max(m.tmax, eT);
        for (int k = 0; k < 2; ++k) {
            double t = y.at<double>(i, k), q = p.at<double>(i, k);
            if (t != 0) { ++na; m.sign += ((q > 0) - (q < 0)) == ((t > 0) - (t < 0)); }
            else { ++ni; m.inact += std::abs(q); m.inact_max = std::max(m.inact_max, std::abs(q)); }
        }
    }
    m.ymae /= m.n; m.tmae /= m.n; m.sign /= std::max(na, 1); m.inact = ni ? m.inact / ni : kNaN;
    return m;
}
using Rank = std::tuple<int, double, double, int>;
Rank rank_score(const Mat& p, const Mat& t, int k) {
    int f = 0;
    double mx = 0, s = 0;
    for (int i = 0; i < p.rows; ++i) {
        double a = std::abs(p.at<double>(i, 0) - t.at<double>(i, 0)), b = std::abs(p.at<double>(i, 1) - t.at<double>(i, 1));
        f += a > .09 || b > .09; mx = std::max({mx, a, b}); s += a + b;
    }
    return Rank{f, mx, s / (2.0 * p.rows), k};
}

// experiment.Study.diagnostics: P12 에 [1, Y, T] 직선, 잔차 표준화 공분산(0.999C + 0.001I)의 고유분해 백색화
struct Diag { double cos, angle, cond, ystd, tstd; };
Diag diagnostics(const Mat& X, const Mat& y, const std::vector<int>& tr) {
    const int n = (int)tr.size(), p = X.cols;
    Mat A(n, 3, CV_64F), Xt = num::rows(X, tr);
    for (int i = 0; i < n; ++i) { A.at<double>(i, 0) = 1; A.at<double>(i, 1) = y.at<double>(tr[i], 0); A.at<double>(i, 2) = y.at<double>(tr[i], 1); }
    Mat co = jm::lstsq(A, Xt), r = Xt - A * co;
    std::vector<double> sd(p);
    for (int j = 0; j < p; ++j) {
        double m = cv::mean(r.col(j))[0], s = 0;
        for (int i = 0; i < n; ++i) s += (r.at<double>(i, j) - m) * (r.at<double>(i, j) - m);
        sd[j] = std::max(std::sqrt(s / n), 1e-8);
    }
    Mat z = r.clone();
    for (int j = 0; j < p; ++j) z.col(j) /= sd[j];
    Mat C = 0.999 * (z.t() * z / n) + 0.001 * Mat::eye(p, p, CV_64F), ev, U;
    cv::eigen(C, ev, U);  // 행 = 고유벡터
    Mat W = Mat::zeros(p, p, CV_64F);
    for (int k = 0; k < p; ++k) {
        Mat u = U.row(k).t();
        W += u * u.t() / std::sqrt(std::max(ev.at<double>(k), 1e-8));
    }
    for (int i = 0; i < p; ++i) W.row(i) /= sd[i];
    Mat d = co.rowRange(1, 3) * W;  // 2 x p
    double n0 = cv::norm(d.row(0)), n1 = cv::norm(d.row(1));
    Diag g;
    g.cos = d.row(0).dot(d.row(1)) / (n0 * n1);
    g.angle = std::acos(std::min(1.0, std::max(-1.0, g.cos))) * 180 / M_PI;
    Mat dn = d.t();
    dn.col(0) /= n0;
    dn.col(1) /= n1;
    g.cond = num::cond2(dn);
    Mat cov = (d * d.t()).inv();
    g.ystd = std::sqrt(cov.at<double>(0, 0));
    g.tstd = std::sqrt(cov.at<double>(1, 1));
    return g;
}

// engine.Forward(cov='shrink', quad=False): [1, Y, T] 순방향 + 백색화 최소제곱 (원래 첫 단계 'stage0_common_origin')
Mat forward_linear(const Mat& X, const Mat& y, const std::vector<int>& tr, const Mat& Xte) {
    const int n = (int)tr.size(), p = X.cols;
    Mat A(n, 3, CV_64F), Xt = num::rows(X, tr);
    for (int i = 0; i < n; ++i) { A.at<double>(i, 0) = 1; A.at<double>(i, 1) = y.at<double>(tr[i], 0); A.at<double>(i, 2) = y.at<double>(tr[i], 1); }
    Mat co = jm::lstsq(A, Xt), r = Xt - A * co;
    std::vector<double> sd(p);
    for (int j = 0; j < p; ++j) { Mat c = r.col(j); sd[j] = std::max(std::sqrt(c.dot(c) / n), 1e-8); }
    Mat z = r.clone();
    for (int j = 0; j < p; ++j) z.col(j) /= sd[j];
    Mat C = 0.999 * (z.t() * z / n) + 0.001 * Mat::eye(p, p, CV_64F), ev, U;
    cv::eigen(C, ev, U);
    Mat W = Mat::zeros(p, p, CV_64F);
    for (int k = 0; k < p; ++k) { Mat u = U.row(k).t(); W += u * u.t() / std::sqrt(std::max(ev.at<double>(k), 1e-8)); }
    for (int i = 0; i < p; ++i) W.row(i) /= sd[i];
    Mat Aw = co * W, zz = Xte * W;
    for (int i = 0; i < zz.rows; ++i) zz.row(i) -= Aw.row(0);
    return jm::lstsq(Aw.rowRange(1, 3).t(), zz.t()).t();
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
    begin_results();
    // ---- 630점 측정 -> D (사진 x 630 x 2), P12 ----
    auto shots = load_1014();
    const cv::Mat center = s630::read_gray(center_1014(shots).path);
    auto all = s630::structural_corners(s630::decode_qr(center));
    s630::refine_center(center, all);
    auto cs = s630::non_diagonal(all);
    std::vector<int> order(shots.size());
    std::iota(order.begin(), order.end(), 0);
    auto fname = [&](int i) { auto r = shots[i].rel; return r.substr(r.rfind('/') + 1); };
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        bool ca = fname(a) == "center.bmp", cb = fname(b) == "center.bmp";
        if (ca != cb) return ca;
        return fname(a) < fname(b);
    });
    const int N = (int)order.size(), P = (int)cs.size();
    std::vector<s630::Track> tracks(N);
#pragma omp parallel for schedule(dynamic) num_threads(4)
    for (int k = 0; k < N; ++k) tracks[k] = s630::track(s630::read_gray(shots[order[k]].path), center, cs);
    std::cout << "tracked " << N << "\n";
    std::vector<cv::Point2d> pts(P);
    for (int j = 0; j < P; ++j) pts[j] = {tracks[0].obs[j].x, tracks[0].obs[j].y};
    Mat X12(N, 12, CV_64F), RAW(N, 2 * P, CV_64F), y = Mat::zeros(N, 2, CV_64F);
    std::vector<std::string> names(N), axis(N), session(N);
    std::vector<double> value(N, 0);
    for (int k = 0; k < N; ++k) {
        std::vector<cv::Point2d> d(P);
        for (int j = 0; j < P; ++j) {
            d[j] = {tracks[k].obs[j].x - pts[j].x, tracks[k].obs[j].y - pts[j].y};
            RAW.at<double>(k, 2 * j) = d[j].x;
            RAW.at<double>(k, 2 * j + 1) = d[j].y;
        }
        auto f = s630::p12(pts, d);
        for (int c = 0; c < 12; ++c) X12.at<double>(k, c) = f[c];
        names[k] = fname(order[k]);
        if (names[k] == "center.bmp") { axis[k] = "C"; session[k] = "C"; continue; }
        axis[k] = names[k].substr(0, 1);
        value[k] = std::stod(names[k].substr(1, names[k].size() - 5));
        session[k] = axis[k] == "T" ? "T" : ((std::llabs(std::llround(value[k] / .05)) % 2) ? "Y2" : "Y1");
        session[k] += value[k] > 0 ? "+" : "-";
        y.at<double>(k, axis[k] == "Y" ? 0 : 1) = value[k];
    }
    std::vector<int> ids;
    for (int i = 0; i < N; ++i) if (axis[i] != "C") ids.push_back(i);
    std::vector<std::string> lab;
    for (int i : ids) lab.push_back(session[i]);
    auto tf = num::stratified_test_folds(lab, 4, 20261021);
    std::vector<std::vector<int>> TR(5), TE(4);
    for (size_t k = 0; k < ids.size(); ++k)
        for (int f = 0; f < 4; ++f) (tf[k] == f ? TE[f] : TR[f]).push_back(ids[k]);
    TR[4] = ids;

    // ---- 모델별 OOF ----
    const std::vector<double> alphas{.001, .01, .1, .5, 1.};
    Mat P_tree(0, 2, CV_64F), P_none(0, 2, CV_64F), P_coord(0, 2, CV_64F), P_fwd(0, 2, CV_64F), Ytrue(0, 2, CV_64F);
    Mat S_tree(0, 2, CV_64F), S_coord(0, 2, CV_64F), S_true(0, 2, CV_64F);
    std::vector<int> oof_idx, chosen_alpha(5);
    std::vector<Diag> diags;
    // 원래 fullfield_search.json 의 raw gls_full 내부 점수 (실패 수) [fold][alpha]
    const int orig_fail[5][5] = {{95, 95, 94, 88, 194}, {90, 90, 88, 95, 188}, {72, 72, 71, 69, 196}, {89, 88, 89, 85, 187}, {104, 103, 106, 111, 252}};
    Table aT{{"fold", "alpha", "내부 실패 수 (C++)", "내부 최대 오차", "내부 평균 오차", "원래 내부 실패 수"}};
    for (int f = 0; f < 5; ++f) {
        const auto& tr = TR[f];
        // 좌표 GLS: 학습 안 3-fold(StratifiedKFold(3, shuffle, 1200+fold))로 alpha 선택
        std::vector<std::string> tl;
        for (int i : tr) tl.push_back(session[i]);
        auto itf = num::stratified_test_folds(tl, 3, 1200 + f);
        std::vector<Rank> ranks;
        for (int k = 0; k < (int)alphas.size(); ++k) {
            Mat ip(0, 2, CV_64F), it(0, 2, CV_64F);
            for (int g = 0; g < 3; ++g) {
                std::vector<int> a, b;
                for (size_t q = 0; q < tr.size(); ++q) (itf[q] == g ? b : a).push_back(tr[q]);
                jm::GlsFull m;
                m.alpha = alphas[k];
                m.fit(num::rows(RAW, a), num::rows(y, a));
                ip.push_back(m.predict(num::rows(RAW, b)));
                it.push_back(num::rows(y, b));
            }
            ranks.push_back(rank_score(ip, it, k));
            aT.add({std::to_string(f), fmt(alphas[k], 3), std::to_string(std::get<0>(ranks.back())), fmt(std::get<1>(ranks.back()), 4), fmt(std::get<2>(ranks.back()), 4),
                    std::to_string(orig_fail[f][k])});
        }
        chosen_alpha[f] = std::get<3>(*std::min_element(ranks.begin(), ranks.end()));
        if (f == 4) break;
        const auto& te = TE[f];
        Mat Xtr = num::rows(X12, tr), ytr = num::rows(y, tr), Xte = num::rows(X12, te);
        jm::Forest ft;
        ft.seed = 20261022 + f;
        ft.fit(Xtr, ytr);
        P_tree.push_back(ft.predict(Xte));
        jm::Forest fn;
        fn.seed = 20261022 + f;
        fn.noise = false;
        fn.fit(Xtr, ytr);
        P_none.push_back(fn.predict(Xte));
        jm::GlsFull gc;
        gc.alpha = alphas[chosen_alpha[f]];
        gc.fit(num::rows(RAW, tr), ytr);
        P_coord.push_back(gc.predict(num::rows(RAW, te)));
        P_fwd.push_back(forward_linear(X12, y, tr, Xte));
        Ytrue.push_back(num::rows(y, te));
        for (int i : te) oof_idx.push_back(i);
        diags.push_back(diagnostics(X12, y, tr));
        // 합성 혼합: 시험 Y 사진 x 시험 T 사진, X_Y + X_T - 원점([1,Y,T,Y²,T²] 절편)
        std::vector<int> yi, ti;
        for (int i : te) { if (y.at<double>(i, 0) != 0) yi.push_back(i); if (y.at<double>(i, 1) != 0) ti.push_back(i); }
        for (int which = 0; which < 2; ++which) {
            const Mat& F = which ? RAW : X12;
            Mat A((int)tr.size(), 5, CV_64F);
            for (size_t q = 0; q < tr.size(); ++q) {
                double Y = y.at<double>(tr[q], 0), T = y.at<double>(tr[q], 1);
                double r[5] = {1, Y, T, Y * Y, T * T};
                for (int c = 0; c < 5; ++c) A.at<double>((int)q, c) = r[c];
            }
            Mat origin = jm::lstsq(A, num::rows(F, tr)).row(0);
            Mat Xs(0, F.cols, CV_64F), ts(0, 2, CV_64F);
            for (int a : yi)
                for (int b : ti) {
                    Xs.push_back(Mat(F.row(a) + F.row(b) - origin));
                    if (!which) ts.push_back(Mat(y.row(a) + y.row(b)));
                }
            if (!which) { S_tree.push_back(ft.predict(Xs)); S_true.push_back(ts); }
            else S_coord.push_back(gc.predict(Xs));
        }
        std::cout << "fold " << f << " alpha " << alphas[chosen_alpha[f]] << "\n";
    }
    auto mt = metrics(P_tree, Ytrue), mn = metrics(P_none, Ytrue), mc = metrics(P_coord, Ytrue), mf = metrics(P_fwd, Ytrue);
    auto st = metrics(S_tree, S_true), sc = metrics(S_coord, S_true);
    {
        Table t{{"file", "axis", "true_Y", "true_T", "trees_noise_Y", "trees_noise_T", "coordinate_Y", "coordinate_T", "forward_Y", "forward_T"}};
        for (size_t r = 0; r < oof_idx.size(); ++r) {
            int i = oof_idx[r];
            t.add({names[i], axis[i], fmt(y.at<double>(i, 0), 2), fmt(y.at<double>(i, 1), 2), fmt(P_tree.at<double>((int)r, 0), 6), fmt(P_tree.at<double>((int)r, 1), 6),
                   fmt(P_coord.at<double>((int)r, 0), 6), fmt(P_coord.at<double>((int)r, 1), 6), fmt(P_fwd.at<double>((int)r, 0), 6), fmt(P_fwd.at<double>((int)r, 1), 6)});
        }
        t.csv(out("oof_predictions.csv"));
    }
    Table mT{{"모델", "통과 / 280", "Y 통과", "T 통과", "Y 평균 / 최대 |오차|", "T 평균 / 최대 |오차|", "활성 축 부호 정확도", "비활성 축 평균 / 최대 |값|"}};
    auto addm = [&](const std::string& nm, const Metrics& m) {
        mT.add({nm, std::to_string(m.passed), std::to_string(m.ypass), std::to_string(m.tpass), fmt(m.ymae, 4) + " / " + fmt(m.ymax, 4), fmt(m.tmae, 4) + " / " + fmt(m.tmax, 4),
                fmt(m.sign, 4), fmt(m.inact, 4) + " / " + fmt(m.inact_max, 4)});
    };
    addm("P12 + ExtraTrees + 잡음 증강", mt);
    addm("P12 + ExtraTrees (증강 없음)", mn);
    addm("630점 좌표 선형 GLS", mc);
    addm("P12 선형 GLS 순방향 (첫 단계)", mf);
    Table sT{{"모델", "합성 혼합 통과 / " + std::to_string(st.n), "Y 평균 / 최대", "T 평균 / 최대", "활성 축 부호 정확도"}};
    sT.add({"P12 + ExtraTrees + 잡음", std::to_string(st.passed), fmt(st.ymae, 4) + " / " + fmt(st.ymax, 4), fmt(st.tmae, 4) + " / " + fmt(st.tmax, 4), fmt(st.sign, 4)});
    sT.add({"630점 좌표 선형 GLS", std::to_string(sc.passed), fmt(sc.ymae, 4) + " / " + fmt(sc.ymax, 4), fmt(sc.tmae, 4) + " / " + fmt(sc.tmax, 4), fmt(sc.sign, 4)});
    Table dT{{"fold", "백색화 코사인", "사잇각 (deg)", "정규화 조건수", "잔차 기반 Y 표준편차 (mm)", "잔차 기반 T 표준편차 (deg)", "선택 alpha (좌표 GLS)"}};
    double cmean = 0, amean = 0;
    for (int f = 0; f < 4; ++f) {
        auto& g = diags[f];
        dT.add({std::to_string(f), fmt(g.cos, 5), fmt(g.angle, 3), fmt(g.cond, 3), fmt(g.ystd, 4), fmt(g.tstd, 4), fmt(alphas[chosen_alpha[f]], 3)});
        cmean += g.cos / 4; amean += g.angle / 4;
    }

    // ---- 그림 ----
    std::vector<double> ty, tt, py, pt, ey, et;
    std::vector<std::string> nm;
    for (size_t r = 0; r < oof_idx.size(); ++r) {
        int i = oof_idx[r];
        ty.push_back(y.at<double>(i, 0)); tt.push_back(y.at<double>(i, 1));
        py.push_back(P_tree.at<double>((int)r, 0)); pt.push_back(P_tree.at<double>((int)r, 1));
        ey.push_back(py.back() - ty.back()); et.push_back(pt.back() - tt.back());
        nm.push_back(names[i].substr(0, names[i].size() - 4));
    }
    pred_vs_truth("01_트리증강_OOF오차.png", "P12 + ExtraTrees + noise OOF", ty, py, tt, pt, nm);
    {
        std::vector<double> a, b;
        for (int r = 0; r < P_coord.rows; ++r) { a.push_back(P_coord.at<double>(r, 0)); b.push_back(P_coord.at<double>(r, 1)); }
        pred_vs_truth("02_좌표선형_OOF오차.png", "630-point coordinate GLS OOF", ty, a, tt, b, nm);
    }
    bar_chart("03_모델별_통과수.png", "OOF pass count (of 280)", {"trees+noise", "trees", "coordinate GLS", "P12 forward GLS"},
              {(double)mt.passed, (double)mn.passed, (double)mc.passed, (double)mf.passed}, "passed", 0, 280);
    {
        // 반응 방향: 학습 전체로 P12 [1,Y,T] 기울기를 백색화한 공간에서 Y 1mm, T 1deg 방향의 첫 두 성분
        Chart c("Linear forward error: Y error vs T error (OOF)", "T error (deg)", "Y error (mm)");
        std::vector<double> fy, ft;
        for (int r = 0; r < P_fwd.rows; ++r) { fy.push_back(P_fwd.at<double>(r, 0) - Ytrue.at<double>(r, 0)); ft.push_back(P_fwd.at<double>(r, 1) - Ytrue.at<double>(r, 1)); }
        c.scatter(ft, fy, color(0), "P12 forward GLS").hline(0).vline(0);
        c.save("04_선형오차_Y와T상관.png");
    }
    auto outlier_fig = [&](int idx, const std::string& file, const std::string& cap) {
        cv::Mat im = s630::read_gray(shots[order[idx]].path);
        std::vector<cv::Mat> tiles;
        for (int q = 0; q < 3; ++q) {
            double mx = 0, my = 0;
            int nq = 0;
            for (int j = 0; j < P; ++j) if (cs[j].qr == q + 1) { mx += tracks[idx].obs[j].x - pts[j].x; my += tracks[idx].obs[j].y - pts[j].y; ++nq; }
            mx /= nq; my /= nq;
            auto b = s630::kBoxes[q];
            auto sh = tracks[idx].shift[q];
            cv::Rect roi(b[0] + sh.x - 15, b[1] + sh.y - 15, b[2] - b[0] + 30, b[3] - b[1] + 30);
            std::vector<Mark> mk;
            for (int j = 0; j < P; ++j)
                if (cs[j].qr == q + 1 && j % 3 == 0)
                    mk.push_back({{tracks[idx].obs[j].x, tracks[idx].obs[j].y}, {tracks[idx].obs[j].x - pts[j].x - mx, tracks[idx].obs[j].y - pts[j].y - my}, "", kRed});
            tiles.push_back(zoom(im, roi, 420, mk, 20, q == 0 ? cap : "",
                                 {{cv::Rect2d(b[0], b[1], b[2] - b[0], b[3] - b[1]), kGreen, "ref"},
                                  {cv::Rect2d(b[0] + mx, b[1] + my, b[2] - b[0], b[3] - b[1]), kRed, "QR shift " + fmt(mx, 2) + ", " + fmt(my, 2)}}));
        }
        save_png(file, hstack(tiles));
    };
    size_t wy = worst(ey, 1)[0], wt = worst(et, 1)[0];
    std::string fy = "05_튀는값_Y_" + nm[wy] + ".png", ft = "06_튀는값_T_" + nm[wt] + ".png";
    outlier_fig(oof_idx[wy], fy, nm[wy] + " Y err " + fmts(ey[wy], 3) + " mm");
    outlier_fig(oof_idx[wt], ft, nm[wt] + " T err " + fmts(et[wt], 3) + " deg");
    // 튀는 값 표: 가장 가까운 학습 사진(표준화 P12 거리) 도 보인다 -> 트리는 학습 사진 정답의 평균만 낼 수 있다
    Table outl{{"구분", "사진", "정답 Y", "추정 Y", "정답 T", "추정 T", "P12 c0x (px)", "c2x (px)", "좌표 GLS 추정 (Y / T)", "표준화 P12 가장 가까운 학습 사진"}};
    for (auto [lab, idx] : std::vector<std::pair<std::string, std::vector<size_t>>>{{"Y 오차 상위", worst(ey, 4)}, {"T 오차 상위", worst(et, 3)}})
        for (size_t k : idx) {
            int i = oof_idx[k];
            int f = 0;
            for (int g = 0; g < 4; ++g) if (std::find(TE[g].begin(), TE[g].end(), i) != TE[g].end()) f = g;
            // 학습 fold 의 표준화 거리
            std::vector<double> sdv(12);
            for (int c = 0; c < 12; ++c) {
                double m = 0, s = 0;
                for (int t : TR[f]) m += X12.at<double>(t, c);
                m /= TR[f].size();
                for (int t : TR[f]) s += (X12.at<double>(t, c) - m) * (X12.at<double>(t, c) - m);
                sdv[c] = std::sqrt(s / TR[f].size());
            }
            double bd = 1e300;
            int bj = -1;
            for (int t : TR[f]) {
                double dd = 0;
                for (int c = 0; c < 12; ++c) dd += std::pow((X12.at<double>(t, c) - X12.at<double>(i, c)) / sdv[c], 2);
                if (dd < bd) { bd = dd; bj = t; }
            }
            outl.add({lab, nm[k], fmt(ty[k], 2), fmt(py[k], 3), fmt(tt[k], 2), fmt(pt[k], 3), fmt(X12.at<double>(i, 0), 3), fmt(X12.at<double>(i, 4), 4),
                      fmt(P_coord.at<double>((int)k, 0), 3) + " / " + fmt(P_coord.at<double>((int)k, 1), 3), names[bj].substr(0, names[bj].size() - 4) + " (거리 " + fmt(std::sqrt(bd), 2) + ")"});
        }

    // ---- 원래 결과와 대조 (joint_yt_improved_20261022/fixedfeature_summary.json, coordinate_summary.json, diagnostic_fold*.json) ----
    Parity par;
    par.add("트리+증강 OOF 통과 장수", 251, mt.passed, 8, 0);
    par.add("트리+증강 Y 평균 |오차| (mm)", 0.030531, mt.ymae, 0.006);
    par.add("트리+증강 Y 최대 |오차| (mm)", 0.31, mt.ymax, 0.1);
    par.add("트리+증강 T 평균 |오차| (deg)", 0.008269, mt.tmae, 0.002);
    par.add("트리+증강 T 최대 |오차| (deg)", 0.0685, mt.tmax, 0.03);
    par.add("트리+증강 합성 혼합 통과 (4000)", 198, st.passed, 60, 0);
    par.add("좌표 선형 OOF 통과 장수", 174, mc.passed, 3, 0);
    par.add("좌표 선형 Y 평균 |오차| (mm)", 0.081834, mc.ymae, 0.002);
    par.add("좌표 선형 Y 최대 |오차| (mm)", 0.350433, mc.ymax, 0.02);
    par.add("좌표 선형 T 평균 |오차| (deg)", 0.014836, mc.tmae, 0.001);
    par.add("좌표 선형 T 최대 |오차| (deg)", 0.064705, mc.tmax, 0.005);
    par.add("좌표 선형 합성 혼합 통과 (4000)", 1809, sc.passed, 40, 0);
    {
        const double oa[5] = {0.5, 0.1, 0.5, 0.5, 0.01};
        for (int f = 0; f < 5; ++f) par.add("좌표 GLS 선택 alpha fold " + std::to_string(f) + (f == 4 ? " (전체 280)" : ""), oa[f], alphas[chosen_alpha[f]], 1e-9, 3);
    }
    const double oc[4] = {0.99468, 0.99477, 0.99376, 0.99454}, oy[4] = {0.12937, 0.13191, 0.11793, 0.13058};
    for (int f = 0; f < 4; ++f) {
        par.add("fold " + std::to_string(f) + " Y·T 반응 백색화 코사인", oc[f], diags[f].cos, 0.0005, 5);
        par.add("fold " + std::to_string(f) + " 잔차 기반 Y 표준편차 (mm)", oy[f], diags[f].ystd, 0.005);
    }

    std::ostringstream R;
    R << "# 결과: 동시 추정 오차 원인, 비선형 트리·증강, 좌표 선형\n\n"
      << "입력: `data/clean/clean_20261014` 281장. 고정 630점(track_corners 방식)을 다시 재고 P12(원래 'uniform global2')와 630점 x 2 = 1260 좌표를 특징으로 쓴다.\n"
      << "분할: #68 과 같은 4-fold(세션 층화, 20261021). 모든 수치는 OOF 280장.\n\n"
      << "## OOF 결과\n\n" << mT.md()
      << "\n## 합성 혼합 (시험 fold 의 Y 사진 x T 사진 특징 합 - 학습 원점, " << st.n << "쌍)\n\n" << sT.md()
      << "\n## Y·T 반응 방향 진단 (P12, 학습 fold 마다)\n\n" << dT.md()
      << "\n잔차 기반 표준편차 = 백색화 공간 Fisher 정보 역행렬에서 본 (Y, T) 추정 하한 수준. 반응 방향이 거의 나란해 Y 표준편차가 T보다 약 " << fmt(diags[0].ystd / diags[0].tstd, 1)
      << "배 크다.\n\n"
      << "## 원래 결과와 대조 (" << par.ok << "/" << par.total << " 일치)\n\n" << par.t.md()
      << "\n좌표 GLS alpha 내부 3-fold 점수 (실패 수가 가장 적은 것, 같으면 최대 오차가 작은 것을 고른다):\n\n" << aT.md()
      << "\nfold 3 과 전체 280 의 alpha 는 원래도 후보 사이 내부 실패 수 차이가 1~4장이었고, 다시 잰 좌표로 순위가 바뀌어 불일치로 남긴다.\n"
      << "\n허용오차: 트리 모델은 sklearn 트리 내부 난수와 numpy default_rng 정규 잡음을 C++ 에서 같은 수열로 만들 수 없어 고정 시드 mt19937 로 바꿨다. 그래서 통과 ±8장, 합성 ±60쌍, 최대 오차 ±0.1 mm / ±0.03 deg 를 일치로 본다.\n\n"
      << "## 옮긴 경로와 옮기지 않은 부분\n\n"
      << "- 옮김: refine.py(fixedfeature)의 최종 경로. 원래는 학습 fold 안 3-fold로 회귀 28종(C)·안정화 8종(D)을 골랐고 5번 모두 'trees leaf 1' + 'noise'가 뽑혔다. 여기서는 그 설정을 고정해 fold마다 학습했다.\n"
      << "- 옮김: coordinate_only.py 의 최종 경로 'raw 1260 좌표 + gls_full'. 원래 후보 76개(grid2/4/8/raw x ridge/gls/pcr/pls) 중 5번 모두 raw gls_full 이 뽑혀서, alpha 5개 안에서만 같은 내부 3-fold(1200+fold)로 골랐다.\n"
      << "- 옮기지 않음: experiment.py 의 4단계 순차 탐색(측정 가중 7 x 특징 7 x 회귀 28 x 안정화 8, 원래 210/280), Bayes·ARD·PLS·커널·SVR 회귀, grid 특징·ridge/pcr/pls, 특징 잡음 내성·linear_audit·verify. 모델 joblib 저장.\n"
      << "\n## 파일\n\n- `oof_predictions.csv`: 사진별 OOF 추정(트리+증강, 좌표 GLS, P12 선형)\n- `이미지/`: 그림\n";
    write_text(out("결과.md"), R.str());

    std::ostringstream Pp;
    Pp << "# 보고서: 동시 추정 오차 원인, 비선형 트리·증강 (10-22 15:29)\n\n"
       << "## 문제\n\n첫 동시 추정에서 오차가 커진 원인을 찾고 줄여야 했다.\n\n"
       << "## 해결 방법\n\n원래 P12 특징에 비선형 트리 회귀(ExtraTrees)와 잡음 증강을, 630점 좌표에 선형 GLS를 맞춰 같은 4-fold로 비교하고 Y·T 반응 방향을 진단했다.\n\n"
       << "## 결과\n\n비선형 트리·증강은 " << mt.passed << "/280, 630점 좌표 선형은 " << mc.passed << "/280 통과였고, 원인은 백색화한 Y와 T 반응의 사잇각이 평균 " << fmt(amean, 1)
       << "도로 거의 나란한 것이었다.\n\n"
       << "| 지표 | 트리 + 증강 | 좌표 선형 GLS | P12 선형 GLS |\n|---|---|---|---|\n"
       << "| 통과 / 280 | " << mt.passed << " | " << mc.passed << " | " << mf.passed << " |\n"
       << "| Y 평균 / 최대 \\|오차\\| (mm) | " << fmt(mt.ymae, 4) << " / " << fmt(mt.ymax, 4) << " | " << fmt(mc.ymae, 4) << " / " << fmt(mc.ymax, 4) << " | " << fmt(mf.ymae, 4) << " / " << fmt(mf.ymax, 4) << " |\n"
       << "| T 평균 / 최대 \\|오차\\| (deg) | " << fmt(mt.tmae, 4) << " / " << fmt(mt.tmax, 4) << " | " << fmt(mc.tmae, 4) << " / " << fmt(mc.tmax, 4) << " | " << fmt(mf.tmae, 4) << " / " << fmt(mf.tmax, 4) << " |\n"
       << "| 비활성 축 평균 \\|값\\| | " << fmt(mt.inact, 4) << " | " << fmt(mc.inact, 4) << " | " << fmt(mf.inact, 4) << " |\n"
       << "| 합성 혼합 통과 / " << st.n << " | " << st.passed << " | " << sc.passed << " | - |\n"
       << "| 증강 없는 트리 통과 | " << mn.passed << " | - | - |\n\n"
       << img_md("01_트리증강_OOF오차.png", "트리 오차") << "\n\n" << img_md("02_좌표선형_OOF오차.png", "좌표 선형 오차") << "\n\n" << img_md("03_모델별_통과수.png", "모델별 통과")
       << "\n\n" << img_md("04_선형오차_Y와T상관.png", "선형 오차 상관") << "\n\n"
       << "## 튀는 값\n\n트리 모델이 가장 크게 틀린 사진은 Y는 " << nm[wy] << "(정답 Y " << fmt(ty[wy], 2) << ", 추정 " << fmt(py[wy], 3) << " mm), T는 " << nm[wt] << "(정답 T " << fmt(tt[wt], 2)
       << ", 추정 " << fmt(pt[wt], 3) << ")이고, 트리는 학습 사진 정답들의 평균만 낼 수 있어 특징이 비슷한 다른 축 사진의 정답을 섞어 냈다.\n\n"
       << outl.md() << "\n" << img_md(fy, "Y 튀는 값") << "\n\n" << img_md(ft, "T 튀는 값")
       << "\n\n초록 상자는 기준 사진의 QR 영역, 빨간 상자는 그 QR 점들의 평균 이동만큼 옮긴 영역, 빨간 십자는 맞춘 모서리(3개 중 1개), 화살표는 점 변위 - QR 평균 변위를 20배 키운 것이다.\n\n"
       << "## 한줄평\n\nY 1mm와 T 1deg 가 만드는 특징 변화의 백색화 코사인이 " << fmt(cmean, 4) << "라 선형으로 두 축을 동시에 풀면 Y 불확도가 T의 약 " << fmt(diags[0].ystd / diags[0].tstd, 0)
       << "배가 되고, 단일 축 사진만 외우는 트리는 실제 사진 OOF 는 " << mt.passed << "장 통과하지만 합성 혼합은 " << st.passed << "/" << st.n << "만 통과해 축 선택을 암묵적으로 배운 것이다.\n\n"
       << "## 원래 결과와 대조\n\n원래 값은 `codex_experiments_20261014/joint_yt_improved_20261022/fixedfeature_summary.json`, `coordinate_summary.json`, `diagnostic_fold0~3.json`에서 가져왔다. "
       << par.ok << "/" << par.total << " 일치. sklearn ExtraTreesRegressor 는 수식대로 다시 만들었지만 난수열(트리 내부, numpy default_rng 잡음)이 달라 사진별 추정은 같지 않고 통과 장수·오차 분포만 비교한다(허용오차는 결과.md). "
       << "좌표 GLS 의 alpha 는 fold 3 과 전체 280 에서 원래와 다르게 뽑혔다. 원래도 후보 사이 내부 실패 수 차이가 1~4장인 근소한 순위라 다시 잰 좌표의 작은 차이로 바뀌었다(결과.md 표).\n\n"
       << par.t.md();
    write_text(out("보고서.md"), Pp.str());
    std::cout << "trees " << mt.passed << " none " << mn.passed << " coord " << mc.passed << " fwd " << mf.passed << " synth " << st.passed << "/" << sc.passed << " parity " << par.ok << "/" << par.total << "\n";
    return 0;
}
