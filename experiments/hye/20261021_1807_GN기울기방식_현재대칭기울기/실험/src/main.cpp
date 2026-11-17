// forward GN: 현재 영상 spline 기울기 vs 기준·현재 대칭 기울기, 창 17/31 → 630점 P12 → 64설정 4-fold 중첩 선택
// 원본: codex_experiments_20261014/forward_gn_20261021/measure_forward.py,
//       multiref_4fold_20261021/experiment.py --forward, zero_experiment.py, measurement/spread.py
#include <hye/dataset.hpp>
#include <hye/plot.hpp>
#include <hye/report.hpp>
#include <omp.h>
#include <iostream>
#include <map>
#include <set>
#include "corners.hpp"
#include "gls.hpp"
#include "track.hpp"

using namespace hye;
using g09::Item;
using g09::Row;
using g09::Vec;

namespace {

struct Variant {
    int window;
    bool symmetric;
    std::string key() const { return "forward" + std::to_string(window) + "_symmetric" + (symmetric ? "True" : "False"); }
};

struct Config {
    int feature;
    std::string mode;
    bool quad;
    double shrink;
    std::string name;
};

std::string base_name(const std::string& rel) {
    auto p = rel.find_last_of('/');
    return p == std::string::npos ? rel : rel.substr(p + 1);
}

}  // namespace

int main() {
    omp_set_num_threads(4);
    begin_results();
    // ---- 사진 목록: center 먼저, 나머지는 파일 이름 순 (center_patch_v2 tracks.npz 순서) ----
    auto all = load_1014();
    std::vector<Shot> shots(all.begin(), all.end());
    std::stable_sort(shots.begin(), shots.end(), [](const Shot& a, const Shot& b) {
        std::string x = base_name(a.rel), y = base_name(b.rel);
        bool ca = x != "center.bmp", cb = y != "center.bmp";
        return ca != cb ? ca < cb : x < y;
    });
    std::vector<Item> m;
    std::vector<std::string> names;
    for (auto& s : shots) { names.push_back(base_name(s.rel)); m.push_back(g09::make_item(names.back())); }
    const int N = (int)shots.size();

    // ---- 630 기준점 ----
    cv::Mat center = read_gray(shots[0]);
    g09::Reference R = g09::build_reference(center);
    std::vector<cv::Point2d> pts;
    std::vector<int> regs;
    for (auto& c : R.c630) {
        pts.push_back({c.px, c.py});
        double cx[3] = {1644, 1901, 2159};
        int k = 0;
        for (int j = 1; j < 3; ++j) if (std::abs(c.px - cx[j]) < std::abs(c.px - cx[k])) k = j;
        regs.push_back(k);
    }
    const int NP = (int)pts.size();

    // ---- 4가지 추적 ----
    std::vector<Variant> variants = {{17, false}, {17, true}, {31, false}, {31, true}};  // 파일 이름 정렬 순
    std::vector<std::unique_ptr<g09::SpreadExtractor>> ex;
    for (auto& v : variants) {
        ex.emplace_back(new g09::SpreadExtractor());
        ex.back()->init(center, pts, regs, 1, v.window, v.symmetric);
    }
    const int NV = (int)variants.size();
    std::vector<std::vector<std::vector<cv::Point2d>>> D(NV, std::vector<std::vector<cv::Point2d>>(N));
    std::vector<std::vector<std::vector<bool>>> V(NV, std::vector<std::vector<bool>>(N));
    std::vector<std::vector<std::vector<double>>> rmse(NV, std::vector<std::vector<double>>(N));
#pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < N; ++i) {
        cv::Mat img = read_gray(shots[i]);
        for (int v = 0; v < NV; ++v) {
            g09::TrackResult r = ex[v]->extract(img);
            D[v][i] = r.D;
            V[v][i] = r.valid;
            for (auto& q : r.quality) rmse[v][i].push_back(q[0]);
        }
    }
    for (int v = 0; v < NV; ++v)
        for (int i = N - 1; i >= 0; --i)
            for (int p = 0; p < NP; ++p) D[v][i][p] -= D[v][0][p];
    g09::P12 p12;
    p12.init(pts);
    std::vector<std::vector<Vec>> X(NV, std::vector<Vec>(N));
    std::vector<double> valid_frac(NV);
    for (int v = 0; v < NV; ++v) {
        long ok = 0;
        for (int i = 0; i < N; ++i) {
            X[v][i] = p12.apply(D[v][i]);
            for (bool b : V[v][i]) ok += b;
        }
        valid_frac[v] = (double)ok / ((double)N * NP);
    }

    // ---- 64 설정 ----
    std::vector<Config> configs;
    for (int f = 0; f < NV; ++f)
        for (std::string mode : {"center", "inferred_session"})
            for (bool q : {false, true})
                for (double s : {.001, .05, .2, 1.}) {
                    char b[8];
                    std::snprintf(b, sizeof b, "c%02d", (int)configs.size());
                    configs.push_back({f, mode, q, s, b});
                }
    const int NC = (int)configs.size();
    std::vector<int> ids;
    for (int i = 0; i < N; ++i) if (m[i].axis != 'C') ids.push_back(i);
    std::vector<std::string> sess;
    for (int i : ids) sess.push_back(m[i].session);
    std::vector<int> outer = g09::stratified_kfold(sess, 4, 20261021);

    // 정답 가지(세션 또는 축)의 비용: 튀는 값 진단용
    auto true_cost = [&](const g09::Model& M, const g09::Pred& p, int i, const std::string& mode) {
        std::string key = mode == "inferred_session" ? m[i].session : std::string(1, m[i].axis);
        for (size_t b = 0; b < M.branches.size(); ++b) if (M.branches[b].name == key) return p.costs[b];
        return kNaN;
    };
    struct Ext { Row r; double tcost; std::string config; };
    std::vector<std::vector<Ext>> oof(NC);
    std::vector<Ext> nested;
    std::vector<std::string> selected;
    Table rank_t{{"fold", "선택 설정", "내부 실패", "내부 최대 오차", "2위 설정", "2위 최대 오차"}};
    std::vector<int> fold_sum(4, 0);
    for (int f = 0; f < 4; ++f) {
        std::vector<int> tr, te;
        for (size_t k = 0; k < ids.size(); ++k) (outer[k] == f ? te : tr).push_back(ids[k]);
        for (int i : te) fold_sum[f] += i;
        std::vector<std::string> ts;
        for (int i : tr) ts.push_back(m[i].session);
        std::vector<int> inner = g09::stratified_kfold(ts, 3, 100 + f);
        std::vector<std::tuple<int, double, int>> rank(NC);
        std::vector<std::vector<Ext>> outer_rows(NC);
#pragma omp parallel for schedule(dynamic, 1)
        for (int c = 0; c < NC; ++c) {
            const Config& C = configs[c];
            std::vector<Row> ip;
            for (int g = 0; g < 3; ++g) {
                std::vector<int> a, b;
                for (size_t k = 0; k < tr.size(); ++k) (inner[k] == g ? b : a).push_back(tr[k]);
                g09::Model M = g09::fit_multiref(m, X[C.feature], a, C.mode, C.quad, C.shrink);
                for (int i : b) ip.push_back(g09::enrich(m, i, g09::predict_one(M, X[C.feature][i], C.mode)));
            }
            g09::Stats s = g09::stats(ip);
            rank[c] = {s.failures, std::max(s.Y_max, s.T_max), c};
            g09::Model M = g09::fit_multiref(m, X[C.feature], tr, C.mode, C.quad, C.shrink);
            for (int i : te) {
                g09::Pred p = g09::predict_one(M, X[C.feature][i], C.mode);
                Ext e{g09::enrich(m, i, p), true_cost(M, p, i, C.mode), C.name};
                e.r.fold = f;
                outer_rows[c].push_back(e);
            }
        }
        for (int c = 0; c < NC; ++c) oof[c].insert(oof[c].end(), outer_rows[c].begin(), outer_rows[c].end());
        auto sorted = rank;
        std::sort(sorted.begin(), sorted.end());
        int chosen = std::get<2>(sorted[0]);
        selected.push_back(configs[chosen].name);
        nested.insert(nested.end(), outer_rows[chosen].begin(), outer_rows[chosen].end());
        rank_t.add({std::to_string(f), configs[chosen].name, std::to_string(std::get<0>(sorted[0])), fmt(std::get<1>(sorted[0]), 6),
                    configs[std::get<2>(sorted[1])].name, fmt(std::get<1>(sorted[1]), 6)});
    }
    // 재대입
    std::vector<std::vector<Row>> resub(NC);
#pragma omp parallel for schedule(dynamic, 1)
    for (int c = 0; c < NC; ++c) {
        const Config& C = configs[c];
        g09::Model M = g09::fit_multiref(m, X[C.feature], ids, C.mode, C.quad, C.shrink);
        for (int i : ids) resub[c].push_back(g09::enrich(m, i, g09::predict_one(M, X[C.feature][i], C.mode)));
    }
    auto rows_of = [](const std::vector<Ext>& e) { std::vector<Row> r; for (auto& x : e) r.push_back(x.r); return r; };
    std::vector<g09::Stats> so(NC), sr(NC);
    for (int c = 0; c < NC; ++c) { so[c] = g09::stats(rows_of(oof[c])); sr[c] = g09::stats(resub[c]); }
    g09::Stats sn = g09::stats(rows_of(nested));

    // ---- 표 ----
    Table summ{{"설정", "추적", "원점", "이차", "축소", "OOF Y 맞음", "OOF T 맞음", "OOF 실패", "OOF Y 최대", "OOF T 최대", "재대입 실패", "재대입 Y 최대"}};
    Table csv{{"config", "feature", "mode", "quadratic", "shrinkage", "oof_Y_correct", "oof_T_correct", "oof_failures", "oof_Y_max", "oof_T_max",
               "resub_failures", "resub_Y_max", "resub_T_max"}};
    for (int c = 0; c < NC; ++c) {
        const Config& C = configs[c];
        std::string mode = C.mode == "center" ? "center 1" : "session 6";
        summ.add({C.name, variants[C.feature].key(), mode, C.quad ? "O" : "X", fmt(C.shrink, 3), std::to_string(so[c].Y_correct) + "/80",
                  std::to_string(so[c].T_correct) + "/200", std::to_string(so[c].failures), fmt(so[c].Y_max, 6), fmt(so[c].T_max, 6),
                  std::to_string(sr[c].failures), fmt(sr[c].Y_max, 6)});
        csv.add({C.name, variants[C.feature].key(), C.mode, C.quad ? "True" : "False", fmt(C.shrink, 3), std::to_string(so[c].Y_correct),
                 std::to_string(so[c].T_correct), std::to_string(so[c].failures), fmt(so[c].Y_max, 8), fmt(so[c].T_max, 8), std::to_string(sr[c].failures),
                 fmt(sr[c].Y_max, 8), fmt(sr[c].T_max, 8)});
    }
    csv.csv(out("config_summary.csv"));
    // 추적 방식 x 원점 모드 최선
    Table best_t{{"추적", "원점", "최선 설정", "OOF 실패", "OOF Y 최대 (mm)", "재대입 최선 실패"}};
    std::vector<std::string> bar_names;
    std::vector<double> bar_vals;
    std::map<std::string, std::pair<std::string, g09::Stats>> best_feat;
    std::map<std::string, std::pair<std::string, g09::Stats>> best_feat_resub;
    double mode_mean[2] = {0, 0};
    for (int v = 0; v < NV; ++v)
        for (int md = 0; md < 2; ++md) {
            int bc = -1, br = -1;
            for (int c = 0; c < NC; ++c) {
                if (configs[c].feature != v || (configs[c].mode == "center") != (md == 0)) continue;
                mode_mean[md] += so[c].failures / 32.0;
                if (bc < 0 || std::make_pair(so[c].failures, so[c].Y_max) < std::make_pair(so[bc].failures, so[bc].Y_max)) bc = c;
                if (br < 0 || std::make_pair(sr[c].failures, sr[c].Y_max) < std::make_pair(sr[br].failures, sr[br].Y_max)) br = c;
            }
            best_t.add({variants[v].key(), md ? "session 6" : "center 1", configs[bc].name, std::to_string(so[bc].failures), fmt(so[bc].Y_max, 6),
                        std::to_string(sr[br].failures)});
            bar_names.push_back("w" + std::to_string(variants[v].window) + (variants[v].symmetric ? " sym" : " cur") + (md ? " 6org" : " 1org"));
            bar_vals.push_back(so[bc].failures);
        }
    for (int v = 0; v < NV; ++v) {
        int bc = -1, br = -1;
        for (int c = 0; c < NC; ++c) {
            if (configs[c].feature != v) continue;
            if (bc < 0 || std::make_pair(so[c].failures, so[c].Y_max) < std::make_pair(so[bc].failures, so[bc].Y_max)) bc = c;
            if (br < 0 || std::make_pair(sr[c].failures, sr[c].Y_max) < std::make_pair(sr[br].failures, sr[br].Y_max)) br = c;
        }
        best_feat[variants[v].key()] = {configs[bc].name, so[bc]};
        best_feat_resub[variants[v].key()] = {configs[br].name, sr[br]};
    }

    // 추적 방식 간 측정 차이 (630점 평균 |D 차이|, 기준 = forward17 현재 기울기)
    Table diff_t{{"비교", "사진 평균 (px)", "사진 최대 (px)", "P12 c0x 최대 차이 (px)"}};
    std::vector<std::vector<double>> dmean(NV, std::vector<double>(N, 0));
    for (int v = 1; v < NV; ++v) {
        double mx = 0, mxx = 0, av = 0;
        for (int i = 1; i < N; ++i) {
            double s = 0;
            for (int p = 0; p < NP; ++p) s += std::hypot(D[v][i][p].x - D[0][i][p].x, D[v][i][p].y - D[0][i][p].y);
            dmean[v][i] = s / NP;
            av += dmean[v][i] / (N - 1);
            mx = std::max(mx, dmean[v][i]);
            mxx = std::max(mxx, std::abs(X[v][i][0] - X[0][i][0]));
        }
        diff_t.add({variants[v].key() + " - " + variants[0].key(), fmt(av, 5), fmt(mx, 5), fmt(mxx, 5)});
    }

    // ---- 튀는 값 ----
    std::vector<double> ty, py, tt, pt, ey, et;
    std::vector<std::string> nn;
    for (auto& e : nested) {
        auto& r = e.r;
        ty.push_back(r.axis == 'Y' ? r.value : 0);
        tt.push_back(r.axis == 'T' ? r.value : 0);
        py.push_back(r.pred_axis == 'Y' ? r.pred : 0);
        pt.push_back(r.pred_axis == 'T' ? r.pred : 0);
        ey.push_back(py.back() - ty.back());
        et.push_back(pt.back() - tt.back());
        nn.push_back(r.file);
    }
    size_t wy = worst(ey, 1)[0], wt = worst(et, 1)[0];
    Table outl{{"구분", "사진", "fold", "선택 설정", "정답 Y (mm)", "추정 Y", "정답 T (deg)", "추정 T", "고른 가지", "고른 가지 비용", "정답 가지 비용",
                "P12 c0x 평균 x 이동 (px)", "P12 c0y (px)"}};
    auto add_out = [&](const std::string& lab, size_t k) {
        auto& e = nested[k];
        int c = 0;
        for (int j = 0; j < NC; ++j) if (configs[j].name == e.config) c = j;
        const Vec& x = X[configs[c].feature][e.r.index];
        outl.add({lab, e.r.file, std::to_string(e.r.fold), e.config, fmt(ty[k], 2), fmt(py[k], 4), fmt(tt[k], 2), fmt(pt[k], 4), e.r.branch,
                  fmt(e.r.cost, 2), fmt(e.tcost, 2), fmt(x[0], 3), fmt(x[1], 3)});
    };
    for (size_t k : worst(ey, 3)) add_out("Y 오차 상위", k);
    for (size_t k : worst(et, 3)) add_out("T 오차 상위", k);
    std::vector<std::string> fails;
    for (auto& e : nested) if (!e.r.pass) fails.push_back(e.r.file);

    // ---- 그림 ----
    bar_chart("01_추적방식_원점모드별_OOF실패.png", "Best OOF failures (280) by tracking x origin mode", bar_names, bar_vals, "failures", -1);
    pred_vs_truth("02_내부선택OOF_예측오차.png", "Nested OOF 280", ty, py, tt, pt, nn);
    {
        Chart c("Tracking difference vs forward17 current-gradient", "image index (center first, sorted by name)", "mean |D diff| over 630 pts (px)");
        for (int v = 1; v < NV; ++v) {
            std::vector<double> x, y;
            for (int i = 1; i < N; ++i) { x.push_back(i); y.push_back(dmean[v][i]); }
            c.scatter(x, y, color(v), variants[v].key(), 2);
        }
        c.save("03_추적방식간_측정차이.png");
    }
    // 사진 확대: 세 QR, 초록 = center 기준점, 빨강 = 이 사진 측정점 (이동량에서 QR 평균 이동을 뺀 나머지를 x200 화살표)
    auto outlier_fig = [&](size_t k, const std::string& file, const std::string& cap) {
        auto& e = nested[k];
        int c = 0;
        for (int j = 0; j < NC; ++j) if (configs[j].name == e.config) c = j;
        int v = configs[c].feature, i = e.r.index;
        cv::Mat im = read_color(shots[i]);
        std::vector<cv::Mat> tiles;
        for (int q = 0; q < 3; ++q) {
            double mx = 0, my = 0;
            int n = 0;
            for (int p = 0; p < NP; ++p) if (regs[p] == q) { mx += D[v][i][p].x; my += D[v][i][p].y; ++n; }
            mx /= n; my /= n;
            std::vector<Mark> mk;
            int cnt = 0;
            for (int p = 0; p < NP; ++p) {
                if (regs[p] != q || (cnt++ % 6)) continue;
                mk.push_back({{pts[p].x + D[v][i][p].x, pts[p].y + D[v][i][p].y}, {D[v][i][p].x - mx, D[v][i][p].y - my}, "", kRed});
            }
            const int* b = g09::BOXES[q];
            cv::Rect2d ref(b[0], b[1], b[2] - b[0], b[3] - b[1]);
            cv::Rect2d now(ref.x + mx, ref.y + my, ref.width, ref.height);
            cv::Rect roi = cv::Rect(ref | now) + cv::Size(40, 40) - cv::Point(20, 20);
            tiles.push_back(zoom(im, roi, 420, mk, 200, q == 0 ? cap : "",
                                 {{ref, kGreen, ""}, {now, kRed, "q" + std::to_string(q) + " dx " + fmt(mx, 2) + " dy " + fmt(my, 2)}}));
        }
        save_png(file, hstack(tiles));
    };
    // T 그림은 Y 그림과 다른 실패 사진이 있으면 그것을 쓴다 (같은 사진이면 정보가 겹친다)
    if (wt == wy)
        for (size_t k : worst(et, et.size()))
            if (k != wy && !nested[k].r.pass) { wt = k; break; }
    auto stem = [](std::string s) { return s.size() > 4 && s.substr(s.size() - 4) == ".bmp" ? s.substr(0, s.size() - 4) : s; };
    std::string fy = "04_튀는값_Y_" + stem(nn[wy]) + ".png", ft = "05_튀는값_T_" + stem(nn[wt]) + ".png";
    outlier_fig(wy, fy, nn[wy] + " Y err " + fmts(ey[wy], 3) + " mm");
    outlier_fig(wt, ft, nn[wt] + " T err " + fmts(et[wt], 3) + " deg");

    // ---- 원래 결과와 대조 ----
    // 원래 값 출처: forward_gn_20261021/audit.json, evaluation/summary.csv, evaluation/nested_selection.json,
    //             evaluation/splits.json (시험 사진 색인 합), forward*_features.npy (T+1.00, Y+1.00 행)
    Parity par;
    for (int v = 0; v < NV; ++v) par.add(variants[v].key() + " 유효점 비율", 1.0, valid_frac[v], 0.0, 4);
    const double c0T[4] = {79.3507578, 79.3507183, 79.3495299, 79.3495473};
    int iT = 0, iY = 0;
    for (int i = 0; i < N; ++i) { if (names[i] == "T+1.00.bmp") iT = i; if (names[i] == "Y+1.00.bmp") iY = i; }
    for (int v = 0; v < NV; ++v) par.add(variants[v].key() + " T+1.00 P12 c0x (px)", c0T[v], X[v][iT][0], 0.002, 4);
    par.add("forward17_symmetricFalse Y+1.00 P12 c0x (px)", 13.83635022, X[0][iY][0], 0.002, 4);
    const int fs0[4] = {9784, 9627, 9943, 9986};
    for (int f = 0; f < 4; ++f) par.add("fold " + std::to_string(f) + " 시험 사진 색인 합", fs0[f], fold_sum[f], 0, 0);
    par.add("내부 선택 OOF Y 맞음 (/80)", 80, sn.Y_correct, 0, 0);
    par.add("내부 선택 OOF T 맞음 (/200)", 198, sn.T_correct, 0, 0);
    par.add("내부 선택 OOF 실패 (/280)", 2, sn.failures, 0, 0);
    par.add("내부 선택 OOF Y 최대 (mm)", 0.266138, sn.Y_max, 0.001, 6);
    par.add("내부 선택 OOF T 최대 (deg)", 0.05, sn.T_max, 0.0005, 4);
    {
        std::string s0 = "c63, c40, c60, c40", s1;
        for (size_t k = 0; k < selected.size(); ++k) s1 += (k ? ", " : "") + selected[k];
        par.add_text("fold별 선택 설정", s0, s1, s0 == s1);
        std::string f1;
        for (size_t k = 0; k < fails.size(); ++k) f1 += (k ? ", " : "") + fails[k];
        std::set<std::string> a{"T-0.05.bmp", "T+0.01.bmp"}, b(fails.begin(), fails.end());
        par.add_text("내부 선택 OOF 실패 사진", "T-0.05.bmp, T+0.01.bmp", f1, a == b);
    }
    const char* bk[4] = {"forward17_symmetricFalse", "forward17_symmetricTrue", "forward31_symmetricFalse", "forward31_symmetricTrue"};
    const char* bcfg[4] = {"c15", "c31", "c47", "c63"};
    const double bym[4] = {0.2661241, 0.2661219, 0.2661407, 0.2661384};
    const char* rcfg[4] = {"c08", "c24", "c40", "c56"};
    const double rym[4] = {0.2644421, 0.2644269, 0.2649804, 0.2651608};
    for (int v = 0; v < 4; ++v) {
        auto& b = best_feat[bk[v]];
        par.add(std::string(bk[v]) + " OOF 최선 실패", 1, b.second.failures, 0, 0);
        par.add(std::string(bk[v]) + " OOF 최선 Y 최대 (mm)", bym[v], b.second.Y_max, 0.001, 6);
        par.add_text(std::string(bk[v]) + " OOF 최선 설정", bcfg[v], b.first, b.first == bcfg[v]);
        auto& r = best_feat_resub[bk[v]];
        par.add(std::string(bk[v]) + " 재대입 최선 Y 최대 (mm)", rym[v], r.second.Y_max, 0.001, 6);
        par.add_text(std::string(bk[v]) + " 재대입 최선 설정", rcfg[v], r.first, r.first == rcfg[v]);
    }
    par.add("OOF 평균 실패, center 1원점 32설정", 5.78125, mode_mean[0], 0.2, 3);
    par.add("OOF 평균 실패, session 6원점 32설정", 1.5625, mode_mean[1], 0.2, 3);

    // 630점 기준 좌표 품질: 원래 center 좌표와 비교 못 하므로(입력 금지) 구조만 적는다
    Table qr_t{{"QR", "해독 문자", "모듈", "검출 방법", "사각형 꼭짓점 (px)"}};
    for (size_t k = 0; k < R.qr.size(); ++k) {
        auto& q = R.qr[k];
        std::string how = q.margin == 0 ? "crop 전체 다중 검출" : "상자 여백 " + std::to_string(q.margin) + ", 배율 " + std::to_string(q.scale);
        std::string corners;
        for (int j = 0; j < 4; ++j) corners += (j ? " / " : "") + fmt(q.quad[j].x, 2) + "," + fmt(q.quad[j].y, 2);
        qr_t.add({std::to_string(k + 1), q.payload, std::to_string(q.modules), how, corners});
    }
    int ref_invalid = 0;
    for (auto& c : R.c630) ref_invalid += !c.ref_valid;

    // ---- 결과.md ----
    std::ostringstream Rm;
    Rm << "# 결과: forward GN 기울기 방식 (현재 / 대칭) x 창 17·31\n\n"
       << "입력: `data/clean/clean_20261014` 281장 (center 1, T 200, Y 80). 순서는 center 다음 파일 이름 순이다.\n"
       << "측정: center.bmp에서 QR 세 개를 해독해 격자 꼭짓점 755개를 만들고, 대각 접촉을 뺀 " << NP << "점을 erf 경계 모델로 맞춰 기준점으로 썼다 (진단 불통과 "
       << ref_invalid << "점도 그대로 쓴다). 사진마다 QR 전체 NCC(±110x22 px) → LK 창 15 → forward GN(창 17/31, 최대 20회)으로 추적했다.\n"
       << "특징: 사진별 이동 D(현재 - center)를 QR 좌표의 이차식 6항으로 회귀한 12개 계수(P12).\n"
       << "평가: 280장 StratifiedKFold(4, seed 20261021). fold마다 학습 210장 안에서 3-fold(seed 100+fold)로 64설정을 비교해 실패 수 → 최대 오차 순으로 하나 골랐다.\n\n"
       << "## QR 해독\n\n" << qr_t.md()
       << "\n## 추적 방식 간 측정 차이\n\n" << diff_t.md()
       << "\n## 추적 x 원점 모드별 최선 설정\n\n" << best_t.md()
       << "\n## fold별 내부 선택\n\n" << rank_t.md()
       << "\n## 64설정 전체 (OOF = 4-fold 시험 280장, 재대입 = 280장 학습 후 같은 280장)\n\n" << summ.md()
       << "\n## 원래 결과와 대조 (" << par.ok << "/" << par.total << " 일치)\n\n" << par.t.md()
       << "\n## 옮기지 않은 부분\n\n"
       << "- `numerical_checks.json`의 합성 이동 검사(numpy 난수 96x96 무늬, mode='reflect' 이동)는 numpy 난수열과 reflect 경계 spline을 옮기지 않아 빠졌다.\n"
       << "- 원래 실행 시간(audit.json seconds)은 비교하지 않는다.\n\n"
       << "## 파일\n\n- `config_summary.csv`: 64설정 OOF·재대입 성적\n- `이미지/`: 보고서 그림\n";
    write_text(out("결과.md"), Rm.str());

    // ---- 보고서.md ----
    int bestmin = 1 << 30;
    for (int c = 0; c < NC; ++c) bestmin = std::min(bestmin, so[c].failures);
    std::ostringstream P;
    P << "# 보고서: GN 기울기 방식 비교, 현재 영상 vs 대칭 기울기 (10-21 18:07)\n\n"
      << "## 문제\n\nGN 기울기를 현재 영상이나 대칭으로 구하면 측정과 판정이 나아지는지 확인해야 했다.\n\n"
      << "## 해결 방법\n\nforward GN(창 17/31, 대칭 여부 4가지)으로 630점을 다시 추적하고 P12 특징 위에서 64설정을 4-fold 중첩 선택으로 평가했다.\n\n"
      << "## 결과\n\n내부 선택 OOF는 Y " << sn.Y_correct << "/80, T " << sn.T_correct << "/200(실패 " << sn.failures
      << ")였고, 네 추적 방식 모두 OOF 최선 실패가 " << bestmin << "장으로 같았다.\n\n"
      << "| 지표 | 값 |\n|---|---|\n"
      << "| 평가 장수 | 280 |\n"
      << "| 내부 선택 OOF 통과 | " << 280 - sn.failures << " / 280 |\n"
      << "| 축·부호 맞음 Y / T | " << sn.Y_correct << "/80 · " << sn.T_correct << "/200 |\n"
      << "| 최대 \\|오차\\| Y (mm) / T (deg) | " << fmt(sn.Y_max, 4) << " / " << fmt(sn.T_max, 4) << " |\n"
      << "| 평균 \\|오차\\| Y / T (280장) | " << fmt(mean(absv(ey)), 4) << " / " << fmt(mean(absv(et)), 4) << " |\n"
      << "| fold별 선택 설정 | " << selected[0] << ", " << selected[1] << ", " << selected[2] << ", " << selected[3] << " |\n"
      << "| OOF 평균 실패 center 1원점 / session 6원점 | " << fmt(mode_mean[0], 2) << " / " << fmt(mode_mean[1], 2) << " |\n\n"
      << img_md("01_추적방식_원점모드별_OOF실패.png", "OOF 실패") << "\n\n" << img_md("02_내부선택OOF_예측오차.png", "예측 오차") << "\n\n"
      << img_md("03_추적방식간_측정차이.png", "측정 차이") << "\n\n"
      << "## 튀는 값\n\n가장 크게 틀린 사진은 " << nn[wy] << "(정답 T " << fmt(tt[wy], 2) << ")로, 정답 가지 비용 " << fmt(nested[wy].tcost, 2)
      << "보다 Y 가지(" << nested[wy].r.branch << ") 비용 " << fmt(nested[wy].r.cost, 2) << "이 작아 Y " << fmt(py[wy], 4)
      << " mm로 읽혔다. 작은 T의 평균 이동(P12 c0x)이 Y 원점 사이 간격과 비슷한 크기라서 가지를 잘못 골랐다.\n\n"
      << outl.md() << "\n" << img_md(fy, "Y 튀는 값") << "\n\n" << img_md(ft, "T 튀는 값")
      << "\n\nT 그림은 다른 실패 사진 " << nn[wt] << "(정답 T " << fmt(tt[wt], 2) << ", 추정 " << (nested[wt].r.pred_axis == 'Y' ? "Y " : "T ")
      << fmt(nested[wt].r.pred, 4) << ")이다. 초록 상자는 center의 QR 영역, 빨간 상자는 이 사진에서 잰 QR 평균 이동만큼 옮긴 영역이다. 빨간 십자는 측정점(6점마다 1개), 화살표는 QR 평균 이동을 뺀 나머지 이동을 200배 키운 것이다.\n\n"
      << "## 한줄평\n\n기울기 방식은 630점 평균 측정을 " << diff_t.rows[0][1] << "~" << diff_t.rows[2][1]
      << " px만 바꿔 판정을 움직이지 못했고, 성적을 가른 것은 원점 모드(center 1원점 평균 실패 " << fmt(mode_mean[0], 2) << "장 → session 6원점 "
      << fmt(mode_mean[1], 2) << "장)였다.\n\n"
      << "## 원래 결과와 대조\n\n원래 값은 `forward_gn_20261021/evaluation/summary.csv`, `nested_selection.json`, `splits.json`, `audit.json`, `forward*_features.npy`에서 가져왔다. "
      << par.ok << "/" << par.total << " 일치.\n\n" << par.t.md();
    write_text(out("보고서.md"), P.str());
    std::cout << "parity " << par.ok << "/" << par.total << "\n";
    return 0;
}
