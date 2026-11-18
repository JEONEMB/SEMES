// QR 내부 536점만으로 고정 4-fold·재대입 280/280 재확인 (나사 없음)
// 원본: codex_experiments_20261014/release_qr536_20261021/pipeline.py, verify_raw.py, completion_audit.py
//       (측정 measurement/spread.py + refine.py GN17, 모델 multiref_4fold_20261021/experiment.py --regions c45)
#include <hye/dataset.hpp>
#include <hye/plot.hpp>
#include <hye/report.hpp>
#include <omp.h>
#include <iostream>
#include <set>
#include "corners.hpp"
#include "gls.hpp"
#include "track.hpp"

using namespace hye;
using g09::Item;
using g09::Row;
using g09::Vec;

namespace {
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
std::string stem(std::string s) { return s.size() > 4 && s.substr(s.size() - 4) == ".bmp" ? s.substr(0, s.size() - 4) : s; }
bool same_model(const g09::Model& a, const g09::Model& b) {
    if (a.branches.size() != b.branches.size() || cv::norm(a.W - b.W, cv::NORM_INF) > 1e-9) return false;
    for (size_t k = 0; k < a.branches.size(); ++k) {
        if (a.branches[k].name != b.branches[k].name) return false;
        for (size_t j = 0; j < a.branches[k].s.size(); ++j)
            if (std::abs(a.branches[k].s[j] - b.branches[k].s[j]) > 1e-9 || std::abs(a.branches[k].c[j] - b.branches[k].c[j]) > 1e-9 ||
                std::abs(a.branches[k].base[j] - b.branches[k].base[j]) > 1e-9)
                return false;
    }
    return true;
}
}  // namespace

int main() {
    omp_set_num_threads(4);
    begin_results();
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

    // ---- 630 기준점과 구조 영역 ----
    cv::Mat center = read_gray(shots[0]);
    g09::Reference R = g09::build_reference(center);
    std::vector<cv::Point2d> pts;
    std::vector<int> regs;
    std::vector<bool> finder, interior;
    for (auto& c : R.c630) {
        pts.push_back({c.px, c.py});
        double cx[3] = {1644, 1901, 2159};
        int k = 0;
        for (int j = 1; j < 3; ++j) if (std::abs(c.px - cx[j]) < std::abs(c.px - cx[k])) k = j;
        regs.push_back(k);
        finder.push_back((c.gx <= 7 && c.gy <= 7) || (c.gx >= 14 && c.gy <= 7) || (c.gx <= 7 && c.gy >= 14));
        interior.push_back(c.gx > 0 && c.gx < 21 && c.gy > 0 && c.gy < 21);
    }
    const int NP = (int)pts.size();

    // ---- GN17 추적 (전체 QR NCC → LK15 → 역방향 GN 창 17) ----
    g09::SpreadExtractor ex;
    ex.init(center, pts, regs, 0, 17, false);
    std::vector<std::vector<cv::Point2d>> D(N);
    std::vector<std::vector<bool>> V(N);
#pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < N; ++i) {
        cv::Mat img = read_gray(shots[i]);
        g09::TrackResult r = ex.extract(img);
        D[i] = r.D;
        V[i] = r.valid;
    }
    for (int i = N - 1; i >= 0; --i)
        for (int p = 0; p < NP; ++p) D[i][p] -= D[0][p];  // zero = center 자기 자신 측정
    // 영역별 P12: finder / data / interior
    const char* fname[3] = {"finder", "data", "interior"};
    std::vector<std::vector<int>> sel(3);
    for (int p = 0; p < NP; ++p) {
        if (finder[p]) sel[0].push_back(p); else sel[1].push_back(p);
        if (interior[p]) sel[2].push_back(p);
    }
    std::vector<std::vector<Vec>> X(3, std::vector<Vec>(N));
    for (int f = 0; f < 3; ++f) {
        std::vector<cv::Point2d> sp;
        for (int p : sel[f]) sp.push_back(pts[p]);
        g09::P12 p12;
        p12.init(sp);
        for (int i = 0; i < N; ++i) {
            std::vector<cv::Point2d> d;
            for (int p : sel[f]) d.push_back(D[i][p]);
            X[f][i] = p12.apply(d);
        }
    }
    int all_valid_images = 0;
    for (int i = 0; i < N; ++i) {
        int v = 0;
        for (int p : sel[2]) v += V[i][p];
        all_valid_images += v == (int)sel[2].size();
    }

    // ---- 구조 영역 48설정 중첩 선택 (multiref --regions) ----
    std::vector<Config> configs;
    for (int f = 0; f < 3; ++f)
        for (std::string mode : {"center", "inferred_session"})
            for (bool q : {false, true})
                for (double s : {.001, .05, .2, 1.}) {
                    char b[8];
                    std::snprintf(b, sizeof b, "c%02d", (int)configs.size());
                    configs.push_back({f, mode, q, s, b});
                }
    const int NC = (int)configs.size();
    const int C45 = 45;  // interior, inferred_session, 이차, 축소 .05 (pipeline.py 고정 설정)
    std::vector<int> ids;
    for (int i = 0; i < N; ++i) if (m[i].axis != 'C') ids.push_back(i);
    std::vector<std::string> sess;
    for (int i : ids) sess.push_back(m[i].session);
    std::vector<int> outer = g09::stratified_kfold(sess, 4, 20261021);
    std::vector<std::vector<Row>> oof(NC);
    std::vector<Row> nested;
    std::vector<std::string> selected;
    std::vector<int> fold_sum(4, 0);
    std::vector<g09::Model> fold_model(4);
    std::vector<std::vector<int>> fold_tr(4), fold_te(4);
    for (int f = 0; f < 4; ++f) {
        std::vector<int>& tr = fold_tr[f];
        std::vector<int>& te = fold_te[f];
        for (size_t k = 0; k < ids.size(); ++k) (outer[k] == f ? te : tr).push_back(ids[k]);
        for (int i : te) fold_sum[f] += i;
        std::vector<std::string> ts;
        for (int i : tr) ts.push_back(m[i].session);
        std::vector<int> inner = g09::stratified_kfold(ts, 3, 100 + f);
        std::vector<std::tuple<int, double, int>> rank(NC);
        std::vector<std::vector<Row>> rows(NC);
        std::vector<g09::Model> models(NC);
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
            models[c] = g09::fit_multiref(m, X[C.feature], tr, C.mode, C.quad, C.shrink);
            for (int i : te) {
                Row r = g09::enrich(m, i, g09::predict_one(models[c], X[C.feature][i], C.mode));
                r.fold = f;
                rows[c].push_back(r);
            }
        }
        for (int c = 0; c < NC; ++c) oof[c].insert(oof[c].end(), rows[c].begin(), rows[c].end());
        std::sort(rank.begin(), rank.end());
        int chosen = std::get<2>(rank[0]);
        selected.push_back(configs[chosen].name);
        nested.insert(nested.end(), rows[chosen].begin(), rows[chosen].end());
        fold_model[f] = models[C45];
    }
    // 고정 설정 c45: raw_oof4 = fold 모델로 시험, raw_resub280 = 280장 전체 학습 모델
    const std::vector<Row>& raw_oof = oof[C45];
    g09::Model full = g09::fit_multiref(m, X[2], ids, "inferred_session", true, .05);
    std::vector<Row> resub;
    std::vector<g09::Pred> resub_pred;
    for (int i : ids) {
        resub_pred.push_back(g09::predict_one(full, X[2][i], "inferred_session"));
        resub.push_back(g09::enrich(m, i, resub_pred.back()));
    }
    g09::Stats so = g09::stats(raw_oof), sr = g09::stats(resub), sn = g09::stats(nested);
    std::vector<g09::Stats> sf(4);
    for (int f = 0; f < 4; ++f) {
        std::vector<Row> g;
        for (auto& r : raw_oof) if (r.fold == f) g.push_back(r);
        sf[f] = g09::stats(g);
    }
    // 시험 사진의 라벨·특징을 바꿔도 학습 결과가 같은지 (completion_audit.py)
    bool heldout_ok;
    {
        std::vector<Item> m2(m);
        std::vector<Vec> X2(X[2]);
        for (int i : fold_te[0]) { m2[i].value = 1234; m2[i].axis = 'Y'; m2[i].session = "dummy"; for (auto& v : X2[i]) v += 999; }
        g09::Model a = g09::fit_multiref(m, X[2], fold_tr[0], "inferred_session", true, .05);
        g09::Model b = g09::fit_multiref(m2, X2, fold_tr[0], "inferred_session", true, .05);
        heldout_ok = same_model(a, b);
    }
    std::vector<std::string> nfail;
    for (auto& r : nested) if (!r.pass) nfail.push_back(r.file);
    int iT005 = 0;
    for (size_t k = 0; k < ids.size(); ++k) if (names[ids[k]] == "T-0.05.bmp") iT005 = (int)k;

    // ---- 표 ----
    Table fold_t{{"fold", "시험 장수", "Y 맞음", "T 맞음", "실패", "Y 최대 (mm)", "T 최대 (deg)"}};
    for (int f = 0; f < 4; ++f)
        fold_t.add({std::to_string(f), std::to_string(sf[f].n), std::to_string(sf[f].Y_correct) + "/" + std::to_string(sf[f].Y_n),
                    std::to_string(sf[f].T_correct) + "/" + std::to_string(sf[f].T_n), std::to_string(sf[f].failures), fmt(sf[f].Y_max, 6), fmt(sf[f].T_max, 6)});
    Table proto_t{{"평가", "장수", "Y 맞음", "T 맞음", "실패", "Y 최대 (mm)", "T 최대 (deg)"}};
    auto addp = [&](const std::string& n, const g09::Stats& s) {
        proto_t.add({n, std::to_string(s.n), std::to_string(s.Y_correct) + "/80", std::to_string(s.T_correct) + "/200", std::to_string(s.failures),
                     fmt(s.Y_max, 6), fmt(s.T_max, 6)});
    };
    addp("raw_oof4 (c45 고정, fold 학습)", so);
    addp("raw_resub280 (c45, 280장 학습)", sr);
    addp("구조 영역 48설정 내부 선택 OOF", sn);
    Table reg_t{{"영역", "점 수", "규칙", "c 최선 OOF 실패", "OOF Y 최대"}};
    for (int f = 0; f < 3; ++f) {
        int bc = -1;
        for (int c = 0; c < NC; ++c) {
            if (configs[c].feature != f) continue;
            g09::Stats s = g09::stats(oof[c]);
            if (bc < 0) bc = c;
            g09::Stats b = g09::stats(oof[bc]);
            if (std::make_pair(s.failures, s.Y_max) < std::make_pair(b.failures, b.Y_max)) bc = c;
        }
        g09::Stats b = g09::stats(oof[bc]);
        const char* rule[3] = {"격자 x<=7·y<=7 또는 x>=14·y<=7 또는 x<=7·y>=14", "finder 나머지", "0<격자 x<21 이고 0<격자 y<21"};
        reg_t.add({fname[f], std::to_string(sel[f].size()), rule[f], configs[bc].name + " " + std::to_string(b.failures), fmt(b.Y_max, 6)});
    }
    Table pred_csv{{"protocol", "fold", "file", "axis", "value", "session", "pred_axis", "pred", "branch", "cost", "eY", "eT", "pass"}};
    for (const std::vector<Row>* v : std::vector<const std::vector<Row>*>{&raw_oof, &resub})
        for (auto& r : *v)
            pred_csv.add({v == &raw_oof ? "raw_oof4" : "raw_resub280", std::to_string(r.fold), r.file, std::string(1, r.axis), fmt(r.value, 2), r.session,
                          std::string(1, r.pred_axis), fmt(r.pred, 8), r.branch, fmt(r.cost, 6), fmt(r.eY, 8), fmt(r.eT, 8), r.pass ? "True" : "False"});
    pred_csv.csv(out("raw_predictions.csv"));

    // ---- 튀는 값 (raw_oof4) ----
    std::vector<double> ty, py, tt, pt, ey, et;
    std::vector<std::string> nn;
    for (auto& r : raw_oof) {
        ty.push_back(r.axis == 'Y' ? r.value : 0);
        tt.push_back(r.axis == 'T' ? r.value : 0);
        py.push_back(r.pred_axis == 'Y' ? r.pred : 0);
        pt.push_back(r.pred_axis == 'T' ? r.pred : 0);
        ey.push_back(py.back() - ty.back());
        et.push_back(pt.back() - tt.back());
        nn.push_back(r.file);
    }
    size_t wy = worst(ey, 1)[0], wt = worst(et, 1)[0];
    // 원인 진단: 가장 큰 이동 잔차 점 (QR 평균 이동을 뺀 나머지)
    auto resid_point = [&](int i, int& qbest, double& rmax) {
        int pbest = -1;
        rmax = -1;
        for (int q = 0; q < 3; ++q) {
            double mx = 0, my = 0;
            int n = 0;
            for (int p : sel[2]) if (regs[p] == q) { mx += D[i][p].x; my += D[i][p].y; ++n; }
            mx /= n; my /= n;
            for (int p : sel[2])
                if (regs[p] == q) {
                    double r = std::hypot(D[i][p].x - mx, D[i][p].y - my);
                    if (r > rmax) { rmax = r; pbest = p; qbest = q; }
                }
        }
        return pbest;
    };
    Table outl{{"구분", "사진", "fold", "정답 Y (mm)", "추정 Y", "정답 T (deg)", "추정 T", "고른 가지", "비용", "P12 c0x (px)", "QR 평균 대비 최대 잔차 점 (QR, 격자, px)"}};
    auto add_out = [&](const std::string& lab, size_t k) {
        auto& r = raw_oof[k];
        int q = 0;
        double rm = 0;
        int p = resid_point(r.index, q, rm);
        outl.add({lab, r.file, std::to_string(r.fold), fmt(ty[k], 2), fmt(py[k], 4), fmt(tt[k], 2), fmt(pt[k], 4), r.branch, fmt(r.cost, 2),
                  fmt(X[2][r.index][0], 3), "QR" + std::to_string(q + 1) + " (" + std::to_string(R.c630[p].gx) + "," + std::to_string(R.c630[p].gy) + ") " + fmt(rm, 3)});
    };
    for (size_t k : worst(ey, 3)) add_out("Y 오차 상위", k);
    for (size_t k : worst(et, 3)) add_out("T 오차 상위", k);

    // ---- 그림 ----
    {
        std::vector<std::string> cats;
        std::vector<double> vals;
        for (int f = 0; f < 4; ++f) { cats.push_back("fold" + std::to_string(f) + " Y"); vals.push_back(sf[f].Y_max); }
        for (int f = 0; f < 4; ++f) { cats.push_back("fold" + std::to_string(f) + " T"); vals.push_back(sf[f].T_max); }
        bar_chart("01_fold별_최대오차.png", "raw_oof4 max |error| per fold (limit 0.09)", cats, vals, "max |error|", -1, 0.09);
    }
    pred_vs_truth("02_raw_oof4_예측오차.png", "QR536 raw OOF4", ty, py, tt, pt, nn);
    {
        // 536점(초록) / 바깥 테두리 94점(빨강)
        cv::Mat crop = center(cv::Rect(1540, 960, 725, 220)).clone();
        cv::cvtColor(crop, crop, cv::COLOR_GRAY2BGR);
        cv::resize(crop, crop, cv::Size(), 2, 2, cv::INTER_NEAREST);
        for (int p = 0; p < NP; ++p) {
            cv::Point c((int)std::lround((pts[p].x - 1540) * 2), (int)std::lround((pts[p].y - 960) * 2));
            cv::circle(crop, c, 3, interior[p] ? kGreen : kRed, -1, cv::LINE_AA);
        }
        text(crop, "green: interior 536 used, red: outer boundary 94 excluded", {8, 22}, 0.6, kBlack, 2);
        save_png("03_사용점_536.png", crop);
    }
    auto outlier_fig = [&](size_t k, const std::string& file, const std::string& cap) {
        auto& r = raw_oof[k];
        int i = r.index;
        cv::Mat im = read_color(shots[i]);
        int q = 0;
        double rm = 0;
        int pb = resid_point(i, q, rm);
        std::vector<cv::Mat> tiles;
        for (int qq = 0; qq < 3; ++qq) {
            double mx = 0, my = 0;
            int n = 0;
            for (int p : sel[2]) if (regs[p] == qq) { mx += D[i][p].x; my += D[i][p].y; ++n; }
            mx /= n; my /= n;
            std::vector<Mark> mk;
            int cnt = 0;
            for (int p : sel[2]) {
                if (regs[p] != qq || ((cnt++ % 6) && p != pb)) continue;
                mk.push_back({{pts[p].x + D[i][p].x, pts[p].y + D[i][p].y}, {D[i][p].x - mx, D[i][p].y - my}, p == pb ? "max" : "", p == pb ? kGreen : kRed});
            }
            const int* b = g09::BOXES[qq];
            cv::Rect2d ref(b[0], b[1], b[2] - b[0], b[3] - b[1]);
            cv::Rect2d now(ref.x + mx, ref.y + my, ref.width, ref.height);
            cv::Rect roi = cv::Rect(ref | now) + cv::Size(40, 40) - cv::Point(20, 20);
            tiles.push_back(zoom(im, roi, 420, mk, 200, qq == 0 ? cap : "",
                                 {{ref, kGreen, ""}, {now, kRed, "q" + std::to_string(qq) + " dx " + fmt(mx, 2) + " dy " + fmt(my, 2)}}));
        }
        save_png(file, hstack(tiles));
    };
    std::string fy = "04_튀는값_Y_" + stem(nn[wy]) + ".png", ft = "05_튀는값_T_" + stem(nn[wt]) + ".png";
    outlier_fig(wy, fy, nn[wy] + " Y err " + fmts(ey[wy], 4) + " mm");
    outlier_fig(wt, ft, nn[wt] + " T err " + fmts(et[wt], 4) + " deg");

    // ---- 원래 결과와 대조 ----
    // 출처: release_qr536_20261021/raw_summary.csv, raw_fold_results.csv, audit.json, completion_audit.json,
    //       multiref_4fold_20261021/structural_regions/summary.csv, nested_selection.json, selection_audit.json, splits.json
    Parity par;
    par.add("finder 영역 점 수", 108, (double)sel[0].size(), 0, 0);
    par.add("data 영역 점 수", 522, (double)sel[1].size(), 0, 0);
    par.add("interior 점 수", 536, (double)sel[2].size(), 0, 0);
    par.add("536점 모두 유효인 사진 (/281)", 281, all_valid_images, 0, 0);
    const int fs0[4] = {9784, 9627, 9943, 9986};
    for (int f = 0; f < 4; ++f) par.add("fold " + std::to_string(f) + " 시험 사진 색인 합", fs0[f], fold_sum[f], 0, 0);
    par.add("raw_oof4 축·부호 맞음 (/280)", 280, so.Y_correct + so.T_correct, 0, 0);
    par.add("raw_oof4 실패", 0, so.failures, 0, 0);
    par.add("raw_oof4 Y 최대 (mm)", 0.036550229, so.Y_max, 0.0005, 6);
    par.add("raw_oof4 T 최대 (deg)", 0.002993403, so.T_max, 0.0005, 6);
    const double fy0[4] = {0.036550229, 0.004233478, 0.009949631, 0.002740174}, ft0[4] = {0.001119885, 0.002993403, 0.001315298, 0.002276907};
    for (int f = 0; f < 4; ++f) {
        par.add("raw_oof4 fold " + std::to_string(f) + " Y 최대", fy0[f], sf[f].Y_max, 0.0005, 6);
        par.add("raw_oof4 fold " + std::to_string(f) + " T 최대", ft0[f], sf[f].T_max, 0.0005, 6);
    }
    par.add("raw_resub280 실패", 0, sr.failures, 0, 0);
    par.add("raw_resub280 Y 최대 (mm)", 0.036372779, sr.Y_max, 0.0005, 6);
    par.add("raw_resub280 T 최대 (deg)", 0.002898078, sr.T_max, 0.0005, 6);
    par.add("T-0.05 재대입 추정 (중립 파일 이름 예)", -0.049755950, resub[iT005].pred, 0.0005, 6);
    par.add("구조 영역 내부 선택 OOF 통과 (/280)", 279, 280 - sn.failures, 0, 0);
    par.add("구조 영역 내부 선택 OOF Y 최대 (mm)", 0.266347493, sn.Y_max, 0.001, 6);
    {
        std::string s0 = "c28, c45, c46, c24", s1;
        for (size_t k = 0; k < selected.size(); ++k) s1 += (k ? ", " : "") + selected[k];
        par.add_text("구조 영역 fold별 선택 설정", s0, s1, s0 == s1);
        std::string f1;
        for (size_t k = 0; k < nfail.size(); ++k) f1 += (k ? ", " : "") + nfail[k];
        par.add_text("내부 선택 OOF 실패 사진", "T-0.05.bmp", f1, f1 == "T-0.05.bmp");
    }
    par.add_text("시험 라벨·특징을 바꿔도 학습 결과 같음", "True", heldout_ok ? "True" : "False", heldout_ok);

    // ---- 결과.md ----
    std::ostringstream Rm;
    Rm << "# 결과: QR 내부 536점 고정 모델, 원본 재측정 확인\n\n"
       << "입력: `data/clean/clean_20261014` 281장. center.bmp에서 QR을 해독해 대각 접촉을 뺀 구조 모서리 630점을 erf 경계 모델로 맞추고, 그중 바깥 테두리(격자 0, 21)를 뺀 536점만 쓴다.\n"
       << "측정: 사진마다 QR 전체 NCC(±110x22) → LK 창 15 → 역방향 GN 창 17 (QR 상자 안 픽셀만). 이동 = 현재 - center 자기 측정, 536점을 이차식 6항으로 회귀한 12계수(P12).\n"
       << "모델: 6원점(inferred_session) Y/T 이차곡선 GLS, 공분산 축소 0.05 (c45). 4-fold는 StratifiedKFold(4, seed 20261021).\n\n"
       << "## 평가별 성적\n\n" << proto_t.md() << "\n## raw_oof4 fold별\n\n" << fold_t.md()
       << "\n## 구조 영역별 (48설정 중 각 영역 최선)\n\n" << reg_t.md()
       << "\n## 원래 결과와 대조 (" << par.ok << "/" << par.total << " 일치)\n\n" << par.t.md()
       << "\n## 옮기지 않은 부분\n\n- `report.py`의 HTML 보고서와 `fold_errors.png` 원본 서식은 옮기지 않고 같은 내용을 이 문서와 `이미지/`에 담았다.\n"
       << "- 원래 `verify_raw.py`는 fold 모델을 캐시 특징으로 학습하고 새로 잰 특징으로 시험했다(둘 차이 1.4e-8). C++은 한 번 잰 특징을 둘 다에 쓴다.\n"
       << "- 웹 추론 도구(`종합/`)는 이 실험의 범위가 아니다.\n\n"
       << "## 파일\n\n- `raw_predictions.csv`: raw_oof4·raw_resub280 280장씩 예측\n- `이미지/`: 보고서 그림\n";
    write_text(out("결과.md"), Rm.str());

    // ---- 보고서.md ----
    std::ostringstream P;
    P << "# 보고서: QR만으로 통과 확인, 내부 536점 고정 (10-21 18:10)\n\n"
      << "## 문제\n\n나사 없이 QR만으로 고정 4-fold와 재대입 모두 280/280을 원본 재실행으로 확인해야 했다.\n\n"
      << "## 해결 방법\n\n내부 536점을 NCC→LK15→GN17로 추적해 P12로 요약하고 6원점·이차곡선 GLS c45 고정 모델로 검증했다.\n\n"
      << "## 결과\n\nraw_oof4 " << 280 - so.failures << "/280, 최대 오차 Y " << fmt(so.Y_max, 6) << " mm·T " << fmt(so.T_max, 6)
      << " deg였고, 설정을 학습 안에서 고른 구조 영역 내부 선택 OOF는 " << 280 - sn.failures << "/280이었다.\n\n"
      << "| 지표 | 값 |\n|---|---|\n"
      << "| raw_oof4 통과 | " << 280 - so.failures << " / 280 |\n"
      << "| raw_oof4 최대 \\|오차\\| Y (mm) / T (deg) | " << fmt(so.Y_max, 6) << " / " << fmt(so.T_max, 6) << " |\n"
      << "| raw_oof4 평균 \\|오차\\| Y / T | " << fmt(mean(absv(ey)), 6) << " / " << fmt(mean(absv(et)), 6) << " |\n"
      << "| raw_resub280 통과 | " << 280 - sr.failures << " / 280 |\n"
      << "| raw_resub280 최대 \\|오차\\| Y / T | " << fmt(sr.Y_max, 6) << " / " << fmt(sr.T_max, 6) << " |\n"
      << "| 구조 영역 내부 선택 OOF 통과 | " << 280 - sn.failures << " / 280 (선택 " << selected[0] << ", " << selected[1] << ", " << selected[2] << ", " << selected[3] << ") |\n"
      << "| 536점 모두 유효인 사진 | " << all_valid_images << " / 281 |\n\n"
      << img_md("01_fold별_최대오차.png", "fold별 최대 오차") << "\n\n" << img_md("02_raw_oof4_예측오차.png", "예측 오차") << "\n\n"
      << img_md("03_사용점_536.png", "사용점") << "\n\n"
      << "## 튀는 값\n\nraw_oof4에서 가장 크게 틀린 사진은 Y는 " << nn[wy] << "(정답 Y " << fmt(ty[wy], 2) << ", 추정 " << fmt(py[wy], 4) << " mm), T는 " << nn[wt]
      << "(정답 T " << fmt(tt[wt], 2) << ", 추정 " << fmt(pt[wt], 4) << " deg)이고, 둘 다 통과 기준 0.09 안이다.\n\n"
      << outl.md() << "\n" << img_md(fy, "Y 튀는 값") << "\n\n" << img_md(ft, "T 튀는 값")
      << "\n\n초록 상자는 center QR 영역, 빨간 상자는 QR 평균 이동만큼 옮긴 영역이다. 빨간 십자는 536점 중 6점마다 1개, 초록 십자(max)는 QR 평균 이동에서 가장 벗어난 점이고 화살표는 그 나머지 이동을 200배 키운 것이다.\n\n"
      << "## 한줄평\n\n바깥 테두리 94점을 빼고 QR 내부 536점의 공간 이차 요약에 세션별 원점 6개를 두니 작은 T도 원점 사이를 구분해 raw_oof4 최대 오차가 Y "
      << fmt(so.Y_max, 4) << " mm로 기준 0.09의 " << fmt(so.Y_max / 0.09 * 100, 0) << "%에 그쳤지만, 같은 자료에서 설정을 고른 결과라 내부 선택 OOF는 "
      << 280 - sn.failures << "/280으로 한 장(" << (nfail.empty() ? "-" : nfail[0]) << ")을 놓쳤다.\n\n"
      << "## 원래 결과와 대조\n\n원래 값은 `release_qr536_20261021/raw_summary.csv`, `raw_fold_results.csv`, `completion_audit.json`과 `multiref_4fold_20261021/structural_regions/summary.csv`, `nested_selection.json`, `selection_audit.json`에서 가져왔다. "
      << par.ok << "/" << par.total << " 일치.\n\n" << par.t.md();
    write_text(out("보고서.md"), P.str());
    std::cout << "parity " << par.ok << "/" << par.total << "\n";
    return 0;
}
