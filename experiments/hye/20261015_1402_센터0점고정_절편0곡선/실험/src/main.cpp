// center 0점 고정 + 절편 0 곡선 (원본: codex_experiments_20261014/center_fixed_20261015/run.py, zero_experiment.py fit/predict/score)
// center 에서 205점을 한 번 고르고 280장을 직접 추적한다. 모델은 축마다 12값 = v·s + v²·c (절편 0, 0점 후보 없음).
// 비교: center_linear (1차), legacy_six_origins (구간 6개 0점 후보 + 2차, 원래 방식)
#include <hye/dataset.hpp>
#include <hye/plot.hpp>
#include <hye/report.hpp>
#include <omp.h>
#include <iostream>
#include "spread.hpp"
#include "splits.hpp"

using namespace hye;

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
    omp_set_num_threads(4);
    auto shots = load_1014();
    struct Row { cx::Item it; const Shot* s; };
    std::vector<Row> rows;
    for (auto& s : shots) {
        std::string f = s.rel.substr(s.rel.find('/') == std::string::npos ? 0 : s.rel.find('/') + 1);
        char a = s.condition == "center" ? 'C' : s.condition == "T" ? 'T' : 'Y';
        double v = a == 'T' ? s.t : a == 'Y' ? s.y : 0;
        rows.push_back({{f, a, v, cx::session_of(a, v)}, &s});
    }
    std::sort(rows.begin(), rows.end(), [](const Row& p, const Row& q) {
        int ka = p.it.axis == 'C' ? 0 : p.it.axis == 'T' ? 1 : 2, kb = q.it.axis == 'C' ? 0 : q.it.axis == 'T' ? 1 : 2;
        return ka != kb ? ka < kb : p.it.file < q.it.file;
    });
    const int NI = (int)rows.size();
    std::vector<cx::Item> m;
    for (auto& r : rows) m.push_back(r.it);
    int ic = 0;
    for (int i = 0; i < NI; ++i) if (m[i].axis == 'C') ic = i;

    cv::Mat ref = cx::crop(read_gray(*rows[ic].s));
    cx::Extractor ex(ref, 14);
    const int NP = (int)ex.P.p.size();
    std::vector<cv::Point2d> pts;
    for (int i = 0; i < NP; ++i) pts.push_back(ex.global(i));
    std::vector<std::vector<cv::Point2d>> D(NI, std::vector<cv::Point2d>(NP, {0, 0}));
    std::vector<int> nvalid(NI, NP);
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < NI; ++i) {
        if (i == ic) continue;  // center 는 0 으로 둔다
        auto r = ex.extract(cx::crop(read_gray(*rows[i].s)));
        int nv = 0;
        for (int k = 0; k < NP; ++k) { D[i][k] = {r.D[k].x, r.D[k].y}; nv += r.valid[k]; }
        nvalid[i] = nv;
    }
    int invalid = 0;
    for (int v : nvalid) invalid += NP - v;
    std::cout << "points " << NP << " invalid " << invalid << "\n";
    std::vector<std::vector<double>> X(NI);
    for (int i = 0; i < NI; ++i) X[i] = cx::p12(pts, D[i]);

    // 분할: 재대입 1개 + 원래 splits.json 의 118개를 같은 규칙으로 다시 만든다
    auto folds = sp::folds(m);
    std::vector<int> ids;
    for (int i = 0; i < NI; ++i) if (i != ic) ids.push_back(i);
    std::vector<sp::Fold> all{{"full280_resubstitution", "all", ids, ids}};
    all.insert(all.end(), folds.begin(), folds.end());

    struct Cfg { std::string method, mode; bool quad; };
    const std::vector<Cfg> cfgs = {{"center_quadratic", "center", true}, {"center_linear", "center", false}, {"legacy_six_origins", "inferred_session", true}};
    std::vector<sp::PredRow> P;
    std::map<std::string, cx::Model> full_model;
    for (auto& c : cfgs) {
        std::vector<std::vector<sp::PredRow>> per(all.size());
#pragma omp parallel for schedule(dynamic)
        for (int f = 0; f < (int)all.size(); ++f) {
            auto md = cx::fit(m, X, all[f].train, c.mode, c.quad);
            for (int i : all[f].test) {
                auto p = cx::predict_one(X[i], md, c.mode == "inferred_session");
                per[f].push_back({i, m[i].axis, p.axis, m[i].value, p.value, p.cost, m[i].session, p.branch, c.method, all[f].protocol, all[f].fold});
            }
        }
        for (auto& v : per) P.insert(P.end(), v.begin(), v.end());
        full_model[c.method] = cx::fit(m, X, ids, c.mode, c.quad);
    }
    // 점수
    std::vector<std::string> protocols;
    for (auto& f : all) if (std::find(protocols.begin(), protocols.end(), f.protocol) == protocols.end()) protocols.push_back(f.protocol);
    std::map<std::pair<std::string, std::string>, sp::Score> S;
    Table met{{"method", "protocol", "n", "axis_sign_accuracy", "strict_pass09", "Y_axis_sign_accuracy", "T_axis_sign_accuracy", "vector_Y_max", "vector_T_max",
               "Y_matched_MAE", "T_matched_MAE"}};
    for (auto& c : cfgs)
        for (auto& pr : protocols) {
            std::vector<sp::PredRow> d;
            for (auto& r : P) if (r.method == c.method && r.protocol == pr) d.push_back(r);
            auto s = sp::score(d);
            S[{c.method, pr}] = s;
            met.add({c.method, pr, std::to_string(s.n), fmt(s.axis_sign_accuracy, 6), fmt(s.strict_pass09, 6), fmt(s.Y_axis_sign_accuracy, 6), fmt(s.T_axis_sign_accuracy, 6),
                     fmt(s.vector_Y_max, 6), fmt(s.vector_T_max, 6), fmt(s.Y_matched_MAE, 6), fmt(s.T_matched_MAE, 6)});
        }
    met.csv(out("metrics.csv"));
    std::map<std::string, sp::Score> rep;
    for (auto& c : cfgs) {
        std::vector<sp::PredRow> d;
        for (auto& r : P) if (r.method == c.method && r.protocol.rfind("repeated5_", 0) == 0) d.push_back(r);
        rep[c.method] = sp::score(d);
    }
    {
        Table pt{{"file", "axis", "value", "session", "pred_axis", "pred", "branch", "cost", "method", "protocol", "fold"}};
        for (auto& r : P)
            pt.add({m[r.i].file, std::string(1, r.axis), fmt(r.value, 2), r.session, std::string(1, r.pred_axis), fmt(r.pred, 10), r.branch, fmt(r.cost, 6), r.method, r.protocol, r.fold});
        pt.csv(out("predictions.csv"));
    }

    // 주 결과: center_quadratic 재대입
    std::vector<sp::PredRow> full;
    for (auto& r : P) if (r.method == "center_quadratic" && r.protocol == "full280_resubstitution") full.push_back(r);
    std::vector<double> ty, py, tt, pt, eY, eT;
    std::vector<std::string> names;
    int npass = 0, passY = 0, passT = 0;
    std::vector<int> fails;
    for (size_t k = 0; k < full.size(); ++k) {
        auto& r = full[k];
        double a = r.axis == 'Y' ? r.value : 0, b = r.axis == 'T' ? r.value : 0, c = r.pred_axis == 'Y' ? r.pred : 0, d = r.pred_axis == 'T' ? r.pred : 0;
        ty.push_back(a); tt.push_back(b); py.push_back(c); pt.push_back(d);
        eY.push_back(c - a); eT.push_back(d - b);
        names.push_back(m[r.i].file.substr(0, m[r.i].file.size() - 4));
        bool ok = r.axis == r.pred_axis && sp::sgn(r.value) == sp::sgn(r.pred), pass = ok && std::abs(c - a) <= kPass && std::abs(d - b) <= kPass;
        npass += pass;
        (r.axis == 'Y' ? passY : passT) += pass;
        if (!pass) fails.push_back((int)k);
    }
    auto& fq = S[{"center_quadratic", "full280_resubstitution"}];

    // ---- 그림 ----
    pred_vs_truth("01_재대입_예측오차.png", "center_quadratic full 280", ty, py, tt, pt, names);
    {
        std::vector<std::string> cats;
        std::vector<double> v;
        for (auto& c : cfgs) {
            cats.push_back(c.method.substr(0, 6) + (c.method[0] == 'l' ? "" : c.quad ? " quad" : " lin") + " Y");
            v.push_back(rep[c.method].Y_axis_sign_accuracy * 100);
            cats.push_back(c.method.substr(0, 6) + (c.method[0] == 'l' ? "" : c.quad ? " quad" : " lin") + " T");
            v.push_back(rep[c.method].T_axis_sign_accuracy * 100);
        }
        bar_chart("02_반복5분할_축부호정확도.png", "Repeated 5-fold x20: axis+sign accuracy (%)", cats, v, "%", 0, 100);
    }
    // c0x(평균 가로 이동) vs 값: 구간마다 0점이 달라 center 하나로는 못 맞춘다
    {
        Chart c("Mean x shift c0x vs command, by session (zero = center)", "command (Y mm, T deg x5)", "c0x (px)");
        const std::vector<std::string> ss = {"Y1+", "Y1-", "Y2+", "Y2-"};
        for (size_t q = 0; q < ss.size(); ++q) {
            std::vector<double> x, y;
            for (int i : ids) if (m[i].session == ss[q] && std::abs(m[i].value) <= 0.5) { x.push_back(m[i].value); y.push_back(X[i][0]); }
            c.scatter(x, y, color((int)q), ss[q], 4);
        }
        std::vector<double> lx, ly;
        auto& br = full_model["center_quadratic"].branches;
        for (auto& b : br)
            if (b.second.axis == 'Y')
                for (double v = -0.5; v <= 0.5001; v += 0.01) { lx.push_back(v); ly.push_back(v * b.second.s[0] + v * v * b.second.c[0]); }
        c.line(lx, ly, kBlack, "center model Y");
        std::vector<double> tx, tyy;
        for (int i : ids) if (m[i].axis == 'T' && std::abs(m[i].value) <= 0.1) { tx.push_back(m[i].value * 5); tyy.push_back(X[i][0]); }
        c.scatter(tx, tyy, kGray, "T (x5)", 2);
        c.hline(0).vline(0);
        c.save("03_구간별_평균이동_0점.png");
    }
    // 튀는 값 그림
    auto outlier_fig = [&](int k, const std::string& file, const std::string& cap) {
        auto& r = full[k];
        cv::Mat im = read_color(*rows[r.i].s);
        std::vector<cv::Mat> tiles;
        for (int q = 0; q < 3; ++q) {
            cv::Point2d md(0, 0);
            int n = 0;
            for (int j = 0; j < NP; ++j) if (ex.P.region[j] == q) { md += D[r.i][j]; ++n; }
            md /= (double)n;
            cv::Rect2d refb(cx::BOXES[q][0], cx::BOXES[q][1], cx::BOXES[q][2] - cx::BOXES[q][0], cx::BOXES[q][3] - cx::BOXES[q][1]);
            cv::Rect2d now(refb.x + md.x, refb.y + md.y, refb.width, refb.height);
            cv::Rect roi = cv::Rect(refb | now) + cv::Size(40, 40) - cv::Point(20, 20);
            std::vector<Mark> mk;
            for (int j = 0; j < NP; ++j) if (ex.P.region[j] == q) mk.push_back({pts[j] + D[r.i][j], D[r.i][j] - md, "", kRed});
            tiles.push_back(zoom(im, roi, 460, mk, 200, q == 0 ? cap : "",
                                 {{refb, kGreen, ""}, {now, kRed, "QR" + std::to_string(q + 1) + " mean (" + fmts(md.x, 2) + ", " + fmts(md.y, 3) + ")"}}));
        }
        save_png(file, hstack(tiles));
    };
    size_t wy = worst(eY, 1)[0], wt = worst(eT, 1)[0];
    std::string fy = "04_튀는값_Y_" + names[wy] + ".png", ft = "05_튀는값_T_" + names[wt] + ".png";
    outlier_fig((int)wy, fy, names[wy] + " (" + std::string(1, full[wy].axis) + " " + fmts(full[wy].value, 2) + ") -> " + std::string(1, full[wy].pred_axis) + " " + fmts(full[wy].pred, 3));
    outlier_fig((int)wt, ft, names[wt] + " (" + std::string(1, full[wt].axis) + " " + fmts(full[wt].value, 2) + ") -> " + std::string(1, full[wt].pred_axis) + " " + fmts(full[wt].pred, 3));

    // 실패 사진 표: 다른 축 비용, legacy 가 고른 후보, 그 구간 0점
    auto& cq = full_model["center_quadratic"];
    auto& lg = full_model["legacy_six_origins"];
    auto branch_zero = [&](const std::string& s) {  // 구간 0점 c0x 를 그 축 기울기로 나눈 값 (mm 또는 deg)
        for (auto& b : lg.branches) if (b.first == s) return b.second.base[0] / b.second.s[0];
        return kNaN;
    };
    Table ft_tab{{"사진", "구간", "정답", "고른 축", "추정", "고른 축 비용", "맞는 축 비용", "평균 가로 이동 c0x (px)", "세로 c0y (px)", "그 구간 0점 위치 (legacy, center 대비 단위값)"}};
    double gap_min = 1e9, gap_max = 0;
    for (int k : fails) {
        auto& r = full[k];
        auto p = cx::predict_one(X[r.i], cq, false);
        double right = kNaN;
        for (auto& a : p.all) if (a.first[0] == r.axis) right = a.second.second;
        ft_tab.add({m[r.i].file, r.session, std::string(1, r.axis) + " " + fmts(r.value, 2), std::string(1, r.pred_axis), fmts(r.pred, 4), fmt(p.cost, 2), fmt(right, 2),
                    fmt(X[r.i][0], 3), fmt(X[r.i][1], 4), fmts(branch_zero(r.session), 4)});
        gap_min = std::min(gap_min, right - p.cost); gap_max = std::max(gap_max, right - p.cost);
    }
    Table zt{{"구간", "0점 위치 = c0x 기준값 / 기울기 (center 대비, Y mm 또는 T deg)"}};
    double zmin = 1e9, zmax = -1e9;
    for (auto& b : lg.branches) if (b.second.axis == 'Y') { double z = b.second.base[0] / b.second.s[0]; zmin = std::min(zmin, z); zmax = std::max(zmax, z); }
    for (auto& b : lg.branches) zt.add({b.first, fmts(b.second.base[0] / b.second.s[0], 4)});

    // ---- 원래 결과와 대조 (center_fixed_20261015/metrics.csv, predictions.csv, checks.json, 결과.html; 서명은 gls_comparison_20261014/splits.json) ----
    Parity par;
    par.add("특징점 수", 205, NP, 0, 0);
    par.add("유효하지 않은 점", 0, invalid, 0, 0);
    const std::vector<std::pair<std::string, long long>> sig = {
        {"interleaved", 59500}, {"stratified5", 118343}, {"blocked_within_session5", 118860}, {"repeated5_seed0", 117733}, {"repeated5_seed1", 119296},
        {"repeated5_seed2", 118411}, {"repeated5_seed3", 118813}, {"repeated5_seed4", 118759}, {"repeated5_seed5", 117935}, {"repeated5_seed6", 118214},
        {"repeated5_seed7", 118822}, {"repeated5_seed8", 118776}, {"repeated5_seed9", 118812}, {"repeated5_seed10", 118562}, {"repeated5_seed11", 119043},
        {"repeated5_seed12", 118596}, {"repeated5_seed13", 118560}, {"repeated5_seed14", 119449}, {"repeated5_seed15", 118369}, {"repeated5_seed16", 118452},
        {"repeated5_seed17", 118321}, {"repeated5_seed18", 118995}, {"repeated5_seed19", 118285}, {"session_out_missing_origin", 123150}};
    int sig_ok = 0;
    for (auto& [p, v] : sig) sig_ok += sp::signature(folds, p) == v;
    par.add("분할 서명 일치 (24 protocol)", 24, sig_ok, 0, 0);
    par.add("center_quadratic 재대입 통과 (/280)", 274, npass, 0, 0);
    par.add("center_quadratic 재대입 Y 통과 (/80)", 75, passY, 0, 0);
    par.add("center_quadratic 재대입 T 통과 (/200)", 199, passT, 0, 0);
    par.add("center_quadratic 재대입 Y 맞은 축 MAE", 0.011780, fq.Y_matched_MAE, 0.0002, 6);
    par.add("center_quadratic 재대입 T 맞은 축 MAE", 0.002261, fq.T_matched_MAE, 0.00005, 6);
    par.add("center_quadratic 반복 5분할 Y 축·부호 (%)", 93.4375, rep["center_quadratic"].Y_axis_sign_accuracy * 100, 0.05, 4);
    par.add("center_quadratic 반복 5분할 T 축·부호 (%)", 99.5, rep["center_quadratic"].T_axis_sign_accuracy * 100, 0.05, 4);
    par.add("center_linear 반복 5분할 Y 축·부호 (%)", 93.375, rep["center_linear"].Y_axis_sign_accuracy * 100, 0.05, 4);
    par.add("center_linear 반복 5분할 T 축·부호 (%)", 99.5, rep["center_linear"].T_axis_sign_accuracy * 100, 0.05, 4);
    par.add("legacy 반복 5분할 Y 축·부호 (%)", 99.9375, rep["legacy_six_origins"].Y_axis_sign_accuracy * 100, 0.05, 4);
    par.add("legacy 반복 5분할 T 축·부호 (%)", 99.7, rep["legacy_six_origins"].T_axis_sign_accuracy * 100, 0.05, 4);
    par.add("center_linear 재대입 Y 맞은 축 MAE", 0.011834, S[{"center_linear", "full280_resubstitution"}].Y_matched_MAE, 0.0002, 6);
    par.add("legacy 재대입 Y 맞은 축 MAE", 0.001613, S[{"legacy_six_origins", "full280_resubstitution"}].Y_matched_MAE, 0.0001, 6);
    par.add("legacy 재대입 T 맞은 축 MAE", 0.000367, S[{"legacy_six_origins", "full280_resubstitution"}].T_matched_MAE, 0.00002, 6);
    par.add("center_quadratic session_out 축·부호", 0.978571, S[{"center_quadratic", "session_out_missing_origin"}].axis_sign_accuracy, 0.0005, 6);
    par.add("center_quadratic session_out T 최대 벡터 오차", 0.053600, S[{"center_quadratic", "session_out_missing_origin"}].vector_T_max, 0.001, 6);
    par.add("legacy session_out 축·부호", 0.207143, S[{"legacy_six_origins", "session_out_missing_origin"}].axis_sign_accuracy, 0.0005, 6);
    par.add("center_quadratic blocked 축·부호", 0.978571, S[{"center_quadratic", "blocked_within_session5"}].axis_sign_accuracy, 0.0005, 6);
    par.add("center_quadratic interleaved 축·부호", 0.978571, S[{"center_quadratic", "interleaved"}].axis_sign_accuracy, 0.0005, 6);
    std::string fl;
    for (int k : fails) fl += (fl.empty() ? "" : ",") + names[k];
    const std::string flo = "T-0.05,Y+0.05,Y+0.15,Y-0.10,Y-0.20,Y-0.30";
    par.add_text("재대입 실패 사진", flo, fl, fl == flo);
    auto pv = [&](const std::string& f) { for (auto& r : full) if (m[r.i].file == f) return r.pred; return kNaN; };
    const std::vector<std::pair<std::string, double>> sample = {{"T-0.05.bmp", -0.25238981504697705}, {"T+0.01.bmp", 0.0059175177342439},
                                                                {"Y+1.00.bmp", 0.9904202068426072}, {"T-0.50.bmp", -0.4986981395958026},
                                                                {"Y-2.00.bmp", -1.98600496156774}, {"Y+0.05.bmp", 0.0098691041823932}};
    for (auto& [f, v] : sample) par.add(f + " 재대입 추정", v, pv(f), 0.0005, 6);

    // ---- 결과.md ----
    Table key{{"method", "protocol", "축·부호 정확도", "엄격 통과", "Y 축·부호", "T 축·부호", "Y 최대 벡터 오차", "T 최대 벡터 오차", "Y 맞은 축 MAE", "T 맞은 축 MAE"}};
    for (auto& c : cfgs)
        for (auto& pr : protocols) {
            if (pr.rfind("repeated5_seed", 0) == 0) continue;
            auto& s = S[{c.method, pr}];
            key.add({c.method, pr, fmt(s.axis_sign_accuracy * 100, 2) + "%", fmt(s.strict_pass09 * 100, 2) + "%", fmt(s.Y_axis_sign_accuracy * 100, 2) + "%",
                     fmt(s.T_axis_sign_accuracy * 100, 2) + "%", fmt(s.vector_Y_max, 4), fmt(s.vector_T_max, 4), fmt(s.Y_matched_MAE, 5), fmt(s.T_matched_MAE, 5)});
        }
    Table rk{{"method", "반복 5분할 x20 (5600 예측) Y 축·부호", "T 축·부호", "엄격 통과"}};
    for (auto& c : cfgs)
        rk.add({c.method, fmt(rep[c.method].Y_axis_sign_accuracy * 100, 4) + "%", fmt(rep[c.method].T_axis_sign_accuracy * 100, 4) + "%",
                fmt(rep[c.method].strict_pass09 * 100, 4) + "%"});
    std::ostringstream R;
    R << "# 결과: center 0점 고정 + 절편 0 곡선\n\n"
      << "입력: `data/clean/clean_20261014` 281장. center 에서 QR 세 상자마다 Shi-Tomasi(간격 14)로 " << NP
      << "점을 한 번 고르고, 280장을 center 와 직접 비교해 추적했다(상자 NCC → LK 15px → GN17). center 의 이동은 0 이다. 사진마다 점 이동을 판 좌표 2차식 12값으로 요약했다.\n\n"
      << "모델: center_quadratic = 축마다 12값 ~ v·s + v²·c (절편 0, 후보 2개 Y/T), center_linear = v·s, legacy_six_origins = 원래 6후보 0점 + 2차. 잔차 상관 백색화 후 비용 최소 축을 고른다.\n\n"
      << "분할: 재대입 1개 + 원래 `splits.json` 과 같은 규칙으로 다시 만든 118개(교차·층화·덩어리·반복 20회·구간 빼기). sklearn StratifiedKFold 의 섞기는 MT19937 로 그대로 재현해 서명이 모두 같다.\n\n"
      << "## 반복 5분할 20회 (사진마다 20번 예측)\n\n" << rk.md() << "\n## 분할별\n\n" << key.md()
      << "\n## 구간별 0점 (legacy 모델의 구간 기준값 / 기울기)\n\n" << zt.md()
      << "\n## 원래 결과와 대조 (" << par.ok << "/" << par.total << " 일치)\n\n" << par.t.md()
      << "\n## 옮기지 않은 부분\n\n- 원래의 6개 trial 전체 경로 그림(png·svg·미리보기 jpg), 결과.html, 기록 zip, frozen_model.npz 저장은 옮기지 않았다. 대신 구간별 평균 이동 그림(03)을 넣었다.\n"
      << "- `종합/` 아래 ECC 설명 그림·중간발표 자료는 옮기지 않았다.\n\n"
      << "## 파일\n\n- `metrics.csv`: 방법 x 분할 지표\n- `predictions.csv`: 모든 분할의 사진별 예측 (3 방법 x 119 분할)\n- `이미지/`: 그림 5장\n";
    write_text(out("결과.md"), R.str());

    // ---- 보고서.md ----
    auto& RY = full[wy];
    auto& RT = full[wt];
    std::ostringstream Pm;
    Pm << "# 보고서: center 0점 고정 + 절편 0 곡선 (10-15 14:02)\n\n"
       << "## 문제\n\n시행별 6개 0점 후보 없이 센터 하나만 0점으로 고정해도 Y/T를 맞힐 수 있는지 확인하려 했다.\n\n"
       << "## 해결 방법\n\n센터 " << NP << "점을 280장 추적하고 절편 0인 선형/2차 축 곡선을 6후보 방식과 같은 분할로 비교했다.\n\n"
       << "## 결과\n\ncenter_quadratic 은 재대입 " << npass << "/280(Y " << passY << "/80, T " << passT << "/200), 반복 5분할 Y "
       << fmt(rep["center_quadratic"].Y_axis_sign_accuracy * 100, 4) << "%·T " << fmt(rep["center_quadratic"].T_axis_sign_accuracy * 100, 4)
       << "%로 6후보 방식(Y " << fmt(rep["legacy_six_origins"].Y_axis_sign_accuracy * 100, 4) << "%)보다 나빠 목표에 못 미쳤다.\n\n"
       << "| 지표 | center_quadratic | center_linear | legacy_six_origins |\n|---|---|---|---|\n";
    auto rowf = [&](const std::string& lab, auto fn) {
        Pm << "| " << lab;
        for (auto& c : cfgs) Pm << " | " << fn(c.method);
        Pm << " |\n";
    };
    rowf("재대입 축·부호 정확도", [&](const std::string& k) { return fmt(S[{k, "full280_resubstitution"}].axis_sign_accuracy * 100, 2) + "%"; });
    rowf("재대입 Y 맞은 축 MAE (mm)", [&](const std::string& k) { return fmt(S[{k, "full280_resubstitution"}].Y_matched_MAE, 4); });
    rowf("재대입 T 맞은 축 MAE (deg)", [&](const std::string& k) { return fmt(S[{k, "full280_resubstitution"}].T_matched_MAE, 4); });
    rowf("반복 5분할 Y 축·부호", [&](const std::string& k) { return fmt(rep[k].Y_axis_sign_accuracy * 100, 4) + "%"; });
    rowf("반복 5분할 T 축·부호", [&](const std::string& k) { return fmt(rep[k].T_axis_sign_accuracy * 100, 4) + "%"; });
    rowf("구간 빼기 축·부호", [&](const std::string& k) { return fmt(S[{k, "session_out_missing_origin"}].axis_sign_accuracy * 100, 2) + "%"; });
    Pm << "\n" << img_md("01_재대입_예측오차.png", "재대입 예측 오차") << "\n\n" << img_md("02_반복5분할_축부호정확도.png", "반복 5분할 정확도") << "\n\n"
       << img_md("03_구간별_평균이동_0점.png", "구간별 평균 이동") << "\n\n"
       << "## 튀는 값\n\n재대입에서 틀린 " << fails.size() << "장은 모두 |값| 0.3 이하의 작은 이동이고, 가장 크게 틀린 " << m[RY.i].file << "(정답 " << RY.axis << " "
       << fmts(RY.value, 2) << ")는 " << RY.pred_axis << " " << fmts(RY.pred, 4) << "로 읽혔다. 이 사진의 평균 가로 이동 c0x " << fmt(X[RY.i][0], 3) << " px는 T " << fmts(RT.value, 2) << " 사진(" << fmt(X[RT.i][0], 3) << " px)과 거의 같아 나머지 11값의 모양 차이로만 갈리는데, 틀린 6장의 맞는 축과 고른 축 비용 차이가 " << fmt(gap_min, 2) << "~" << fmt(gap_max, 2) << "로 작아 뒤집혔다.\n\n"
       << ft_tab.md() << "\n" << img_md(fy, "Y 튀는 값") << "\n\n" << img_md(ft, "T 튀는 값")
       << "\n\n초록 상자는 center 의 QR 상자, 빨간 상자는 점 평균 이동만큼 옮긴 상자, 빨간 십자는 측정 점, 화살표는 QR 평균을 뺀 점 이동 x200이다. T 쪽 최대 오차는 " << m[RT.i].file
       << "(정답 " << RT.axis << " " << fmts(RT.value, 2) << " → " << RT.pred_axis << " " << fmts(RT.pred, 4) << ")다.\n\n"
       << "## 한줄평\n\n절편 0 곡선은 legacy 가 구간마다 따로 맞추는 12값 기준점(c0x 만 봐도 Y 구간 0점이 center 대비 "
       << fmts(zmin, 3) << "~" << fmts(zmax, 3) << " mm)을 center 하나로 묶어 버려서, 평균 이동이 같은 작은 Y와 작은 T를 모양 차이만으로 가르다 |Y| <= 0.3 mm 사진을 T 로 읽어 Y 축·부호 정확도가 "
       << fmt(rep["center_quadratic"].Y_axis_sign_accuracy * 100, 2) << "%에 머물렀다.\n\n"
       << "## 원래 결과와 대조\n\n원래 값은 `codex_experiments_20261014/center_fixed_20261015/metrics.csv`, `predictions.csv`(반복 5분할 합산과 사진별 추정), `checks.json`, 분할 서명은 `gls_comparison_20261014/splits.json`에서 계산했다. "
       << par.ok << "/" << par.total << " 일치.\n\n" << par.t.md();
    write_text(out("보고서.md"), Pm.str());
    std::cout << "pass " << npass << " repY " << rep["center_quadratic"].Y_axis_sign_accuracy << " parity " << par.ok << "/" << par.total << "\n";
    return 0;
}
