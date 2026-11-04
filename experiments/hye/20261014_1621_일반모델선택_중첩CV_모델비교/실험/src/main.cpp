// 중첩 교차검증으로 측정 방식·모델을 학습 자료 안에서만 고른다.
// 원본: codex_experiments_20261014/modeling/unified.py (+ experiment.py MotionModel, measurement/measure.py·refine.py·spread.py)
// 이 C++ 판은 unified.py 의 "isolated" 후보 25개(LK31, LK51, GN17 spread 17px/14px, LK51 QR18)를 사진에서 다시 재서 옮겼다.
#include <hye/dataset.hpp>
#include <hye/plot.hpp>
#include <hye/report.hpp>
#include <omp.h>
#include <iostream>
#include <map>
#include <set>
#include "kfold.hpp"
#include "motion.hpp"
#include "track.hpp"

using namespace hye;

namespace {

std::string file_of(const Shot& s) { return s.rel.substr(s.rel.rfind('/') + 1); }

struct Meas {  // 사진 한 장의 네 가지 측정
    std::vector<cv::Point2f> lk31, lk51, sp17, sp14;
    std::array<cv::Point, 3> init;
};

std::vector<mm::Cand> candidates() {
    // unified.py candidate_list 순서 그대로 (번호도 원래 전체 목록 번호)
    std::vector<mm::Cand> out;
    struct S { int deg; double sh; const char* nu; bool grid; };
    const S settings[6] = {{1, .1, "none", false}, {2, .1, "none", false}, {2, .1, "dy", false},
                           {2, .1, "both", false}, {2, 1., "none", false}, {2, .1, "dy", true}};
    int n = 0;
    for (std::string name : {"lk31", "lk51", "gn17", "gn31", "direct_p12_newinit", "gn17_spread", "gn17_spread14"})
        for (auto& s : settings) {
            char id[64];
            std::snprintf(id, sizeof id, "%s_%02d", name.c_str(), n++);
            mm::Cand c;
            c.meas = name; c.id = id; c.degree = s.deg; c.shrink = s.sh; c.nuisance = s.nu; c.grid = s.grid;
            out.push_back(c);
        }
    for (std::string name : {"lk51", "gn17"}) {
        mm::Cand c;
        c.meas = name; c.id = name + "_qr18"; c.kind = "qr18"; c.degree = 1; c.shrink = .001;
        out.push_back(c);
    }
    return out;
}
bool isolated(const mm::Cand& c) { return c.meas == "lk31" || c.meas == "lk51" || c.meas == "gn17_spread" || c.meas == "gn17_spread14"; }

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
    omp_set_num_threads(4);
    auto shots = load_1014();
    const cv::Mat ref = trk::crop(read_gray(center_1014(shots)));
    // 특징점: 928점(LK), spread 17px / 14px (GN17)
    trk::Features f928 = trk::features928(ref), f17 = trk::features_spread(ref, 17), f14 = trk::features_spread(ref, 14);
    trk::Spread sp17(ref, f17), sp14(ref, f14);
    std::vector<Meas> meas(shots.size());
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < (int)shots.size(); ++i) {
        cv::Mat cur = trk::crop(read_gray(shots[i]));
        Meas m;
        m.init = trk::coarse(ref, cur);
        std::vector<cv::Point2f> init;
        for (int r : f928.region) init.push_back(cv::Point2f((float)m.init[r].x, (float)m.init[r].y));
        m.lk31 = trk::lk_track(ref, cur, f928.pts, init, 31).d;
        m.lk51 = trk::lk_track(ref, cur, f928.pts, init, 51).d;
        m.sp17 = sp17.extract(cur).D;
        m.sp14 = sp14.extract(cur).D;
        meas[i] = std::move(m);
    }
    std::cout << "measured " << shots.size() << "\n";

    // 대상 280장: (축, 값) 순서. 시행(session) = T+/T-, Y1±/Y2± (Y 단계 홀짝)
    size_t ci = 0;
    std::vector<size_t> target;
    for (size_t i = 0; i < shots.size(); ++i) {
        if (shots[i].condition == "center") ci = i;
        else target.push_back(i);
    }
    auto fval = [&](size_t i) { auto f = file_of(shots[i]); return std::stod(f.substr(1, f.size() - 5)); };
    std::stable_sort(target.begin(), target.end(), [&](size_t a, size_t b) {
        char xa = file_of(shots[a])[0], xb = file_of(shots[b])[0];
        return xa != xb ? xa < xb : fval(a) < fval(b);
    });
    const int n = (int)target.size();
    std::vector<char> axes(n);
    std::vector<double> values(n);
    std::vector<std::string> sessions(n), names(n);
    for (int k = 0; k < n; ++k) {
        size_t i = target[k];
        names[k] = file_of(shots[i]);
        axes[k] = names[k][0];
        values[k] = fval(i);
        std::string s = axes[k] == 'T' ? "T" : (((long long)std::llabs((long long)std::nearbyint(values[k] / .05)) % 2) ? "Y2" : "Y1");
        sessions[k] = s + (values[k] > 0 ? "+" : "-");
    }
    // 측정 방식별 자료 (center 이동을 뺀다, float32 그대로)
    auto make = [&](const trk::Features& F, std::vector<cv::Point2f> Meas::*mp) {
        mm::Data d;
        for (auto& p : F.global()) { d.px.push_back(p.x); d.py.push_back(p.y); }
        d.region = F.region;
        const auto& c0 = meas[ci].*mp;
        for (int k = 0; k < n; ++k) {
            const auto& v = meas[target[k]].*mp;
            cv::Mat D((int)v.size(), 2, CV_64F);
            for (int j = 0; j < D.rows; ++j) {
                float dx = v[j].x - c0[j].x, dy = v[j].y - c0[j].y;
                D.at<double>(j, 0) = dx;
                D.at<double>(j, 1) = dy;
            }
            d.D.push_back(D);
        }
        return d;
    };
    std::map<std::string, mm::Data> data;
    data["lk31"] = make(f928, &Meas::lk31);
    data["lk51"] = make(f928, &Meas::lk51);
    data["gn17_spread"] = make(f17, &Meas::sp17);
    data["gn17_spread14"] = make(f14, &Meas::sp14);

    std::vector<mm::Cand> cs;
    for (auto& c : candidates()) if (isolated(c)) cs.push_back(c);

    auto evaluate = [&](const mm::Cand& c, const std::vector<kf::Split>& folds, std::vector<int>* fold_of = nullptr) {
        std::vector<mm::Rec> rec;
        const auto& d = data.at(c.meas);
        for (size_t f = 0; f < folds.size(); ++f) {
            mm::Model m;
            m.c = c;
            m.fit(d, folds[f].first, axes, values);
            for (int i : folds[f].second) {
                auto p = m.predict_one(d.D[i]);
                rec.push_back({i, axes[i], p.axis, values[i], p.value});
                if (fold_of) fold_of->push_back((int)f);
            }
        }
        return rec;
    };
    std::vector<int> all(n);
    for (int i = 0; i < n; ++i) all[i] = i;

    // ---- 전체 3분할 후보별 (배포 후보 선택용, 서술) ----
    auto folds0 = kf::split(all, sessions, 3, 20261014);
    std::vector<mm::Metrics> fullm;
    for (auto& c : cs) fullm.push_back(mm::metrics(evaluate(c, folds0)));

    // ---- 중첩: 바깥 층화 3분할 / 시행 제외 6분할, 안쪽 층화 3분할 (seed 20261015+fold) ----
    struct Nested {
        std::vector<mm::Rec> rec;
        std::vector<std::string> chosen, test_sessions;
        std::vector<mm::Metrics> fold_m, inner_best;
    };
    auto nested = [&](bool strat) {
        Nested out;
        std::vector<kf::Split> outer;
        std::vector<std::string> labels;
        if (strat) outer = folds0;
        else {
            std::set<std::string> us(sessions.begin(), sessions.end());
            for (auto& s : us) {
                kf::Split sp;
                for (int i = 0; i < n; ++i) (sessions[i] == s ? sp.second : sp.first).push_back(i);
                outer.push_back(sp);
                labels.push_back(s);
            }
        }
        for (size_t f = 0; f < outer.size(); ++f) {
            auto inner = kf::split(outer[f].first, sessions, 3, 20261015 + (uint32_t)f);
            int best = -1;
            mm::Metrics bm;
            for (size_t j = 0; j < cs.size(); ++j) {
                auto m = mm::metrics(evaluate(cs[j], inner));
                if (best < 0 || mm::better(m, bm)) { best = (int)j; bm = m; }
            }
            auto r = evaluate(cs[best], {outer[f]});
            out.rec.insert(out.rec.end(), r.begin(), r.end());
            out.chosen.push_back(cs[best].id);
            std::set<std::string> ts;
            for (int i : outer[f].second) ts.insert(sessions[i]);
            std::string t;
            for (auto& s : ts) t += (t.empty() ? "" : ",") + s;
            out.test_sessions.push_back(t);
            out.fold_m.push_back(mm::metrics(r));
            out.inner_best.push_back(bm);
        }
        return out;
    };
    Nested ns = nested(true), so = nested(false);
    auto mS = mm::metrics(ns.rec), mO = mm::metrics(so.rec);
    std::cout << "stratified Y " << mS.Yok << "/" << mS.Yn << " T " << mS.Tok << "/" << mS.Tn << "  session_out Y " << mO.Yok << "/" << mO.Yn << " T " << mO.Tok
              << "/" << mO.Tn << "\n";

    // ---- 원래 결과 (codex_experiments_20261014/modeling/unified_*.json, unified_full_cv_selection_metrics.csv,
    //      unified_isolated_*_selections.csv, measurement/*.npz 요약) ----
    Parity par;
    par.add("928점 특징 수 (QR)", 928, (double)f928.pts.size(), 0, 0);
    par.add("spread 17px 점 수", 157, (double)f17.pts.size(), 0, 0);
    par.add("spread 14px 점 수", 205, (double)f14.pts.size(), 0, 0);
    auto meanD = [&](const mm::Data& d, const std::string& file, int col) {
        for (int k = 0; k < n; ++k)
            if (names[k] == file) return cv::mean(d.D[k].col(col))[0];
        return kNaN;
    };
    // measurement/*.npz 에서 계산한 값 (center 이동 뺀 점 평균 x 이동, px)
    par.add("LK31 T+1.00 평균 x 이동 (px)", 79.496301, meanD(data["lk31"], "T+1.00.bmp", 0), 0.002, 6);
    par.add("LK51 Y-2.00 평균 x 이동 (px)", -28.699798, meanD(data["lk51"], "Y-2.00.bmp", 0), 0.002, 6);
    par.add("GN17 spread T-0.01 평균 x 이동 (px)", -1.059701, meanD(data["gn17_spread"], "T-0.01.bmp", 0), 0.001, 6);
    par.add("GN17 spread14 T+1.00 평균 x 이동 (px)", 79.494913, meanD(data["gn17_spread14"], "T+1.00.bmp", 0), 0.001, 6);
    par.add("GN17 spread14 Y-2.00 평균 y 이동 (px)", -0.015820, meanD(data["gn17_spread14"], "Y-2.00.bmp", 1), 0.001, 6);
    // 전체 3분할 후보별 축 오류 수 (unified_full_cv_selection_metrics.csv)
    const std::map<std::string, int> orig_err{{"lk31_00", 8}, {"lk31_01", 8}, {"lk31_02", 8}, {"lk31_03", 9}, {"lk31_04", 8}, {"lk31_05", 7},
                                              {"lk51_06", 7}, {"lk51_07", 6}, {"lk51_08", 7}, {"lk51_09", 7}, {"lk51_10", 7}, {"lk51_11", 8},
                                              {"gn17_spread_30", 7}, {"gn17_spread_31", 8}, {"gn17_spread_32", 8}, {"gn17_spread_33", 8},
                                              {"gn17_spread_34", 11}, {"gn17_spread_35", 7}, {"gn17_spread14_36", 6}, {"gn17_spread14_37", 6},
                                              {"gn17_spread14_38", 5}, {"gn17_spread14_39", 5}, {"gn17_spread14_40", 6}, {"gn17_spread14_41", 5},
                                              {"lk51_qr18", 3}};
    int same_err = 0;
    for (size_t j = 0; j < cs.size(); ++j) same_err += orig_err.at(cs[j].id) == fullm[j].axis_err;
    par.add("전체 3분할 후보 25개 중 축 오류 수가 같은 후보", 25, same_err, 0, 0);
    par.add("중첩 층화 Y 정확도 (%)", 92.50, 100.0 * mS.Yok / mS.Yn, 0.01, 2);
    par.add("중첩 층화 T 정확도 (%)", 98.50, 100.0 * mS.Tok / mS.Tn, 0.01, 2);
    par.add("중첩 층화 Y 일치 최대 오차 (mm)", 0.031236, mS.Ymax, 0.002, 6);
    par.add("시행 제외 Y 정확도 (%)", 85.00, 100.0 * mO.Yok / mO.Yn, 0.01, 2);
    par.add("시행 제외 T 정확도 (%)", 97.50, 100.0 * mO.Tok / mO.Tn, 0.01, 2);
    const std::vector<std::string> orig_sel_s{"lk51_06", "lk51_09", "gn17_spread_30"},
        orig_sel_o{"lk51_08", "lk51_06", "gn17_spread14_36", "lk31_01", "lk51_qr18", "gn17_spread_34"};
    auto join = [](const std::vector<std::string>& v) { std::string s; for (auto& x : v) s += (s.empty() ? "" : ", ") + x; return s; };
    par.add_text("층화 바깥 분할별 선택 후보", join(orig_sel_s), join(ns.chosen), orig_sel_s == ns.chosen);
    par.add_text("시행 제외 바깥 분할별 선택 후보", join(orig_sel_o), join(so.chosen), orig_sel_o == so.chosen);

    // ---- CSV ----
    Table tc{{"후보", "측정", "모델", "차수", "shrink", "nuisance", "grid", "축 오류", "원래 축 오류", "엄격 실패", "Y 일치 최대 (mm)", "T 일치 최대 (deg)"}};
    for (size_t j = 0; j < cs.size(); ++j) {
        auto& c = cs[j];
        auto& m = fullm[j];
        tc.add({c.id, c.meas, c.kind, std::to_string(c.degree), fmt(c.shrink, 3), c.nuisance, c.grid ? "yes" : "no", std::to_string(m.axis_err),
                std::to_string(orig_err.at(c.id)), std::to_string(m.strict_fail), fmt(m.Ymax, 4), fmt(m.Tmax, 4)});
    }
    tc.csv(out("full_cv_candidates.csv"));
    auto pred_table = [&](const std::vector<mm::Rec>& r, const std::string& file) {
        Table t{{"file", "session", "axis", "value", "pred_axis", "pred", "joint_correct"}};
        for (auto& x : r) t.add({names[x.i], sessions[x.i], std::string(1, x.axis), fmt(x.value, 2), std::string(1, x.pred_axis), fmt(x.pred, 5), x.joint() ? "1" : "0"});
        t.csv(out(file));
    };
    pred_table(ns.rec, "nested_stratified_predictions.csv");
    pred_table(so.rec, "nested_session_out_predictions.csv");
    auto sel_table = [&](const Nested& N, const std::vector<std::string>& orig) {
        Table t{{"바깥 분할", "시험 시행", "선택 후보", "원래 선택", "안쪽 축 오류", "시험 장수", "시험 축 오류", "시험 Y 최대 벡터 오차", "시험 T 최대 벡터 오차"}};
        for (size_t f = 0; f < N.chosen.size(); ++f)
            t.add({std::to_string(f), N.test_sessions[f], N.chosen[f], orig[f], std::to_string(N.inner_best[f].joint_err), std::to_string(N.fold_m[f].n),
                   std::to_string(N.fold_m[f].axis_err), fmt(N.fold_m[f].maxY, 4), fmt(N.fold_m[f].maxT, 4)});
        return t;
    };
    Table selS = sel_table(ns, orig_sel_s), selO = sel_table(so, orig_sel_o);

    // ---- 그림 ----
    {
        std::vector<std::string> cats;
        std::vector<double> v;
        int hi = 0;
        for (size_t j = 0; j < cs.size(); ++j) {
            cats.push_back(cs[j].id);
            v.push_back(fullm[j].axis_err);
            if (fullm[j].axis_err < fullm[hi].axis_err) hi = (int)j;
        }
        bar_chart("01_후보별_축오류_전체3분할.png", "Axis errors per candidate (3-fold CV, 280 images)", cats, v, "axis errors", hi);
    }
    auto pv = [&](const std::vector<mm::Rec>& r, const std::string& file, const std::string& title) {
        std::vector<double> ty, py, tt, pt;
        std::vector<std::string> nm;
        for (auto& x : r) {
            ty.push_back(x.axis == 'Y' ? x.value : 0); tt.push_back(x.axis == 'T' ? x.value : 0);
            py.push_back(x.pred_axis == 'Y' ? x.pred : 0); pt.push_back(x.pred_axis == 'T' ? x.pred : 0);
            nm.push_back(names[x.i].substr(0, names[x.i].size() - 4));
        }
        pred_vs_truth(file, title, ty, py, tt, pt, nm);
    };
    pv(ns.rec, "02_중첩층화_예측오차.png", "Nested stratified 3-fold");
    pv(so.rec, "03_시행제외_예측오차.png", "Leave-session-out");

    // 튀는 값: 시행 제외에서 축을 틀린 사진 중 벡터 오차 최대 (Y, T 각각)
    auto worst_of = [&](const std::vector<mm::Rec>& r, char a) {
        int w = -1;
        double e = -1;
        for (size_t k = 0; k < r.size(); ++k)
            if (r[k].axis == a) {
                double v = std::max(r[k].eY(), r[k].eT());
                if (v > e + 1e-12) { e = v; w = (int)k; }
            }
        return w;
    };
    // 사진별 QR 세 영역: 초록 = 기준 QR, 빨강 = 이 사진 GN17 spread14 평균 이동만큼 옮긴 QR. 점 = 측정 점(이동 벡터 x 배율, 평균 이동 뺀 잔차)
    auto outlier_fig = [&](const mm::Rec& r, const std::string& file, const std::string& cap) {
        size_t si = target[r.i];
        cv::Mat im = read_color(shots[si]);
        const auto& d = data["gn17_spread14"];
        std::vector<cv::Mat> tiles;
        for (int q = 0; q < 3; ++q) {
            auto b = trk::BOXES[q];
            double mx = 0, my = 0;
            int cnt = 0;
            for (size_t j = 0; j < d.px.size(); ++j)
                if (d.region[j] == q) { mx += d.D[r.i].at<double>((int)j, 0); my += d.D[r.i].at<double>((int)j, 1); ++cnt; }
            mx /= cnt; my /= cnt;
            cv::Rect2d refb(b[0], b[1], b[2] - b[0], b[3] - b[1]), now(b[0] + mx, b[1] + my, b[2] - b[0], b[3] - b[1]);
            std::vector<Mark> marks;
            for (size_t j = 0; j < d.px.size(); ++j)
                if (d.region[j] == q)
                    marks.push_back({{d.px[j] + mx, d.py[j] + my}, {d.D[r.i].at<double>((int)j, 0) - mx, d.D[r.i].at<double>((int)j, 1) - my}, "", kRed});
            cv::Rect roi = cv::Rect(refb | now) + cv::Size(40, 40) - cv::Point(20, 20);
            tiles.push_back(zoom(im, roi, 420, marks, 300, q == 0 ? cap : "",
                                 {{refb, kGreen, ""}, {now, kRed, "q" + std::to_string(q) + " dx " + fmt(mx, 2) + " dy " + fmt(my, 3)}}));
        }
        save_png(file, hstack(tiles));
    };
    int wy = worst_of(so.rec, 'Y'), wt = worst_of(so.rec, 'T');
    auto nm = [&](int k) { return names[so.rec[k].i].substr(0, names[so.rec[k].i].size() - 4); };
    auto cap = [&](int k) {
        auto& r = so.rec[k];
        return nm(k) + " -> " + std::string(1, r.pred_axis) + " " + fmts(r.pred, 3);
    };
    outlier_fig(so.rec[wy], "04_튀는값_Y_" + nm(wy) + ".png", cap(wy));
    outlier_fig(so.rec[wt], "05_튀는값_T_" + nm(wt) + ".png", cap(wt));

    // 틀린 사진 표 (시행 제외)
    auto chosen_of = [&](const Nested& N, int i) {
        for (size_t f = 0; f < N.chosen.size(); ++f)
            if (N.test_sessions[f].find(sessions[i]) != std::string::npos) return N.chosen[f];
        return std::string("?");
    };
    Table bad{{"평가", "사진", "시행", "정답 Y (mm)", "정답 T (deg)", "추정 축", "추정값", "Y 벡터 오차", "T 벡터 오차", "선택 후보"}};
    for (auto* pr : {&ns, &so})
        for (auto& r : pr->rec)
            if (!r.joint())
                bad.add({pr == &ns ? "층화" : "시행 제외", names[r.i], sessions[r.i], fmt(r.axis == 'Y' ? r.value : 0, 2), fmt(r.axis == 'T' ? r.value : 0, 2),
                         std::string(1, r.pred_axis), fmts(r.pred, 3), fmt(r.eY(), 3), fmt(r.eT(), 3),
                         pr == &ns ? std::string("-") : chosen_of(so, r.i)});

    auto metric_rows = [&](Table& t, const std::string& lab, const mm::Metrics& m) {
        t.add({lab, std::to_string(m.Yok) + " / " + std::to_string(m.Yn) + " (" + fmt(100.0 * m.Yok / m.Yn, 2) + "%)",
               std::to_string(m.Tok) + " / " + std::to_string(m.Tn) + " (" + fmt(100.0 * m.Tok / m.Tn, 2) + "%)", std::to_string(m.n - m.strict_fail) + " / " + std::to_string(m.n),
               fmt(m.Ymax, 4) + " / " + fmt(m.Ymae, 4), fmt(m.Tmax, 4) + " / " + fmt(m.Tmae, 4), fmt(m.maxY, 3), fmt(m.maxT, 3)});
    };
    Table head{{"평가", "Y 축·부호 정답", "T 축·부호 정답", "통과 (축·부호·0.09)", "Y 일치 최대 / 평균 (mm)", "T 일치 최대 / 평균 (deg)", "Y 최대 벡터 오차", "T 최대 벡터 오차"}};
    metric_rows(head, "중첩 층화 3분할", mS);
    metric_rows(head, "시행 제외 6분할", mO);

    // ---- 결과.md ----
    std::ostringstream R;
    R << "# 결과: 측정·모델 중첩 선택 (isolated 후보 25개)\n\n"
      << "입력: `data/clean/clean_20261014` 281장. center.bmp를 기준으로 280장(T 200, Y 80)의 QR 내부 점 이동을 사진에서 다시 쟀다.\n"
      << "측정 4가지: LK31·LK51(928점, QR 전체 NCC 초기화, 단일 층 LK), GN17 spread(점 간격 17px, 157점)·spread14(14px, 205점, QR별 잘라 LK15 후 3차 스플라인 GN17).\n"
      << "모델: 점 이동 -> P12 공간 계수(또는 QR별 affine 18) -> 축별 2차 곡선, 백색화 거리로 축·값 선택. 후보는 측정 x 6설정 + LK51 QR18 = 25개.\n"
      << "평가: 바깥 층화 3분할(seed 20261014)과 시행 제외 6분할. 바깥 학습 자료 안에서만 층화 3분할(seed 20261015+분할)로 후보를 골랐다.\n\n"
      << "## 요약\n\n" << head.md() << "\n## 바깥 분할별 선택 (층화)\n\n" << selS.md() << "\n## 바깥 분할별 선택 (시행 제외)\n\n" << selO.md()
      << "\n## 전체 3분할 후보별 성적 (서술용, 배포 후보 선택에 쓰던 표)\n\n" << tc.md() << "\n## 축·부호를 틀린 사진\n\n" << bad.md()
      << "\n## 원래 결과와 대조 (" << par.ok << "/" << par.total << " 일치)\n\n" << par.t.md()
      << "\n## 옮기지 않은 부분\n\n"
      << "- unified.py의 strict 후보(gn17·gn31 전체 crop GN, direct_p12_newinit)와 그 중첩 결과. direct_p12는 `20261014_1622_QR직접정합_픽셀P12정합`에서 옮긴다.\n"
      << "- experiment.py의 옛 71후보(analysis_v6 hp8 추적 캐시 입력), new_measurement_models.py의 gn17/direct_p12 단독 중첩, screws_fallback, 배포 pkl 저장.\n\n"
      << "## 파일\n\n- `full_cv_candidates.csv`: 후보 25개의 전체 3분할 성적\n- `nested_stratified_predictions.csv`, `nested_session_out_predictions.csv`: 바깥 시험 사진별 예측\n";
    write_text(out("결과.md"), R.str());

    // ---- 보고서.md ----
    auto& WY = so.rec[wy];
    auto& WT = so.rec[wt];
    std::ostringstream P;
    {
        int lo = 1 << 30, hi = 0;
        for (auto& m : fullm) { lo = std::min(lo, m.axis_err); hi = std::max(hi, m.axis_err); }
        double bad_y = 0, bad_t = 0;
        for (auto& r : so.rec)
            if (!r.joint()) {
                double& b = r.axis == 'Y' ? bad_y : bad_t;
                b = std::max(b, std::abs(r.value));
            }
        size_t distinct = std::set<std::string>(so.chosen.begin(), so.chosen.end()).size();
        P << "# 보고서: 측정·모델 중첩 교차검증 선택 (10-14 16:21)\n\n"
          << "## 문제\n\n점 이동을 Y/T 축·부호·이동량으로 바꾸는 모델을 학습 자료 안에서만 골라 새 자료에서의 성능을 재야 했다.\n\n"
          << "## 해결 방법\n\nLK31·LK51·GN17 spread 두 가지 측정과 P12·QR18 모델 설정 25개를 바깥 층화 3분할과 시행 제외 6분할 안에서 안쪽 3분할로 고르고, 고른 후보를 바깥 시험 사진에만 적용했다.\n\n"
          << "## 결과\n\n중첩 층화에서 Y " << fmt(100.0 * mS.Yok / mS.Yn, 2) << "%·T " << fmt(100.0 * mS.Tok / mS.Tn, 2) << "%, 시행 제외에서 Y "
          << fmt(100.0 * mO.Yok / mO.Yn, 2) << "%·T " << fmt(100.0 * mO.Tok / mO.Tn, 2) << "%로 전부 맞히지 못했다.\n\n"
          << head.md() << "\n" << img_md("01_후보별_축오류_전체3분할.png", "후보별 축 오류") << "\n\n" << img_md("02_중첩층화_예측오차.png", "층화 예측 오차")
          << "\n\n" << img_md("03_시행제외_예측오차.png", "시행 제외 예측 오차") << "\n\n"
          << "## 튀는 값\n\n시행 제외에서 가장 크게 틀린 Y 사진은 " << nm(wy) << "(정답 Y " << fmt(WY.value, 2) << " mm를 " << WY.pred_axis << " " << fmts(WY.pred, 3)
          << "로 읽음), T 사진은 " << nm(wt) << "(정답 T " << fmt(WT.value, 2) << " deg를 " << WT.pred_axis << " " << fmts(WT.pred, 3)
          << "로 읽음)이다. 시험 시행의 0점 이동을 학습에서 못 봐서 작은 이동의 축·부호를 다른 곡선에 붙였다.\n\n"
          << bad.md() << "\n" << img_md("04_튀는값_Y_" + nm(wy) + ".png", "Y 튀는 값") << "\n\n" << img_md("05_튀는값_T_" + nm(wt) + ".png", "T 튀는 값")
          << "\n\n초록 상자는 기준 사진의 QR, 빨간 상자는 GN17 spread14 점 평균 이동만큼 옮긴 QR이다. 빨간 십자는 측정 점, 화살표는 QR 평균을 뺀 점별 잔차 x300이다.\n\n"
          << "## 한줄평\n\n모든 후보가 점 이동을 축별 곡선 하나에 맞추는 구조라 작은 Y(시행 제외에서 틀린 Y는 |정답| " << fmt(bad_y, 2)
          << " mm 이하)와 작은 T(틀린 T는 |정답| " << fmt(bad_t, 2) << " deg 이하)가 같은 방향 이동으로 겹쳤고, 후보 25개의 전체 3분할 축 오류가 " << lo << "~" << hi
          << "개로 흩어져 안쪽 분할마다 다른 후보(시행 제외 6분할에서 " << distinct << "가지)가 뽑혀 선택만으로 이 겹침을 없애지 못했다.\n\n"
          << "## 원래 결과와 대조\n\n원래 값은 `modeling/unified_isolated_stratified_metrics.json`, `unified_isolated_session_out_metrics.json`, "
             "`unified_isolated_*_selections.csv`, `unified_full_cv_selection_metrics.csv`, `measurement/*.npz`에서 가져왔다. "
          << par.ok << "/" << par.total << " 일치.\n\n" << par.t.md();
    }
    write_text(out("보고서.md"), P.str());
    std::cout << "parity " << par.ok << "/" << par.total << "\n";
    return 0;
}
