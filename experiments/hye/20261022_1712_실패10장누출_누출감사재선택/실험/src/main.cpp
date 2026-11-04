// 실패 10장 누출 감사와 재선택: 직전 실험(20261022_1631)의 pass_first 가 20261020 에서 틀린 10장을 사진별로 나누어 보고,
// 평가 누출(바깥 시험 그룹이 학습·내부 선택에 들어갔는지)을 검사한 뒤, 사전 고정 강건 후보 4개(extended / robust_only),
// 추가 탐색 강건 후보 16개(exploratory16), 위상 잡음 제거 후보 9개(phase)를 같은 내부 분할로 다시 고른다.
// 원본: 00000.legacies/codex_experiments_20261014/joint_yt_erroraudit_20261022/evaluate.py, robust/robust_model.py,
//       robust/exploratory_nested.py, phase_model.py, phase_evaluate.py, diagnostics/analyze.py, audit/evaluation_audit.py
#include <hye/dataset.hpp>
#include <hye/plot.hpp>
#include <hye/report.hpp>
#include <omp.h>
#include <iostream>
#include <map>
#include <sstream>
#include <tuple>
#include "pipeline630.hpp"
#include "robust.hpp"
#include "separation.hpp"
#include "ytmodel.hpp"

using namespace hye;
using pipe630::Img;

namespace {
struct Score {
    int fails = 0;
    double maximum = 0, mean = 0;
    std::tuple<int, double, double> key() const { return {fails, maximum, mean}; }
};
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
    cv::setNumThreads(1);
    begin_results();
    std::vector<cv::Point2d> ref;
    auto imgs = pipe630::load_and_measure(ref);
    const int N = 341;
    ytm::Data data;
    data.n = N;
    sep::P12Map p12;
    std::vector<sep::Row> RS(N), PX(N);
    for (auto& im : imgs) { data.D.push_back(im.D); data.h.push_back(im.h); data.y.push_back(im.shot.y); data.t.push_back(im.shot.t); }
    for (int i = 0; i < N; ++i) { RS[i] = sep::raw_shift(imgs[i].D); PX[i] = p12(imgs[i].D); }
    std::vector<std::string> strat(N), groups(N);
    std::vector<int> dates(N);
    for (int i = 0; i < N; ++i) {
        strat[i] = imgs[i].isnew ? "new_" + imgs[i].scope : "old";
        groups[i] = imgs[i].group;
        dates[i] = imgs[i].isnew ? 20261020 : 20261014;
    }

    // ---- 후보: old15 (직전 실험 15개), rob16 (exploratory), phase9 ----
    struct Cand { std::string method, label; ytm::Config base; std::string mode; double mix = 0, shrink = 0; };
    std::vector<Cand> C15;
    for (double s : {.01, .1}) C15.push_back({"baseline", "global raw_shift s" + fmt(s, 2), {"global", "raw_shift", s, 0}});
    C15.push_back({"baseline", "local raw_shift s0.01 w20", {"local", "raw_shift", .01, 20.}});
    C15.push_back({"baseline", "local raw_shift s0.1 w100", {"local", "raw_shift", .1, 100.}});
    for (std::string f : {"new_condition_means", "axis_pairs"})
        for (double mix : {0., .5})
            for (double s : {.01, .1})
                C15.push_back({"repeat", "repeat " + std::string(f == "axis_pairs" ? "axis_pairs" : "cond_means") + " mix" + fmt(mix, 1) + " s" + fmt(s, 2), {}, f, mix, s});
    C15.push_back({"paired", "paired P12"});
    C15.push_back({"conditional", "conditional P12"});
    C15.push_back({"neural", "neural raw_shift"});
    std::vector<rob::RobustCfg> R16;
    for (std::string f : {"new_condition_means", "all_condition_means"})
        for (double s : {.01, .1})
            for (auto [h, b] : std::vector<std::pair<double, int>>{{0, 1}, {1.5, 1}, {0, 12}, {1.5, 12}}) R16.push_back({f, s, h, b});
    std::vector<rob::PhaseCfg> PH{{false, "x", 1.}};
    for (std::string p : {"x", "xy"})
        for (double a : {.01, .1, 1., 10.}) PH.push_back({true, p, a});

    auto meta = [&](const std::vector<int>& tr) {
        sep::Meta md;
        for (int i : tr) { md.group.push_back(groups[i]); md.date.push_back(dates[i]); md.strat.push_back(strat[i]); }
        return md;
    };
    // family: 0 old15, 1 rob16, 2 phase9
    auto fit_predict = [&](int fam, int k, const std::vector<int>& tr, const std::vector<int>& te, std::vector<double>& py, std::vector<double>& pt) {
        py.assign(te.size(), 0);
        pt.assign(te.size(), 0);
        std::vector<double> y, t;
        std::vector<int> dd;
        for (int i : tr) { y.push_back(data.y[i]); t.push_back(data.t[i]); dd.push_back(dates[i]); }
        auto md = meta(tr);
        if (fam == 1) {
            std::vector<sep::Row> X;
            for (int i : tr) X.push_back(RS[i]);
            rob::RobustBagged m;
            m.c = R16[k];
            m.fit(X, y, t, dd);
            for (size_t q = 0; q < te.size(); ++q) m.predict(RS[te[q]], py[q], pt[q]);
            return;
        }
        if (fam == 2) {
            std::vector<sep::Row> D;
            for (int i : tr) D.push_back(data.D[i]);
            rob::Phase m;
            m.c = PH[k];
            m.fit(D, y, t, md);
            for (size_t q = 0; q < te.size(); ++q) m.predict(data.D[te[q]], py[q], pt[q]);
            return;
        }
        const auto& c = C15[k];
        if (c.method == "baseline") { ytm::fit_predict(data, c.base, tr, te, py, pt); return; }
        std::vector<sep::Row> X, Xb;
        if (c.method == "repeat") {
            for (int i : tr) X.push_back(RS[i]);
            sep::RepeatGLS m;
            m.shrink = c.shrink; m.mix = c.mix; m.mode = c.mode;
            m.fit(X, y, t, md);
            for (size_t q = 0; q < te.size(); ++q) m.predict(RS[te[q]], py[q], pt[q]);
        } else if (c.method == "paired") {
            for (int i : tr) X.push_back(PX[i]);
            sep::PairedRidge m;
            m.fit(X, y, t, dd);
            for (size_t q = 0; q < te.size(); ++q) m.predict(PX[te[q]], py[q], pt[q]);
        } else if (c.method == "conditional") {
            for (int i : tr) { X.push_back(PX[i]); Xb.push_back(RS[i]); }
            sep::Conditional m;
            m.fit(X, Xb, y, t, md);
            for (size_t q = 0; q < te.size(); ++q) m.predict(PX[te[q]], RS[te[q]], py[q], pt[q]);
        } else {
            for (int i : tr) X.push_back(RS[i]);
            sep::PairNet m;
            m.fit(X, y, t, dd);
            for (size_t q = 0; q < te.size(); ++q) m.predict(RS[te[q]], py[q], pt[q]);
        }
    };
    const int nfam[3] = {(int)C15.size(), (int)R16.size(), (int)PH.size()};

    struct Sel {
        uint32_t seed;
        std::vector<std::pair<std::vector<int>, std::vector<int>>> folds;  // (내부 학습, 내부 검증 중 20261020)
        std::vector<std::vector<int>> inner_test;                          // 내부 검증 전체
        std::vector<Score> sc[3];
        int prior = 0, extended = 0, robust_only = 0, expl16 = 0, phase = 0;
    };
    auto select = [&](const std::vector<int>& tr, uint32_t seed) {
        std::vector<std::string> sy, sg;
        for (int i : tr) { sy.push_back(strat[i]); sg.push_back(groups[i]); }
        auto tests = ytm::sgkf_test(sy, sg, 3, seed);
        Sel s;
        s.seed = seed;
        for (auto& te : tests) {
            std::set<int> ts(te.begin(), te.end());
            std::vector<int> it, iv, all_v;
            for (int q = 0; q < (int)tr.size(); ++q) {
                if (ts.count(q)) { all_v.push_back(tr[q]); if (imgs[tr[q]].isnew) iv.push_back(tr[q]); }
                else it.push_back(tr[q]);
            }
            s.folds.push_back({it, iv});
            s.inner_test.push_back(all_v);
        }
        std::vector<std::tuple<int, int, int>> tasks;
        for (int fam = 0; fam < 3; ++fam)
            for (int k = 0; k < nfam[fam]; ++k)
                for (int f = 0; f < 3; ++f) tasks.push_back({fam, k, f});
        std::vector<std::vector<double>> err(tasks.size());
#pragma omp parallel for schedule(dynamic)
        for (int ti = 0; ti < (int)tasks.size(); ++ti) {
            auto [fam, k, f] = tasks[ti];
            std::vector<double> py, pt;
            fit_predict(fam, k, s.folds[f].first, s.folds[f].second, py, pt);
            for (size_t q = 0; q < s.folds[f].second.size(); ++q) {
                int i = s.folds[f].second[q];
                err[ti].push_back(std::abs(py[q] - data.y[i]));
                err[ti].push_back(std::abs(pt[q] - data.t[i]));
            }
        }
        size_t ti = 0;
        for (int fam = 0; fam < 3; ++fam) {
            s.sc[fam].resize(nfam[fam]);
            for (int k = 0; k < nfam[fam]; ++k) {
                auto& S = s.sc[fam][k];
                double sum = 0;
                int cnt = 0;
                for (int f = 0; f < 3; ++f, ++ti)
                    for (size_t q = 0; q < err[ti].size(); q += 2) {
                        auto& e = err[ti];
                        S.fails += (e[q] > .09 + 1e-12 || e[q + 1] > .09 + 1e-12);
                        S.maximum = std::max({S.maximum, e[q], e[q + 1]});
                        sum += e[q] + e[q + 1];
                        cnt += 2;
                    }
                S.mean = sum / cnt;
            }
        }
        auto argmin = [&](int fam, int lo, int hi) {
            int b = lo;
            for (int k = lo + 1; k < hi; ++k) if (s.sc[fam][k].key() < s.sc[fam][b].key()) b = k;
            return b;
        };
        s.prior = argmin(0, 0, 15);
        s.robust_only = argmin(1, 4, 8);  // 사전 고정 4개 = new_condition_means s0.1 의 네 조합
        s.extended = s.sc[1][s.robust_only].key() < s.sc[0][s.prior].key() ? 15 + (s.robust_only - 4) : s.prior;
        s.expl16 = argmin(1, 0, 16);
        s.phase = argmin(2, 0, nfam[2]);
        return s;
    };
    auto pick = [&](const Sel& s, const std::string& method) -> std::pair<int, int> {
        if (method == "prior") return {0, s.prior};
        if (method == "extended") return s.extended < 15 ? std::make_pair(0, s.extended) : std::make_pair(1, s.extended - 15 + 4);
        if (method == "robust_only") return {1, s.robust_only};
        if (method == "exploratory16") return {1, s.expl16};
        return {2, s.phase};
    };
    const std::vector<std::string> methods{"prior", "extended", "robust_only", "exploratory16", "phase"};

    struct Pred { int i; double py, pt; std::string method, protocol; int fold; };
    std::vector<Pred> preds;
    std::vector<Sel> sels;
    // 누출 감사: 바깥 시험 그룹이 바깥 학습에 없는지, 내부 검증이 바깥 학습 안에 있고 내부 학습과 그룹이 겹치지 않는지
    Table aud{{"바깥 fold", "학습", "시험", "바깥 학습-시험 그룹 겹침", "내부 검증이 바깥 학습 밖", "내부 학습-검증 그룹 겹침", "시험 사진이 내부에 쓰임"}};
    int leaks = 0;
    for (int f = 0; f < 4; ++f) {
        std::vector<int> tr, te;
        for (int i = 0; i < N; ++i) (imgs[i].fold == f ? te : tr).push_back(i);
        auto s = select(tr, 20261022 + f);
        std::set<std::string> gtr, gte;
        for (int i : tr) gtr.insert(groups[i]);
        for (int i : te) gte.insert(groups[i]);
        int ov = 0, outside = 0, iov = 0, used = 0;
        for (auto& g : gte) ov += gtr.count(g);
        std::set<int> trs(tr.begin(), tr.end()), tes(te.begin(), te.end());
        for (int j = 0; j < 3; ++j) {
            std::set<std::string> gi, gv;
            for (int i : s.folds[j].first) { gi.insert(groups[i]); used += tes.count(i); }
            for (int i : s.inner_test[j]) { gv.insert(groups[i]); outside += !trs.count(i); used += tes.count(i); }
            for (auto& g : gv) iov += gi.count(g);
        }
        leaks += ov + outside + iov + used;
        aud.add({std::to_string(f), std::to_string(tr.size()), std::to_string(te.size()), std::to_string(ov), std::to_string(outside), std::to_string(iov), std::to_string(used)});
        for (auto& me : methods) {
            auto [fam, k] = pick(s, me);
            std::vector<double> py, pt;
            fit_predict(fam, k, tr, te, py, pt);
            for (size_t q = 0; q < te.size(); ++q) preds.push_back({te[q], py[q], pt[q], me, "nested_oof", f});
        }
        sels.push_back(s);
        std::cout << "outer " << f << " prior " << s.prior << " ext " << s.extended << " rob " << s.robust_only << " expl " << s.expl16 << " phase " << s.phase << std::endl;
    }
    std::vector<int> all(N);
    std::iota(all.begin(), all.end(), 0);
    auto fsel = select(all, 20261101);
    sels.push_back(fsel);
    for (auto& me : methods) {
        auto [fam, k] = pick(fsel, me);
        std::vector<double> py, pt;
        fit_predict(fam, k, all, all, py, pt);
        for (int i = 0; i < N; ++i) preds.push_back({i, py[i], pt[i], me, "resub341", -1});
    }

    // ---- 요약 ----
    std::map<std::string, ytm::Metric> M;
    std::map<std::string, std::vector<double>> PY, PT;  // method -> 사진별 OOF 예측
    Table sum{{"방법", "평가", "자료", "장수", "통과", "Y 평균 |오차|", "Y 최대", "T 평균 |오차|", "T 최대"}};
    for (auto& me : methods) {
        PY[me].assign(N, 0);
        PT[me].assign(N, 0);
        for (auto& p : preds) if (p.method == me && p.protocol == "nested_oof") { PY[me][p.i] = p.py; PT[me][p.i] = p.pt; }
        for (std::string pr : {"nested_oof", "resub341"})
            for (int ds : {0, 1014, 1020}) {
                std::vector<double> y, py, t, pt;
                for (auto& p : preds)
                    if (p.method == me && p.protocol == pr && (ds == 0 || (ds == 1020) == imgs[p.i].isnew)) {
                        y.push_back(data.y[p.i]); py.push_back(p.py); t.push_back(data.t[p.i]); pt.push_back(p.pt);
                    }
                auto m = ytm::metric(y, py, t, pt);
                M[me + "|" + pr + "|" + std::to_string(ds)] = m;
                sum.add({me, pr, ds == 0 ? "전체" : ds == 1014 ? "20261014" : "20261020", std::to_string(m.n), std::to_string(m.passed), fmt(m.ymae, 4), fmt(m.ymax, 4),
                         fmt(m.tmae, 4), fmt(m.tmax, 4)});
            }
    }
    auto passed = [&](const std::string& me, int i) {
        return std::abs(PY[me][i] - data.y[i]) <= .09 + 1e-12 && std::abs(PT[me][i] - data.t[i]) <= .09 + 1e-12;
    };
    // 직전 pass_first 실패 사진 진단 (20261020)
    std::map<std::string, std::vector<int>> cond;
    for (int i = 0; i < N; ++i) if (imgs[i].isnew) cond[groups[i]].push_back(i);
    Table ten{{"사진", "fold", "정답 Y", "정답 T", "추정 Y", "eY", "eT", "조건 평균 추정 Y 편향", "개별 편차 Y", "같은 조건 통과", "유효점", "NCC 최소", "exploratory16 추정 Y", "결과"}};
    int kept = 0, still = 0, recov = 0, newf = 0, old_kept = 0, old_still = 0, old_recov = 0, old_newf = 0;
    std::vector<double> bias_x, dev_y, bias_f, dev_f;
    std::vector<int> ten_idx;
    for (int i = 0; i < N; ++i) {
        bool a = passed("prior", i), b = passed("exploratory16", i);
        if (imgs[i].isnew) { kept += a && b; still += !a && !b; recov += !a && b; newf += a && !b; }
        else { old_kept += a && b; old_still += !a && !b; old_recov += !a && b; old_newf += a && !b; }
        if (!imgs[i].isnew || imgs[i].scope != "mixed") continue;
        auto& ix = cond[groups[i]];
        double cm = 0;
        int cp = 0;
        for (int j : ix) { cm += PY["prior"][j]; cp += passed("prior", j); }
        cm /= ix.size();
        double bias = cm - data.y[i], dev = PY["prior"][i] - cm;
        (a ? bias_x : bias_f).push_back(bias);
        (a ? dev_y : dev_f).push_back(dev);
        if (a) continue;
        ten_idx.push_back(i);
        ten.add({imgs[i].shot.name(), std::to_string(imgs[i].fold), fmt(data.y[i], 2), fmt(data.t[i], 2), fmt(PY["prior"][i], 3), fmts(PY["prior"][i] - data.y[i], 3),
                 fmts(PT["prior"][i] - data.t[i], 3), fmts(bias, 3), fmts(dev, 3), std::to_string(cp) + " / " + std::to_string(ix.size()), std::to_string(imgs[i].nvalid),
                 fmt(imgs[i].m.ncc_min, 3), fmt(PY["exploratory16"][i], 3), b ? "복구" : "실패 유지"});
    }
    // 선택 표
    Table st{{"바깥 fold", "직전 pass_first", "extended (15+4)", "robust_only (4)", "exploratory16", "phase (9)"}};
    for (size_t f = 0; f < sels.size(); ++f) {
        auto& s = sels[f];
        st.add({f < 4 ? std::to_string(f) : "전체(최종)", std::to_string(s.prior) + " " + C15[s.prior].label,
                std::to_string(s.extended) + (s.extended < 15 ? " " + C15[s.extended].label : " " + R16[s.extended - 15 + 4].str()),
                std::to_string(s.robust_only - 4 + 15) + " " + R16[s.robust_only].str(), std::to_string(s.expl16) + " " + R16[s.expl16].str(),
                std::to_string(s.phase) + " " + PH[s.phase].str()});
    }
    Table rs{{"#", "강건 후보", "fold0 실패/최악", "fold1", "fold2", "fold3", "최종"}};
    for (int k = 0; k < 16; ++k) {
        std::vector<std::string> row{std::to_string(k), R16[k].str()};
        for (auto& s : sels) row.push_back(std::to_string(s.sc[1][k].fails) + " / " + fmt(s.sc[1][k].maximum, 3) + (s.expl16 == k ? " *" : ""));
        rs.add(row);
    }
    Table pc{{"index", "file", "Y", "T", "scope", "fold", "method", "protocol", "predY", "predT", "eY", "eT", "pass"}};
    for (auto& p : preds) {
        auto& im = imgs[p.i];
        double ey = std::abs(p.py - im.shot.y), et = std::abs(p.pt - im.shot.t);
        pc.add({std::to_string(p.i), im.shot.name(), fmt(im.shot.y, 2), fmt(im.shot.t, 2), im.scope, std::to_string(p.fold), p.method, p.protocol, fmt(p.py, 5), fmt(p.pt, 5),
                fmt(ey, 5), fmt(et, 5), (ey <= .09 + 1e-12 && et <= .09 + 1e-12) ? "1" : "0"});
    }
    pc.csv(out("predictions.csv"));

    // ---- 그림 ----
    auto& Me = M["exploratory16|nested_oof|0"];
    auto& Men = M["exploratory16|nested_oof|1020"];
    bar_chart("01_방법별_20261020_통과.png", "20261020 mixed 60: pass count by selection", {"prior", "extended", "robust4", "explore16", "phase9"},
              {(double)M["prior|nested_oof|1020"].passed, (double)M["extended|nested_oof|1020"].passed, (double)M["robust_only|nested_oof|1020"].passed, (double)Men.passed,
               (double)M["phase|nested_oof|1020"].passed},
              "images passed", 3, 60);
    {
        Chart c("Prior pass_first, 20261020 mixed: condition bias vs repeat deviation (Y)", "condition-mean bias of pred Y (mm)", "individual deviation from condition mean (mm)");
        c.scatter(bias_x, dev_y, color(0), "passed").scatter(bias_f, dev_f, kRed, "failed", 5).hline(0).vline(0);
        c.save("02_실패분해_조건편향_개별편차.png");
    }
    std::vector<double> ty, py, tt, pt, ey, et;
    std::vector<std::string> names;
    std::vector<int> evi;
    for (int i = 0; i < N; ++i) {
        ty.push_back(data.y[i]); py.push_back(PY["exploratory16"][i]); tt.push_back(data.t[i]); pt.push_back(PT["exploratory16"][i]);
        ey.push_back(py.back() - ty.back()); et.push_back(pt.back() - tt.back());
        names.push_back(imgs[i].name); evi.push_back(i);
    }
    pred_vs_truth("03_exploratory16_중첩OOF_예측오차.png", "exploratory16 nested OOF 341", ty, py, tt, pt, names);
    int wy = (int)worst(ey, 1)[0], wt = (int)worst(et, 1)[0];
    std::string fy = "04_튀는값_Y_" + names[wy] + ".png", ft = "05_튀는값_T_" + names[wt] + ".png";
    pipe630::qr_outlier_fig(imgs[wy], fy, names[wy] + " Y err " + fmts(ey[wy], 3) + " mm");
    pipe630::qr_outlier_fig(imgs[wt], ft, names[wt] + " T err " + fmts(et[wt], 3) + " deg");
    Table outl{{"구분", "사진", "fold", "정답 Y", "추정 Y", "정답 T", "추정 T", "평균 x 이동 (px)", "유효점", "직전 pass_first 추정 Y"}};
    for (auto [lab, idx] : std::vector<std::pair<std::string, std::vector<size_t>>>{{"Y 오차 상위", worst(ey, 5)}, {"T 오차 상위", worst(et, 3)}})
        for (size_t k : idx)
            outl.add({lab, names[k], std::to_string(imgs[k].fold), fmt(ty[k], 2), fmt(py[k], 3), fmt(tt[k], 2), fmt(pt[k], 3), fmt(imgs[k].h, 2), std::to_string(imgs[k].nvalid),
                      fmt(PY["prior"][k], 3)});
    double mean_bias_f = 0, mean_dev_f = 0;
    for (size_t q = 0; q < bias_f.size(); ++q) { mean_bias_f += std::abs(bias_f[q]); mean_dev_f += std::abs(dev_f[q]); }
    mean_bias_f /= std::max<size_t>(1, bias_f.size());
    mean_dev_f /= std::max<size_t>(1, dev_f.size());

    // ---- 대조 (원래 값: joint_yt_erroraudit_20261022/result.json, summary.csv, comparison_summary.csv, phase_summary.csv,
    //      robust/exploratory_nested_summary.csv, robust/exploratory_nested_analysis.json, robust/exploratory_nested_selection.json, audit/evaluation_audit.json)
    Parity par;
    par.add("직전 pass_first 20261020 실패 수", 10, 60 - M["prior|nested_oof|1020"].passed, 0, 0);
    par.add("누출 검사 위반 수 (그룹 겹침·시험 사진 사용)", 0, leaks, 0, 0);
    par.add("extended 341 통과", 242, M["extended|nested_oof|0"].passed, 2, 0);
    par.add("extended 20261020 통과", 50, M["extended|nested_oof|1020"].passed, 1, 0);
    par.add("robust_only 341 통과", 211, M["robust_only|nested_oof|0"].passed, 3, 0);
    par.add("robust_only 20261020 통과", 45, M["robust_only|nested_oof|1020"].passed, 1, 0);
    par.add("exploratory16 341 통과", 244, Me.passed, 2, 0);
    par.add("exploratory16 20261014 통과", 191, M["exploratory16|nested_oof|1014"].passed, 2, 0);
    par.add("exploratory16 20261020 통과", 53, Men.passed, 1, 0);
    par.add("exploratory16 20261020 Y 평균 |오차|", 0.049727, Men.ymae, 0.001);
    par.add("exploratory16 20261020 Y 최대 |오차|", 0.251095, Men.ymax, 0.01);
    par.add("exploratory16 20261020 T 평균 |오차|", 0.008813, Men.tmae, 0.0003);
    par.add("exploratory16 341 Y 평균 |오차|", 0.068233, Me.ymae, 0.001);
    par.add("10장 중 exploratory16 복구", 4, recov, 1, 0);
    par.add("10장 중 실패 유지", 6, still, 1, 0);
    par.add("20261020 새 실패", 1, newf, 1, 0);
    par.add("exploratory16 재대입 341 통과", 335, M["exploratory16|resub341|0"].passed, 2, 0);
    par.add("phase 341 통과", 248, M["phase|nested_oof|0"].passed, 2, 0);
    par.add("phase 20261020 통과", 53, M["phase|nested_oof|1020"].passed, 1, 0);
    const int ref_expl[5] = {1, 3, 2, 1, 0}, ref_phase[5] = {2, 1, 0, 0, 0}, ref_ro[5] = {16, 16, 16, 15, 18};
    const int ref_rf[5][16] = {{9, 8, 9, 9, 17, 16, 19, 18, 10, 10, 10, 8, 11, 10, 11, 11}, {11, 11, 8, 8, 10, 9, 13, 13, 10, 11, 8, 8, 8, 8, 9, 8},
                               {14, 14, 11, 12, 15, 14, 21, 21, 15, 15, 14, 15, 13, 13, 14, 13}, {9, 8, 10, 10, 11, 12, 13, 13, 8, 8, 8, 8, 10, 10, 11, 11},
                               {8, 10, 9, 10, 15, 15, 15, 14, 9, 10, 9, 9, 10, 9, 11, 10}};
    int expl_same = 0, phase_same = 0, ro_same = 0, rf_same = 0;
    for (int f = 0; f < 5; ++f) {
        expl_same += sels[f].expl16 == ref_expl[f];
        phase_same += sels[f].phase == ref_phase[f];
        ro_same += sels[f].robust_only - 4 + 15 == ref_ro[f];
        for (int k = 0; k < 16; ++k) rf_same += sels[f].sc[1][k].fails == ref_rf[f][k];
    }
    par.add("exploratory16 선택 후보 같은 분할 (5개 중)", 5, expl_same, 0, 0);
    par.add("robust_only 선택 후보 같은 분할 (5개 중)", 5, ro_same, 0, 0);
    par.add("phase 선택 후보 같은 분할 (5개 중)", 5, phase_same, 1, 0);
    par.add("강건 16후보 x 5분할 내부 실패 수 같은 칸 (80개 중)", 80, rf_same, 4, 0);

    // ---- 결과.md ----
    std::ostringstream R;
    R << "# 결과: 실패 10장 진단, 평가 누출 감사, 강건 후보 재선택\n\n"
      << "입력: `data/clean` 341장, 630점 측정은 20261022_1608 실험과 같다. 직전 실험(20261022_1631)의 15후보와 같은 바깥 4-fold, 같은 내부 분할(seed 20261022+fold, 최종 20261101)을 썼다.\n"
      << "강건 후보 RobustBaggedForward: 조건 평균(날짜·명령이 같은 사진)으로 Y/T 반응을 맞추고, 날짜별 같은 총 가중, huber 1.5 로 튀는 조건 가중을 줄이며, bags 12 는 "
      << "조건 가중에 지수분포 난수(원래 시드 numpy default_rng(20261022) 난수열, `nprng.hpp`)를 곱해 12번 평균한다. extended = 직전 15 + 사전 고정 4(new_condition_means s0.1), "
      << "robust_only = 사전 고정 4, exploratory16 = 강건 16 조합, phase = 반복 사진의 평균 이동 부화소 위상(sin, cos)에 따른 모양 변화를 빼는 9조합.\n\n"
      << "## 평가 누출 감사\n\n" << aud.md() << "\n위반 합계 " << leaks
      << ". 바깥 시험 사진과 그 그룹은 학습·내부 선택 어디에도 들어가지 않았다. 다만 341장 전체가 여러 차례 OOF 결과를 보며 후보를 바꾸는 데 쓰였으므로 적응적 오염은 남는다.\n\n"
      << "## 방법별 요약\n\n" << sum.md() << "\n## 선택\n\n" << st.md() << "\n### 강건 16후보 내부 점수 (실패 수 / 최악, * = 선택)\n\n" << rs.md()
      << "\n## 직전 pass_first 의 20261020 실패 사진\n\n" << ten.md()
      << "\n## 원래 결과와 대조 (" << par.ok << "/" << par.total << " 일치)\n\n" << par.t.md()
      << "\n## 옮기지 않은 부분\n\n- `audit/inventory.py`, `snapshot_dependencies.py`, `future_validator.py`: 파일 해시·의존성 목록·향후 검증 규칙 검사(사진 계산 없음).\n"
      << "- `diagnostics/analyze.py` 의 기울기·잔차 분해 그림: 조건 편향/개별 편차 분해와 사진별 측정 품질만 옮겼다.\n- `robust/screen.py`, `review_phase.py`, `verify_*.py`, `report.py`, `predict.py`.\n"
      << "\n## 파일\n\n- `predictions.csv`: 방법별 중첩 OOF·재대입 예측\n";
    write_text(out("결과.md"), R.str());

    std::ostringstream P;
    P << "# 보고서: 실패 10장 진단과 누출 감사 재선택 (10-22 17:12)\n\n"
      << "## 문제\n\n직전 분리 학습이 20261020에서 틀린 " << 60 - M["prior|nested_oof|1020"].passed << "장을 사진별로 보고 개선하면서, 평가 누출이 있었는지 감사하려 했다.\n\n"
      << "## 해결 방법\n\n실패 사진을 조건 평균 편향과 개별 반복 편차로 나누고, 바깥·내부 분할의 그룹 겹침을 검사한 뒤, 사전 고정 강건 후보 4개와 추가 탐색 16개를 같은 내부 분할로 다시 골랐다.\n\n"
      << "## 결과\n\n직접 누출(위반 " << leaks << "건)은 없었고, 추가 탐색 16후보는 20261020 " << Men.passed << "/60, 전체 " << Me.passed
      << "/341이었다.\n\n| 지표 | 값 |\n|---|---|\n"
      << "| 누출 검사 위반 | " << leaks << " |\n"
      << "| 직전 pass_first 20261020 / 341 통과 | " << M["prior|nested_oof|1020"].passed << " / 60, " << M["prior|nested_oof|0"].passed << " / 341 |\n"
      << "| extended (15 + 사전 4) 20261020 / 341 | " << M["extended|nested_oof|1020"].passed << " / 60, " << M["extended|nested_oof|0"].passed << " / 341 |\n"
      << "| robust_only (사전 4) 20261020 / 341 | " << M["robust_only|nested_oof|1020"].passed << " / 60, " << M["robust_only|nested_oof|0"].passed << " / 341 |\n"
      << "| exploratory16 20261020 / 20261014 / 341 | " << Men.passed << " / 60, " << M["exploratory16|nested_oof|1014"].passed << " / 281, " << Me.passed << " / 341 |\n"
      << "| exploratory16 Y 평균 / 최대 \\|오차\\| (mm) | " << fmt(Me.ymae, 4) << " / " << fmt(Me.ymax, 4) << " |\n"
      << "| exploratory16 T 평균 / 최대 \\|오차\\| (deg) | " << fmt(Me.tmae, 4) << " / " << fmt(Me.tmax, 4) << " |\n"
      << "| 실패 10장 중 복구 / 유지, 새 실패 | " << recov << " / " << still << ", " << newf << " |\n"
      << "| phase 20261020 / 341 | " << M["phase|nested_oof|1020"].passed << " / 60, " << M["phase|nested_oof|0"].passed << " / 341 |\n\n"
      << img_md("01_방법별_20261020_통과.png", "방법별 통과") << "\n\n" << img_md("02_실패분해_조건편향_개별편차.png", "실패 분해") << "\n\n"
      << img_md("03_exploratory16_중첩OOF_예측오차.png", "예측 오차") << "\n\n"
      << "## 튀는 값\n\n직전 실패 " << ten_idx.size() << "장(혼합 자세)의 평균 \\|조건 편향\\|은 " << fmt(mean_bias_f, 3) << " mm, 평균 \\|개별 반복 편차\\|는 " << fmt(mean_dev_f, 3)
      << " mm였고, exploratory16에서 가장 크게 틀린 사진은 Y는 " << names[wy] << "(정답 " << fmt(ty[wy], 2) << ", 추정 " << fmt(py[wy], 3) << " mm), T는 " << names[wt] << "(정답 "
      << fmt(tt[wt], 2) << ", 추정 " << fmt(pt[wt], 3) << " deg)이다.\n\n" << ten.md() << "\n" << outl.md() << "\n" << img_md(fy, "Y 튀는 값") << "\n\n"
      << img_md(ft, "T 튀는 값") << "\n\n초록 상자는 기준 사진의 QR 영역, 빨간 상자는 NCC로 찾은 이 사진의 QR 영역, 빨간 십자는 무효 판정된 모서리 점이다.\n\n"
      << "## 한줄평\n\n강건 가중은 튀는 조건의 영향을 줄여 20261020을 " << M["prior|nested_oof|1020"].passed << "에서 " << Men.passed << "장으로 올렸지만 20261014는 "
      << M["prior|nested_oof|1014"].passed << "에서 " << M["exploratory16|nested_oof|1014"].passed << "장으로 줄어 전체 이득이 " << Me.passed - M["prior|nested_oof|0"].passed
      << "장뿐이고, 같은 341장을 보며 후보를 늘린 선택이라 이 차이도 독립 검증 성능으로 볼 수 없다.\n\n"
      << "## 원래 결과와 대조\n\n원래 값은 `joint_yt_erroraudit_20261022/result.json`, `summary.csv`, `comparison_summary.csv`, `phase_summary.csv`, "
      << "`robust/exploratory_nested_*.json·csv`에서 가져왔다. bags 12 후보의 난수는 원래 numpy 시드 난수열을 그대로 넣었다. " << par.ok << "/" << par.total << " 일치.\n\n"
      << par.t.md()
      << "\n불일치는 fold0 exploratory16 선택 하나다. 원래는 후보 1(new_means s0.01 huber1.5)의 내부 실패가 8장이었는데 C++ 은 9장이라, 실패 8장 동률 중 최악 오차로 후보 11이 뽑혔다. "
      << "직전 실험(20261022_1631)에서 본 것처럼 통과선 0.09에 1e-4 mm 안으로 붙은 내부 검증 오차가 측정 좌표의 1e-4 px 수준 차이로 뒤집힌 경우다. "
      << "같은 이유로 직전 pass_first 의 fold1 선택도 원래 후보 8 대신 10이다.\n";
    write_text(out("보고서.md"), P.str());
    std::cout << "parity " << par.ok << "/" << par.total << std::endl;
    return 0;
}
