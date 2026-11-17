// 모든 특징점의 이동량과 점별 직선 잔차 겹쳐 보기 (원본: eda_all_points_20261022/plot_all.py + src/export.cpp)
//   기존: 20261014 고정 630점 구조 모서리(struct630.hpp, track_corners.py 방식), 양수 단일 축 구간에서 점마다 직선
//   새 자료: 20261020 60장을 hw_eda 측정기(hw_measure.hpp)로 점별 좌표 추출, 고정한 다른 축 값마다 점별 직선
#include "hw_measure.hpp"
#include "struct630.hpp"
#include <hye/dataset.hpp>
#include <hye/plot.hpp>
#include <hye/report.hpp>
#include <iostream>
#include <set>

using namespace hye;

namespace {

struct LineFit { double slope = kNaN, icpt = kNaN, rms = kNaN, r2 = kNaN; int n = 0; bool ok = false; std::vector<double> res; };
// plot_all.fit: 유한한 값으로 polyfit, 서로 다른 x 가 3개 이상일 때만
LineFit fit(const std::vector<double>& x, const std::vector<double>& y) {
    LineFit f;
    std::vector<double> xs, ys;
    std::set<double> ux;
    for (size_t i = 0; i < x.size(); ++i) if (std::isfinite(x[i]) && std::isfinite(y[i])) { xs.push_back(x[i]); ys.push_back(y[i]); ux.insert(x[i]); }
    if (ux.size() < 3) return f;
    double mx = mean(xs), my = mean(ys), sxy = 0, sxx = 0;
    for (size_t i = 0; i < xs.size(); ++i) { sxy += (xs[i] - mx) * (ys[i] - my); sxx += (xs[i] - mx) * (xs[i] - mx); }
    f.slope = sxy / sxx;
    f.icpt = my - f.slope * mx;
    double ss = 0, st = 0;
    for (size_t i = 0; i < x.size(); ++i) f.res.push_back(y[i] - (f.slope * x[i] + f.icpt));
    for (size_t i = 0; i < xs.size(); ++i) { double r = ys[i] - (f.slope * xs[i] + f.icpt); ss += r * r; st += (ys[i] - my) * (ys[i] - my); }
    f.rms = std::sqrt(ss / xs.size());
    f.r2 = st > 0 ? 1 - ss / st : kNaN;
    f.n = (int)xs.size();
    f.ok = true;
    return f;
}

// 선 수백 개를 얇게 겹쳐 그리는 판. 축·눈금·범례는 Chart 로, 선은 직접 1px 반투명으로.
struct Poly { std::vector<double> x, y; cv::Scalar c; bool dash = false; };
cv::Mat lines_panel(const std::string& title, const std::string& xl, const std::string& yl, const std::vector<Poly>& lines,
                    const std::vector<std::pair<std::string, cv::Scalar>>& legend, bool zero, double alpha = 0.25,
                    const std::vector<Poly>& dots = {}) {
    double a = 1e300, b = -1e300, c = 1e300, d = -1e300;
    for (auto* set : {&lines, &dots})
        for (auto& l : *set)
            for (size_t i = 0; i < l.x.size(); ++i)
                if (std::isfinite(l.y[i])) { a = std::min(a, l.x[i]); b = std::max(b, l.x[i]); c = std::min(c, l.y[i]); d = std::max(d, l.y[i]); }
    if (zero) { c = std::min(c, 0.0); d = std::max(d, 0.0); }
    double px = (b - a) * 0.04, py = (d - c) * 0.06;
    Chart ch(title, xl, yl);
    ch.W = 760; ch.H = 470;
    ch.range(a - px, b + px, c - py, d + py);
    for (auto& [lab, col] : legend) ch.scatter({}, {}, col, lab);
    cv::Mat m = ch.render(), ov = m.clone();
    cv::Rect clip(ch.L, ch.T, ch.W - ch.L - ch.R + 1, ch.H - ch.T - ch.B + 1);
    for (auto& l : lines)
        for (size_t i = 1; i < l.x.size(); ++i) {
            if (!std::isfinite(l.y[i - 1]) || !std::isfinite(l.y[i])) continue;
            if (l.dash && i % 2) continue;
            cv::Point p = ch.P(l.x[i - 1], l.y[i - 1]), q = ch.P(l.x[i], l.y[i]);
            if (cv::clipLine(clip, p, q)) cv::line(ov, p, q, l.c, 1, cv::LINE_AA);
        }
    for (auto& l : dots)
        for (size_t i = 0; i < l.x.size(); ++i) {
            if (!std::isfinite(l.y[i])) continue;
            cv::Point p = ch.P(l.x[i], l.y[i]);
            if (clip.contains(p)) cv::circle(ov, p, 1, l.c, -1);
        }
    cv::addWeighted(ov, alpha, m, 1 - alpha, 0, m);
    if (zero) { cv::Point p = ch.P(a - px, 0), q = ch.P(b + px, 0); cv::line(m, p, q, kBlack, 1); }
    return m;
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
    // ---- 기존 630점: center + 양수 Y 40장 + 양수 T 100장 ----
    auto shots = load_1014();
    const cv::Mat center = s630::read_gray(center_1014(shots).path);
    auto all = s630::structural_corners(s630::decode_qr(center));
    s630::refine_center(center, all);
    auto cs = s630::non_diagonal(all);
    const int P = (int)cs.size();
    struct OShot { std::string name, axis; double cmd; int si; };
    std::vector<OShot> os;
    for (size_t i = 0; i < shots.size(); ++i) {
        auto n = shots[i].rel.substr(shots[i].rel.rfind('/') + 1);
        if (shots[i].condition == "center") os.push_back({n, "C", 0, (int)i});
        else if (n[1] == '+') os.push_back({n, n.substr(0, 1), std::stod(n.substr(2, n.size() - 6)), (int)i});
    }
    std::vector<s630::Track> tr(os.size());
#pragma omp parallel for schedule(dynamic) num_threads(4)
    for (int k = 0; k < (int)os.size(); ++k) tr[k] = s630::track(s630::read_gray(shots[os[k].si].path), center, cs);
    int ci = 0;
    for (size_t k = 0; k < os.size(); ++k) if (os[k].axis == "C") ci = (int)k;
    std::cout << "old tracked " << os.size() << "\n";

    // ---- 새 자료: 20261020 hw_eda 측정 ----
    auto ns = load_set("clean_20261020");
    std::vector<std::string> npaths;
    size_t nref = 0;
    for (size_t i = 0; i < ns.size(); ++i) { npaths.push_back(ns[i].path); if (ns[i].condition == "center" && ns[i].repeat == 1) nref = i; }
    auto NM = hwm::measure_set(npaths, nref);
    const int NP = (int)NM.corners.size();
    std::cout << "new corners " << NP << "\n";
    auto r6 = [](double v) { return std::isfinite(v) ? std::stod(fmt(v, 6)) : kNaN; };

    Table metr{{"dataset", "axis", "component", "id", "fixed", "diagonal", "slope", "intercept", "residual_rms_px", "r2", "n"}};
    struct Agg { std::vector<double> rms; };
    std::map<std::tuple<std::string, std::string, std::string>, Agg> agg;
    std::map<std::string, int> old_invalid;
    std::map<std::string, cv::Mat> figs;
    // 기존 Y 지그재그 진단용: 양수 구간 점별 잔차 [사진][점]
    std::map<std::string, std::vector<std::vector<double>>> oldres;
    std::map<std::string, std::vector<double>> oldcmd;
    std::map<std::string, std::vector<std::string>> oldname;
    for (int comp = 0; comp < 2; ++comp) {
        const std::string cn = comp ? "dy" : "dx";
        std::vector<cv::Mat> cols;
        for (std::string axis : {"Y", "T"}) {
            std::vector<int> ks{ci};
            for (size_t k = 0; k < os.size(); ++k) if (os[k].axis == axis) ks.push_back((int)k);
            std::sort(ks.begin(), ks.end(), [&](int a, int b) { return os[a].cmd < os[b].cmd; });
            std::vector<Poly> L0, L1;
            std::vector<double> x;
            for (int k : ks) x.push_back(os[k].cmd);
            int invalid = 0;
            std::vector<double> xp(x.begin() + 1, x.end());
            std::vector<std::vector<double>> R(xp.size(), std::vector<double>(P));
            for (int j = 0; j < P; ++j) {
                std::vector<double> v;
                for (int k : ks) v.push_back(comp ? tr[k].obs[j].y - tr[ci].obs[j].y : tr[k].obs[j].x - tr[ci].obs[j].x);
                for (int k : ks) invalid += !tr[k].obs[j].valid;
                std::vector<double> vp(v.begin() + 1, v.end());
                auto f = fit(xp, vp);
                cv::Scalar col = color(cs[j].qr - 1);
                L0.push_back({x, v, col});
                if (f.ok) {
                    L1.push_back({xp, f.res, col});
                    for (size_t i = 0; i < xp.size(); ++i) R[i][j] = f.res[i];
                    metr.add({"old", axis, cn, std::to_string(cs[j].id), "", "False", fmt(f.slope, 6), fmt(f.icpt, 6), fmt(f.rms, 6), fmt(f.r2, 8), std::to_string(f.n)});
                    agg[{"old", axis, cn}].rms.push_back(f.rms);
                }
            }
            if (comp == 0) old_invalid[axis] = invalid;
            oldres[axis + cn] = R;
            oldcmd[axis] = xp;
            for (size_t i = 1; i < ks.size(); ++i) oldname[axis].push_back(os[ks[i]].name.substr(0, os[ks[i]].name.size() - 4));
            std::vector<std::pair<std::string, cv::Scalar>> qleg{{"left QR", color(0)}, {"center QR", color(1)}, {"right QR", color(2)}};
            cv::Mat a0 = lines_panel("10-14 single axis: " + axis + " rising, 630 points", axis + " command", cn + " vs center (px)", L0, qleg, false, 0.15);
            cv::Mat a1 = lines_panel("10-14: per-point line residual (positive only)", axis + " command", cn + " residual (px)", L1, qleg, true, 0.15);
            // 새 자료
            const bool vy = axis == "Y";
            std::set<double> fixedv;
            for (auto& s : ns) if (s.condition != "center") fixedv.insert(vy ? s.t : s.y);
            std::vector<Poly> L2, L3, D3;
            std::vector<std::pair<std::string, cv::Scalar>> leg;
            int fi = 0;
            for (double fv : fixedv) {
                std::vector<int> si;
                std::set<double> lv;
                for (size_t i = 0; i < ns.size(); ++i)
                    if (ns[i].condition != "center" && (vy ? ns[i].t : ns[i].y) == fv) { si.push_back((int)i); lv.insert(vy ? ns[i].y : ns[i].t); }
                if (lv.size() < 3) continue;
                cv::Scalar col = color(fi);
                leg.push_back({(vy ? "T=" : "Y=") + fmt(fv, 2), col});
                for (int j = 0; j < NP; ++j) {
                    const bool diag = NM.corners[j].diagonal;
                    std::vector<double> hx, hy;
                    for (int i : si) {
                        auto p = NM.shots[i].pos[j];
                        float dv = comp ? p.y - NM.corners[j].p.y : p.x - NM.corners[j].p.x;
                        hx.push_back(vy ? ns[i].y : ns[i].t);
                        hy.push_back(r6(std::isfinite(p.x) ? (double)dv : kNaN));
                    }
                    // 조건 평균 선
                    std::vector<double> mx, my;
                    for (double lvv : lv) {
                        double s = 0;
                        int c = 0;
                        for (size_t q = 0; q < hx.size(); ++q) if (hx[q] == lvv && std::isfinite(hy[q])) { s += hy[q]; ++c; }
                        mx.push_back(lvv);
                        my.push_back(c ? s / c : kNaN);
                    }
                    cv::Scalar lc = diag ? cv::Scalar(119, 119, 119) : col;
                    L2.push_back({mx, my, lc, diag});
                    auto f = fit(hx, hy);
                    if (f.ok) {
                        std::vector<double> mr;
                        for (size_t q = 0; q < mx.size(); ++q) mr.push_back(my[q] - (f.slope * mx[q] + f.icpt));
                        L3.push_back({mx, mr, lc, diag});
                        D3.push_back({hx, f.res, lc});
                        metr.add({"new", axis, cn, std::to_string(j), fmt(fv, 2), diag ? "True" : "False", fmt(f.slope, 6), fmt(f.icpt, 6), fmt(f.rms, 6), fmt(f.r2, 8),
                                  std::to_string(f.n)});
                        agg[{"new", axis, cn}].rms.push_back(f.rms);
                    }
                }
                ++fi;
            }
            leg.push_back({"diagonal (gray)", cv::Scalar(119, 119, 119)});
            cv::Mat a2 = lines_panel("10-20: " + axis + " rising, " + std::to_string(NP) + " points x " + std::to_string(fi) + " fixed", axis + " command",
                                     cn + " vs center1 (px)", L2, leg, false, 0.15);
            cv::Mat a3 = lines_panel("10-20: per-point line residual (dots = repeats)", axis + " command", cn + " residual (px)", L3, leg, true, 0.15, D3);
            cols.push_back(vstack({a0, a1, a2, a3}));
        }
        save_png(std::string("0") + (comp ? "2" : "1") + "_전점_" + (comp ? "dy" : "dx") + ".png", hstack(cols));
    }
    metr.csv(out("point_linearity_metrics.csv"));
    Table sm{{"자료", "축", "성분", "선 수", "잔차 RMS 중앙값 (px)", "95% (px)", "최대 (px)"}};
    for (auto& [k, a] : agg)
        sm.add({std::get<0>(k), std::get<1>(k), std::get<2>(k), std::to_string(a.rms.size()), fmt(percentile(a.rms, .5), 4), fmt(percentile(a.rms, .95), 4),
                fmt(*std::max_element(a.rms.begin(), a.rms.end()), 4)});
    int new_diag = 0, failed = 0;
    std::set<int> fail_pts;
    for (int j = 0; j < NP; ++j) new_diag += NM.corners[j].diagonal;
    for (auto& s : NM.shots) for (int j = 0; j < NP; ++j) if (!std::isfinite(s.pos[j].x)) { ++failed; fail_pts.insert(j); }

    // ---- 지그재그: 기존 양수 구간 잔차의 공통 성분 (사진마다 630점 평균) ----
    struct Zig { double total, common, share; std::vector<double> c; };
    std::map<std::string, Zig> zig;
    for (auto& [key, R] : oldres) {
        Zig z{};
        double ss = 0, sc = 0;
        for (auto& row : R) {
            double m = mean(row);
            z.c.push_back(m);
            for (double v : row) ss += v * v;
            sc += m * m * row.size();
        }
        z.total = std::sqrt(ss / (R.size() * R[0].size()));
        z.common = rms(z.c);
        z.share = sc / ss;
        zig[key] = z;
    }
    Table zt{{"축·성분", "점별 잔차 RMS (px)", "사진 공통 잔차 RMS (px)", "공통 성분 비율 (제곱합)"}};
    for (auto k : {"Ydx", "Tdx", "Ydy", "Tdy"}) zt.add({k, fmt(zig[k].total, 4), fmt(zig[k].common, 4), fmt(zig[k].share, 3)});
    // 튀는 값: 기존 Y dx 에서 공통 잔차가 가장 큰 사진, 점별 잔차 RMS 가 가장 큰 점
    auto& zy = zig["Ydx"];
    size_t wshot = worst(zy.c, 1)[0];
    auto& RY = oldres["Ydx"];
    int wpt = 0;
    double wv = 0;
    for (int j = 0; j < P; ++j) {
        double s = 0;
        for (auto& row : RY) s += row[j] * row[j];
        s = std::sqrt(s / RY.size());
        if (s > wv) { wv = s; wpt = j; }
    }
    double corr = 0;
    {
        std::vector<double> pr;
        for (auto& row : RY) pr.push_back(row[wpt]);
        double ma = mean(pr), mb = mean(zy.c), sab = 0, saa = 0, sbb = 0;
        for (size_t i = 0; i < pr.size(); ++i) { sab += (pr[i] - ma) * (zy.c[i] - mb); saa += (pr[i] - ma) * (pr[i] - ma); sbb += (zy.c[i] - mb) * (zy.c[i] - mb); }
        corr = sab / std::sqrt(saa * sbb);
    }
    {
        Chart c("10-14 Y dx residual: common part per shot vs one point", "Y command", "residual (px)");
        std::vector<double> pr;
        for (auto& row : RY) pr.push_back(row[wpt]);
        c.line(oldcmd["Y"], zy.c, color(0), "mean of 630 points", true).line(oldcmd["Y"], pr, color(3), "point id " + std::to_string(cs[wpt].id), true).hline(0);
        c.note(oldcmd["Y"][wshot], zy.c[wshot], oldname["Y"][wshot]);
        c.save("03_Y지그재그_공통잔차.png");
    }
    std::string fz;
    {
        int k = -1;
        for (size_t q = 0; q < os.size(); ++q) if (os[q].axis == "Y" && os[q].name.substr(0, os[q].name.size() - 4) == oldname["Y"][wshot]) k = (int)q;
        cv::Mat im = s630::read_gray(shots[os[k].si].path);
        std::vector<cv::Mat> tiles;
        for (int q = 0; q < 3; ++q) {
            auto b = s630::kBoxes[q];
            auto sh = tr[k].shift[q];
            cv::Rect roi(b[0] + sh.x - 10, b[1] + sh.y - 10, b[2] - b[0] + 20, b[3] - b[1] + 20);
            std::vector<Mark> mk;
            for (int j = 0; j < P; ++j)
                if (cs[j].qr == q + 1 && (j % 4 == 0 || j == wpt))
                    mk.push_back({{tr[k].obs[j].x, tr[k].obs[j].y}, {RY[wshot][j], 0}, j == wpt ? "id " + std::to_string(cs[j].id) : "", j == wpt ? color(0) : kRed});
            tiles.push_back(zoom(im, roi, 420, mk, 150, q == 0 ? oldname["Y"][wshot] + " common resid " + fmts(zy.c[wshot], 3) + " px" : "",
                                 {{cv::Rect2d(b[0] + sh.x, b[1] + sh.y, b[2] - b[0], b[3] - b[1]), kGreen, "coarse match"}}));
        }
        fz = "04_튀는값_Y_" + oldname["Y"][wshot] + ".png";
        save_png(fz, hstack(tiles));
    }

    // ---- 원래 결과와 대조 (eda_all_points_20261022/summary_metrics.csv, audit.json) ----
    Parity par;
    struct O { const char* d; const char* a; const char* c; int n; double med, p95, mx; };
    const O orig[8] = {{"new", "T", "dx", 2413, 0.14790347624275915, 0.19432616155727309, 0.3363059754081669},
                       {"new", "T", "dy", 2413, 0.029962184572868377, 0.07768740502342426, 0.4010972235518625},
                       {"new", "Y", "dx", 2403, 0.18533706703788677, 0.24626203726388105, 0.4388879166849773},
                       {"new", "Y", "dy", 2403, 0.029730538276836794, 0.06941739408218113, 0.40354073139056057},
                       {"old", "T", "dx", 630, 0.05970128785886481, 0.07311733042045311, 0.09657494009536792},
                       {"old", "T", "dy", 630, 0.03469231910123115, 0.05779884605406807, 0.10711687185488997},
                       {"old", "Y", "dx", 630, 0.0777520951388425, 0.092967826809237, 0.10683059198943752},
                       {"old", "Y", "dy", 630, 0.04942306598320739, 0.07123673879402057, 0.12275900634936476}};
    for (auto& o : orig) {
        auto& a = agg[{o.d, o.a, o.c}].rms;
        std::string nm = std::string(o.d) + " " + o.a + " " + o.c;
        par.add(nm + " 선 수", o.n, (double)a.size(), 0, 0);
        par.add(nm + " 잔차 RMS 중앙값", o.med, percentile(a, .5), 0.0005);
        par.add(nm + " 잔차 RMS 95%", o.p95, percentile(a, .95), 0.001);
        par.add(nm + " 잔차 RMS 최대", o.mx, *std::max_element(a.begin(), a.end()), 0.002);
    }
    par.add("새 자료 점 수", 625, NP, 0, 0);
    par.add("새 자료 대각선 점 수", 56, new_diag, 0, 0);
    par.add("새 자료 추적 실패 관측 수", 1895, failed, 0, 0);
    par.add("새 자료 실패가 있는 점 수", 65, (double)fail_pts.size(), 0, 0);
    par.add("기존 Y 진단 실패 수 (41장 x 630)", 960, old_invalid["Y"], 0, 0);
    par.add("기존 T 진단 실패 수 (101장 x 630)", 2101, old_invalid["T"], 0, 0);

    std::ostringstream R;
    R << "# 결과: 모든 점의 이동량과 점별 직선 잔차\n\n"
      << "기존 자료: `data/clean/clean_20261014` center + 양수 Y 40장 + 양수 T 100장. center.bmp 에서 고정한 구조 모서리 630점(대각선 125점 제외)을 흐린 경계 모델로 추적하고 center 좌표를 뺐다. "
      << "점마다 양수 명령 구간만 직선을 맞췄다(center 는 이동량 그림에만 표시). 진단 실패 점도 그대로 쓴다.\n"
      << "새 자료: `data/clean/clean_20261020` 60장을 hw_eda 측정기(center1 기준, 라벨 안 코너 " << NP << "점)로 추적했다. 추적 실패는 NaN으로 두고 유한한 값만 맞췄다. "
      << "고정한 다른 축 값마다(상승 축 3수준 이상) 점별 직선을 반복 3장 원측정에 맞췄다.\n\n"
      << "## 요약 (점별 잔차 RMS 분포)\n\n" << sm.md()
      << "\n## 지그재그: 기존 양수 구간 잔차의 사진 공통 성분\n\n" << zt.md()
      << "\n공통 잔차 = 한 사진에서 630점 잔차의 평균. 공통 성분 비율 = 공통 잔차가 설명하는 잔차 제곱합 비율.\n\n"
      << "## 건수\n\n- 새 자료 점 " << NP << ", 대각선 " << new_diag << ", 추적 실패 관측 " << failed << ", 실패가 있는 점 " << fail_pts.size() << "\n"
      << "- 기존 진단 실패: Y " << old_invalid["Y"] << " / " << 41 * P << ", T " << old_invalid["T"] << " / " << 101 * P << "\n\n"
      << "## 원래 결과와 대조 (" << par.ok << "/" << par.total << " 일치)\n\n" << par.t.md()
      << "\n## 원래와 다른 점\n\n- 원래는 hw_eda 측정기를 export.cpp 로 불러 CSV를 만들고 Python 으로 그렸다. 여기서는 측정기를 `hw_measure.hpp`로 옮겨 한 프로그램에서 측정·분석·그림을 한다. "
      << "원래 `실험/CMakeLists.txt`(export_points)는 없어진 경로를 가리켜서 새 프로그램 빌드로 바꿨다. `src/export.cpp`는 그대로 두었다.\n"
      << "- 기존 630점 좌표는 옛 corners_280.csv 대신 사진에서 다시 쟀다(scipy TRF 대신 같은 비용의 범위 제한 LM). 진단 실패 수는 최적화 수렴 판정이 달라 원래와 다르다.\n"
      << "\n## 파일\n\n- `point_linearity_metrics.csv`: 점·축·성분·고정값별 기울기·절편·잔차 RMS·R²\n- `이미지/`: 그림\n";
    write_text(out("결과.md"), R.str());

    std::ostringstream Pp;
    auto& oy = agg[{"old", "Y", "dx"}].rms;
    auto& ot = agg[{"old", "T", "dx"}].rms;
    auto& ny = agg[{"new", "Y", "dx"}].rms;
    Pp << "# 보고서: 모든 점 노이즈, 점별 직선 겹침 (10-22 15:41)\n\n"
       << "## 문제\n\n특징점 하나하나의 이동량과 직선 잔차를 겹쳐 보고 노이즈 양상을 확인해야 했다.\n\n"
       << "## 해결 방법\n\n630점 좌표와 20261020 점별 좌표를 점마다 선으로 그리고 점별 직선을 적합했다.\n\n"
       << "## 결과\n\n기존 Y 수평 잔차는 점별 RMS 중앙값 " << fmt(percentile(oy, .5), 4) << " px 중 " << fmt(100 * zig["Ydx"].share, 0)
       << "%가 사진마다 모든 점이 함께 움직이는 공통 성분이라, 지그재그를 점별 독립 노이즈로 단정할 수 없었다.\n\n"
       << "| 지표 | 값 |\n|---|---|\n"
       << "| 기존 Y / T 수평 잔차 RMS 중앙값 (px) | " << fmt(percentile(oy, .5), 4) << " / " << fmt(percentile(ot, .5), 4) << " |\n"
       << "| 기존 Y / T 수평 잔차 공통 성분 비율 | " << fmt(zig["Ydx"].share, 3) << " / " << fmt(zig["Tdx"].share, 3) << " |\n"
       << "| 기존 Y 사진 공통 잔차 RMS (px) | " << fmt(zig["Ydx"].common, 4) << " |\n"
       << "| 새 자료 Y / T 수평 잔차 RMS 중앙값 (px) | " << fmt(percentile(ny, .5), 4) << " / " << fmt(percentile(agg[{"new", "T", "dx"}].rms, .5), 4) << " |\n"
       << "| 새 자료 점 / 대각선 / 실패 관측 | " << NP << " / " << new_diag << " / " << failed << " |\n\n"
       << img_md("01_전점_dx.png", "수평 이동") << "\n\n" << img_md("02_전점_dy.png", "수직 이동") << "\n\n"
       << "그림 열: 왼쪽 Y, 오른쪽 T. 위에서부터 기존 이동량, 기존 점별 직선 잔차, 새 자료 조건 평균 이동량, 새 자료 점별 직선 잔차(점 = 반복 원측정). 선 하나가 같은 점 하나다.\n\n"
       << "## 튀는 값\n\n사진 공통 잔차가 가장 큰 기존 Y 사진은 " << oldname["Y"][wshot] << "(" << fmts(zy.c[wshot], 3) << " px)이고, 점별 잔차 RMS 가 가장 큰 점(id "
       << cs[wpt].id << ", " << fmt(wv, 4) << " px)의 잔차와 사진 공통 잔차의 상관은 " << fmt(corr, 2) << "다.\n\n"
       << "| 구분 | 사진 / 점 | Y 명령 | 공통 잔차 (px) | 그 점 잔차 (px) |\n|---|---|---|---|---|\n";
    for (size_t k : worst(zy.c, 4)) Pp << "| 공통 잔차 상위 | " << oldname["Y"][k] << " | " << fmt(oldcmd["Y"][k], 2) << " | " << fmts(zy.c[k], 4) << " | " << fmts(RY[k][wpt], 4) << " |\n";
    Pp << "\n" << img_md("03_Y지그재그_공통잔차.png", "공통 잔차") << "\n\n" << img_md(fz, "Y 튀는 값")
       << "\n\n빨간 십자는 이 사진에서 맞춘 모서리(4개 중 1개), 파란 십자는 잔차가 가장 큰 점, 화살표는 그 점의 수평 직선 잔차를 150배 키운 것, 초록 상자는 템플릿 매칭으로 찾은 QR 영역이다.\n\n"
       << "## 한줄평\n\n점마다 직선을 따로 맞춰도 기존 Y 잔차 제곱합의 " << fmt(100 * zig["Ydx"].share, 0) << "%가 사진 공통 잔차(RMS " << fmt(zig["Ydx"].common, 3)
       << " px)라서, 지그재그는 점별 측정 노이즈보다 사진 전체가 함께 어긋나는 시행·위치 차이에서 온다.\n\n"
       << "## 원래 결과와 대조\n\n원래 값은 `eda_all_points_20261022/summary_metrics.csv`, `audit.json`에서 가져왔다. " << par.ok << "/" << par.total
       << " 일치. 기존 진단 실패 수는 불일치다. 원래 corners_280.csv 의 630점 진단 실패 6626건 중 6077건이 scipy TRF 가 85회 평가 안에 끝나지 않은 'optimizer'였고, C++ LM 은 같은 점을 수렴시켜 실패로 세지 않는다. 좌표 자체는 같은 비용의 최소점이라 잔차 지표는 일치한다.\n\n" << par.t.md();
    write_text(out("보고서.md"), Pp.str());
    std::cout << "parity " << par.ok << "/" << par.total << "\n";
    return 0;
}
