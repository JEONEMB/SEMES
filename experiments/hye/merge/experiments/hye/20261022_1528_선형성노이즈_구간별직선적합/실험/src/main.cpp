// 20261020 60장 Y·T 명령 대비 수평 이동의 선형성과 반복 노이즈 (원본: eda_linearity_20261020_20261022/analyze.py)
//   원래 analyze.py 는 hw_eda(C++) 가 만든 shots.csv 를 읽었다. 여기서는 그 측정(hw_measure.hpp)을 사진에서 다시 하고 같은 분석을 한다.
#include "hw_measure.hpp"
#include <hye/dataset.hpp>
#include <hye/plot.hpp>
#include <hye/report.hpp>
#include <iostream>
#include <map>
#include <set>

using namespace hye;

namespace {

double r4(double v) { return std::stod(fmt(v, 4)); }  // shots.csv 에 4자리로 적던 값과 같게

// np.polyfit(x, y, 1)
std::pair<double, double> polyfit1(const std::vector<double>& x, const std::vector<double>& y) {
    double mx = mean(x), my = mean(y), sxy = 0, sxx = 0;
    for (size_t i = 0; i < x.size(); ++i) { sxy += (x[i] - mx) * (y[i] - my); sxx += (x[i] - mx) * (x[i] - mx); }
    double k = sxy / sxx;
    return {k, my - k * mx};
}
double sd1(const std::vector<double>& v) {
    double m = mean(v), s = 0;
    for (double x : v) s += (x - m) * (x - m);
    return std::sqrt(s / (v.size() - 1));
}
// 정규화 불완전 베타 (연분수) -> F 분포 위꼬리 확률 (scipy.stats.f.sf)
double betacf(double a, double b, double x) {
    const double FPMIN = 1e-300, EPS = 1e-15;
    double qab = a + b, qap = a + 1, qam = a - 1, c = 1, d = 1 - qab * x / qap;
    if (std::abs(d) < FPMIN) d = FPMIN;
    d = 1 / d;
    double h = d;
    for (int m = 1; m <= 300; ++m) {
        int m2 = 2 * m;
        double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1 + aa * d; if (std::abs(d) < FPMIN) d = FPMIN;
        c = 1 + aa / c; if (std::abs(c) < FPMIN) c = FPMIN;
        d = 1 / d; h *= d * c;
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1 + aa * d; if (std::abs(d) < FPMIN) d = FPMIN;
        c = 1 + aa / c; if (std::abs(c) < FPMIN) c = FPMIN;
        d = 1 / d;
        double del = d * c;
        h *= del;
        if (std::abs(del - 1) < EPS) break;
    }
    return h;
}
double ibeta(double a, double b, double x) {
    if (x <= 0) return 0;
    if (x >= 1) return 1;
    double bt = std::exp(std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b) + a * std::log(x) + b * std::log(1 - x));
    if (x < (a + 1) / (a + b + 2)) return bt * betacf(a, b, x) / a;
    return 1 - bt * betacf(b, a, 1 - x) / b;
}
double f_sf(double F, double d1, double d2) { return ibeta(d2 / 2, d1 / 2, d2 / (d2 + d1 * F)); }

// 최소제곱 (열 우선 X)
std::vector<double> lstsq(const std::vector<std::vector<double>>& cols, const std::vector<double>& y) {
    cv::Mat A((int)y.size(), (int)cols.size(), CV_64F), B((int)y.size(), 1, CV_64F), C;
    for (int i = 0; i < A.rows; ++i) {
        for (int j = 0; j < A.cols; ++j) A.at<double>(i, j) = cols[j][i];
        B.at<double>(i) = y[i];
    }
    cv::solve(A, B, C, cv::DECOMP_SVD);
    return std::vector<double>(C.begin<double>(), C.end<double>());
}

struct Rec {  // 사진 한 장
    std::string rel, name;
    bool center;
    double y, t;
    int rep;
    double corner_dx, ecc_dx;
};
std::string short_name(const std::string& rel) {
    std::string s = rel.substr(rel.rfind('/') + 1);
    s = s.substr(0, s.rfind('.'));
    std::string o;
    for (char c : s) o += (c == ' ' ? '_' : c);
    return o;
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
    // ---- 측정: 20261020 (기준 = 기준 위치/center1.bmp) ----
    auto shots = load_set("clean_20261020");
    std::vector<std::string> paths;
    size_t ref = 0;
    for (size_t i = 0; i < shots.size(); ++i) {
        paths.push_back(shots[i].path);
        if (shots[i].condition == "center" && shots[i].repeat == 1) ref = i;
    }
    auto M = hwm::measure_set(paths, ref);
    int used = 0, diag = 0;
    for (size_t k = 0; k < M.corners.size(); ++k) { used += M.usable[k] && !M.corners[k].diagonal; diag += M.corners[k].diagonal; }
    std::cout << "20261020 corners " << M.corners.size() << " used " << used << "\n";
    std::vector<Rec> d;
    for (size_t i = 0; i < shots.size(); ++i)
        d.push_back({shots[i].rel, short_name(shots[i].rel), shots[i].condition == "center", shots[i].y, shots[i].t, shots[i].repeat,
                     r4(M.shots[i].corner_dx), r4(M.shots[i].ecc_dx)});
    // ---- 측정: 20261014 (기준 = center.bmp) ----
    auto old = load_1014();
    std::vector<std::string> opaths;
    size_t oref = 0;
    for (size_t i = 0; i < old.size(); ++i) { opaths.push_back(old[i].path); if (old[i].condition == "center") oref = i; }
    auto OM = hwm::measure_set(opaths, oref);
    int oused = 0;
    for (size_t k = 0; k < OM.corners.size(); ++k) oused += OM.usable[k] && !OM.corners[k].diagonal;
    std::cout << "20261014 corners " << OM.corners.size() << " used " << oused << "\n";

    {
        Table s{{"file", "center", "y", "t", "rep", "ecc_dx", "ecc_dy", "corner_dx", "corner_dy"}};
        for (size_t i = 0; i < shots.size(); ++i)
            s.add({shots[i].rel, d[i].center ? "1" : "0", fmt(d[i].y, 2), fmt(d[i].t, 2), std::to_string(d[i].rep), fmt(M.shots[i].ecc_dx, 4),
                   fmt(M.shots[i].ecc_dy, 4), fmt(M.shots[i].corner_dx, 4), fmt(M.shots[i].corner_dy, 4)});
        s.csv(out("shots_20261020.csv"));
        Table o{{"file", "y", "t", "ecc_dx", "corner_dx"}};
        for (size_t i = 0; i < old.size(); ++i)
            o.add({old[i].rel, fmt(old[i].y, 2), fmt(old[i].t, 2), fmt(OM.shots[i].ecc_dx, 4), fmt(OM.shots[i].corner_dx, 4)});
        o.csv(out("shots_20261014.csv"));
    }

    // ---- 조건 통계 (center 제외, (y,t) 묶음) ----
    struct Cond { double y, t, mean, sd, ecc; int n; std::vector<size_t> idx; };
    std::map<std::pair<double, double>, Cond> cmap;
    for (size_t i = 0; i < d.size(); ++i)
        if (!d[i].center) { auto& c = cmap[{d[i].y, d[i].t}]; c.y = d[i].y; c.t = d[i].t; c.idx.push_back(i); }
    std::vector<Cond> g;
    for (auto& [k, c] : cmap) {
        std::vector<double> v, e;
        for (auto i : c.idx) { v.push_back(d[i].corner_dx); e.push_back(d[i].ecc_dx); }
        c.mean = mean(v); c.sd = sd1(v); c.n = (int)v.size(); c.ecc = mean(e);
        g.push_back(c);
    }
    Table condT{{"Y", "T", "평균 corner_dx (px)", "반복 SD (px)", "장수", "평균 ecc_dx (px)"}};
    std::vector<double> sds;
    for (auto& c : g) { condT.add({fmt(c.y, 2), fmt(c.t, 2), fmt(c.mean, 4), fmt(c.sd, 4), std::to_string(c.n), fmt(c.ecc, 4)}); sds.push_back(c.sd); }
    const double sd_med = percentile(sds, 0.5), sd_max = *std::max_element(sds.begin(), sds.end());

    // ---- 구간별 직선 ----
    struct Met { std::string axis, fixed; double v; int levels; double slope, icpt, r2, rms, maxres, pooled, p; std::vector<double> x, y, sd, res; };
    std::vector<Met> mets;
    for (int col = 0; col < 2; ++col) {
        const bool ax_y = col == 0;  // 0: y 상승 (t 고정), 1: t 상승 (y 고정)
        std::set<double> fixed_vals;
        for (auto& c : g) fixed_vals.insert(ax_y ? c.t : c.y);
        for (double v : fixed_vals) {
            std::vector<const Cond*> sub;
            for (auto& c : g) if ((ax_y ? c.t : c.y) == v) sub.push_back(&c);
            if (sub.size() < 3) continue;
            std::sort(sub.begin(), sub.end(), [&](auto a, auto b) { return (ax_y ? a->y : a->t) < (ax_y ? b->y : b->t); });
            Met m{ax_y ? "y" : "t", ax_y ? "t" : "y", v, (int)sub.size()};
            for (auto c : sub) { m.x.push_back(ax_y ? c->y : c->t); m.y.push_back(c->mean); m.sd.push_back(c->sd); }
            std::tie(m.slope, m.icpt) = polyfit1(m.x, m.y);
            double ssr = 0, sst = 0, my = mean(m.y), sse_lof = 0, sse_pure = 0;
            int nraw = 0;
            for (size_t i = 0; i < m.x.size(); ++i) {
                double r = m.y[i] - (m.slope * m.x[i] + m.icpt);
                m.res.push_back(r);
                ssr += r * r; sst += (m.y[i] - my) * (m.y[i] - my);
                sse_lof += sub[i]->n * r * r;
                for (auto k : sub[i]->idx) { sse_pure += (d[k].corner_dx - sub[i]->mean) * (d[k].corner_dx - sub[i]->mean); ++nraw; }
            }
            int df_lof = (int)sub.size() - 2, df_pure = nraw - (int)sub.size();
            double F = (sse_lof / df_lof) / (sse_pure / df_pure);
            m.r2 = 1 - ssr / sst;
            m.rms = std::sqrt(ssr / m.x.size());
            m.maxres = max_abs(m.res);
            m.pooled = std::sqrt(sse_pure / df_pure);
            m.p = f_sf(F, df_lof, df_pure);
            mets.push_back(m);
        }
    }
    Table metT{{"상승 축", "고정 축 값", "수준 수", "기울기 (px/단위)", "절편 (px)", "R²", "잔차 RMS (px)", "최대 잔차 (px)", "합동 반복 SD (px)", "적합결여 p"}};
    double r2y_min = 1, r2y_max = 0, r2t_min = 1, rmsy_min = 1e9, rmsy_max = 0, rmst_min = 1e9, rmst_max = 0, p_min = 1;
    for (auto& m : mets) {
        metT.add({m.axis == "y" ? "Y" : "T", (m.fixed == "y" ? "Y=" : "T=") + fmt(m.v, 2), std::to_string(m.levels), fmt(m.slope, 4), fmt(m.icpt, 4),
                  fmt(m.r2, 6), fmt(m.rms, 4), fmt(m.maxres, 4), fmt(m.pooled, 4), fmt(m.p, 4)});
        if (m.axis == "y") { r2y_min = std::min(r2y_min, m.r2); r2y_max = std::max(r2y_max, m.r2); rmsy_min = std::min(rmsy_min, m.rms); rmsy_max = std::max(rmsy_max, m.rms); }
        else { r2t_min = std::min(r2t_min, m.r2); rmst_min = std::min(rmst_min, m.rms); rmst_max = std::max(rmst_max, m.rms); }
        p_min = std::min(p_min, m.p);
    }

    // ---- 원점 통과 / 자유 절편 가산 직선 (center 제외 57장) ----
    std::vector<size_t> raw;
    for (size_t i = 0; i < d.size(); ++i) if (!d[i].center) raw.push_back(i);
    std::vector<double> ry, rt, rx, one(raw.size(), 1.0);
    for (auto i : raw) { ry.push_back(d[i].y); rt.push_back(d[i].t); rx.push_back(d[i].corner_dx); }
    auto bo = lstsq({ry, rt}, rx), bf = lstsq({one, ry, rt}, rx);
    std::vector<double> res_o, res_f;
    for (size_t k = 0; k < raw.size(); ++k) {
        res_o.push_back(rx[k] - (bo[0] * ry[k] + bo[1] * rt[k]));
        res_f.push_back(rx[k] - (bf[0] + bf[1] * ry[k] + bf[2] * rt[k]));
    }
    const double rms_o = rms(res_o), rms_f = rms(res_f);

    // ---- 기존 20261014 단일 축 양수 구간 ----
    struct OldFit { double slope, icpt, rms, r2; int n; std::vector<double> x, y; };
    std::map<std::string, OldFit> oldfit;
    for (int col = 0; col < 2; ++col) {
        const bool ax_y = col == 0;
        std::vector<std::pair<double, double>> h;
        for (size_t i = 0; i < old.size(); ++i) {
            double fixedv = ax_y ? old[i].t : old[i].y, axv = ax_y ? old[i].y : old[i].t;
            if (fixedv == 0 && axv >= 0) h.push_back({axv, r4(OM.shots[i].corner_dx)});
        }
        std::sort(h.begin(), h.end());
        OldFit f;
        for (auto& p : h) { f.x.push_back(p.first); f.y.push_back(p.second); }
        std::tie(f.slope, f.icpt) = polyfit1(f.x, f.y);
        std::vector<double> r;
        double sst = 0, my = mean(f.y);
        for (size_t i = 0; i < f.x.size(); ++i) { r.push_back(f.y[i] - (f.slope * f.x[i] + f.icpt)); sst += (f.y[i] - my) * (f.y[i] - my); }
        f.rms = rms(r);
        f.r2 = 1 - std::pow(f.rms, 2) * r.size() / sst;
        f.n = (int)h.size();
        oldfit[ax_y ? "y" : "t"] = f;
    }
    Table cmpT{{"상승 축", "고정 축 값", "기존 기울기", "새 기울기", "차이 (%)"}};
    double dmin[2] = {1e9, 1e9}, dmax[2] = {-1e9, -1e9};
    for (auto& m : mets) {
        auto& o = oldfit[m.axis];
        double pc = 100 * (m.slope / o.slope - 1);
        int a = m.axis == "y" ? 0 : 1;
        dmin[a] = std::min(dmin[a], pc); dmax[a] = std::max(dmax[a], pc);
        cmpT.add({m.axis == "y" ? "Y" : "T", fmt(m.v, 2), fmt(o.slope, 4), fmt(m.slope, 4), fmts(pc, 3)});
    }
    double ce_max = 0, cmin = 1e9, cmax = -1e9;
    for (auto& r : d) {
        ce_max = std::max(ce_max, std::abs(r.corner_dx - r.ecc_dx));
        if (r.center) { cmin = std::min(cmin, r.corner_dx); cmax = std::max(cmax, r.corner_dx); }
    }

    // ---- 그림 ----
    for (int col = 0; col < 2; ++col) {
        const bool ax_y = col == 0;
        Chart a(std::string(ax_y ? "Y" : "T") + " rising, " + (ax_y ? "T" : "Y") + " fixed: condition mean", std::string(ax_y ? "Y" : "T") + " command",
                "horizontal shift vs reference (px)");
        Chart b("Residual from free-intercept line (condition mean)", std::string(ax_y ? "Y" : "T") + " command", "mean - line (px)");
        int k = 0;
        for (auto& m : mets) {
            if ((m.axis == "y") != ax_y) continue;
            std::vector<double> fx = m.x, fy;
            for (double x : fx) fy.push_back(m.slope * x + m.icpt);
            std::string lab = std::string(ax_y ? "T=" : "Y=") + fmt(m.v, 2);
            a.scatter(m.x, m.y, color(k), lab, 4).line(fx, fy, color(k));
            // 반복 각 장도 작은 점으로
            std::vector<double> px, py;
            for (auto& c : g)
                if ((ax_y ? c.t : c.y) == m.v)
                    for (auto i : c.idx) { px.push_back(ax_y ? d[i].y : d[i].t); py.push_back(d[i].corner_dx - (m.slope * (ax_y ? d[i].y : d[i].t) + m.icpt)); }
            b.line(m.x, m.res, color(k), lab, true).scatter(px, py, color(k), "", 2);
            ++k;
        }
        b.hline(0);
        save_png(std::string("01_축별선형성_") + (ax_y ? "Y" : "T") + ".png", hstack({a.render(), b.render()}));
    }
    {
        Chart a("Repeat scatter: shot - condition mean", "repeat number", "deviation (px)");
        int k = 0;
        for (auto& c : g) {
            std::vector<double> x, y;
            for (auto i : c.idx) { x.push_back(d[i].rep); y.push_back(d[i].corner_dx - c.mean); }
            a.line(x, y, color(k++ % 6), "", true);
        }
        a.hline(0);
        Chart b("Reference (0,0) shots", "center repeat", "shift vs center1 (px)");
        std::vector<double> x, y1, y2;
        for (auto& r : d) if (r.center) { x.push_back(r.rep); y1.push_back(r.corner_dx); y2.push_back(r.ecc_dx); }
        b.line(x, y1, color(0), "corner", true).line(x, y2, color(1), "ECC", true);
        Chart c("Corner minus ECC", "corner shift (px)", "corner - ECC (px)");
        std::vector<double> cx, cy;
        for (auto& r : d) { cx.push_back(r.corner_dx); cy.push_back(r.corner_dx - r.ecc_dx); }
        c.scatter(cx, cy, color(2)).hline(0);
        save_png("02_반복노이즈.png", hstack({a.render(), b.render(), c.render()}));
    }
    std::vector<double> order;
    for (size_t k = 0; k < raw.size(); ++k) order.push_back((double)k);
    {
        Chart a("Origin-through additive line: RMS " + fmt(rms_o, 3) + " px", "shot (manifest order)", "measured - fit (px)");
        a.scatter(order, res_o, color(0)).hline(0);
        Chart b("Free-intercept additive line: RMS " + fmt(rms_f, 3) + " px", "shot (manifest order)", "measured - fit (px)");
        b.scatter(order, res_f, color(1)).hline(0);
        save_png("03_원점오프셋.png", hstack({a.render(), b.render()}));
    }
    {
        std::vector<cv::Mat> tiles;
        for (int col = 0; col < 2; ++col) {
            const bool ax_y = col == 0;
            auto& o = oldfit[ax_y ? "y" : "t"];
            Chart a(std::string("Slope compare (") + (ax_y ? "Y" : "T") + "): intercept removed", std::string(ax_y ? "Y" : "T") + " command", "shift - intercept (px)");
            std::vector<double> oy, ol;
            for (size_t i = 0; i < o.x.size(); ++i) { oy.push_back(o.y[i] - o.icpt); ol.push_back(o.slope * o.x[i]); }
            a.scatter(o.x, oy, kBlack, "10-14 single axis", 2).line(o.x, ol, kBlack);
            Chart b(std::string("Line residual (") + (ax_y ? "Y" : "T") + ")", std::string(ax_y ? "Y" : "T") + " command", "residual (px)");
            std::vector<double> orr;
            for (size_t i = 0; i < o.x.size(); ++i) orr.push_back(o.y[i] - (o.slope * o.x[i] + o.icpt));
            b.line(o.x, orr, kBlack, "10-14", true);
            int k = 0;
            for (auto& m : mets) {
                if ((m.axis == "y") != ax_y) continue;
                std::vector<double> yy;
                for (size_t i = 0; i < m.x.size(); ++i) yy.push_back(m.y[i] - m.icpt);
                std::string lab = std::string("10-20 ") + (ax_y ? "T=" : "Y=") + fmt(m.v, 2);
                a.scatter(m.x, yy, color(k), lab, 4);
                b.line(m.x, m.res, color(k), lab, true);
                ++k;
            }
            b.hline(0);
            tiles.push_back(vstack({a.render(), b.render()}));
        }
        save_png("04_기존단일축과비교.png", hstack(tiles));
    }

    // ---- 튀는 값: 자유 절편 직선에서 가장 벗어난 사진, 반복 평균에서 가장 벗어난 사진 ----
    size_t wf = raw[worst(res_f, 1)[0]];
    size_t wr = raw[0];
    double wrv = 0;
    std::map<size_t, double> cond_mean;
    for (auto& c : g) for (auto i : c.idx) cond_mean[i] = c.mean;
    for (auto i : raw) if (std::abs(d[i].corner_dx - cond_mean[i]) > wrv) { wrv = std::abs(d[i].corner_dx - cond_mean[i]); wr = i; }
    auto res_of = [&](size_t i) { for (size_t k = 0; k < raw.size(); ++k) if (raw[k] == i) return res_f[k]; return kNaN; };
    Table outl{{"구분", "사진", "Y", "T", "corner_dx (px)", "조건 평균 (px)", "평균 대비 (px)", "자유 절편 직선 잔차 (px)", "ecc_dx (px)", "corner - ECC (px)"}};
    auto addrow = [&](const std::string& lab, size_t i) {
        outl.add({lab, d[i].rel, fmt(d[i].y, 2), fmt(d[i].t, 2), fmt(d[i].corner_dx, 4), fmt(cond_mean[i], 4), fmts(d[i].corner_dx - cond_mean[i], 4),
                  fmts(res_of(i), 4), fmt(d[i].ecc_dx, 4), fmts(d[i].corner_dx - d[i].ecc_dx, 4)});
    };
    for (auto k : worst(res_f, 3)) addrow("직선 잔차 상위", raw[k]);
    addrow("반복 편차 최대", wr);
    // 사진 확대: 세 QR 라벨, 초록 = 기준 사진 라벨, 빨강 = ECC 로 잰 이 사진 라벨 / 모서리 점 (사진 측정 위치)
    auto outlier_fig = [&](size_t i, const std::string& file, const std::string& cap) {
        cv::Mat im = hwm::read_gray0(paths[i]);
        std::vector<cv::Mat> tiles;
        for (int q = 0; q < 3; ++q) {
            cv::Rect2d refb(M.qrs[q].label);
            cv::Rect2d now(refb.x + M.shots[i].ecc[q].dx, refb.y + M.shots[i].ecc[q].dy, refb.width, refb.height);
            cv::Rect roi = cv::Rect(refb | now) + cv::Size(40, 40) - cv::Point(20, 20);
            std::vector<Mark> mk;
            for (size_t k = 0; k < M.corners.size(); ++k)
                if (M.corners[k].qr == q && M.usable[k] && !M.corners[k].diagonal && (k % 6 == 0))
                    mk.push_back({cv::Point2d(M.shots[i].pos[k]), {}, "", kRed});
            tiles.push_back(zoom(im, roi, 420, mk, 0, q == 0 ? cap : "",
                                 {{refb, kGreen, ""}, {now, kRed, "dx " + fmt(M.shots[i].ecc[q].dx, 2)}}));
        }
        save_png(file, hstack(tiles));
    };
    std::string f1 = "05_튀는값_직선잔차_" + d[wf].name + ".png", f2 = "06_튀는값_반복편차_" + d[wr].name + ".png";
    outlier_fig(wf, f1, d[wf].name + " resid " + fmts(res_of(wf), 3) + " px");
    outlier_fig(wr, f2, d[wr].name + " dev " + fmts(d[wr].corner_dx - cond_mean[wr], 3) + " px");

    // ---- 원래 결과와 대조 (eda_linearity_20261020_20261022/linearity_metrics.csv, summary.json, condition_statistics.csv) ----
    Parity par;
    const double o_slope[8] = {14.44073182957393, 14.543909774436091, 14.542713211600455, 14.360161833154327,
                               80.58878815039074, 80.38249512629655, 80.45382949310239, 80.43725782754485};
    const double o_rms[8] = {0.028934065497244234, 0.07485856058973615, 0.09277776256473967, 0.15844816474924428,
                             0.04121379074108669, 0.018680960235156842, 0.04991978692938418, 0.02208048868492254};
    const double o_p[8] = {0.43596084621525977, 0.4890969845274239, 0.2967583540155664, 0.0763405631359294,
                           0.7301002413768256, 0.9887665151232489, 0.6070250482601075, 0.899001991605423};
    par.add_text("구간 수 (Y 상승 4, T 상승 4)", "8", std::to_string(mets.size()), mets.size() == 8);
    for (size_t k = 0; k < mets.size() && k < 8; ++k) {
        std::string nm = std::string(mets[k].axis == "y" ? "Y 상승 T=" : "T 상승 Y=") + fmt(mets[k].v, 2);
        par.add(nm + " 기울기", o_slope[k], mets[k].slope, 0.002);
        par.add(nm + " 잔차 RMS (px)", o_rms[k], mets[k].rms, 0.0005);
        par.add(nm + " 적합결여 p", o_p[k], mets[k].p, 0.01);
    }
    par.add("Y 상승 R² 최소", 0.9991637758003042, r2y_min, 2e-6, 6);
    par.add("Y 상승 R² 최대", 0.9999724031939045, r2y_max, 2e-6, 6);
    par.add("T 상승 R² 최소", 0.9999994058936772, r2t_min, 2e-7, 7);
    par.add("반복 SD 중앙값 (px)", 0.164447651245008, sd_med, 0.0005);
    par.add("반복 SD 최대 (px)", 0.3339131773380619, sd_max, 0.0005);
    par.add("원점 통과 RMS (px)", 0.5337973553500955, rms_o, 0.0005);
    par.add("자유 절편 RMS (px)", 0.16863594046857477, rms_f, 0.0005);
    par.add("자유 절편 (px)", -0.7993964667644078, bf[0], 0.001);
    par.add("기존 Y 기울기 (px/mm)", 14.335315331010454, oldfit["y"].slope, 0.002);
    par.add("기존 T 기울기 (px/deg)", 80.25423727431566, oldfit["t"].slope, 0.002);
    par.add("기존 Y 잔차 RMS (px)", 0.09549696482313379, oldfit["y"].rms, 0.0005);
    par.add("기존 T 잔차 RMS (px)", 0.0901832351338937, oldfit["t"].rms, 0.0005);
    par.add("모서리-ECC 최대 차이 (px)", 0.014999999999999902, ce_max, 0.0002);
    par.add("기준 사진 3장 범위 (px)", 0.0764, cmax - cmin, 0.0002);
    par.add("20261020 평균에 쓴 모서리 수", 558, used, 0, 0);

    // ---- 결과.md ----
    std::ostringstream R;
    R << "# 결과: 20261020 축별 선형성과 반복 노이즈\n\n"
      << "입력: `data/clean/clean_20261020` 60장(기준 위치 center1~3 + 19조건 x 3회), 비교용 `data/clean/clean_20261014` 281장.\n"
      << "측정: hw_eda 와 같은 방법. 기준 사진(20261020은 center1, 20261014는 center.bmp)에서 QR 라벨 3개와 격자 모서리를 찾고, 사진마다 ECC 아핀으로 자리를 예측해 흐린 2x2 모듈 밝기 모델로 모서리를 다시 맞춘다. "
      << "corner_dx = 대각선이 아니고 모든 사진에서 추적된 모서리의 평균 수평 이동. 20261020은 모서리 " << M.corners.size() << "개 중 대각선 " << diag << "개를 빼고 "
      << used << "개를 평균했다. 20261014는 " << OM.corners.size() << "개 중 " << oused << "개.\n"
      << "원래 분석과 같게 corner_dx·ecc_dx를 소수 4자리로 반올림한 값으로 분석했다.\n\n"
      << "## 조건별 통계 (center 제외)\n\n" << condT.md()
      << "\n## 구간별 자유 절편 직선\n\n" << metT.md()
      << "\n적합결여 p는 조건 평균의 직선 잔차(장수 가중)와 반복 안 산포의 F 비다.\n\n"
      << "## 가산 직선 (center 제외 57장)\n\n| 모델 | 계수 | RMS (px) |\n|---|---|---|\n"
      << "| 원점 통과 dx = aY + bT | a " << fmt(bo[0], 4) << ", b " << fmt(bo[1], 4) << " | " << fmt(rms_o, 4) << " |\n"
      << "| 자유 절편 dx = c + aY + bT | c " << fmt(bf[0], 4) << ", a " << fmt(bf[1], 4) << ", b " << fmt(bf[2], 4) << " | " << fmt(rms_f, 4) << " |\n\n"
      << "## 기존 20261014 단일 축 양수 구간\n\n| 축 | 장수 | 기울기 | 절편 | 잔차 RMS (px) | R² |\n|---|---|---|---|---|---|\n";
    for (auto a : {"y", "t"})
        R << "| " << (std::string(a) == "y" ? "Y" : "T") << " | " << oldfit[a].n << " | " << fmt(oldfit[a].slope, 4) << " | " << fmt(oldfit[a].icpt, 4) << " | "
          << fmt(oldfit[a].rms, 4) << " | " << fmt(oldfit[a].r2, 6) << " |\n";
    R << "\n" << cmpT.md()
      << "\n## 원래 결과와 대조 (" << par.ok << "/" << par.total << " 일치)\n\n" << par.t.md()
      << "\n## 원래와 다른 점\n\n- 원래 그림의 가로축 '촬영 시각(분)'은 파일 수정 시각이었다. `data/clean`에는 촬영 시각이 없어서 매니페스트 순서·반복 번호로 대신 그렸다. 수치 지표는 시각을 쓰지 않는다.\n"
      << "- 원래 그림의 오차막대(반복 SD)는 그리지 않고 반복 각 장을 작은 점으로 그렸다.\n"
      << "\n## 파일\n\n- `shots_20261020.csv`: 20261020 사진별 ECC·모서리 평균 이동\n- `shots_20261014.csv`: 20261014 사진별 이동\n- `이미지/`: 그림\n";
    write_text(out("결과.md"), R.str());

    // ---- 보고서.md ----
    std::ostringstream P;
    P << "# 보고서: 20261020 선형성·반복 노이즈, 구간별 직선 적합 (10-22 15:28)\n\n"
      << "## 문제\n\n20261020 60장에서 Y·T 명령에 대한 영상 수평 이동이 직선인지, 반복 노이즈는 얼마인지 확인해야 했다.\n\n"
      << "## 해결 방법\n\n다른 축을 고정한 구간마다 조건 평균에 자유 절편 직선을 맞추고 잔차 RMS·반복 SD·적합결여 검정을 계산했다.\n\n"
      << "## 결과\n\nY 상승 구간 R²는 " << fmt(r2y_min, 6) << "~" << fmt(r2y_max, 6) << ", T 상승 구간은 " << fmt(r2t_min, 7)
      << " 이상이고, 반복 SD 중앙값은 " << fmt(sd_med, 3) << " px로 직선 잔차 RMS(Y " << fmt(rmsy_min, 3) << "~" << fmt(rmsy_max, 3) << " px)보다 크다.\n\n"
      << "| 지표 | 값 |\n|---|---|\n"
      << "| 사진 / 조건 | " << d.size() << " / " << g.size() << " |\n"
      << "| Y 상승 R² 범위 | " << fmt(r2y_min, 6) << " ~ " << fmt(r2y_max, 6) << " |\n"
      << "| T 상승 R² 최소 | " << fmt(r2t_min, 7) << " |\n"
      << "| 직선 잔차 RMS Y / T (px) | " << fmt(rmsy_min, 3) << "~" << fmt(rmsy_max, 3) << " / " << fmt(rmst_min, 3) << "~" << fmt(rmst_max, 3) << " |\n"
      << "| 반복 SD 중앙값 / 최대 (px) | " << fmt(sd_med, 4) << " / " << fmt(sd_max, 4) << " |\n"
      << "| 적합결여 p 최소 | " << fmt(p_min, 4) << " |\n"
      << "| 원점 통과 → 자유 절편 RMS (px) | " << fmt(rms_o, 4) << " → " << fmt(rms_f, 4) << " (절편 " << fmt(bf[0], 3) << ") |\n"
      << "| 기존 대비 기울기 차이 Y / T (%) | " << fmts(dmin[0], 2) << "~" << fmts(dmax[0], 2) << " / " << fmts(dmin[1], 2) << "~" << fmts(dmax[1], 2) << " |\n\n"
      << img_md("01_축별선형성_Y.png", "Y 상승 선형성") << "\n\n" << img_md("01_축별선형성_T.png", "T 상승 선형성") << "\n\n"
      << img_md("02_반복노이즈.png", "반복 노이즈") << "\n\n" << img_md("03_원점오프셋.png", "원점 오프셋") << "\n\n" << img_md("04_기존단일축과비교.png", "기존 비교") << "\n\n"
      << "## 튀는 값\n\nY·T 추정이 없는 실험이라 자유 절편 가산 직선에서 가장 벗어난 사진과 반복 평균에서 가장 벗어난 사진을 보였고, 가장 큰 직선 잔차는 "
      << d[wf].name << "의 " << fmts(res_of(wf), 3) << " px로 반복 SD 최대 " << fmt(sd_max, 3) << " px와 같은 크기다.\n\n"
      << outl.md() << "\n" << img_md(f1, "직선 잔차 튀는 값") << "\n\n" << img_md(f2, "반복 편차 튀는 값")
      << "\n\n초록 상자는 기준 사진(center1)의 QR 라벨, 빨간 상자는 ECC로 잰 이 사진의 라벨 위치, 빨간 십자는 이 사진에서 맞춘 모서리 일부(6개 중 1개)다.\n\n"
      << "## 한줄평\n\n조건 평균은 직선에 가깝지만(직선 잔차 RMS 최대 " << fmt(std::max(rmsy_max, rmst_max), 3) << " px) 같은 명령 3회 반복이 중앙값 " << fmt(sd_med, 3)
      << " px 흔들리고 원점 통과 직선이 자유 절편보다 RMS가 " << fmt(rms_o / rms_f, 1) << "배라서, 비선형보다 반복 산포와 원점 어긋남이 이동 오차의 주성분이다.\n\n"
      << "## 원래 결과와 대조\n\n원래 값은 `eda_linearity_20261020_20261022/linearity_metrics.csv`, `summary.json`, `condition_statistics.csv`에서 가져왔다. "
      << par.ok << "/" << par.total << " 일치.\n\n" << par.t.md();
    write_text(out("보고서.md"), P.str());
    std::cout << "parity " << par.ok << "/" << par.total << "\n";
    return 0;
}
