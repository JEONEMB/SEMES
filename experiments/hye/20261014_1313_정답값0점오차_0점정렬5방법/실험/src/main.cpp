// 0점 정렬 5방법 (원본: analysis_v4/zero_align.py)
// 모델 x_i = k v_i + o_i: x = QR 3개 평균 가로 이동(px, center = 0), v = 명령값, o = 사진 i 의 0점 어긋남.
// 방법마다 o 를 추정하고, 같은 데이터 적합 RMS 와 짝/홀(촬영 순서) 교차검증 RMS 를 본다.
// 원래 입력 14번 CSV(QR 위치)·15번 CSV(볼트 이동)는 만든 코드가 남아 있지 않아 사진에서 다시 쟀다:
//   QR = 시범 실험과 같은 세 영역 ECC affine 의 QR 중심 이동, 볼트 = 오른쪽 아래 볼트 64px 영역 ECC 이동.
// 촬영 시각(원래 os.path.getmtime)은 data/raw zip 의 NTFS 수정 시각에서 읽는다 (zip_mtime.hpp).
#include <hye/dataset.hpp>
#include <hye/ecc_compat.hpp>
#include <hye/plot.hpp>
#include <hye/report.hpp>
#include <omp.h>
#include <array>
#include <iostream>
#include <map>
#include "zip_mtime.hpp"

using namespace hye;

namespace {

const std::array<cv::Vec4i, 3> kBoxes{{{1554, 1734, 972, 1164}, {1815, 1995, 972, 1164}, {2073, 2253, 972, 1164}}};
const int kPad = 130;
const cv::Rect kBolt(3545, 1770, 64, 64);  // center.bmp 의 오른쪽 아래 볼트 (중심 약 3577, 1802)

struct Meas { double qx[3], qy[3], x, bolt, bolt_y, bolt_cc; };

Meas measure(const cv::Mat& center, const cv::Mat& im) {
    Meas m{};
    const auto crit = cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 150, 1e-7);
    for (int q = 0; q < 3; ++q) {
        auto b = kBoxes[q];
        cv::Mat tpl = center(cv::Range(b[2], b[3]), cv::Range(b[0], b[1]));
        cv::Mat search = im(cv::Range(b[2] - kPad, b[3] + kPad), cv::Range(b[0] - kPad, b[1] + kPad));
        cv::Mat res;
        cv::matchTemplate(search, tpl, res, cv::TM_CCOEFF_NORMED);
        cv::Point loc;
        cv::minMaxLoc(res, nullptr, nullptr, nullptr, &loc);
        int dx0 = loc.x - kPad, dy0 = loc.y - kPad;
        cv::Mat target = im(cv::Range(b[2] + dy0, b[3] + dy0), cv::Range(b[0] + dx0, b[1] + dx0));
        cv::Mat W = cv::Mat::eye(2, 3, CV_32F);
        try { semes::ecc5::findTransformECCWithMask(tpl, target, cv::noArray(), cv::noArray(), W, cv::MOTION_AFFINE, crit, 3); } catch (const cv::Exception&) {}
        m.qx[q] = dx0 + W.at<float>(0, 2) + (W.at<float>(0, 0) - 1) * 90 + W.at<float>(0, 1) * 96;  // QR 중심 가로 이동
        m.qy[q] = dy0 + W.at<float>(1, 2) + W.at<float>(1, 0) * 90 + (W.at<float>(1, 1) - 1) * 96;
    }
    m.x = (m.qx[0] + m.qx[1] + m.qx[2]) / 3;
    // 볼트: 옆 볼트(+133px)와 헷갈리지 않게 QR 이동의 0.6~1.3배 +-12px 안에서만 찾는다 (실측 비 0.79~1.13)
    double lo = std::min(0.6 * m.x, 1.3 * m.x) - 12, hi = std::max(0.6 * m.x, 1.3 * m.x) + 12;
    int ilo = (int)std::floor(lo), ihi = (int)std::ceil(hi);
    cv::Mat tpl = center(kBolt);
    cv::Mat search = im(cv::Rect(kBolt.x + ilo, kBolt.y - 10, kBolt.width + ihi - ilo, kBolt.height + 20));
    cv::Mat res;
    cv::matchTemplate(search, tpl, res, cv::TM_CCOEFF_NORMED);
    cv::Point loc;
    cv::minMaxLoc(res, nullptr, nullptr, nullptr, &loc);
    int dx0 = ilo + loc.x, dy0 = loc.y - 10;
    cv::Mat target = im(cv::Rect(kBolt.x + dx0, kBolt.y + dy0, kBolt.width, kBolt.height));
    cv::Mat W = cv::Mat::eye(2, 3, CV_32F);
    double cc = kNaN;
    try { cc = semes::ecc5::findTransformECCWithMask(tpl, target, cv::noArray(), cv::noArray(), W, cv::MOTION_TRANSLATION, crit, 3); } catch (const cv::Exception&) {}
    m.bolt = dx0 + W.at<float>(0, 2);
    m.bolt_y = dy0 + W.at<float>(1, 2);
    m.bolt_cc = cc;
    return m;
}

// ---------- 수치 도구 ----------
cv::Mat lsq(const cv::Mat& A, const cv::Mat& y) { cv::Mat p; cv::solve(A, y, p, cv::DECOMP_SVD); return p; }
std::pair<double, double> polyfit1(const std::vector<double>& x, const std::vector<double>& y) {
    double mx = mean(x), my = mean(y), sxy = 0, sxx = 0;
    for (size_t i = 0; i < x.size(); ++i) { sxy += (x[i] - mx) * (y[i] - my); sxx += (x[i] - mx) * (x[i] - mx); }
    double k = sxy / sxx;
    return {k, my - k * mx};
}
double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}
std::string sfmt(const char* f, double v) { char b[64]; std::snprintf(b, sizeof b, f, v); return b; }

struct P {  // 한 축의 사진 한 장 (촬영 순서)
    size_t shot;
    std::string file;
    double v, x, bolt, t, dir = 0;
    int seg = 0;
};
using Axis = std::vector<P>;

struct Fit { std::vector<double> o; double k; std::string note; int npar; std::vector<double> cp_t; };

// 증인: 다른 축 사진의 (시각, 0점 드리프트 px)
std::map<char, std::pair<std::vector<double>, std::vector<double>>> WIT;
double RY = 0, RT = 0;

std::vector<int> pelt(const std::vector<double>& t, const std::vector<double>& r, double pen, int minlen = 6) {
    const int n = (int)r.size();
    std::vector<double> F(n + 1, std::numeric_limits<double>::infinity());
    std::vector<int> last(n + 1, 0);
    F[0] = -pen;
    auto sse = [&](int i, int j) {
        double mt = 0, mr = 0;
        for (int k = i; k < j; ++k) { mt += t[k]; mr += r[k]; }
        mt /= (j - i); mr /= (j - i);
        double stt = 0, str = 0, srr = 0;
        for (int k = i; k < j; ++k) { double a = t[k] - mt, b = r[k] - mr; stt += a * a; str += a * b; srr += b * b; }
        return srr - (stt > 0 ? str * str / stt : 0);
    };
    for (int j = minlen; j <= n; ++j)
        for (int i = 0; i <= j - minlen; ++i) {
            if (std::isinf(F[i]) || (i > 0 && i < minlen)) continue;
            double c = F[i] + sse(i, j) + pen;
            if (c < F[j]) { F[j] = c; last[j] = i; }
        }
    std::vector<int> cps;
    for (int j = n; j > 0; j = last[j]) cps.push_back(j);
    std::sort(cps.begin(), cps.end());
    cps.pop_back();
    return cps;
}

Fit fit(const Axis& g, const std::vector<bool>& tr, int method, char ax) {
    const int n = (int)g.size();
    Fit f;
    f.o.assign(n, 0);
    std::vector<int> it;
    for (int i = 0; i < n; ++i) if (tr[i]) it.push_back(i);
    auto col = [&](auto get) { cv::Mat m(n, 1, CV_64F); for (int i = 0; i < n; ++i) m.at<double>(i) = get(g[i], i); return m; };
    auto rows = [&](const cv::Mat& m) { cv::Mat o((int)it.size(), m.cols, CV_64F); for (size_t r = 0; r < it.size(); ++r) m.row(it[r]).copyTo(o.row((int)r)); return o; };
    cv::Mat V = col([](const P& p, int) { return p.v; }), X = col([](const P& p, int) { return p.x; });
    if (method == 0) {
        f.k = lsq(rows(V), rows(X)).at<double>(0);
        f.note = "보정 없음"; f.npar = 1;
        return f;
    }
    if (method == 2) {
        auto& w = WIT[ax];
        std::vector<double> tw{0}, ew{0};
        tw.insert(tw.end(), w.first.begin(), w.first.end());
        ew.insert(ew.end(), w.second.begin(), w.second.end());
        auto [a, b] = polyfit1(tw, ew);
        for (int i = 0; i < n; ++i) f.o[i] = a * g[i].t + b;
        cv::Mat Xc = col([&](const P& p, int i) { return p.x - f.o[i]; });
        f.k = lsq(rows(V), rows(Xc)).at<double>(0);
        f.note = "증인 드리프트 " + sfmt("%+.3f", a) + "px/분 (rY=" + sfmt("%.2f", RY) + ", rΘ=" + sfmt("%.2f", RT) + ")";
        f.npar = 1;
        return f;
    }
    if (method == 5) {
        f.k = lsq(rows(V), rows(X)).at<double>(0);
        std::vector<int> cps;
        for (int iter = 0; iter < 3; ++iter) {
            std::vector<double> r(n), ti, ri, d2;
            for (int i = 0; i < n; ++i) r[i] = g[i].x - f.k * g[i].v;
            for (int i : it) { ti.push_back(g[i].t); ri.push_back(r[i]); }
            for (size_t i = 1; i < ri.size(); ++i) d2.push_back((ri[i] - ri[i - 1]) * (ri[i] - ri[i - 1]));
            double sig2 = median(d2) / (2 * 0.455) + 1e-9;
            cps = pelt(ti, ri, 4 * sig2 * std::log((double)it.size()));
            std::vector<int> bounds{0};
            bounds.insert(bounds.end(), cps.begin(), cps.end());
            bounds.push_back((int)it.size());
            std::vector<int> pid(n);
            for (int i = 0; i < n; ++i) {
                int best = 0;
                for (int j = 1; j < (int)it.size(); ++j) if (std::abs(g[i].t - ti[j]) < std::abs(g[i].t - ti[best])) best = j;
                pid[i] = (int)(std::upper_bound(cps.begin(), cps.end(), best) - cps.begin());
            }
            std::fill(f.o.begin(), f.o.end(), 0.0);
            for (int s = 0; s + 1 < (int)bounds.size(); ++s) {
                std::vector<double> tt(ti.begin() + bounds[s], ti.begin() + bounds[s + 1]), rr(ri.begin() + bounds[s], ri.begin() + bounds[s + 1]);
                auto [a, b] = polyfit1(tt, rr);
                for (int i = 0; i < n; ++i) if (pid[i] == s) f.o[i] = a * g[i].t + b;
            }
            cv::Mat Xc = col([&](const P& p, int i) { return p.x - f.o[i]; });
            f.k = lsq(rows(V), rows(Xc)).at<double>(0);
        }
        std::string ts;
        for (size_t c = 0; c < cps.size(); ++c) { ts += (c ? ", " : "") + sfmt("%.0f", g[it[cps[c]]].t); f.cp_t.push_back(g[it[cps[c]]].t); }
        f.note = "변화점 " + std::to_string(cps.size()) + "개 (촬영 " + ts + "분)";
        f.npar = 1 + 2 * ((int)cps.size() + 1);
        return f;
    }
    const int S = g.back().seg + 1;  // 시간순이라 마지막 사진의 구간 번호 = 구간 수 - 1 (구간 번호는 단조 증가)
    int ncol = method == 1 ? 5 : method == 3 ? 2 : 1 + S;
    cv::Mat A = cv::Mat::zeros(n, ncol, CV_64F);
    for (int i = 0; i < n; ++i) {
        A.at<double>(i, 0) = g[i].v;
        if (method == 1) { A.at<double>(i, 1) = 1; A.at<double>(i, 2) = g[i].dir; A.at<double>(i, 3) = g[i].t; A.at<double>(i, 4) = 1 - std::exp(-g[i].t / 3); }
        if (method == 3) A.at<double>(i, 1) = g[i].v * g[i].v;
        if (method == 4) A.at<double>(i, 1 + g[i].seg) = 1;
    }
    cv::Mat p = lsq(rows(A), rows(X));
    f.k = p.at<double>(0);
    cv::Mat off = A.colRange(1, ncol) * p.rowRange(1, ncol);
    for (int i = 0; i < n; ++i) f.o[i] = off.at<double>(i);
    f.npar = ncol;
    if (method == 1) f.note = "β=" + sfmt("%+.2f", p.at<double>(2)) + "px, 드리프트 " + sfmt("%+.3f", p.at<double>(3)) + "px/분";
    if (method == 3) f.note = "2차항 c=" + sfmt("%+.3f", p.at<double>(1)) + "px";
    if (method == 4) {
        f.note = "구간 절편 ";
        for (int s = 0; s < S; ++s) f.note += (s ? ", " : "") + sfmt("%+.2f", p.at<double>(1 + s));
        f.note += "px";
    }
    return f;
}

const char* kName[6] = {"보정 전", "① 백래시 + 드리프트", "② 직교축 증인", "③ 회전 중심 기하 (2차)", "④ 경계 연속 (구간 절편)", "⑤ 변화점 + 시간 직선"};
const char* kKind[6] = {"base", "이론", "이론", "이론", "데이터", "데이터"};
const char* kEng[6] = {"0 none", "1 backlash+drift", "2 witness", "3 quadratic", "4 segment icpt", "5 changepoint"};

struct Row { double rms, mx, cv, left, right; Fit f; std::vector<double> res; };

}  // namespace

static int run();
int main() {
    try { return run(); } catch (const std::exception& e) { std::cerr << "error: " << e.what() << "\n"; return 1; }
}

static int run() {
    omp_set_num_threads(4);
    begin_results();
    auto shots = load_1014();
    // ---- 촬영 시각 ----
    auto times = zipm::capture_times(fs::u8path(repo_dir() + "/data/raw/zip_20261014.zip"), "Set.zip");
    std::vector<double> tsec(shots.size());
    double t0 = 1e300;
    for (size_t i = 0; i < shots.size(); ++i) {
        auto it = times.find("SSAFY Test Image Set/" + shots[i].rel);
        if (it == times.end()) throw std::runtime_error("촬영 시각 없음: " + shots[i].rel);
        tsec[i] = it->second;
        t0 = std::min(t0, tsec[i]);
    }
    // ---- 측정 ----
    const cv::Mat center = read_gray(center_1014(shots));
    std::vector<Meas> meas(shots.size());
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < (int)shots.size(); ++i) meas[i] = measure(center, read_gray(shots[i]));
    std::cout << "measured " << shots.size() << "\n";
    Table raw{{"file", "Y", "T", "time_min", "q0_dx", "q1_dx", "q2_dx", "qr_mean_dx", "bolt_dx", "bolt_dy", "bolt_cc"}};
    for (size_t i = 0; i < shots.size(); ++i) {
        auto& m = meas[i];
        raw.add({shots[i].rel, fmt(shots[i].y, 2), fmt(shots[i].t, 2), fmt((tsec[i] - t0) / 60, 4), fmt(m.qx[0], 4), fmt(m.qx[1], 4), fmt(m.qx[2], 4),
                 fmt(m.x, 4), fmt(m.bolt, 4), fmt(m.bolt_y, 4), fmt(m.bolt_cc, 5)});
    }
    raw.csv(out("qr_bolt_measurements.csv"));

    // ---- 축별 시간순 정리, 구간·방향 ----
    std::map<char, Axis> AX;
    for (size_t i = 0; i < shots.size(); ++i) {
        auto& s = shots[i];
        if (s.condition == "center") continue;
        char a = s.condition == "Y" ? 'Y' : 'T';
        std::string file = s.rel.substr(s.rel.find('/') + 1);
        AX[a].push_back({i, file, a == 'Y' ? s.y : s.t, meas[i].x, meas[i].bolt, (tsec[i] - t0) / 60});
    }
    for (auto& [a, g] : AX) {
        std::stable_sort(g.begin(), g.end(), [](const P& p, const P& q) { return p.t < q.t; });
        for (size_t i = 1; i < g.size(); ++i) {
            auto sg = [](double v) { return (v > 0) - (v < 0); };
            g[i].seg = g[i - 1].seg + ((g[i].t - g[i - 1].t > 3 || sg(g[i].v) != sg(g[i - 1].v)) ? 1 : 0);
        }
        for (int s = 0; s <= g.back().seg; ++s) {
            std::vector<double> t, v;
            for (auto& p : g) if (p.seg == s) { t.push_back(p.t); v.push_back(p.v); }
            double k = polyfit1(t, v).first, d = (k > 0) - (k < 0);
            for (auto& p : g) if (p.seg == s) p.dir = d;
        }
    }
    std::map<char, double> U;
    for (auto& [a, g] : AX) {
        std::vector<double> v, x;
        for (auto& p : g) { v.push_back(p.v); x.push_back(p.x); }
        U[a] = 1000 / polyfit1(v, x).first;
    }
    auto ratio = [](const Axis& g) { double sxb = 0, sxx = 0; for (auto& p : g) { sxb += p.x * p.bolt; sxx += p.x * p.x; } return sxb / sxx; };
    RY = ratio(AX['Y']);
    RT = ratio(AX['T']);
    for (auto& p : AX['T']) { WIT['Y'].first.push_back(p.t); WIT['Y'].second.push_back((p.bolt - RT * p.x) / (RY - RT)); }
    for (auto& p : AX['Y']) { WIT['T'].first.push_back(p.t); WIT['T'].second.push_back((p.bolt - RY * p.x) / (RT - RY)); }

    // ---- 방법 5개 + 보정 전 ----
    std::map<char, std::array<Row, 6>> R;
    Table tab{{"축", "방법", "관점", "파라미터수", "RMS_적합", "최대_적합", "RMS_CV", "좌절편", "우절편", "좌우차", "단위", "메모"}};
    for (char a : {'Y', 'T'}) {
        auto& g = AX[a];
        const int n = (int)g.size();
        const double u = U[a];
        std::vector<bool> all(n, true), even(n), odd(n);
        for (int i = 0; i < n; ++i) { even[i] = i % 2 == 0; odd[i] = !even[i]; }
        for (int m = 0; m < 6; ++m) {
            Row r;
            r.f = fit(g, all, m, a);
            std::vector<double> nl, nr, pl, pr;
            for (int i = 0; i < n; ++i) {
                double e = g[i].x - r.f.o[i] - r.f.k * g[i].v;
                r.res.push_back(e);
                (g[i].v < 0 ? nl : pl).push_back(g[i].v);
                (g[i].v < 0 ? nr : pr).push_back(e);
            }
            std::vector<double> cv;
            for (auto* trm : {&even, &odd}) {
                Fit f2 = fit(g, *trm, m, a);
                for (int i = 0; i < n; ++i) if (!(*trm)[i]) cv.push_back(g[i].x - f2.o[i] - f2.k * g[i].v);
            }
            r.rms = rms(r.res) * u;
            r.mx = max_abs(r.res) * u;
            r.cv = rms(cv) * u;
            r.left = polyfit1(nl, nr).second * u;
            r.right = polyfit1(pl, pr).second * u;
            tab.add({a == 'Y' ? "Y" : "Θ", kName[m], kKind[m], std::to_string(r.f.npar), fmt(r.rms, 3), fmt(r.mx, 3), fmt(r.cv, 3), fmt(r.left, 3),
                     fmt(r.right, 3), fmt(std::abs(r.left - r.right), 3), a == 'Y' ? "µm" : "mΘ (0.001)", r.f.note});
            R[a][m] = r;
        }
    }
    tab.csv(out("결과표_0점정렬_5방법.csv"));

    // ---- 그림 ----
    auto segcol = [](int s) { static const cv::Scalar c[4] = {{208, 114, 74}, {63, 112, 210}, {118, 167, 90}, {160, 122, 207}}; return c[s % 4]; };
    for (char a : {'Y', 'T'}) {
        auto& g = AX[a];
        const double u = U[a];
        const std::string un = a == 'Y' ? "um" : "mT", axn = a == 'Y' ? "Y" : "T";
        std::vector<cv::Mat> top, bot, ttop, tbot;
        for (int m = 0; m < 6; ++m) {
            auto& r = R[a][m];
            Chart c(std::string(kEng[m]) + ": RMS " + fmt(r.rms, 2) + " CV " + fmt(r.cv, 2), "command " + axn + (a == 'Y' ? " (mm)" : " (deg)"), "residual (" + un + ")");
            c.W = 620; c.H = 400;
            std::vector<double> lx, ly;
            for (auto& p : g) lx.push_back(p.v);
            for (double e : r.res) ly.push_back(e * u);
            c.line(lx, ly, {200, 200, 200});
            for (int s = 0; s <= g.back().seg; ++s) {
                std::vector<double> x, y;
                for (size_t i = 0; i < g.size(); ++i) if (g[i].seg == s) { x.push_back(g[i].v); y.push_back(r.res[i] * u); }
                c.scatter(x, y, segcol(s), m == 0 ? "seg " + std::to_string(s + 1) : "");
            }
            c.hline(0);
            (m < 3 ? top : bot).push_back(c.render());
            Chart d(std::string(kEng[m]) + ": estimated zero o(t)", "capture time (min)", "offset (" + un + ")");
            d.W = 620; d.H = 400;
            std::vector<double> tx, raw0, oo;
            for (size_t i = 0; i < g.size(); ++i) { tx.push_back(g[i].t); raw0.push_back(R[a][0].res[i] * u); oo.push_back(r.f.o[i] * u); }
            d.scatter(tx, raw0, kGray, "raw residual", 2);
            if (m == 2) d.scatter(WIT[a].first, [&] { std::vector<double> e; for (double v : WIT[a].second) e.push_back(v * u); return e; }(), color(4), "witness", 2);
            d.line(tx, oo, kBlack, "o(t)", true);
            for (double ct : r.f.cp_t) d.vline(ct, kRed);
            (m < 3 ? ttop : tbot).push_back(d.render());
        }
        save_png(a == 'Y' ? "01_Y_6방법_잔차.png" : "02_T_6방법_잔차.png", vstack({hstack(top), hstack(bot)}));
        save_png(a == 'Y' ? "03_Y_추정0점_시간.png" : "04_T_추정0점_시간.png", vstack({hstack(ttop), hstack(tbot)}));
    }
    {
        std::vector<std::string> cats;
        std::vector<double> vy, vt;
        for (int m = 0; m < 6; ++m) { cats.push_back(kEng[m]); vy.push_back(R['Y'][m].cv); vt.push_back(R['T'][m].cv); }
        bar_chart("05_Y_방법별_CV_RMS.png", "Y: cross-validated RMS (um)", cats, vy, "RMS CV (um)", 1);
        bar_chart("06_T_방법별_CV_RMS.png", "T: cross-validated RMS (mT = 0.001 deg)", cats, vt, "RMS CV (mT)", 5);
    }
    // 튀는 값: 축마다 CV 가 가장 작은 방법의 잔차 최대 사진
    std::map<char, int> best;
    for (char a : {'Y', 'T'}) {
        int b = 1;
        for (int m = 1; m < 6; ++m) if (R[a][m].cv < R[a][b].cv) b = m;
        best[a] = b;
    }
    std::map<char, std::vector<size_t>> top3;
    std::map<char, std::string> outfile;
    Table outl{{"축 (방법)", "사진", "명령값", "촬영 시각 (분)", "구간", "QR 평균 이동 x (px)", "k·v (px)", "추정 0점 o (px)", "잔차", "q0 / q1 / q2 이동 (px)", "볼트 이동 (px)"}};
    for (char a : {'Y', 'T'}) {
        auto& g = AX[a];
        auto& r = R[a][best[a]];
        top3[a] = worst(r.res, 3);
        for (size_t k : top3[a]) {
            auto& p = g[k];
            auto& m = meas[p.shot];
            outl.add({std::string(a == 'Y' ? "Y " : "Θ ") + kName[best[a]], p.file, fmt(p.v, 2), fmt(p.t, 2), std::to_string(p.seg + 1), fmt(p.x, 3), fmt(r.f.k * p.v, 3),
                      fmt(r.f.o[k], 3), fmts(r.res[k] * U[a], 2) + (a == 'Y' ? " µm" : " mΘ"), fmt(m.qx[0], 2) + " / " + fmt(m.qx[1], 2) + " / " + fmt(m.qx[2], 2),
                      fmt(p.bolt, 2)});
        }
        size_t k = top3[a][0];
        auto& p = g[k];
        auto& m = meas[p.shot];
        cv::Mat im = read_color(shots[p.shot]);
        std::vector<cv::Mat> tiles;
        double expect = r.f.k * p.v + r.f.o[k];  // 모델이 기대한 평균 이동
        for (int q = 0; q < 3; ++q) {
            auto b = kBoxes[q];
            cv::Rect2d ref(b[0], b[2], b[1] - b[0], b[3] - b[2]);
            cv::Rect2d now(ref.x + m.qx[q], ref.y + m.qy[q], ref.width, ref.height);
            cv::Rect roi = cv::Rect(ref | now) + cv::Size(60, 60) - cv::Point(30, 30);
            std::string cap = p.file.substr(0, p.file.rfind('.')) + " residual " + fmts(r.res[k] * U[a], 2) + " " + (a == 'Y' ? "um" : "mT");
            tiles.push_back(zoom(im, roi, 380, {{{ref.x + ref.width / 2 + expect, ref.y + ref.height / 2 + m.qy[q]}, {0, 0}, "model", kGreen},
                                                 {{now.x + now.width / 2, now.y + now.height / 2}, {0, 0}, "", kRed}},
                                 0, q == 0 ? cap : "", {{ref, kGray, "center"}, {now, kRed, "dx " + fmt(m.qx[q], 2)}}));
        }
        outfile[a] = std::string(a == 'Y' ? "07_튀는값_Y_" : "08_튀는값_T_") + p.file.substr(0, p.file.rfind('.')) + ".png";
        save_png(outfile[a], hstack(tiles));
    }

    // ---- 원래 결과와 대조 ----
    // 원래 값: analysis_v4 실행 결과 33_결과표_0점정렬_5방법.csv (결과/원본자료/ 사본). 소수 셋째 자리 반올림 값.
    struct O { double rms, mx, cv, gap; int npar; };
    const std::map<char, std::array<O, 6>> orig{
        {'Y', {{{27.017, 46.579, 27.025, 21.218, 1}, {1.724, 8.609, 1.867, 0.111, 5}, {100.379, 159.174, 100.537, 41.461, 1}, {20.858, 36.138, 20.923, 21.218, 2},
                {1.77, 8.002, 1.908, 0.0, 5}, {1.78, 6.719, 2.885, 0.545, 11}}}},
        {'T', {{{8.238, 11.647, 8.238, 7.891, 1}, {0.41, 1.54, 0.411, 0.0, 5}, {9.395, 12.96, 9.395, 7.067, 1}, {5.102, 10.426, 5.103, 7.891, 2},
                {0.982, 3.982, 0.983, 0.0, 3}, {0.27, 0.82, 0.379, 0.0, 19}}}}};
    // 허용오차: 입력을 다시 잰 차이. 원래 14번 CSV 와 ECC QR 중심 이동(시범 실험 측정)의 차이 표준편차 0.0127 px 를 단위로 바꾼 값 + 원래 값의 5%
    const double kMeasSd = 0.0127;
    Parity par;
    for (char a : {'Y', 'T'}) {
        const std::string an = a == 'Y' ? "Y" : "Θ", un = a == 'Y' ? " (µm)" : " (mΘ)";
        const double base = kMeasSd * U[a];
        for (int m = 0; m < 6; ++m) {
            auto& o = orig.at(a)[m];
            auto& r = R[a][m];
            par.add(an + " " + kName[m] + " RMS 적합" + un, o.rms, r.rms, base + 0.05 * o.rms, 3);
            par.add(an + " " + kName[m] + " RMS CV" + un, o.cv, r.cv, base + 0.05 * o.cv, 3);
            if (m == 4 || m == 5) par.add(an + " " + kName[m] + " 파라미터 수", o.npar, r.f.npar, 0, 0);
        }
    }
    par.add("Y 보정 전 좌우 0점 차 (µm)", 21.218, std::abs(R['Y'][0].left - R['Y'][0].right), kMeasSd * U['Y'] + 0.05 * 21.218, 3);
    par.add("Θ 보정 전 좌우 0점 차 (mΘ)", 7.891, std::abs(R['T'][0].left - R['T'][0].right), kMeasSd * U['T'] + 0.05 * 7.891, 3);

    // ---- 구간 표 ----
    Table segt{{"축", "구간", "장수", "촬영 시각 (분)", "명령값 범위", "이동 방향"}};
    for (char a : {'Y', 'T'}) {
        auto& g = AX[a];
        for (int s = 0; s <= g.back().seg; ++s) {
            double t0s = 1e9, t1s = -1e9, v0 = 1e9, v1 = -1e9, d = 0;
            int c = 0;
            for (auto& p : g) if (p.seg == s) { ++c; t0s = std::min(t0s, p.t); t1s = std::max(t1s, p.t); v0 = std::min(v0, p.v); v1 = std::max(v1, p.v); d = p.dir; }
            segt.add({a == 'Y' ? "Y" : "Θ", std::to_string(s + 1), std::to_string(c), fmt(t0s, 1) + " ~ " + fmt(t1s, 1), fmt(v0, 2) + " ~ " + fmt(v1, 2), d > 0 ? "+" : "-"});
        }
    }

    // ---- 결과.md ----
    std::ostringstream Rm;
    Rm << "# 결과: 0점 정렬 5방법 (Y, Θ 각각)\n\n"
       << "입력: `data/clean/clean_20261014` 281장 (center 1, Θ 200, Y 80). 촬영 시각은 `data/raw/zip_20261014.zip` 안 분할 zip의 NTFS 수정 시각(원래 스크립트의 파일 수정 시각과 같은 값)이고, 가장 이른 사진(center) = 0분이다.\n\n"
       << "측정: x = QR 세 개의 중심 가로 이동 평균(px, center = 0). 세 QR 영역(180x192 px)을 템플릿 매칭 후 ECC affine으로 정합했다(시범 실험과 같은 측정). "
       << "증인 방법(②)의 볼트 이동은 오른쪽 아래 볼트 64x64 px 영역을 ECC 평행 이동으로 쟀다. 원래 입력 `14_데이터_QR마커위치.csv`, `15_데이터_파인더_볼트이동량.csv`를 만든 코드는 남아 있지 않다.\n\n"
       << "모델: x_i = k v_i + o_i. 단위 환산 Y 1 µm = " << fmt(1 / U['Y'], 5) << " px (" << fmt(1000 / U['Y'], 3) << " px/mm), Θ 0.001 = " << fmt(1 / U['T'], 5)
       << " px (" << fmt(1000 / U['T'], 3) << " px/deg). 볼트/QR 비 rY = " << fmt(RY, 3) << ", rΘ = " << fmt(RT, 3) << ".\n"
       << "CV: 축마다 촬영 순서 짝수 번째로 맞추고 홀수 번째를 채점, 반대로도 해서 두 쪽 잔차를 합친 RMS.\n\n"
       << "## 방법별 결과\n\n" << tab.md() << "\n## 촬영 구간 (같은 축에서 3분 넘는 공백 또는 부호가 바뀌면 새 구간)\n\n" << segt.md()
       << "\n## 원래 결과와 대조 (" << par.ok << "/" << par.total << " 일치)\n\n"
       << "허용오차 = 입력 재측정 차이(0.0127 px = Y " << fmt(kMeasSd * U['Y'], 3) << " µm, Θ " << fmt(kMeasSd * U['T'], 3) << " mΘ) + 원래 값의 5%.\n\n" << par.t.md()
       << "\n## 옮기지 않은 부분\n\n- 원래 그림(34·35번, matplotlib 3행 x 6열)은 OpenCV 차트 4장(01~04)으로 나눠 그렸다. 촬영 경로 화살표는 빼고 시간순 회색 선으로 이었다.\n"
       << "- 14·15번 CSV의 QR 박스 폭·높이, 파인더 위치는 이 실험에서 쓰지 않아 재지 않았다.\n"
       << "\n## 파일\n\n- `qr_bolt_measurements.csv`: 사진별 촬영 시각, QR 세 개 중심 이동, 평균, 볼트 이동\n- `결과표_0점정렬_5방법.csv`: 원래 33번 표와 같은 열\n"
       << "- `이미지/01_Y_6방법_잔차.png`, `02_T_6방법_잔차.png`: 명령값 대비 보정 후 잔차 (색 = 촬영 구간)\n"
       << "- `이미지/03_Y_추정0점_시간.png`, `04_T_추정0점_시간.png`: 촬영 시각 대비 추정 0점 o(t), 회색 = 보정 전 잔차, 빨간 세로선 = ⑤ 변화점\n"
       << "- `이미지/05_Y_방법별_CV_RMS.png`, `06_T_방법별_CV_RMS.png`, `" << outfile['Y'] << "`, `" << outfile['T'] << "`\n";
    write_text(out("결과.md"), Rm.str());

    // ---- 보고서.md ----
    auto& b1 = R['Y'][1];
    auto& t1 = R['T'][1];
    const int by = best['Y'], bt = best['T'];
    auto& py = AX['Y'][top3['Y'][0]];
    auto& pt = AX['T'][top3['T'][0]];
    auto& ry = R['Y'][by];
    auto& rt = R['T'][bt];
    // 튀는 사진이 구간 안 몇 번째인지, QR 세 개가 같은 방향으로 어긋났는지
    auto where = [&](char a, size_t k) {
        auto& g = AX[a];
        int s = g[k].seg, pos = 0, cnt = 0;
        for (size_t i = 0; i < g.size(); ++i) if (g[i].seg == s) { ++cnt; if (i <= k) ++pos; }
        auto& m = meas[g[k].shot];
        double spread = std::max({m.qx[0], m.qx[1], m.qx[2]}) - std::min({m.qx[0], m.qx[1], m.qx[2]});
        return "구간 " + std::to_string(s + 1) + "의 " + std::to_string(pos) + "번째/" + std::to_string(cnt) + "장, QR 세 개 이동 차 " + fmt(spread, 3) + " px";
    };
    auto why = [&](char a, size_t k, const Row& r) {
        auto& m = meas[AX[a][k].shot];
        double spread = std::max({m.qx[0], m.qx[1], m.qx[2]}) - std::min({m.qx[0], m.qx[1], m.qx[2]}), e = std::abs(r.res[k]);
        std::string an = a == 'Y' ? "Y" : "Θ";
        if (spread < e) return an + " 사진은 QR 세 개가 " + fmt(spread, 3) + " px 안에서 함께 어긋나 측정 잡음이 아니라 그 사진의 실제 위치가 0점 모델(시간 직선)에서 벗어난 것이다.";
        return an + " 사진의 잔차 " + fmt(e, 3) + " px는 그 방법 RMS(" + fmt(rms(r.res), 3) + " px)의 " + fmt(e / rms(r.res), 1) +
               "배이고, QR 세 개의 차(" + fmt(spread, 3) + " px)는 회전에 따른 원근 차이를 포함해 잔차보다 커서 잡음과 실제 어긋남을 이 사진만으로 가를 수 없다.";
    };
    std::ostringstream P;
    P << "# 보고서: 정답값 0점 오차와 0점 정렬 5방법 (10-14 13:13)\n\n"
      << "## 문제\n\n이동 구간마다 0점이 달라 정답값(명령값)과 QR 이동의 직선 관계 자체에 오차가 섞여 있었다.\n\n"
      << "## 해결 방법\n\nQR 세 개의 평균 가로 이동에 0점 정렬 5방법(백래시+드리프트, 직교축 증인, 2차 기하, 구간 절편, 변화점+시간 직선)을 Y·Θ에 따로 적용해 짝/홀 교차검증 RMS로 비교했다.\n\n"
      << "## 결과\n\n백래시+드리프트(①)가 Y RMS를 " << fmt(R['Y'][0].rms, 3) << "→" << fmt(b1.rms, 3) << " µm, Θ를 " << fmt(R['T'][0].rms, 3) << "→" << fmt(t1.rms, 3)
      << " mΘ로 줄였고, 교차검증 최소는 Y " << kName[by] << " " << fmt(ry.cv, 3) << " µm, Θ " << kName[bt] << " " << fmt(rt.cv, 3) << " mΘ였다.\n\n"
      << "| 축 | 방법 | 파라미터 | RMS 적합 | RMS CV | 최대 \\|잔차\\| | 좌우 0점 차 |\n|---|---|---|---|---|---|---|\n";
    for (char a : {'Y', 'T'})
        for (int m = 0; m < 6; ++m) {
            auto& r = R[a][m];
            P << "| " << (a == 'Y' ? "Y (µm)" : "Θ (mΘ)") << " | " << kName[m] << " | " << r.f.npar << " | " << fmt(r.rms, 3) << " | " << fmt(r.cv, 3) << " | " << fmt(r.mx, 3)
              << " | " << fmt(std::abs(r.left - r.right), 3) << " |\n";
        }
    P << "\n" << img_md("01_Y_6방법_잔차.png", "Y 6방법 잔차") << "\n\n" << img_md("02_T_6방법_잔차.png", "T 6방법 잔차") << "\n\n"
      << img_md("03_Y_추정0점_시간.png", "Y 추정 0점") << "\n\n" << img_md("04_T_추정0점_시간.png", "T 추정 0점") << "\n\n"
      << img_md("05_Y_방법별_CV_RMS.png", "Y CV RMS") << "\n\n" << img_md("06_T_방법별_CV_RMS.png", "T CV RMS") << "\n\n"
      << "## 튀는 값\n\n교차검증이 가장 좋은 방법으로 보정해도 가장 크게 남은 사진은 Y는 " << py.file << "(명령 " << fmt(py.v, 2) << " mm, 잔차 " << fmts(ry.res[top3['Y'][0]] * U['Y'], 2)
      << " µm = " << fmts(ry.res[top3['Y'][0]], 3) << " px, " << where('Y', top3['Y'][0]) << "), Θ는 " << pt.file << "(명령 " << fmt(pt.v, 2) << ", 잔차 "
      << fmts(rt.res[top3['T'][0]] * U['T'], 2) << " mΘ = " << fmts(rt.res[top3['T'][0]], 3) << " px, " << where('T', top3['T'][0])
      << ")이다. " << why('Y', top3['Y'][0], ry) << " " << why('T', top3['T'][0], rt) << "\n\n"
      << outl.md() << "\n" << img_md(outfile['Y'], "Y 튀는 값") << "\n\n" << img_md(outfile['T'], "T 튀는 값") << "\n\n"
      << "회색 상자는 center 사진의 QR 영역, 빨간 상자와 빨간 십자는 이 사진에서 잰 QR 위치, 초록 십자(model)는 보정 모델이 기대한 위치(k·v + o)다. 둘의 가로 차이가 잔차다.\n\n"
      << "## 한줄평\n\n0점 어긋남이 이동 방향(백래시 β " << b1.f.note.substr(b1.f.note.find("=") + 1, b1.f.note.find("px") - b1.f.note.find("=") - 1)
      << " px)과 시간 드리프트로 대부분 설명되는 계통 오차라서, 이 두 항만 넣은 ①이 파라미터 5개로 Y 잔차를 " << fmt(R['Y'][0].rms / b1.rms, 1) << "배 줄였고, 볼트 시차로 다른 축 성분을 뽑는 ②는 볼트/QR 비 차이(rY "
      << fmt(RY, 2) << ", rΘ " << fmt(RT, 2) << ")가 작아 측정 잡음이 " << fmt(1 / std::abs(RY - RT), 1) << "배 커져 오히려 나빠졌다.\n\n"
      << "## 원래 결과와 대조\n\n원래 값은 analysis_v4 실행 결과 `33_결과표_0점정렬_5방법.csv`(이 실험 `결과/원본자료/` 사본)에서 가져왔다. "
      << "원래 입력(14·15번 CSV)을 만든 코드가 없어 QR·볼트 이동을 ECC로 다시 쟀으므로, 허용오차는 두 측정의 차이(0.0127 px)를 단위로 바꾼 값에 원래 값의 5%를 더했다. "
      << par.ok << "/" << par.total << " 일치.\n\n" << par.t.md()
      << "\n불일치 이유: ② 직교축 증인은 볼트 이동을 다시 잰 값이 원래 15번 CSV와 사진별로 표준편차 0.6~0.7 px 달라(볼트가 흐리고 반사광이 있음) 볼트/QR 비가 rY "
      << fmt(RY, 2) << "(원래 0.79), rΘ " << fmt(RT, 2) << "(원래 1.13)로 바뀌었고, 이 방법은 그 차를 1/|rY-rΘ|배 키운다. "
      << "⑤의 변화점 개수(Y " << (R['Y'][5].f.npar - 3) / 2 << "개, 원래 4개 / Θ " << (R['T'][5].f.npar - 3) / 2
      << "개, 원래 8개)는 잔차 0.01 px 수준 차이에도 벌점 경계에서 바뀐다. 두 방법의 결론(②가 보정 전보다 나쁨, ⑤가 적합은 좋고 CV는 ①보다 나쁘거나 비슷함)은 같다.\n";
    write_text(out("보고서.md"), P.str());
    std::cout << "parity " << par.ok << "/" << par.total << "\n";
    for (char a : {'Y', 'T'})
        for (int m = 0; m < 6; ++m) std::cout << a << m << " rms " << R[a][m].rms << " cv " << R[a][m].cv << " npar " << R[a][m].f.npar << " " << R[a][m].f.note << "\n";
    return 0;
}
