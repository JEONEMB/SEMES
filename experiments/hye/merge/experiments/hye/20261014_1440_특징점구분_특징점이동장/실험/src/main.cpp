// 특징점 이동장으로 Y/Θ 구분, QR만 (원본: analysis_v6/common.py, t1_synthetic.py(특징점 고르기), t2_track_all.py, t6_F1.py, t7_F4.py, t8_eval.py)
// 1) 흐린 center 에서 Shi-Tomasi 모서리(17px 간격) -> 포화·라벨·케이블 제외, QR 세 영역 안 점만 쓴다
// 2) 점마다 81x81 큰 템플릿으로 정수 위치 -> 17x17 템플릿 매칭+포물선 -> 흐린 영상 17x17 가우스-뉴턴(3차 스플라인, 밝기 a,b)
// 3) 이론 한계: Y 1mm 무늬를 Θ 무늬 + 방해 무늬로 흉내 내고 남은 차이 n(σ/mm) -> 3σ 최소 |Y| = 3/n
// 4) 판정: 점 성분마다 1차/2차 매핑, Y 가설/Θ 가설 Huber GN 비용 비교 (v5 와 같은 맞추기·채점). 폴드·보정본 라벨은 v5 와 같다.
#include <hye/dataset.hpp>
#include <hye/ecc_compat.hpp>
#include <hye/plot.hpp>
#include <hye/report.hpp>
#include <omp.h>
#include <array>
#include <iostream>
#include <map>
#include "bspline.hpp"
#include "zip_mtime.hpp"

using namespace hye;

namespace {

const int HP = 8, NT = 2 * HP + 1, HPC = 40, SX = 110, SY = 16;
// center.bmp QR 중심 (원래 14번 CSV center 행 = 영역 정의 상수)
const double kQRc[3][2] = {{1642.4546, 1067.5000}, {1900.4866, 1068.2500}, {2159.2039, 1070.5367}};
const int PX0 = 1420, PX1 = 2392, PY0 = 948, PY1 = 1192;  // QR 판
const int kScrew[2][2] = {{1492, 1063}, {2312, 1073}};
const std::array<cv::Vec4i, 3> kBoxes{{{1554, 1734, 972, 1164}, {1815, 1995, 972, 1164}, {2073, 2253, 972, 1164}}};
const double THETA_TO_Y = 80.66 / 14.39;

// 영역: 0..2 QR좌중우, 3 판나사, 4 판, 5 판밖, 6 라벨, 7 케이블
int region_of(int x, int y) {
    if (1740 < x && x < 2060 && 1195 < y && y < 1360) return 6;
    if (1840 < x && x < 2300 && 0 < y && y < 590) return 7;
    if (PX0 <= x && x <= PX1 && PY0 <= y && y <= PY1) {
        for (auto& s : kScrew) if (std::hypot(x - s[0], y - s[1]) < 60) return 3;
        for (int i = 0; i < 3; ++i) if (std::abs(x - kQRc[i][0]) < 95 && std::abs(y - kQRc[i][1]) < 100) return i;
        return 4;
    }
    return 5;
}
const char* kReg[8] = {"QR좌", "QR중", "QR우", "판나사", "판", "판밖", "라벨", "케이블"};

double qr_mean_dx(const cv::Mat& center, const cv::Mat& im) {
    const auto crit = cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 150, 1e-7);
    double sx = 0;
    for (int q = 0; q < 3; ++q) {
        auto b = kBoxes[q];
        cv::Mat tpl = center(cv::Range(b[2], b[3]), cv::Range(b[0], b[1]));
        cv::Mat search = im(cv::Range(b[2] - 130, b[3] + 130), cv::Range(b[0] - 130, b[1] + 130));
        cv::Mat res;
        cv::matchTemplate(search, tpl, res, cv::TM_CCOEFF_NORMED);
        cv::Point loc;
        cv::minMaxLoc(res, nullptr, nullptr, nullptr, &loc);
        int dx0 = loc.x - 130, dy0 = loc.y - 130;
        cv::Mat target = im(cv::Range(b[2] + dy0, b[3] + dy0), cv::Range(b[0] + dx0, b[1] + dx0));
        cv::Mat W = cv::Mat::eye(2, 3, CV_32F);
        try { semes::ecc5::findTransformECCWithMask(tpl, target, cv::noArray(), cv::noArray(), W, cv::MOTION_AFFINE, crit, 3); } catch (const cv::Exception&) {}
        sx += dx0 + W.at<float>(0, 2) + (W.at<float>(0, 0) - 1) * 90 + W.at<float>(0, 1) * 96;
    }
    return sx / 3;
}

struct Pt { int x, y, region; };

double parab(const float* a, int n, int i, ptrdiff_t st) {
    if (0 < i && i < n - 1) {
        double l = a[(i - 1) * st], c = a[i * st], r = a[(i + 1) * st], d = l - 2 * c + r;
        return i + (d != 0 ? 0.5 * (l - r) / d : 0);
    }
    return i;
}

// 한 사진의 점 이동 (TM, GN). img = 원본 float, pad = 가장자리 복제 40px, bimg = 흐림 sigma 1
void track(const cv::Mat& img, const cv::Mat& pad, const cv::Mat& bimg, const std::vector<Pt>& F, const std::vector<cv::Mat>& BIG, const std::vector<cv::Mat>& Tr,
           const std::vector<cv::Mat>& Tb, std::vector<cv::Point2d>& gn) {
    const int n = (int)F.size();
    gn.resize(n);
    for (int k = 0; k < n; ++k) {
        // 큰 템플릿으로 정수 이동
        int x = F[k].x + HPC, y = F[k].y + HPC;
        int y0 = std::max(y - HPC - SY, 0), x0 = std::max(x - HPC - SX, 0);
        cv::Mat s = pad(cv::Range(y0, std::min(y + HPC + SY + 1, pad.rows)), cv::Range(x0, std::min(x + HPC + SX + 1, pad.cols)));
        cv::Mat r;
        cv::matchTemplate(s, BIG[k], r, cv::TM_CCOEFF_NORMED);
        cv::Point bl;
        cv::minMaxLoc(r, nullptr, nullptr, nullptr, &bl);
        double cx0 = bl.x + x0 + HPC - x, cy0 = bl.y + y0 + HPC - y;
        // 17x17 템플릿 + 포물선
        int ox = (int)std::nearbyint(cx0), oy = (int)std::nearbyint(cy0);
        int xs = F[k].x + ox - HP - 4, ys = F[k].y + oy - HP - 4;
        cv::Mat s2 = img(cv::Rect(xs, ys, NT + 8, NT + 8));
        cv::matchTemplate(s2, Tr[k], r, cv::TM_CCOEFF_NORMED);
        cv::minMaxLoc(r, nullptr, nullptr, nullptr, &bl);
        double tmx = parab(r.ptr<float>(bl.y), r.cols, bl.x, 1) - 4 + ox;
        double tmy = parab(r.ptr<float>(0) + bl.x, r.rows, bl.y, r.step1()) - 4 + oy;
        // GN: 창 29x29 (np.roll 기울기, 창 단위 스플라인)
        int cx = F[k].x + (int)std::nearbyint(tmx), cy = F[k].y + (int)std::nearbyint(tmy);
        const int L = HP + 3 + 3, WN = 2 * L + 1;
        cv::Mat win;
        bimg(cv::Rect(cx - L, cy - L, WN, WN)).convertTo(win, CV_64F);
        cv::Mat gx(WN, WN, CV_64F), gy(WN, WN, CV_64F);
        for (int yy = 0; yy < WN; ++yy)
            for (int xx = 0; xx < WN; ++xx) {
                gx.at<double>(yy, xx) = (win.at<double>(yy, (xx + 1) % WN) - win.at<double>(yy, (xx - 1 + WN) % WN)) / 2;
                gy.at<double>(yy, xx) = (win.at<double>((yy + 1) % WN, xx) - win.at<double>((yy - 1 + WN) % WN, xx)) / 2;
            }
        cv::Mat cI = bsp::prefilter(win), cX = bsp::prefilter(gx), cY = bsp::prefilter(gy);
        double dx = tmx - (cx - F[k].x), dy = tmy - (cy - F[k].y), a = 1, b = 0;
        for (int it = 0; it < 15; ++it) {
            double A[4][4] = {{0}}, g[4] = {0};
            for (int ii = -HP; ii <= HP; ++ii)
                for (int jj = -HP; jj <= HP; ++jj) {
                    bsp::Tap tp(jj + L + dx, ii + L + dy, WN, WN);
                    double I = tp.at(cI), Ix = tp.at(cX), Iy = tp.at(cY);
                    double rr = a * I + b - Tb[k].at<float>(ii + HP, jj + HP);
                    double J[4] = {a * Ix, a * Iy, I, 1};
                    for (int u = 0; u < 4; ++u) { g[u] += J[u] * rr; for (int v = 0; v < 4; ++v) A[u][v] += J[u] * J[v]; }
                }
            cv::Mat Am(4, 4, CV_64F), gm(4, 1, CV_64F), sol;
            for (int u = 0; u < 4; ++u) { gm.at<double>(u) = g[u]; for (int v = 0; v < 4; ++v) Am.at<double>(u, v) = A[u][v] + (u == v ? 1e-9 : 0); }
            cv::solve(Am, gm, sol, cv::DECOMP_LU);
            double d0 = -sol.at<double>(0), d1 = -sol.at<double>(1);
            dx += std::min(0.5, std::max(-0.5, d0));
            dy += std::min(0.5, std::max(-0.5, d1));
            a += -sol.at<double>(2);
            b += -sol.at<double>(3);
            if (std::max(std::abs(d0), std::abs(d1)) < 1e-5) break;
        }
        gn[k] = {dx + (cx - F[k].x), dy + (cy - F[k].y)};
    }
}

cv::Mat lsq(const cv::Mat& A, const cv::Mat& B) { cv::Mat X; cv::solve(A, B, X, cv::DECOMP_SVD); return X; }

// 방해 무늬 (2n x m): kind 1 = 세로 평행 이동, 화면 회전
cv::Mat nuis(const std::vector<double>& x, const std::vector<double>& y, int kind) {
    const int n = (int)x.size();
    double mx = mean(x), my = mean(y);
    cv::Mat U = cv::Mat::zeros(2 * n, kind == 0 ? 0 : kind == 1 ? 2 : 3, CV_64F);
    for (int i = 0; i < n && kind > 0; ++i) {
        double xc = (x[i] - mx) / 100, yc = (y[i] - my) / 100;
        U.at<double>(2 * i + 1, 0) = 1;
        U.at<double>(2 * i, 1) = -yc; U.at<double>(2 * i + 1, 1) = xc;
        if (kind >= 2) { U.at<double>(2 * i, 2) = xc; U.at<double>(2 * i + 1, 2) = yc; }
    }
    return U;
}

struct Map { cv::Mat s, c, b; std::vector<double> var; };
Map fit_map(const cv::Mat& D, const std::vector<double>& v, bool quad) {
    const int n = D.rows, p = quad ? 3 : 2;
    cv::Mat A(n, p, CV_64F);
    for (int i = 0; i < n; ++i) {
        A.at<double>(i, 0) = v[i];
        if (quad) { A.at<double>(i, 1) = v[i] * v[i]; A.at<double>(i, 2) = 1; } else A.at<double>(i, 1) = 1;
    }
    cv::Mat co = lsq(A, D), res = D - A * co;
    Map m{co.row(0).clone(), quad ? co.row(1).clone() : cv::Mat::zeros(1, D.cols, CV_64F), co.row(p - 1).clone(), std::vector<double>(D.cols)};
    for (int j = 0; j < D.cols; ++j) {
        double mu = 0, s2 = 0;
        for (int i = 0; i < n; ++i) mu += res.at<double>(i, j);
        mu /= n;
        for (int i = 0; i < n; ++i) { double e = res.at<double>(i, j) - mu; s2 += e * e; }
        m.var[j] = s2 / n * n / std::max(n - p, 1);
    }
    return m;
}

struct Sol { double v, cost; std::vector<double> resid; };
Sol solve(const double* d, const Map& m, const std::vector<double>& w, const cv::Mat& U, double huber = 3.0, int iters = 6) {
    const int n = U.rows, q = U.cols;
    std::vector<double> ww = w;
    double v = 0;
    auto f = [&](int k) { return m.s.at<double>(k) * v + m.c.at<double>(k) * v * v + m.b.at<double>(k); };
    for (int it = 0; it < iters; ++it) {
        cv::Mat A(n, 1 + q, CV_64F), z(n, 1, CV_64F);
        for (int k = 0; k < n; ++k) {
            double sw = std::sqrt(ww[k]);
            A.at<double>(k, 0) = (m.s.at<double>(k) + 2 * m.c.at<double>(k) * v) * sw;
            for (int j = 0; j < q; ++j) A.at<double>(k, 1 + j) = U.at<double>(k, j) * sw;
            z.at<double>(k) = (d[k] - f(k)) * sw;
        }
        cv::Mat p = lsq(A, z);
        v += p.at<double>(0);
        for (int k = 0; k < n; ++k) {
            double e = d[k] - f(k);
            for (int j = 0; j < q; ++j) e -= U.at<double>(k, j) * p.at<double>(1 + j);
            double r = e * std::sqrt(w[k]);
            ww[k] = w[k] * (std::abs(r) <= huber ? 1.0 : huber / std::max(std::abs(r), 1e-12));
        }
    }
    cv::Mat A(n, q, CV_64F), z(n, 1, CV_64F);
    std::vector<double> zz(n);
    for (int k = 0; k < n; ++k) {
        double sw = std::sqrt(ww[k]);
        zz[k] = d[k] - f(k);
        for (int j = 0; j < q; ++j) A.at<double>(k, j) = U.at<double>(k, j) * sw;
        z.at<double>(k) = zz[k] * sw;
    }
    cv::Mat u = lsq(A, z);
    Sol o{v, 0, std::vector<double>(n)};
    for (int k = 0; k < n; ++k) {
        double e = zz[k];
        for (int j = 0; j < q; ++j) e -= U.at<double>(k, j) * u.at<double>(j);
        o.resid[k] = e;
        double rr = e * std::sqrt(w[k]);
        o.cost += std::abs(rr) <= huber ? rr * rr : 2 * huber * std::abs(rr) - huber * huber;
    }
    return o;
}

double polyslope(const std::vector<double>& x, const std::vector<double>& y) {
    double mx = mean(x), my = mean(y), sxy = 0, sxx = 0;
    for (size_t i = 0; i < x.size(); ++i) { sxy += (x[i] - mx) * (y[i] - my); sxx += (x[i] - mx) * (x[i] - mx); }
    return sxy / sxx;
}

}  // namespace

static int run();
int main() {
    try { return run(); } catch (const std::exception& e) { std::cerr << "error: " << e.what() << "\n"; return 1; }
}

static int run() {
    omp_set_num_threads(4);
    begin_results();
    auto shots = load_1014();
    const int NS = (int)shots.size();
    int ic = -1;
    for (int i = 0; i < NS; ++i) if (shots[i].condition == "center") ic = i;
    auto times = zipm::capture_times(fs::u8path(repo_dir() + "/data/raw/zip_20261014.zip"), "Set.zip");
    std::vector<double> tmin(NS);
    double t0 = 1e300;
    for (int i = 0; i < NS; ++i) {
        auto it = times.find("SSAFY Test Image Set/" + shots[i].rel);
        if (it == times.end()) throw std::runtime_error("촬영 시각 없음: " + shots[i].rel);
        tmin[i] = it->second;
        t0 = std::min(t0, tmin[i]);
    }
    for (auto& t : tmin) t = (t - t0) / 60;
    std::vector<char> AX(NS);
    std::vector<double> VAL(NS);
    std::vector<std::string> NAME(NS);
    for (int i = 0; i < NS; ++i) {
        AX[i] = shots[i].condition == "center" ? 'C' : shots[i].condition == "Y" ? 'Y' : 'T';
        VAL[i] = AX[i] == 'Y' ? shots[i].y : AX[i] == 'T' ? shots[i].t : 0;
        NAME[i] = shots[i].rel.substr(shots[i].rel.find('/') + 1);
    }
    std::vector<int> FOLD(NS, -1);
    std::map<char, std::vector<int>> order;
    for (char a : {'Y', 'T'}) {
        for (int i = 0; i < NS; ++i) if (AX[i] == a) order[a].push_back(i);
        std::stable_sort(order[a].begin(), order[a].end(), [&](int p, int q) { return tmin[p] < tmin[q]; });
        for (size_t k = 0; k < order[a].size(); ++k) FOLD[order[a][k]] = (int)(k % 2);
    }

    // ---------- 특징점 고르기 (흐린 center) ----------
    const cv::Mat CEN8 = read_gray(shots[ic]);
    cv::Mat CENF, CENB;
    CEN8.convertTo(CENF, CV_32F);
    cv::GaussianBlur(CENF, CENB, cv::Size(0, 0), 1.0);
    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(CENB, corners, 3000, 0.02, NT, cv::noArray(), 7);
    std::vector<Pt> ALL;
    std::array<int, 8> nreg{};
    for (auto& c : corners) {
        int xi = (int)std::nearbyint(c.x), yi = (int)std::nearbyint(c.y);
        if (!(160 < xi && xi < 3840 - 160 && 60 < yi && yi < 2160 - 60)) continue;
        bool sat = false;
        for (int y = yi - HP; y <= yi + HP && !sat; ++y)
            for (int x = xi - HP; x <= xi + HP; ++x) if (CENF.at<float>(y, x) >= 250) { sat = true; break; }
        int r = region_of(xi, yi);
        if (sat || r >= 6) continue;
        ALL.push_back({xi, yi, r});
        nreg[r]++;
    }
    std::vector<Pt> F;
    for (auto& p : ALL) if (p.region < 3) F.push_back(p);
    const int N = (int)F.size();
    std::cout << "features " << ALL.size() << " QR " << N << "\n";
    cv::Mat CENP;
    cv::copyMakeBorder(CENF, CENP, HPC, HPC, HPC, HPC, cv::BORDER_REPLICATE);
    std::vector<cv::Mat> BIG, Tr, Tb;
    for (auto& p : F) {
        BIG.push_back(CENP(cv::Rect(p.x + HPC - HPC, p.y + HPC - HPC, 2 * HPC + 1, 2 * HPC + 1)).clone());
        Tr.push_back(CENF(cv::Rect(p.x - HP, p.y - HP, NT, NT)).clone());
        Tb.push_back(CENB(cv::Rect(p.x - HP, p.y - HP, NT, NT)).clone());
    }

    // ---------- 사진마다: QR 평균 이동(보정본 라벨용), 점 추적 ----------
    std::vector<double> qx(NS);
    cv::Mat D(NS, 2 * N, CV_64F);
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < NS; ++i) {
        cv::Mat im8 = read_gray(shots[i]), img, pad, bimg;
        im8.convertTo(img, CV_32F);
        qx[i] = qr_mean_dx(CEN8, im8);
        cv::copyMakeBorder(img, pad, HPC, HPC, HPC, HPC, cv::BORDER_REPLICATE);
        cv::GaussianBlur(img, bimg, cv::Size(0, 0), 1.0);
        std::vector<cv::Point2d> gn;
        track(img, pad, bimg, F, BIG, Tr, Tb, gn);
        for (int k = 0; k < N; ++k) { D.at<double>(i, 2 * k) = gn[k].x; D.at<double>(i, 2 * k + 1) = gn[k].y; }
    }
    std::cout << "tracked\n";

    // ---------- 보정본 라벨 (v5 와 같음: ① 백래시 + 드리프트, 폴드마다 학습 사진으로만) ----------
    std::map<char, std::vector<int>> dir;
    for (char a : {'Y', 'T'}) {
        auto& o = order[a];
        std::vector<int> seg(o.size(), 0);
        dir[a].assign(o.size(), 0);
        auto sg = [](double v) { return (v > 0) - (v < 0); };
        for (size_t k = 1; k < o.size(); ++k) seg[k] = seg[k - 1] + ((tmin[o[k]] - tmin[o[k - 1]] > 3 || sg(VAL[o[k]]) != sg(VAL[o[k - 1]])) ? 1 : 0);
        for (int s = 0; s <= seg.back(); ++s) {
            std::vector<double> t, v;
            for (size_t k = 0; k < o.size(); ++k) if (seg[k] == s) { t.push_back(tmin[o[k]]); v.push_back(VAL[o[k]]); }
            double sl = polyslope(t, v);
            for (size_t k = 0; k < o.size(); ++k) if (seg[k] == s) dir[a][k] = (sl > 0) - (sl < 0);
        }
    }
    std::array<std::vector<double>, 2> CORR;
    for (int k = 0; k < 2; ++k) {
        CORR[k] = VAL;
        for (char a : {'Y', 'T'}) {
            auto& o = order[a];
            const int n = (int)o.size();
            cv::Mat A(n, 5, CV_64F);
            std::vector<int> tr;
            for (int i = 0; i < n; ++i) {
                double t = tmin[o[i]];
                A.at<double>(i, 0) = VAL[o[i]]; A.at<double>(i, 1) = 1; A.at<double>(i, 2) = dir[a][i]; A.at<double>(i, 3) = t; A.at<double>(i, 4) = 1 - std::exp(-t / 3);
                if (FOLD[o[i]] != k) tr.push_back(i);
            }
            cv::Mat At((int)tr.size(), 5, CV_64F), xt((int)tr.size(), 1, CV_64F);
            for (size_t r = 0; r < tr.size(); ++r) { A.row(tr[r]).copyTo(At.row((int)r)); xt.at<double>((int)r) = qx[o[tr[r]]]; }
            cv::Mat p = lsq(At, xt), off = A.colRange(1, 5) * p.rowRange(1, 5);
            for (int i = 0; i < n; ++i) CORR[k][o[i]] = VAL[o[i]] + off.at<double>(i) / p.at<double>(0);
        }
    }

    std::vector<double> FX, FY;
    for (auto& p : F) { FX.push_back(p.x); FY.push_back(p.y); }

    // ---------- 이론 한계 (t6_F1): 라벨 = 보정본(폴드0 학습), center 제외 ----------
    std::map<std::string, double> theory;
    std::vector<double> sYx(N), sTx(N);
    {
        std::map<char, cv::Mat> S;
        cv::Mat R(NS, 2 * N, CV_64F, cv::Scalar(kNaN));
        for (char a : {'Y', 'T'}) {
            std::vector<int> idx;
            for (int i = 0; i < NS; ++i) if (AX[i] == a) idx.push_back(i);
            cv::Mat A((int)idx.size(), 3, CV_64F), Dm((int)idx.size(), 2 * N, CV_64F);
            for (size_t r = 0; r < idx.size(); ++r) {
                double l = CORR[0][idx[r]];
                A.at<double>((int)r, 0) = 1; A.at<double>((int)r, 1) = l; A.at<double>((int)r, 2) = l * l;
                D.row(idx[r]).copyTo(Dm.row((int)r));
            }
            cv::Mat co = lsq(A, Dm), res = Dm - A * co;
            S[a] = co.row(1).clone();
            for (size_t r = 0; r < idx.size(); ++r) res.row((int)r).copyTo(R.row(idx[r]));
        }
        for (int k = 0; k < N; ++k) { sYx[k] = S['Y'].at<double>(2 * k); sTx[k] = S['T'].at<double>(2 * k); }
        std::vector<int> ok;
        for (int i = 0; i < NS; ++i) if (AX[i] != 'C') ok.push_back(i);
        std::vector<double> var1(2 * N), var2(2 * N);
        cv::Mat Rs((int)ok.size(), 2 * N, CV_64F);
        for (size_t r = 0; r < ok.size(); ++r) {
            // 점마다 독립: 사진마다 성분별 중앙값(모든 점이 판 위 한 묶음)을 뺀다
            for (int c = 0; c < 2; ++c) {
                std::vector<double> v;
                for (int k = 0; k < N; ++k) v.push_back(R.at<double>(ok[r], 2 * k + c));
                std::sort(v.begin(), v.end());
                double med = N % 2 ? v[N / 2] : 0.5 * (v[N / 2 - 1] + v[N / 2]);
                for (int k = 0; k < N; ++k) Rs.at<double>((int)r, 2 * k + c) = R.at<double>(ok[r], 2 * k + c) - med;
            }
        }
        for (int j = 0; j < 2 * N; ++j) {
            double m1 = 0, m2 = 0;
            for (size_t r = 0; r < ok.size(); ++r) { m1 += R.at<double>(ok[r], j); m2 += Rs.at<double>((int)r, j); }
            m1 /= ok.size(); m2 /= ok.size();
            double v1 = 0, v2 = 0;
            for (size_t r = 0; r < ok.size(); ++r) { v1 += std::pow(R.at<double>(ok[r], j) - m1, 2); v2 += std::pow(Rs.at<double>((int)r, j) - m2, 2); }
            var1[j] = v1 / ok.size(); var2[j] = v2 / ok.size();
        }
        for (int kind = 0; kind < 3; ++kind)
            for (int vv = 0; vv < 2; ++vv) {
                const auto& var = vv ? var2 : var1;
                cv::Mat U = nuis(FX, FY, kind), A(2 * N, 1 + U.cols, CV_64F), y(2 * N, 1, CV_64F);
                for (int j = 0; j < 2 * N; ++j) {
                    double sw = std::sqrt(1 / std::max(var[j], 0.003 * 0.003));
                    A.at<double>(j, 0) = S['T'].at<double>(j) * sw;
                    for (int c = 0; c < U.cols; ++c) A.at<double>(j, 1 + c) = U.at<double>(j, c) * sw;
                    y.at<double>(j) = S['Y'].at<double>(j) * sw;
                }
                cv::Mat p = lsq(A, y), r = y - A * p;
                double n = std::sqrt(r.dot(r));
                std::string key = std::string(kind == 0 ? "없음" : kind == 1 ? "v5" : "v5+크기") + "/" + (vv ? "점마다 독립" : "맞추기 잔차(v5식)");
                theory[key + "/n"] = n;
                theory[key + "/a"] = p.at<double>(0);
                theory[key + "/minY"] = 3 / n;
            }
    }

    // ---------- 판정 (t7_F4) ----------
    cv::Mat U = nuis(FX, FY, 1);
    struct Rec { int i; char axis; double cmd, act; char pa; double pred, ratio; std::vector<double> rtrue; };
    struct Cond { std::string key, label, map, cv; std::vector<Rec> recs; };
    std::vector<Cond> conds;
    auto session = [&](int i) {
        if (AX[i] == 'T') return std::string(VAL[i] > 0 ? "Θ+" : "Θ-");
        bool odd = std::abs((int)std::lround(VAL[i] / 0.05)) % 2 == 1;
        return std::string(odd ? "Y2" : "Y1") + (VAL[i] > 0 ? "+" : "-");
    };
    for (std::string label : {"raw", "보정본"})
        for (bool quad : {false, true})
            for (std::string cvn : {"짝홀", "구간빼기"}) {
                if (cvn == "구간빼기" && quad) continue;
                Cond C{"17px / QR만 / " + label + " / " + (quad ? "2차" : "1차") + " / " + cvn, label, quad ? "2차" : "1차", cvn};
                std::vector<std::pair<int, std::vector<bool>>> splits;
                if (cvn == "짝홀")
                    for (int k = 0; k < 2; ++k) { std::vector<bool> te(NS); for (int i = 0; i < NS; ++i) te[i] = FOLD[i] == k; splits.push_back({k, te}); }
                else
                    for (std::string s : {"Y1+", "Y1-", "Y2+", "Y2-", "Θ+", "Θ-"}) { std::vector<bool> te(NS); for (int i = 0; i < NS; ++i) te[i] = AX[i] != 'C' && session(i) == s; splits.push_back({0, te}); }
                for (auto& [k, te] : splits) {
                    const auto& lab = label == "raw" ? VAL : CORR[k];
                    std::map<char, Map> maps;
                    for (char a : {'Y', 'T'}) {
                        std::vector<int> idx;
                        for (int i = 0; i < NS; ++i) if (!te[i] && (AX[i] == a || AX[i] == 'C')) idx.push_back(i);
                        cv::Mat Dm((int)idx.size(), 2 * N, CV_64F);
                        std::vector<double> v;
                        for (size_t r = 0; r < idx.size(); ++r) { D.row(idx[r]).copyTo(Dm.row((int)r)); v.push_back(lab[idx[r]]); }
                        maps[a] = fit_map(Dm, v, quad);
                    }
                    std::vector<double> w(2 * N);
                    for (int j = 0; j < 2 * N; ++j) w[j] = 1 / std::max((maps['Y'].var[j] + maps['T'].var[j]) / 2, 0.003 * 0.003);
                    std::vector<int> test;
                    for (int i = 0; i < NS; ++i) if (te[i]) test.push_back(i);
                    std::vector<Rec> rk(test.size());
#pragma omp parallel for schedule(dynamic)
                    for (int t = 0; t < (int)test.size(); ++t) {
                        int i = test[t];
                        Sol sy = solve(D.ptr<double>(i), maps['Y'], w, U), st = solve(D.ptr<double>(i), maps['T'], w, U);
                        char pa = sy.cost <= st.cost ? 'Y' : 'T';
                        rk[t] = {i, AX[i], VAL[i], CORR[k][i], pa, pa == 'Y' ? sy.v : st.v, sy.cost / st.cost, AX[i] == 'Y' ? sy.resid : st.resid};
                    }
                    C.recs.insert(C.recs.end(), rk.begin(), rk.end());
                }
                conds.push_back(std::move(C));
            }
    // ---------- 채점 (t8_eval) ----------
    struct Ev { double acc, accY, accT; int wy, wt; double yMean, yRms, yMax, yFull, tMean, tRms, tMax, tFull, pass; };
    auto evaluate = [&](const Cond& C, bool act) {
        Ev e{};
        int n = 0, ny = 0, nt = 0, oky = 0, okt = 0, pass = 0;
        std::vector<double> ey, et, fy, ft;
        for (auto& r : C.recs) {
            double t = act ? r.act : r.cmd, er = std::abs(r.pred - t);
            bool ok = r.pa == r.axis;
            ++n;
            if (r.axis == 'Y') { ++ny; oky += ok; if (ok) ey.push_back(er); fy.push_back(ok ? er : std::abs(t)); e.wy += !ok; }
            else { ++nt; okt += ok; if (ok) et.push_back(er); ft.push_back(ok ? er : std::abs(t)); e.wt += !ok; }
            pass += ok ? er <= 0.09 : (std::abs(t) <= 0.09 && std::abs(r.pred) <= 0.09);
        }
        e.acc = 100.0 * (oky + okt) / n; e.accY = 100.0 * oky / ny; e.accT = 100.0 * okt / nt;
        e.yMean = mean(ey) * 1000; e.yRms = rms(ey) * 1000; e.yMax = max_abs(ey) * 1000; e.yFull = mean(fy) * 1000;
        e.tMean = mean(et) * 1000; e.tRms = rms(et) * 1000; e.tMax = max_abs(et) * 1000; e.tFull = mean(ft) * 1000;
        e.pass = 100.0 * pass / n;
        return e;
    };
    Table tab{{"조건", "정답기준", "사진수", "축정확도", "축정확도_Y", "축정확도_Θ", "축틀림_Y", "축틀림_Θ", "Y_평균오차", "Y_RMS", "Y_최대", "Y_평균오차_전체", "Θ_평균오차", "Θ_RMS",
               "Θ_최대", "Θ_평균오차_전체", "통과09"}};
    std::map<std::string, Ev> EV;
    for (auto& C : conds)
        for (bool act : {false, true}) {
            Ev e = evaluate(C, act);
            EV[C.key + (act ? "|act" : "|cmd")] = e;
            tab.add({C.key, act ? "실제 위치 (보정함)" : "명령값 (보정 안 함)", std::to_string(C.recs.size()), fmt(e.acc, 3), fmt(e.accY, 3), fmt(e.accT, 3), std::to_string(e.wy),
                     std::to_string(e.wt), fmt(e.yMean, 3), fmt(e.yRms, 3), fmt(e.yMax, 3), fmt(e.yFull, 3), fmt(e.tMean, 3), fmt(e.tRms, 3), fmt(e.tMax, 3), fmt(e.tFull, 3), fmt(e.pass, 3)});
        }
    tab.csv(out("평가표_특징점_QR만.csv"));
    const std::string kRaw = "17px / QR만 / raw / 2차 / 짝홀", kCor = "17px / QR만 / 보정본 / 2차 / 짝홀";
    const Cond* MC = nullptr;
    for (auto& C : conds) if (C.key == kCor) MC = &C;
    {
        Table pr{{"file", "axis", "cmd", "act", "pred_axis", "pred", "cost_ratio"}};
        for (auto& r : MC->recs) pr.add({shots[r.i].rel, std::string(1, r.axis), fmt(r.cmd, 2), fmt(r.act, 4), std::string(1, r.pa), fmt(r.pred, 4), fmt(r.ratio, 4)});
        pr.csv(out("사진별_예측_대표설정.csv"));
    }
    Table th{{"방해 무늬", "잡음", "n (σ/mm)", "Θ 흉내 계수", "3σ 최소 |Y| (mm)"}};
    for (std::string k1 : {"없음", "v5", "v5+크기"})
        for (std::string k2 : {"맞추기 잔차(v5식)", "점마다 독립"}) {
            std::string key = k1 + "/" + k2;
            th.add({k1, k2, fmt(theory[key + "/n"], 3), fmt(theory[key + "/a"], 4), fmt(theory[key + "/minY"], 4)});
        }

    // ---------- 그림 ----------
    {   // 01 특징점 지도
        cv::Mat im = read_color(shots[ic]);
        std::vector<Mark> mk;
        for (auto& p : ALL) {
            if (p.region > 4) continue;
            Mark m;
            m.p = {(double)p.x, (double)p.y};
            m.c = p.region < 3 ? kRed : kGray;
            mk.push_back(m);
        }
        save_png("01_특징점지도_QR.png", zoom(im, cv::Rect(PX0 - 10, PY0 - 10, PX1 - PX0 + 20, PY1 - PY0 + 20), 1600, mk, 0,
                                              "Shi-Tomasi 17px: red = QR points used (" + std::to_string(N) + "), gray = other plate points"));
    }
    {   // 02 대표 설정 예측 오차 (축 틀리면 그 축 0)
        std::vector<double> ty, py, tt, pt;
        std::vector<std::string> nm;
        for (auto& r : MC->recs) {
            ty.push_back(r.axis == 'Y' ? r.act : 0); tt.push_back(r.axis == 'T' ? r.act : 0);
            py.push_back(r.pa == 'Y' ? r.pred : 0); pt.push_back(r.pa == 'T' ? r.pred : 0);
            nm.push_back(NAME[r.i].substr(0, NAME[r.i].rfind('.')));
        }
        pred_vs_truth("02_대표설정_예측오차.png", "features QR corr quad", ty, py, tt, pt, nm);
    }
    {   // 03 조건별 축 정확도
        std::vector<std::string> cats;
        std::vector<double> v;
        for (auto& C : conds)
            for (bool act : {false, true}) {
                cats.push_back(std::string(C.label == "raw" ? "raw" : "corr") + (C.map == "2차" ? " quad" : " lin") + (C.cv == "짝홀" ? " even/odd" : " leave-session") + (act ? " act" : " cmd"));
                v.push_back(EV[C.key + (act ? "|act" : "|cmd")].acc);
            }
        bar_chart("03_조건별_축정확도.png", "Axis accuracy (%), QR points only", cats, v, "axis accuracy %", -1);
    }
    {   // 04 점별 Y/Θ 기울기 비
        std::vector<double> x, y;
        for (int k = 0; k < N; ++k) { x.push_back(F[k].x); y.push_back(sYx[k] / sTx[k]); }
        Chart c("Per-point slope ratio sY/sT (x component)", "image x (px)", "sY / sT");
        c.scatter(x, y, color(0));
        c.save("04_점별_기울기비.png");
    }
    // 튀는 값: Y 는 축 틀린 Y 사진 중 비용 비 최대, Θ 는 축 틀린 Θ 중 비용 비 최소 (없으면 오차 최대)
    auto pick = [&](char a) {
        int best = -1;
        for (size_t k = 0; k < MC->recs.size(); ++k) {
            auto& r = MC->recs[k];
            if (r.axis != a || r.pa == a) continue;
            if (best < 0 || (a == 'Y' ? r.ratio > MC->recs[best].ratio : r.ratio < MC->recs[best].ratio)) best = (int)k;
        }
        if (best >= 0) return best;
        for (size_t k = 0; k < MC->recs.size(); ++k) {
            auto& r = MC->recs[k];
            if (r.axis == a && (best < 0 || std::abs(r.pred - r.act) > std::abs(MC->recs[best].pred - MC->recs[best].act))) best = (int)k;
        }
        return best;
    };
    std::map<char, int> sel{{'Y', pick('Y')}, {'T', pick('T')}};
    std::map<char, std::string> ofile;
    for (char a : {'Y', 'T'}) {
        auto& r = MC->recs[sel[a]];
        cv::Mat im = read_color(shots[r.i]);
        std::vector<Mark> mk;
        double sx = 0;
        for (int k = 0; k < N; ++k) sx += D.at<double>(r.i, 2 * k);
        sx /= N;
        for (int k = 0; k < N; ++k) {
            Mark m;
            m.p = {F[k].x + D.at<double>(r.i, 2 * k), F[k].y + D.at<double>(r.i, 2 * k + 1)};
            m.d = {r.rtrue[2 * k], r.rtrue[2 * k + 1]};
            mk.push_back(m);
        }
        std::string nm = NAME[r.i].substr(0, NAME[r.i].rfind('.'));
        std::string cap = nm + ": true " + (a == 'Y' ? "Y " : "T ") + fmt(r.act, 3) + ", judged " + (r.pa == 'Y' ? "Y " : "T ") + fmt(r.pred, 3) + ", cost Y/T " + fmt(r.ratio, 3) +
                          " (arrows: residual of true-axis model x300)";
        ofile[a] = std::string(a == 'Y' ? "05_튀는값_Y_" : "06_튀는값_T_") + nm + ".png";
        save_png(ofile[a], zoom(im, cv::Rect((int)(1540 + sx), 960, 720, 220), 1500, mk, 300, cap));
    }

    // ---------- 원래 결과와 대조 ----------
    // 원래 값: analysis_v6 결과 42_평가표_특징점vs패치.csv(결과/원본자료/), t6_F1.csv, features_hp8.csv
    struct O { std::string key; bool act; double acc; int wy, wt; double yMean, pass; };
    const std::vector<O> orig{{"17px / QR만 / raw / 1차 / 짝홀", false, 96.071, 3, 8, 12.094, 96.786},     {"17px / QR만 / raw / 1차 / 짝홀", true, 96.071, 3, 8, 22.302, 96.786},
                              {"17px / QR만 / raw / 1차 / 구간빼기", false, 96.429, 8, 2, 21.683, 97.143}, {"17px / QR만 / raw / 1차 / 구간빼기", true, 96.429, 8, 2, 24.202, 97.5},
                              {"17px / QR만 / raw / 2차 / 짝홀", false, 93.929, 12, 5, 11.885, 94.643},    {"17px / QR만 / raw / 2차 / 짝홀", true, 93.929, 12, 5, 22.336, 95.0},
                              {"17px / QR만 / 보정본 / 1차 / 짝홀", false, 92.143, 0, 22, 26.145, 93.214}, {"17px / QR만 / 보정본 / 1차 / 짝홀", true, 92.143, 0, 22, 1.192, 93.214},
                              {"17px / QR만 / 보정본 / 1차 / 구간빼기", false, 93.571, 0, 18, 26.137, 94.286}, {"17px / QR만 / 보정본 / 1차 / 구간빼기", true, 93.571, 0, 18, 1.156, 94.286},
                              {"17px / QR만 / 보정본 / 2차 / 짝홀", false, 96.071, 8, 3, 26.481, 96.786}, {"17px / QR만 / 보정본 / 2차 / 짝홀", true, 96.071, 8, 3, 1.226, 96.786}};
    Parity par;
    par.add("특징점 수 (전체, 포화·라벨·케이블 제외)", 309, (double)ALL.size(), 0, 0);
    par.add("특징점 수 (QR만)", 177, N, 0, 0);
    par.add("이론 한계 최소 |Y| mm (v5 방해, 맞추기 잔차)", 0.2419, theory["v5/맞추기 잔차(v5식)/minY"], 0.02, 4);
    par.add("이론 한계 최소 |Y| mm (v5 방해, 점마다 독립)", 0.1403, theory["v5/점마다 독립/minY"], 0.015, 4);
    par.add("이론 한계 Θ 흉내 계수 (v5 방해)", 0.1783, theory["v5/맞추기 잔차(v5식)/a"], 0.002, 4);
    for (auto& o : orig) {
        auto& e = EV[o.key + (o.act ? "|act" : "|cmd")];
        std::string b = o.act ? " (실제 위치)" : " (명령값)";
        par.add(o.key + b + " 축 정확도 %", o.acc, e.acc, 0.72, 3);
        par.add(o.key + b + " 축 틀림 Y/Θ 합", o.wy + o.wt, e.wy + e.wt, 2, 0);
        par.add(o.key + b + " Y 평균 오차 µm", o.yMean, e.yMean, 0.1 * o.yMean, 3);
    }

    // ---------- 결과.md ----------
    Table reg{{"영역", "점 수"}};
    for (int r = 0; r < 6; ++r) reg.add({kReg[r], std::to_string(nreg[r])});
    std::ostringstream Rm;
    Rm << "# 결과: 특징점 이동장, QR만\n\n"
       << "입력: `data/clean/clean_20261014` 281장. 폴드·보정본 라벨은 v5(#13)와 같다: 축마다 촬영 순서(`data/raw` zip 수정 시각)로 짝/홀, 보정본 = 명령값 + o/k (① 백래시+드리프트, 학습 폴드로만, QR 평균 이동은 ECC).\n\n"
       << "특징점: 흐린 center(sigma 1)에서 goodFeaturesToTrack(최대 3000, 품질 0.02, 간격 17, 블록 7) -> 가장자리·포화(>=250)·라벨·케이블 제외 " << ALL.size() << "개, 그중 QR 세 영역(중심 ±95 x ±100 px) " << N << "개를 쓴다.\n\n"
       << reg.md() << "\n추적: 81x81 템플릿으로 가로 ±110, 세로 ±16 px 탐색 -> 17x17 템플릿 매칭(±4 px)+포물선 -> 흐린 사진 29x29 창에서 17x17 가우스-뉴턴(3차 스플라인, 밝기 a,b, 최대 15회).\n\n"
       << "## 이론 한계 (3σ로 구분되는 최소 |Y|, 라벨 = 보정본 폴드0)\n\n" << th.md()
       << "\n## 판정 (v5와 같은 맞추기·채점)\n\n축을 틀리면 0.09 통과는 참값과 예측이 둘 다 0.09 안일 때만. 오차는 축을 맞힌 사진만, `_전체`는 축을 틀린 사진을 정답 축 예측 0으로 넣은 값.\n\n" << tab.md()
       << "\n## 원래 결과와 대조 (" << par.ok << "/" << par.total << " 일치)\n\n허용오차: 축 정확도 ±0.72%(2장), 축 틀림 합 ±2장, Y 평균 오차 ±10%, 이론 한계 ±0.02 mm.\n\n" << par.t.md()
       << "\n## 옮기지 않은 부분\n\n- 판밖 점·QR+판밖 조건(원래 리포트 정정: 설치 환경마다 바뀌어 무효), 25px(hp12) 템플릿, v5 패치 행(#13에 있음).\n"
       << "- t1 합성 이동 시험, t3~t5 실제 정밀도 진단, t9~t10 방법 그림, t11~t14 (QR 9px 점, 상위 k개, Y->Θ 숨은 성분, 점 대 점).\n"
       << "\n## 파일\n\n- `평가표_특징점_QR만.csv`: 원래 42번 표와 같은 열 (QR만 6조건 x 정답 기준 2)\n- `사진별_예측_대표설정.csv`: " << kCor << "\n"
       << "- `이미지/01_특징점지도_QR.png`, `02_대표설정_예측오차.png`, `03_조건별_축정확도.png`, `04_점별_기울기비.png`, `" << ofile['Y'] << "`, `" << ofile['T'] << "`\n";
    write_text(out("결과.md"), Rm.str());

    // ---------- 보고서.md ----------
    auto& eR = EV[kRaw + "|cmd"];
    auto& eC = EV[kCor + "|act"];
    Table outl{{"사진", "정답 축", "정답 값 (실제 위치)", "판정 축", "추정 값", "비용 비 Y/Θ"}};
    double maxY = 0, maxT = 0;
    for (auto& r : MC->recs)
        if (r.pa != r.axis) {
            outl.add({NAME[r.i], r.axis == 'Y' ? "Y" : "Θ", fmt(r.act, 3), r.pa == 'Y' ? "Y" : "Θ", fmt(r.pred, 4), fmt(r.ratio, 3)});
            (r.axis == 'Y' ? maxY : maxT) = std::max(r.axis == 'Y' ? maxY : maxT, std::abs(r.cmd));
        }
    double rmin = 1e9, rmax = 0;
    for (auto& r : MC->recs) if (r.pa != r.axis) { double q = r.ratio > 1 ? r.ratio : 1 / r.ratio; rmin = std::min(rmin, q); rmax = std::max(rmax, q); }
    auto& rY = MC->recs[sel['Y']];
    auto& rT = MC->recs[sel['T']];
    std::ostringstream P;
    P << "# 보고서: 특징점 이동장으로 Y/Θ 구분, QR만 (10-14 14:40)\n\n"
      << "## 문제\n\n패치 대신 특징점 이동장으로 바꾸면 QR만으로 Y와 Θ를 구분할 수 있는지 확인하려 했다.\n\n"
      << "## 해결 방법\n\nQR 안 Shi-Tomasi 특징점 " << N << "개를 17px 템플릿 가우스-뉴턴으로 추적해, 점별 2차 이동 모델과 Y/Θ 가설 비용 비교로 판정하고 이론 한계를 계산했다.\n\n"
      << "## 결과\n\n판정 정확도는 명령값 학습·채점 " << fmt(eR.acc, 1) << "%, 보정본 학습·실제 위치 채점 " << fmt(eC.acc, 1) << "%였고, 이론 한계(3σ)는 QR만 " << fmt(theory["v5/맞추기 잔차(v5식)/minY"], 2)
      << " mm였다.\n\n| 지표 | 명령값 (raw 학습, 2차) | 실제 위치 (보정본 학습, 2차) |\n|---|---|---|\n"
      << "| 축 정확도 | " << fmt(eR.acc, 3) << "% | " << fmt(eC.acc, 3) << "% |\n"
      << "| 축 틀림 (Y / Θ 사진) | " << eR.wy << " / " << eR.wt << " | " << eC.wy << " / " << eC.wt << " |\n"
      << "| Y 평균 / 최대 오차 (µm, 축 맞힌 사진) | " << fmt(eR.yMean, 2) << " / " << fmt(eR.yMax, 2) << " | " << fmt(eC.yMean, 2) << " / " << fmt(eC.yMax, 2) << " |\n"
      << "| Θ 평균 / 최대 오차 (mΘ) | " << fmt(eR.tMean, 3) << " / " << fmt(eR.tMax, 3) << " | " << fmt(eC.tMean, 3) << " / " << fmt(eC.tMax, 3) << " |\n"
      << "| 0.09 통과 | " << fmt(eR.pass, 3) << "% | " << fmt(eC.pass, 3) << "% |\n"
      << "| 이론 한계 최소 \\|Y\\| (맞추기 잔차 / 점마다 독립) | " << fmt(theory["v5/맞추기 잔차(v5식)/minY"], 3) << " / " << fmt(theory["v5/점마다 독립/minY"], 3) << " mm | |\n\n"
      << img_md("02_대표설정_예측오차.png", "대표 설정 예측 오차") << "\n\n" << img_md("03_조건별_축정확도.png", "조건별 축 정확도") << "\n\n" << img_md("01_특징점지도_QR.png", "특징점 지도") << "\n\n"
      << img_md("04_점별_기울기비.png", "점별 기울기 비") << "\n\n"
      << "## 튀는 값\n\n대표 설정(" << kCor << ")에서 축을 틀린 사진은 Y " << eC.wy << "장(|Y| <= " << fmt(maxY, 2) << " mm), Θ " << eC.wt << "장(|Θ| <= " << fmt(maxT, 2)
      << ")으로 모두 0 근처였고, 두 가설의 비용 비가 " << fmt(rmin, 3) << "~" << fmt(rmax, 3) << "배로 거의 같았다.\n\n" << outl.md() << "\n"
      << img_md(ofile['Y'], "Y 튀는 값") << "\n\n" << img_md(ofile['T'], "T 튀는 값") << "\n\n"
      << "빨간 십자는 이 사진에서 잰 특징점 위치(center 위치 + 측정 이동), 화살표는 참 축 매핑으로 맞춘 뒤 남은 점 잔차 x300이다. Y 그림 " << NAME[rY.i] << "(비용 비 " << fmt(rY.ratio, 3)
      << "), Θ 그림 " << NAME[rT.i] << "(비용 비 " << fmt(rT.ratio, 3) << ").\n\n"
      << "## 한줄평\n\nQR 판 위 점은 깊이가 같아 Y/Θ 기울기 비가 점마다 거의 같고(표준편차 " << [&] { std::vector<double> v; for (int k = 0; k < N; ++k) v.push_back(sYx[k] / sTx[k]); double m = mean(v), s = 0; for (double x : v) s += (x - m) * (x - m); return fmt(std::sqrt(s / v.size()), 4); }()
      << ", 평균 " << [&] { std::vector<double> v; for (int k = 0; k < N; ++k) v.push_back(sYx[k] / sTx[k]); return fmt(mean(v), 4); }()
      << "), 남는 구분 단서가 Y 1mm당 " << fmt(theory["v5/맞추기 잔차(v5식)/n"], 1) << "σ뿐이라서, 3σ 한계 " << fmt(theory["v5/맞추기 잔차(v5식)/minY"], 2)
      << " mm보다 작은 Y에서는 두 가설 비용이 같아져 점을 패치에서 특징점으로 바꿔도 판정이 나아지지 않았다.\n\n"
      << "## 원래 결과와 대조\n\n원래 값은 analysis_v6 결과 `42_평가표_특징점vs패치.csv`(이 실험 `결과/원본자료/` 사본), `t6_F1.csv`, `features_hp8.csv`에서 가져왔다. "
      << "보정본 라벨의 QR 이동은 원래 14번 CSV 대신 ECC로 다시 쟀다(#12와 같은 측정). 허용오차: 축 정확도 ±0.72%, 축 틀림 합 ±2장, Y 평균 오차 ±10%, 이론 한계 ±0.02 mm. "
      << par.ok << "/" << par.total << " 일치.\n\n" << par.t.md();
    write_text(out("보고서.md"), P.str());
    std::cout << "parity " << par.ok << "/" << par.total << "\n";
    return 0;
}
