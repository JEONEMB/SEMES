// center + 현재 2장, QR 3개 아핀 18값 GLS 판정 (원본: analysis_v7/lib7.py, judge7.py, e4_track.py, e11_final.py; 특징점은 analysis_v6/t11_qr_features.py)
// 1) 흐린 center 의 QR 판 안 Shi-Tomasi 점 (간격 17) -> QR 점 + 판 나사 가장자리 점
// 2) 33px 템플릿, 흐림 sigma 2 가우스-뉴턴 추적 (81px 큰 템플릿 -> 템플릿 매칭 -> GN)
// 3) QR마다 점 이동을 1, x, y 로 맞춘 아핀 6값 -> QR만 18값 (+나사 평균 이동 4값 = 22값)
// 4) 축마다 f = s v + c v^2 (0점 = center), 전체 공분산 GLS 비용을 v 격자에서 최소화, 비용 작은 축
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

const int HPF = 8, HP = 16, NT = 2 * HP + 1, HPC = 40, SX = 110, SY = 16;
const double SIG = 2.0;
const double kQRc[3][2] = {{1642.4546, 1067.5000}, {1900.4866, 1068.2500}, {2159.2039, 1070.5367}};  // 원래 14번 CSV center 행 (영역 정의 상수)
const int PX0 = 1420, PX1 = 2392, PY0 = 948, PY1 = 1192;
const int kScrew[2][2] = {{1492, 1063}, {2312, 1073}};
const std::array<cv::Vec4i, 3> kBoxes{{{1554, 1734, 972, 1164}, {1815, 1995, 972, 1164}, {2073, 2253, 972, 1164}}};
const double THETA_TO_Y = 80.66 / 14.39;
const char* SES[6] = {"Y1+", "Y1-", "Y2+", "Y2-", "Θ+", "Θ-"};

// 0..2 QR좌중우, 3 판나사, 4 판, 5 판밖, 6 라벨, 7 케이블
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

struct Pt { int x, y, region; double sat; };

double parab(const float* a, int n, int i, ptrdiff_t st) {
    if (0 < i && i < n - 1) {
        double l = a[(i - 1) * st], c = a[i * st], r = a[(i + 1) * st], d = l - 2 * c + r;
        return i + (d != 0 ? 0.5 * (l - r) / d : 0);
    }
    return i;
}

// v6 common.py coarse_init -> track_tm -> track_gn (hp = 템플릿 반폭)
void track(const cv::Mat& img, const cv::Mat& pad, const cv::Mat& bimg, const std::vector<Pt>& F, const std::vector<cv::Mat>& BIG, const std::vector<cv::Mat>& Tr,
           const std::vector<cv::Mat>& Tb, int hp, std::vector<cv::Point2d>& gn) {
    const int n = (int)F.size(), nt = 2 * hp + 1;
    gn.resize(n);
    for (int k = 0; k < n; ++k) {
        int x = F[k].x + HPC, y = F[k].y + HPC;
        int y0 = std::max(y - HPC - SY, 0), x0 = std::max(x - HPC - SX, 0);
        cv::Mat s = pad(cv::Range(y0, std::min(y + HPC + SY + 1, pad.rows)), cv::Range(x0, std::min(x + HPC + SX + 1, pad.cols)));
        cv::Mat r;
        cv::matchTemplate(s, BIG[k], r, cv::TM_CCOEFF_NORMED);
        cv::Point bl;
        cv::minMaxLoc(r, nullptr, nullptr, nullptr, &bl);
        double cx0 = bl.x + x0 + HPC - x, cy0 = bl.y + y0 + HPC - y;
        int ox = (int)std::nearbyint(cx0), oy = (int)std::nearbyint(cy0);
        cv::Mat s2 = img(cv::Rect(F[k].x + ox - hp - 4, F[k].y + oy - hp - 4, nt + 8, nt + 8));
        cv::matchTemplate(s2, Tr[k], r, cv::TM_CCOEFF_NORMED);
        cv::minMaxLoc(r, nullptr, nullptr, nullptr, &bl);
        double tmx = parab(r.ptr<float>(bl.y), r.cols, bl.x, 1) - 4 + ox;
        double tmy = parab(r.ptr<float>(0) + bl.x, r.rows, bl.y, r.step1()) - 4 + oy;
        int cx = F[k].x + (int)std::nearbyint(tmx), cy = F[k].y + (int)std::nearbyint(tmy);
        const int L = hp + 6, WN = 2 * L + 1;
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
            for (int ii = -hp; ii <= hp; ++ii)
                for (int jj = -hp; jj <= hp; ++jj) {
                    bsp::Tap tp(jj + L + dx, ii + L + dy, WN, WN);
                    double I = tp.at(cI), Ix = tp.at(cX), Iy = tp.at(cY);
                    double rr = a * I + b - Tb[k].at<float>(ii + hp, jj + hp);
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
cv::Mat cov(const cv::Mat& R) {  // np.cov(R.T), ddof 1
    cv::Mat c, m;
    cv::calcCovarMatrix(R, c, m, cv::COVAR_NORMAL | cv::COVAR_ROWS | cv::COVAR_SCALE, CV_64F);
    return c * (double)R.rows / (R.rows - 1);
}
cv::Mat pinv(const cv::Mat& S) {  // numpy pinv (rcond 1e-15)
    cv::SVD svd(S, cv::SVD::FULL_UV);
    cv::Mat wi = cv::Mat::zeros(S.cols, S.rows, CV_64F);
    double mx = svd.w.at<double>(0);
    for (int i = 0; i < svd.w.rows; ++i) if (svd.w.at<double>(i) > 1e-15 * mx) wi.at<double>(i, i) = 1 / svd.w.at<double>(i);
    return svd.vt.t() * wi * svd.u.t();
}

double polyslope(const std::vector<double>& x, const std::vector<double>& y) {
    double mx = mean(x), my = mean(y), sxy = 0, sxx = 0;
    for (size_t i = 0; i < x.size(); ++i) { sxy += (x[i] - mx) * (y[i] - my); sxx += (x[i] - mx) * (x[i] - mx); }
    return sxy / sxx;
}
double norm_cdf(double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); }

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
    std::vector<int> SESA(NS, -1);
    for (int i = 0; i < NS; ++i) {
        AX[i] = shots[i].condition == "center" ? 'C' : shots[i].condition == "Y" ? 'Y' : 'T';
        VAL[i] = AX[i] == 'Y' ? shots[i].y : AX[i] == 'T' ? shots[i].t : 0;
        NAME[i] = shots[i].rel.substr(shots[i].rel.find('/') + 1);
        if (AX[i] == 'T') SESA[i] = VAL[i] > 0 ? 4 : 5;
        if (AX[i] == 'Y') SESA[i] = (std::abs((int)std::lround(VAL[i] / 0.05)) % 2 == 1 ? 2 : 0) + (VAL[i] > 0 ? 0 : 1);
    }
    std::vector<int> FOLD(NS, -1);
    std::map<char, std::vector<int>> order;
    for (char a : {'Y', 'T'}) {
        for (int i = 0; i < NS; ++i) if (AX[i] == a) order[a].push_back(i);
        std::stable_sort(order[a].begin(), order[a].end(), [&](int p, int q) { return tmin[p] < tmin[q]; });
        for (size_t k = 0; k < order[a].size(); ++k) FOLD[order[a][k]] = (int)(k % 2);
    }

    // ---------- 특징점 (t11 pick_qr, hp 8) ----------
    const cv::Mat CEN8 = read_gray(shots[ic]);
    cv::Mat CENF, CENB1, CENB2;
    CEN8.convertTo(CENF, CV_32F);
    cv::GaussianBlur(CENF, CENB1, cv::Size(0, 0), 1.0);
    cv::GaussianBlur(CENF, CENB2, cv::Size(0, 0), SIG);
    std::vector<Pt> F;
    {
        cv::Mat mask = cv::Mat::zeros(CENF.size(), CV_8U);
        mask(cv::Rect(PX0, PY0, PX1 - PX0, PY1 - PY0)).setTo(255);
        std::vector<cv::Point2f> corners;
        cv::goodFeaturesToTrack(CENB1, corners, 5000, 0.01, 2 * HPF + 1, mask, 7);
        for (auto& c : corners) {
            int xi = (int)std::nearbyint(c.x), yi = (int)std::nearbyint(c.y), r = region_of(xi, yi);
            if (r > 3) continue;
            int n = 0;
            for (int y = yi - HPF; y <= yi + HPF; ++y)
                for (int x = xi - HPF; x <= xi + HPF; ++x) n += CENF.at<float>(y, x) >= 250;
            double sat = n / double((2 * HPF + 1) * (2 * HPF + 1));
            if ((r < 3 && sat > 0) || sat > 0.6) continue;
            F.push_back({xi, yi, r, sat});
        }
    }
    const int N = (int)F.size();
    std::array<int, 4> nreg{};
    for (auto& p : F) nreg[p.region]++;
    std::cout << "features " << N << " QR " << nreg[0] + nreg[1] + nreg[2] << "\n";
    cv::Mat CENP;
    cv::copyMakeBorder(CENF, CENP, HPC, HPC, HPC, HPC, cv::BORDER_REPLICATE);
    std::vector<cv::Mat> BIG, Tr, Tb;
    for (auto& p : F) {
        BIG.push_back(CENP(cv::Rect(p.x, p.y, 2 * HPC + 1, 2 * HPC + 1)).clone());
        Tr.push_back(CENF(cv::Rect(p.x - HP, p.y - HP, NT, NT)).clone());
        Tb.push_back(CENB2(cv::Rect(p.x - HP, p.y - HP, NT, NT)).clone());
    }

    // ---------- 추적 ----------
    std::vector<double> qx(NS);
    cv::Mat D(NS, 2 * N, CV_64F);
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < NS; ++i) {
        cv::Mat im8 = read_gray(shots[i]), img, pad, bimg;
        im8.convertTo(img, CV_32F);
        qx[i] = qr_mean_dx(CEN8, im8);
        cv::copyMakeBorder(img, pad, HPC, HPC, HPC, HPC, cv::BORDER_REPLICATE);
        cv::GaussianBlur(img, bimg, cv::Size(0, 0), SIG);
        std::vector<cv::Point2d> gn;
        track(img, pad, bimg, F, BIG, Tr, Tb, HP, gn);
        for (int k = 0; k < N; ++k) { D.at<double>(i, 2 * k) = gn[k].x; D.at<double>(i, 2 * k + 1) = gn[k].y; }
    }
    std::cout << "tracked\n";

    // ---------- 요약 (lib7.summarize affine) ----------
    std::vector<std::string> cols;
    std::vector<cv::Mat> blocks;
    std::array<cv::Mat, 3> QA;  // QR마다 [1, x, y] (점 x 3), 그림에서 점 이동 예측에 쓴다
    std::array<std::vector<int>, 3> QI;
    const char* QK = "LCR";
    for (int q = 0; q < 3; ++q) {
        for (int k = 0; k < N; ++k) if (F[k].region == q) QI[q].push_back(k);
        double mx = 0, my = 0;
        for (int k : QI[q]) { mx += F[k].x; my += F[k].y; }
        mx /= QI[q].size(); my /= QI[q].size();
        cv::Mat A((int)QI[q].size(), 3, CV_64F);
        for (size_t r = 0; r < QI[q].size(); ++r) {
            A.at<double>((int)r, 0) = 1; A.at<double>((int)r, 1) = (F[QI[q][r]].x - mx) / 100; A.at<double>((int)r, 2) = (F[QI[q][r]].y - my) / 100;
        }
        QA[q] = A;
        for (int c = 0; c < 2; ++c) {
            cv::Mat B((int)QI[q].size(), NS, CV_64F);
            for (size_t r = 0; r < QI[q].size(); ++r)
                for (int i = 0; i < NS; ++i) B.at<double>((int)r, i) = D.at<double>(i, 2 * QI[q][r] + c);
            blocks.push_back(lsq(A, B).t());  // NS x 3
            for (std::string t : {"0", "x", "y"}) cols.push_back(std::string(1, QK[q]) + (c ? "_dy_" : "_dx_") + t);
        }
    }
    for (int s = 0; s < 2; ++s) {
        std::vector<int> m;
        for (int k = 0; k < N; ++k) if (F[k].region == 3 && std::hypot(F[k].x - kScrew[s][0], F[k].y - kScrew[s][1]) < 60) m.push_back(k);
        if (m.empty()) continue;
        cv::Mat B(NS, 2, CV_64F, cv::Scalar(0));
        for (int i = 0; i < NS; ++i) {
            for (int k : m) { B.at<double>(i, 0) += D.at<double>(i, 2 * k); B.at<double>(i, 1) += D.at<double>(i, 2 * k + 1); }
            B.at<double>(i, 0) /= m.size(); B.at<double>(i, 1) /= m.size();
        }
        blocks.push_back(B);
        cols.push_back("S" + std::to_string(s) + "_dx");
        cols.push_back("S" + std::to_string(s) + "_dy");
    }
    cv::Mat X;
    cv::hconcat(blocks, X);
    const int NQ = 18;

    // ---------- 보정본 라벨 (v5 와 같음) ----------
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

    // ---------- 판정 (judge7.run, quad, cov full) ----------
    std::vector<double> GY(5201), GT(5201);
    for (int g = 0; g < 5201; ++g) { GY[g] = -2.6 + 5.2 * g / 5200; GT[g] = -1.3 + 2.6 * g / 5200; }  // np.linspace
    struct Rec { int i; char axis; int ses; double cmd, act; char pa; double pred, costY, costT, valY, valT; };
    struct Model { cv::Mat s[2], c[2], sesb[6]; };
    struct Run { std::vector<Rec> recs; std::array<Model, 2> M; };
    auto runj = [&](const cv::Mat& Xs, bool oracle, bool corr) {
        Run out;
        const int p = Xs.cols;
        for (int k = 0; k < 2; ++k) {
            const auto& lab = corr ? CORR[k] : VAL;
            Model& M = out.M[k];
            std::vector<cv::Mat> res;
            for (int ai = 0; ai < 2; ++ai) {
                char a = ai ? 'T' : 'Y';
                std::vector<int> idx;
                for (int i = 0; i < NS; ++i) if (FOLD[i] >= 0 && FOLD[i] != k && AX[i] == a) idx.push_back(i);
                std::vector<int> ss = ai ? std::vector<int>{4, 5} : std::vector<int>{0, 1, 2, 3};
                int nc = 2 + (oracle ? (int)ss.size() : 0);
                cv::Mat A((int)idx.size(), nc, CV_64F, cv::Scalar(0)), Xm((int)idx.size(), p, CV_64F);
                for (size_t r = 0; r < idx.size(); ++r) {
                    double v = lab[idx[r]];
                    A.at<double>((int)r, 0) = v; A.at<double>((int)r, 1) = v * v;
                    if (oracle) for (size_t j = 0; j < ss.size(); ++j) A.at<double>((int)r, 2 + (int)j) = SESA[idx[r]] == ss[j];
                    Xs.row(idx[r]).copyTo(Xm.row((int)r));
                }
                cv::Mat co = lsq(A, Xm);
                M.s[ai] = co.row(0).clone(); M.c[ai] = co.row(1).clone();
                if (oracle) for (size_t j = 0; j < ss.size(); ++j) M.sesb[ss[j]] = co.row(2 + (int)j).clone();
                cv::Mat r = Xm - A.col(0) * M.s[ai] - A.col(1) * M.c[ai];
                if (oracle) for (size_t rr = 0; rr < idx.size(); ++rr) { cv::Mat rw = r.row((int)rr); rw -= M.sesb[SESA[idx[rr]]]; }
                res.push_back(r);
            }
            cv::Mat R;
            cv::vconcat(res, R);
            cv::Mat W = pinv(cov(R));
            // 격자: cost(g) = P_g W P_g - 2 z W P_g + z W z
            std::array<cv::Mat, 2> Q;
            std::array<std::vector<double>, 2> t1;
            for (int ai = 0; ai < 2; ++ai) {
                const auto& G = ai ? GT : GY;
                cv::Mat P(5201, p, CV_64F);
                for (int g = 0; g < 5201; ++g) cv::Mat(G[g] * M.s[ai] + G[g] * G[g] * M.c[ai]).copyTo(P.row(g));
                cv::Mat PW = P * W;
                t1[ai].resize(5201);
                for (int g = 0; g < 5201; ++g) t1[ai][g] = PW.row(g).dot(P.row(g));
                Q[ai] = W * P.t();  // p x 5201
            }
            for (int i = 0; i < NS; ++i) {
                if (FOLD[i] != k) continue;
                Rec rec{i, AX[i], SESA[i], VAL[i], CORR[k][i]};
                double val[2], cst[2];
                for (int ai = 0; ai < 2; ++ai) {
                    cv::Mat z = Xs.row(i).clone();
                    if (oracle) z -= M.sesb[SESA[i]];
                    cv::Mat zq = z * Q[ai];
                    int best = 0;
                    double bc = 1e300;
                    for (int g = 0; g < 5201; ++g) {
                        double cg = t1[ai][g] - 2 * zq.at<double>(g);
                        if (cg < bc) { bc = cg; best = g; }
                    }
                    cv::Mat zw = z * W;
                    val[ai] = (ai ? GT : GY)[best];
                    cst[ai] = bc + zw.dot(z);
                }
                rec.pa = cst[0] <= cst[1] ? 'Y' : 'T';
                rec.pred = rec.pa == 'Y' ? val[0] : val[1];
                rec.costY = cst[0]; rec.costT = cst[1]; rec.valY = val[0]; rec.valT = val[1];
                out.recs.push_back(rec);
            }
        }
        return out;
    };
    struct Ev { double acc; int wy, wt; double maxYeq, yMean, yMax, tMean, tMax, pass, passY, passT; };
    auto evaluate = [&](const std::vector<Rec>& recs, bool act) {
        Ev e{};
        std::vector<double> ey, et;
        int ok = 0, pass = 0, py = 0, pt = 0, ny = 0, nt = 0;
        for (auto& r : recs) {
            double t = act ? r.act : r.cmd, er = std::abs(r.pred - t);
            bool g = r.pa == r.axis;
            bool ps = g ? er <= 0.09 : (std::abs(t) <= 0.09 && std::abs(r.pred) <= 0.09);
            ok += g; pass += ps;
            if (r.axis == 'Y') { ++ny; py += ps; if (g) ey.push_back(er); else { e.wy++; e.maxYeq = std::max(e.maxYeq, std::abs(r.cmd)); } }
            else { ++nt; pt += ps; if (g) et.push_back(er); else { e.wt++; e.maxYeq = std::max(e.maxYeq, std::abs(r.cmd) * THETA_TO_Y); } }
        }
        e.acc = 100.0 * ok / recs.size(); e.pass = 100.0 * pass / recs.size(); e.passY = 100.0 * py / ny; e.passT = 100.0 * pt / nt;
        e.yMean = mean(ey) * 1000; e.yMax = max_abs(ey) * 1000; e.tMean = mean(et) * 1000; e.tMax = max_abs(et) * 1000;
        return e;
    };
    cv::Mat Xq = X.colRange(0, NQ);
    Table tab{{"세트", "학습라벨", "영점", "정답기준", "축정확도", "축틀림_Y", "축틀림_Θ", "틀림최대_Yeq", "Y_평균오차", "Y_최대오차", "Θ_평균오차", "Θ_최대오차", "통과09", "통과09_Y", "통과09_Θ"}};
    std::map<std::string, Ev> EV;
    std::map<std::string, Run> RUN;
    for (int set = 0; set < 2; ++set)
        for (int corr = 0; corr < 2; ++corr)
            for (int oracle = 0; oracle < 2; ++oracle) {
                Run r = runj(set ? X : Xq, oracle, corr);
                std::string key = std::string(set ? "QR+나사" : "QR만") + "|" + (corr ? "보정본" : "명령값") + "|" + (oracle ? "구간 0점 앎 (상한)" : "center (배포 가능)");
                for (int act = 0; act < 2; ++act) {
                    Ev e = evaluate(r.recs, act);
                    EV[key + "|" + std::to_string(act)] = e;
                    tab.add({set ? "QR+나사" : "QR만", corr ? "보정본" : "명령값", oracle ? "구간 0점 앎 (상한)" : "center (배포 가능)", act ? "실제 위치 (보정함)" : "명령값 (보정 안 함)",
                             fmt(e.acc, 3), std::to_string(e.wy), std::to_string(e.wt), fmt(e.maxYeq, 3), fmt(e.yMean, 3), fmt(e.yMax, 3), fmt(e.tMean, 3), fmt(e.tMax, 3), fmt(e.pass, 3),
                             fmt(e.passY, 3), fmt(e.passT, 3)});
                }
                RUN[key] = std::move(r);
            }
    tab.csv(out("평가표_최종.csv"));
    const std::string kMain = "QR만|명령값|center (배포 가능)", kCorr = "QR만|보정본|center (배포 가능)";
    {
        Table pr{{"세트", "학습라벨", "file", "axis", "cmd", "act", "pred_axis", "pred", "cost_Y", "cost_T", "val_Y", "val_T"}};
        for (std::string key : {kMain, kCorr, std::string("QR+나사|명령값|center (배포 가능)"), std::string("QR+나사|보정본|center (배포 가능)")})
            for (auto& r : RUN[key].recs)
                pr.add({key.substr(0, key.find('|')), key.find("보정본") != std::string::npos ? "corr" : "raw", shots[r.i].rel, std::string(1, r.axis), fmt(r.cmd, 2), fmt(r.act, 5),
                        std::string(1, r.pa), fmt(r.pred, 4), fmt(r.costY, 4), fmt(r.costT, 4), fmt(r.valY, 4), fmt(r.valT, 4)});
        pr.csv(out("사진별예측_최종.csv"));
    }

    // ---------- 0점 7개 + 잡음 바닥 (lib7.session_fit, floor; 명령값 라벨) ----------
    std::array<cv::Mat, 6> ints, sl;
    cv::Mat Rz;
    {
        std::vector<cv::Mat> rr;
        for (int s = 0; s < 6; ++s) {
            std::vector<int> idx;
            for (int i = 0; i < NS; ++i) if (SESA[i] == s) idx.push_back(i);
            cv::Mat A((int)idx.size(), 3, CV_64F), Xm((int)idx.size(), NQ, CV_64F);
            for (size_t r = 0; r < idx.size(); ++r) {
                double v = VAL[idx[r]];
                A.at<double>((int)r, 0) = 1; A.at<double>((int)r, 1) = v; A.at<double>((int)r, 2) = v * v;
                Xq.row(idx[r]).copyTo(Xm.row((int)r));
            }
            cv::Mat co = lsq(A, Xm);
            ints[s] = co.row(0).clone(); sl[s] = co.row(1).clone();
            rr.push_back(Xm - A * co);
        }
        cv::vconcat(rr, Rz);
    }
    cv::Mat Si = pinv(cov(Rz));
    cv::Mat sYm = (sl[0] + sl[1] + sl[2] + sl[3]) / 4, sTm = (sl[4] + sl[5]) / 2;
    double aa = cv::Mat(sTm * Si * sYm.t()).at<double>(0) / cv::Mat(sTm * Si * sTm.t()).at<double>(0);
    cv::Mat dd = sYm - aa * sTm;
    const double nfloor = std::sqrt(cv::Mat(dd * Si * dd.t()).at<double>(0));
    cv::Mat u = Si * dd.t() / nfloor / nfloor;
    Table zt{{"구간", "평균이동_px", "Y환산_mm", "Θ환산", "판정방향_치우침_Yeq_mm"}};
    zt.add({"center", "0", "0", "0", "0"});
    std::array<double, 6> zY, zdir;
    for (int s = 0; s < 6; ++s) {
        double mpx = (ints[s].at<double>(0) + ints[s].at<double>(6) + ints[s].at<double>(12)) / 3;  // L_dx_0, C_dx_0, R_dx_0
        zY[s] = mpx / 14.39;
        zdir[s] = cv::Mat(ints[s] * u).at<double>(0);
        zt.add({SES[s], fmt(mpx, 4), fmt(zY[s], 4), fmt(mpx / 80.66, 4), fmt(zdir[s], 4)});
    }
    // 이론 기대 틀림 수
    auto expected = [&](double n) { double s = 0; for (int i = 0; i < NS; ++i) if (AX[i] != 'C') s += norm_cdf(-n * std::abs(VAL[i]) * (AX[i] == 'Y' ? 1 : THETA_TO_Y) / 2); return s; };
    double need = 5;
    while (need < 200 && expected(need) >= 0.5) need += 0.5;

    // ---------- 그림 ----------
    auto& RM = RUN[kCorr];
    {   // 01 특징점
        cv::Mat im = read_color(shots[ic]);
        std::vector<Mark> mk;
        for (auto& p : F) { Mark m; m.p = {(double)p.x, (double)p.y}; m.c = p.region < 3 ? kRed : color(1); mk.push_back(m); }
        save_png("01_특징점_QR232_나사.png", zoom(im, cv::Rect(PX0 - 10, PY0 - 10, PX1 - PX0 + 20, PY1 - PY0 + 20), 1600, mk, 0,
                                                "red = QR points (" + std::to_string(nreg[0] + nreg[1] + nreg[2]) + "), orange = screw edge points (" + std::to_string(nreg[3]) + ")"));
    }
    {   // 02 예측 오차 (QR만, 보정본, center 0점, 실제 위치)
        std::vector<double> ty, py, tt, pt;
        std::vector<std::string> nm;
        for (auto& r : RM.recs) {
            ty.push_back(r.axis == 'Y' ? r.act : 0); tt.push_back(r.axis == 'T' ? r.act : 0);
            py.push_back(r.pa == 'Y' ? r.pred : 0); pt.push_back(r.pa == 'T' ? r.pred : 0);
            nm.push_back(NAME[r.i].substr(0, NAME[r.i].rfind('.')));
        }
        pred_vs_truth("02_QR만_예측오차.png", "QR 18 values GLS (corr, center zero)", ty, py, tt, pt, nm);
    }
    {   // 03 0점 7개
        std::vector<std::string> cats{"center", "Y1+", "Y1-", "Y2+", "Y2-", "T+", "T-"};
        std::vector<double> a{0}, b{0};
        for (int s = 0; s < 6; ++s) { a.push_back(std::abs(zY[s]) * 1000); b.push_back(zdir[s] * 1000); }
        bar_chart("03_0점_값쪽_절대값.png", "Zero offset per session: |mean shift| (Y um)", cats, a, "|offset| (um, Y equiv.)");
        bar_chart("04_0점_판정방향치우침.png", "Zero offset per session: judging bias (Y um)", cats, b, "bias (um, Y equiv.)");
    }
    {   // 05 크기별 (이론): 틀림 확률 곡선과 실제 틀린 사진
        std::vector<double> x, y, wx, wy;
        for (int k = 1; k <= 300; ++k) { double v = 0.01 * std::pow(100.0, k / 300.0) ; x.push_back(v); y.push_back(100 * norm_cdf(-nfloor * v / 2)); }
        for (auto& r : RUN[kMain].recs)
            if (r.pa != r.axis) { double v = std::abs(r.cmd) * (r.axis == 'Y' ? 1 : THETA_TO_Y); wx.push_back(v); wy.push_back(100 * norm_cdf(-nfloor * v / 2)); }
        Chart c("Theory: P(axis wrong) vs size, noise " + fmt(nfloor, 1) + " sigma/mm; red = wrong photos", "size (Y mm equiv.)", "P wrong (%)");
        c.line(x, y, color(0)).scatter(wx, wy, kRed, "wrong (QR, raw)", 6).vline(0.09, kGreen);
        c.save("05_이론_크기별_틀림확률.png");
    }
    // 튀는 값 그림: 축 틀린 Y 사진 중 비용 비(Y/Θ) 최대, Θ 는 축 틀린 Θ 중 최소 (없으면 오차 최대)
    auto pick = [&](char a) {
        int best = -1;
        for (size_t k = 0; k < RM.recs.size(); ++k) {
            auto& r = RM.recs[k];
            if (r.axis != a || r.pa == a) continue;
            double q = r.costY / r.costT;
            if (best < 0 || (a == 'Y' ? q > RM.recs[best].costY / RM.recs[best].costT : q < RM.recs[best].costY / RM.recs[best].costT)) best = (int)k;
        }
        if (best >= 0) return best;
        for (size_t k = 0; k < RM.recs.size(); ++k) {
            auto& r = RM.recs[k];
            if (r.axis == a && (best < 0 || std::abs(r.pred - r.act) > std::abs(RM.recs[best].pred - RM.recs[best].act))) best = (int)k;
        }
        return best;
    };
    std::map<char, std::string> ofile;
    std::map<char, int> sel{{'Y', pick('Y')}, {'T', pick('T')}};
    for (char a : {'Y', 'T'}) {
        auto& r = RM.recs[sel[a]];
        int k = FOLD[r.i], ai = a == 'Y' ? 0 : 1;
        double v = ai ? r.valT : r.valY;
        cv::Mat f = v * RM.M[k].s[ai] + v * v * RM.M[k].c[ai];  // 참 축 가설이 기대한 18값
        cv::Mat im = read_color(shots[r.i]);
        std::vector<Mark> mk;
        double sx = 0;
        for (int q = 0; q < 3; ++q)
            for (size_t rr = 0; rr < QI[q].size(); ++rr) {
                int p = QI[q][rr];
                double ex = 0, ey = 0;
                for (int j = 0; j < 3; ++j) { ex += QA[q].at<double>((int)rr, j) * f.at<double>(6 * q + j); ey += QA[q].at<double>((int)rr, j) * f.at<double>(6 * q + 3 + j); }
                double mx = D.at<double>(r.i, 2 * p), my = D.at<double>(r.i, 2 * p + 1);
                sx += mx;
                Mark m;
                m.p = {F[p].x + mx, F[p].y + my};
                m.d = {mx - ex, my - ey};
                mk.push_back(m);
            }
        sx /= mk.size();
        std::string nm = NAME[r.i].substr(0, NAME[r.i].rfind('.'));
        std::string cap = nm + ": true " + (a == 'Y' ? "Y " : "T ") + fmt(r.act, 3) + ", judged " + (r.pa == 'Y' ? "Y " : "T ") + fmt(r.pred, 3) + ", cost Y/T " + fmt(r.costY / r.costT, 3) +
                          " (arrows: measured - true-axis model x300)";
        ofile[a] = std::string(a == 'Y' ? "06_튀는값_Y_" : "07_튀는값_T_") + nm + ".png";
        save_png(ofile[a], zoom(im, cv::Rect((int)(1540 + sx), 960, 720, 220), 1500, mk, 300, cap));
    }

    // ---------- 원래 결과와 대조 ----------
    // 원래 값: analysis_v7 e11_final 결과 58_v7_평가표_최종.csv, 60_v7_0점7개.csv (00000.legacies 루트), 00_v7결과리포트.md 표 A
    struct O { std::string key; int act; double acc; int wy, wt; double yMean, pass; };
    const std::vector<O> orig{{"QR만|명령값|center (배포 가능)", 0, 98.571, 3, 1, 14.091, 98.571},       {"QR만|명령값|center (배포 가능)", 1, 98.571, 3, 1, 25.875, 98.571},
                              {"QR만|명령값|구간 0점 앎 (상한)", 0, 98.214, 3, 2, 1.519, 98.571},       {"QR만|보정본|center (배포 가능)", 0, 98.571, 3, 1, 26.753, 98.571},
                              {"QR만|보정본|center (배포 가능)", 1, 98.571, 3, 1, 1.396, 98.571},       {"QR만|보정본|구간 0점 앎 (상한)", 1, 98.214, 3, 2, 1.439, 98.571},
                              {"QR+나사|명령값|center (배포 가능)", 0, 98.214, 5, 0, 15.093, 98.571},   {"QR+나사|보정본|center (배포 가능)", 1, 98.571, 4, 0, 1.53, 98.571},
                              {"QR+나사|명령값|구간 0점 앎 (상한)", 0, 98.929, 3, 0, 1.597, 98.929}};
    Parity par;
    par.add("특징점 수 (QR)", 232, nreg[0] + nreg[1] + nreg[2], 0, 0);
    par.add("특징점 수 (판 나사)", 18, nreg[3], 0, 0);
    par.add("잡음 바닥 3σ 최소 |Y| mm (QR만 아핀, 33px σ2)", 0.222, 3 / nfloor, 0.02, 3);
    for (auto& o : orig) {
        auto& e = EV[o.key + "|" + std::to_string(o.act)];
        std::string k = o.key + (o.act ? " / 실제 위치" : " / 명령값");
        par.add(k + " 축 정확도 %", o.acc, e.acc, 0.72, 3);
        par.add(k + " 축 틀림 Y", o.wy, e.wy, 1, 0);
        par.add(k + " 축 틀림 Θ", o.wt, e.wt, 1, 0);
        par.add(k + " Y 평균 오차 µm", o.yMean, e.yMean, 0.1 * o.yMean, 3);
        par.add(k + " 0.09 통과 %", o.pass, e.pass, 0.72, 3);
    }
    const double oz[6] = {-0.0363, 0.0046, -0.0303, -0.0272, -0.0545, -0.0137};
    for (int s = 0; s < 6; ++s) par.add(std::string("0점 ") + SES[s] + " Y 환산 mm", oz[s], zY[s], 0.005, 4);

    // ---------- 결과.md ----------
    std::ostringstream Rm;
    Rm << "# 결과: QR 아핀 18값 GLS 판정 (center + 현재 2장)\n\n"
       << "입력: `data/clean/clean_20261014` 281장. 폴드·보정본 라벨은 v5(#13)와 같다 (축마다 촬영 순서 짝/홀, `data/raw` zip 수정 시각; 보정본 = ① 백래시+드리프트, QR 평균 이동은 ECC).\n\n"
       << "특징점: 흐린 center(sigma 1) QR 판 안 goodFeaturesToTrack(최대 5000, 품질 0.01, 간격 17, 블록 7) -> QR 영역(포화 0) " << nreg[0] + nreg[1] + nreg[2] << "개 (좌 " << nreg[0] << ", 중 " << nreg[1]
       << ", 우 " << nreg[2] << "), 판 나사(포화 60% 이하) " << nreg[3] << "개.\n"
       << "추적: 81x81 큰 템플릿(가로 ±110, 세로 ±16 px) -> 33x33 템플릿 매칭+포물선 -> 흐림 sigma 2, 33x33 가우스-뉴턴(3차 스플라인, 밝기 a,b).\n"
       << "요약: QR마다 dx, dy 를 [1, x, y]로 맞춘 계수 = 18값 (+나사 2개 평균 dx, dy = 22값).\n"
       << "판정: 축마다 f = s v + c v^2 (center = 0점) 또는 구간 절편까지(상한), 잔차 공분산 전체의 역행렬로 GLS 비용, v 격자(Y ±2.6 mm, Θ ±1.3, 5201점) 최소, 비용 작은 축.\n\n"
       << "## 평가표 (짝/홀 교차검증)\n\n" << tab.md()
       << "\n## 0점 7개 (명령값 라벨, 구간마다 [1, v, v^2], QR만)\n\n" << zt.md()
       << "\n잡음 바닥 n = " << fmt(nfloor, 2) << " σ/mm -> 3σ 최소 |Y| " << fmt(3 / nfloor, 3) << " mm. 이론 기대 틀림 " << fmt(expected(nfloor), 2) << "장, 기대 틀림 0.5장 미만에 필요한 잡음 " << fmt(need, 1)
       << " σ/mm (같은 자세 사진 약 " << fmt((need / nfloor) * (need / nfloor), 0) << "장 평균).\n"
       << "\n## 원래 결과와 대조 (" << par.ok << "/" << par.total << " 일치)\n\n허용오차: 축 정확도·통과율 ±0.72%(2장), 축 틀림 ±1장, Y 평균 오차 ±10%, 잡음 바닥 ±0.02 mm, 0점 ±5 µm.\n\n" << par.t.md()
       << "\n## 옮기지 않은 부분\n\n- e11의 무작위 반반 40번 반복(크기별 실측 틀림률, 61번 표): numpy 난수열을 C++에서 같게 만들 수 없어 뺐다. 대신 이론 틀림 확률 곡선(그림 05)만 그렸다.\n"
       << "- e0~e10, e12~e15: 측정 4종 x 요약 x 0점 x 공분산 설정 탐색, 점 순위, ML 비교, 증폭 그림, 이상점 확인.\n"
       << "\n## 파일\n\n- `평가표_최종.csv`: 원래 58번 표와 같은 열\n- `사진별예측_최종.csv`: center 0점 4조건 사진별 판정\n"
       << "- `이미지/01_특징점_QR232_나사.png`, `02_QR만_예측오차.png`, `03_0점_값쪽_절대값.png`, `04_0점_판정방향치우침.png`, `05_이론_크기별_틀림확률.png`, `" << ofile['Y'] << "`, `" << ofile['T'] << "`\n";
    write_text(out("결과.md"), Rm.str());

    // ---------- 보고서.md ----------
    auto& e0 = EV[kMain + "|0"];
    auto& e1 = EV[kCorr + "|1"];
    Table outl{{"사진", "정답 축", "명령값", "실제 위치", "판정 축", "추정 값", "비용 비 Y/Θ", "Y 가설 값", "Θ 가설 값"}};
    for (auto& r : RUN[kMain].recs)
        if (r.pa != r.axis)
            outl.add({NAME[r.i], r.axis == 'Y' ? "Y" : "Θ", fmt(r.cmd, 2), fmt(r.act, 3), r.pa == 'Y' ? "Y" : "Θ", fmt(r.pred, 3), fmt(r.costY / r.costT, 3), fmt(r.valY, 3), fmt(r.valT, 4)});
    auto& rY = RM.recs[sel['Y']];
    std::ostringstream P;
    P << "# 보고서: QR 2장 판정, 아핀 18값 GLS (10-14 15:57)\n\n"
      << "## 문제\n\ncenter와 현재 사진 2장, QR 3개만으로 축 판정 100%와 값 오차 0.09 안을 달성하려 했다.\n\n"
      << "## 해결 방법\n\n33px 가우스-뉴턴으로 QR 점 " << nreg[0] + nreg[1] + nreg[2] << "개를 재고 QR별 아핀 18값으로 요약해, 축마다 2차 매핑과 전체 공분산 GLS 가설 비교로 판정했다.\n\n"
      << "## 결과\n\nQR만, center 0점, 명령값 학습에서 축 정확도 " << fmt(e0.acc, 2) << "%(Y " << e0.wy << "장, Θ " << e0.wt << "장 틀림), 0.09 통과 " << fmt(e0.pass, 1) << "%였고, 축을 맞힌 사진의 값 오차는 모두 0.09 안이었다.\n\n"
      << "| 지표 (QR만, center 0점) | 명령값 학습 → 명령값 채점 | 보정본 학습 → 실제 위치 채점 |\n|---|---|---|\n"
      << "| 축 정확도 | " << fmt(e0.acc, 3) << "% | " << fmt(e1.acc, 3) << "% |\n"
      << "| 축 틀림 (Y / Θ 사진) | " << e0.wy << " / " << e0.wt << " | " << e1.wy << " / " << e1.wt << " |\n"
      << "| Y 평균 / 최대 오차 (µm) | " << fmt(e0.yMean, 2) << " / " << fmt(e0.yMax, 2) << " | " << fmt(e1.yMean, 2) << " / " << fmt(e1.yMax, 2) << " |\n"
      << "| Θ 평균 / 최대 오차 (mΘ) | " << fmt(e0.tMean, 3) << " / " << fmt(e0.tMax, 3) << " | " << fmt(e1.tMean, 3) << " / " << fmt(e1.tMax, 3) << " |\n"
      << "| 0.09 통과 | " << fmt(e0.pass, 3) << "% | " << fmt(e1.pass, 3) << "% |\n"
      << "| 잡음 바닥 (σ/mm, 3σ 최소 \\|Y\\|) | " << fmt(nfloor, 2) << ", " << fmt(3 / nfloor, 3) << " mm | |\n\n"
      << img_md("02_QR만_예측오차.png", "예측 오차") << "\n\n" << img_md("05_이론_크기별_틀림확률.png", "이론 틀림 확률") << "\n\n" << img_md("01_특징점_QR232_나사.png", "특징점") << "\n\n"
      << img_md("03_0점_값쪽_절대값.png", "0점 값") << "\n\n" << img_md("04_0점_판정방향치우침.png", "0점 판정 방향") << "\n\n"
      << "## 튀는 값\n\n축을 틀린 사진은 |Y| " << fmt(e0.maxYeq, 2) << " mm(Θ는 Y 환산) 이하의 작은 움직임뿐이었고, 그림의 " << NAME[rY.i] << "는 실제 Y " << fmt(rY.act, 3) << " mm인데 Y 가설 비용이 Θ 가설의 "
      << fmt(rY.costY / rY.costT, 3) << "배라서 Θ " << fmt(rY.valT, 4) << "로 판정됐다.\n\n" << outl.md() << "\n(위 표는 명령값 학습. 그림은 보정본 학습 조건.)\n\n"
      << img_md(ofile['Y'], "Y 튀는 값") << "\n\n" << img_md(ofile['T'], "T 튀는 값") << "\n\n"
      << "빨간 십자는 이 사진에서 잰 QR 점 위치(center 위치 + 측정 이동), 화살표는 참 축 가설이 기대한 아핀 이동과 잰 이동의 차이 x300이다.\n\n"
      << "## 한줄평\n\nY와 Θ를 가르는 단서가 양 끝 QR의 원근 차이 하나뿐이라 사진 1장 잡음이 Y 1mm당 " << fmt(nfloor, 1) << "σ(1σ = " << fmt(1 / nfloor, 3)
      << " mm)이고, 그래서 이론 기대 틀림 " << fmt(expected(nfloor), 1) << "장과 비슷한 " << e0.wy + e0.wt << "장이 0.3 mm 이하 사진에서 틀려 2장만으로는 축 100%에 닿지 못했다.\n\n"
      << "## 원래 결과와 대조\n\n원래 값은 analysis_v7 e11_final 결과 `58_v7_평가표_최종.csv`, `60_v7_0점7개.csv`(00000.legacies 루트)와 `00_v7결과리포트.md` 표 A에서 가져왔다. "
      << "보정본 라벨의 QR 이동은 원래 14번 CSV 대신 ECC로 다시 쟀다(#12와 같은 측정). 허용오차: 축 정확도·통과율 ±0.72%, 축 틀림 ±1장, Y 평균 오차 ±10%, 잡음 바닥 ±0.02 mm, 0점 ±5 µm. "
      << par.ok << "/" << par.total << " 일치.\n\n" << par.t.md();
    write_text(out("보고서.md"), P.str());
    std::cout << "parity " << par.ok << "/" << par.total << "\n";
    return 0;
}
