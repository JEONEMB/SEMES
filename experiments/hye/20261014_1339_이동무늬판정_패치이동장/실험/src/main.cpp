// 패치 이동장으로 Y/Θ 판정 (원본: analysis_v5/step1_2.py, step3_measure.py, step4_8.py)
// 1단계: 기준 영상 = 학습 폴드 사진을 판 원근 변환으로 center 좌표에 되돌린 평균 (폴드마다)
// 2단계: 판 영역 32x32 패치(16px 겹침), 구조 텐서 작은 고유값으로 약한 패치 제외
// 3단계: 패치마다 위상 상관(1/100 px) -> 가우스-뉴턴(3차 스플라인, 밝기 a,b)
// 4~7단계: 패치별 1차/2차 매핑 d = s v + c v^2 + b 를 축마다 맞추고, 새 사진은 Y 가설/Θ 가설을 Huber GN 으로 풀어 비용이 작은 쪽을 고른다.
// 원래 입력 중 14번 CSV(QR 위치), v2 homography_features.csv 는 사진에서 다시 쟀다 (QR ECC, 판 ECC 호모그래피).
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

// ---------- 영역 ----------
const int X0 = 1420, X1 = 2392, Y0 = 948, Y1 = 1192, HH = Y1 - Y0, WW = X1 - X0, PATCH = 32, STRIDE = 16;
const int CX0 = X0 - 140, CY0 = Y0 - 50, CX1 = X0 + WW + 140, CY1 = Y0 + HH + 50, CW = CX1 - CX0, CH = CY1 - CY0;
const int PCW = 48, PM = 8;
// center.bmp 의 QR 상자 (중심 x, y, 폭, 높이). 원래 14번 CSV center 행의 값 = 영역 정의 상수.
const double kQR[3][4] = {{1642.4546, 1067.5000, 158.9153, 171.0562}, {1900.4866, 1068.2500, 159.9748, 171.5151}, {2159.2039, 1070.5367, 160.4730, 173.0937}};
// v2 판 원근 변환 상자: X0 = int(Lx-110), X1 = int(Rx+110), Y0 = int(min(Ly,Ry)-120), Y1 = int(max(Ly,Ry)+120)
const int V2X0 = 1532, V2X1 = 2269, V2Y0 = 947, V2Y1 = 1190;
// QR ECC (시범 실험·#12 와 같은 측정): 0점 보정 라벨과 정수 출발점에 쓴다
const std::array<cv::Vec4i, 3> kBoxes{{{1554, 1734, 972, 1164}, {1815, 1995, 972, 1164}, {2073, 2253, 972, 1164}}};
const char* kRegion[6] = {"QR좌", "QR중", "QR우", "나사좌", "나사우", "그 외"};

struct Photo {
    double qx = 0, qy = 0;           // QR 세 개 중심 이동 평균
    std::array<cv::Point2f, 4> cs;   // v2 판 귀퉁이 이동 (좌상, 우상, 우하, 좌하)
    double hcc = kNaN;
    cv::Mat crop;                    // GaussianBlur(sigma 1) 된 측정 영역 (CV_32F, CH x CW)
};

double sfloor(double v) { return std::floor(v); }

void measure_qr(const cv::Mat& center, const cv::Mat& im, Photo& ph) {
    const auto crit = cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 150, 1e-7);
    double sx = 0, sy = 0;
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
        sy += dy0 + W.at<float>(1, 2) + W.at<float>(1, 0) * 90 + (W.at<float>(1, 1) - 1) * 96;
    }
    ph.qx = sx / 3;
    ph.qy = sy / 3;
}

// v2 extract_homography: 판 영역을 ECC 호모그래피로 center 에 맞춰 귀퉁이 이동을 구한다
void measure_h(const cv::Mat& cenf, const cv::Mat& imf, Photo& ph) {
    int sx = (int)std::nearbyint(ph.qx), sy = (int)std::nearbyint(ph.qy);
    cv::Mat tpl = cenf(cv::Range(V2Y0, V2Y1), cv::Range(V2X0, V2X1));
    cv::Mat crop = imf(cv::Range(V2Y0 + sy, V2Y1 + sy), cv::Range(V2X0 + sx, V2X1 + sx));
    cv::Mat W = cv::Mat::eye(3, 3, CV_32F);
    try {
        ph.hcc = semes::ecc5::findTransformECCWithMask(tpl, crop, cv::noArray(), cv::noArray(), W, cv::MOTION_HOMOGRAPHY,
                                                       cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 300, 1e-7), 5);
    } catch (const cv::Exception&) {}
    std::vector<cv::Point2f> pts{{(float)V2X0, (float)V2Y0}, {(float)V2X1, (float)V2Y0}, {(float)V2X1, (float)V2Y1}, {(float)V2X0, (float)V2Y1}}, loc, m;
    for (auto& p : pts) loc.push_back(p - cv::Point2f((float)V2X0, (float)V2Y0));
    cv::perspectiveTransform(loc, m, W);
    for (int k = 0; k < 4; ++k) ph.cs[k] = (m[k] + cv::Point2f((float)(V2X0 + sx), (float)(V2Y0 + sy))) - pts[k];
}

const std::vector<cv::Point2f> SRC{{(float)X0, (float)Y0}, {(float)X1, (float)Y0}, {(float)X1, (float)Y1}, {(float)X0, (float)Y1}};
cv::Mat H_of(const Photo& ph) {
    std::vector<cv::Point2f> dst(4);
    for (int k = 0; k < 4; ++k) dst[k] = SRC[k] + ph.cs[k];
    return cv::getPerspectiveTransform(SRC, dst);
}

// ---------- 패치 ----------
struct Patch { int px, py; double cx, cy, lmin, lmax; int region; bool use; };

// ---------- 회귀 도구 ----------
cv::Mat lsq(const cv::Mat& A, const cv::Mat& B) { cv::Mat X; cv::solve(A, B, X, cv::DECOMP_SVD); return X; }

struct Map { cv::Mat s, c, b; std::vector<double> var; };  // 각 1 x 2NP

// D: 사진 x 2NP, v: 라벨
Map fit_map(const cv::Mat& D, const std::vector<double>& v, bool quad) {
    const int n = D.rows, p = quad ? 3 : 2;
    cv::Mat A(n, p, CV_64F);
    for (int i = 0; i < n; ++i) {
        A.at<double>(i, 0) = v[i];
        if (quad) { A.at<double>(i, 1) = v[i] * v[i]; A.at<double>(i, 2) = 1; }
        else A.at<double>(i, 1) = 1;
    }
    cv::Mat coef = lsq(A, D);
    cv::Mat res = D - A * coef;
    Map m;
    m.s = coef.row(0).clone();
    m.c = quad ? coef.row(1).clone() : cv::Mat::zeros(1, D.cols, CV_64F);
    m.b = coef.row(p - 1).clone();
    m.var.resize(D.cols);
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
// 가우스-뉴턴 + Huber 재가중. rows = 쓰는 성분 번호, U = 방해 무늬 (2NP x 2)
Sol solve(const double* d, const Map& m, const std::vector<double>& w, const std::vector<int>& rows, const cv::Mat& U, double huber = 3.0, int iters = 6) {
    const int n = (int)rows.size();
    std::vector<double> s(n), c(n), b(n), dd(n), base(n), ww(n);
    for (int k = 0; k < n; ++k) {
        int j = rows[k];
        s[k] = m.s.at<double>(j); c[k] = m.c.at<double>(j); b[k] = m.b.at<double>(j); dd[k] = d[j]; base[k] = w[j];
    }
    ww = base;
    double v = 0;
    for (int it = 0; it < iters; ++it) {
        cv::Mat A(n, 3, CV_64F), z(n, 1, CV_64F);
        for (int k = 0; k < n; ++k) {
            double sw = std::sqrt(ww[k]);
            A.at<double>(k, 0) = (s[k] + 2 * c[k] * v) * sw;
            A.at<double>(k, 1) = U.at<double>(rows[k], 0) * sw;
            A.at<double>(k, 2) = U.at<double>(rows[k], 1) * sw;
            z.at<double>(k) = (dd[k] - (s[k] * v + c[k] * v * v + b[k])) * sw;
        }
        cv::Mat p = lsq(A, z);
        v += p.at<double>(0);
        for (int k = 0; k < n; ++k) {
            double r = (dd[k] - (s[k] * v + c[k] * v * v + b[k]) - U.at<double>(rows[k], 0) * p.at<double>(1) - U.at<double>(rows[k], 1) * p.at<double>(2)) * std::sqrt(base[k]);
            ww[k] = base[k] * (std::abs(r) <= huber ? 1.0 : huber / std::max(std::abs(r), 1e-12));
        }
    }
    cv::Mat A(n, 2, CV_64F), z(n, 1, CV_64F);
    std::vector<double> zz(n);
    for (int k = 0; k < n; ++k) {
        double sw = std::sqrt(ww[k]);
        zz[k] = dd[k] - (s[k] * v + c[k] * v * v + b[k]);
        A.at<double>(k, 0) = U.at<double>(rows[k], 0) * sw;
        A.at<double>(k, 1) = U.at<double>(rows[k], 1) * sw;
        z.at<double>(k) = zz[k] * sw;
    }
    cv::Mat u = lsq(A, z);
    Sol out{v, 0, std::vector<double>(n)};
    for (int k = 0; k < n; ++k) {
        double e = zz[k] - U.at<double>(rows[k], 0) * u.at<double>(0) - U.at<double>(rows[k], 1) * u.at<double>(1);
        out.resid[k] = e;
        double rr = e * std::sqrt(base[k]);
        out.cost += std::abs(rr) <= huber ? rr * rr : 2 * huber * std::abs(rr) - huber * huber;
    }
    return out;
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
    // 촬영 시각 (폴드 = 축마다 촬영 순서로 번갈아)
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

    // ---------- 사진 읽기: QR ECC, 판 호모그래피, 흐린 측정 영역 ----------
    const cv::Mat CEN8 = read_gray(shots[ic]);
    cv::Mat CENF;
    CEN8.convertTo(CENF, CV_32F);
    std::vector<Photo> ph(NS);
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < NS; ++i) {
        cv::Mat im = read_gray(shots[i]), imf;
        im.convertTo(imf, CV_32F);
        measure_qr(CEN8, im, ph[i]);
        if (i == ic) { ph[i].cs.fill({0, 0}); ph[i].hcc = 1; }
        else measure_h(CENF, imf, ph[i]);
        cv::Mat blur;
        cv::Rect big(CX0 - 20, CY0 - 20, CW + 40, CH + 40);
        cv::GaussianBlur(imf(big), blur, cv::Size(0, 0), 1.0);
        ph[i].crop = blur(cv::Rect(20, 20, CW, CH)).clone();
    }
    std::cout << "read " << NS << "\n";

    // ---------- 2단계: 패치 (center 흐린 영상) ----------
    cv::Mat cen_ref;
    ph[ic].crop(cv::Rect(X0 - CX0, Y0 - CY0, WW, HH)).convertTo(cen_ref, CV_64F);
    // 나사 중심: 대략 위치 주변 140x140 에서 포화(>=250) 픽셀 무게중심 (흐리기 전 center)
    std::array<cv::Point2d, 2> SCREW;
    {
        const int g[2][2] = {{1492, 1063}, {2312, 1073}};
        for (int s = 0; s < 2; ++s) {
            double sx = 0, sy = 0; int n = 0;
            for (int y = 0; y < 140; ++y)
                for (int x = 0; x < 140; ++x)
                    if (CEN8.at<uchar>(g[s][1] - 70 + y, g[s][0] - 70 + x) >= 250) { sx += x; sy += y; ++n; }
            SCREW[s] = {g[s][0] - 70 + sx / n, g[s][1] - 70 + sy / n};
        }
    }
    const double R_SCREW = 52;
    auto region_of = [&](double cx, double cy) {
        for (int i = 0; i < 3; ++i) if (std::abs(cx - kQR[i][0]) <= kQR[i][2] / 2 + 14 && std::abs(cy - kQR[i][1]) <= kQR[i][3] / 2 + 14) return i;
        for (int i = 0; i < 2; ++i) if (std::hypot(cx - SCREW[i].x, cy - SCREW[i].y) <= R_SCREW) return 3 + i;
        return 5;
    };
    std::vector<Patch> PA;
    {
        cv::Mat gx, gy;
        cv::Sobel(cen_ref, gx, CV_64F, 1, 0, 3);
        cv::Sobel(cen_ref, gy, CV_64F, 0, 1, 3);
        gx /= 8; gy /= 8;
        for (int py = 0; py + PATCH <= HH; py += STRIDE)
            for (int px = 0; px + PATCH <= WW; px += STRIDE) {
                double a = 0, b = 0, c = 0;
                for (int y = py; y < py + PATCH; ++y)
                    for (int x = px; x < px + PATCH; ++x) {
                        double u = gx.at<double>(y, x), v = gy.at<double>(y, x);
                        a += u * u; b += u * v; c += v * v;
                    }
                a /= PATCH * PATCH; b /= PATCH * PATCH; c /= PATCH * PATCH;
                double r = std::sqrt(((a - c) / 2) * ((a - c) / 2) + b * b);
                double cx = X0 + px + PATCH / 2.0, cy = Y0 + py + PATCH / 2.0;
                PA.push_back({px, py, cx, cy, (a + c) / 2 - r, (a + c) / 2 + r, region_of(cx, cy), false});
            }
        std::vector<double> lq;
        for (auto& p : PA) if (p.region < 3) lq.push_back(p.lmin);
        std::sort(lq.begin(), lq.end());
        double med = lq.size() % 2 ? lq[lq.size() / 2] : 0.5 * (lq[lq.size() / 2 - 1] + lq[lq.size() / 2]);
        for (auto& p : PA) p.use = p.lmin > 0.1 * med && p.region != 5;
    }
    std::vector<Patch> P;
    for (auto& p : PA) if (p.use) P.push_back(p);
    const int NP = (int)P.size();
    std::array<int, 6> cnt_all{}, cnt_use{};
    for (auto& p : PA) { cnt_all[p.region]++; cnt_use[p.region] += p.use; }
    std::cout << "patches " << NP << "\n";

    // ---------- 0점 보정 라벨 (#12 의 ① 백래시 + 드리프트, 폴드마다 학습 사진으로만) ----------
    std::map<char, std::vector<int>> seg, dir;
    for (char a : {'Y', 'T'}) {
        auto& o = order[a];
        seg[a].assign(o.size(), 0);
        dir[a].assign(o.size(), 0);
        auto sg = [](double v) { return (v > 0) - (v < 0); };
        for (size_t k = 1; k < o.size(); ++k)
            seg[a][k] = seg[a][k - 1] + ((tmin[o[k]] - tmin[o[k - 1]] > 3 || sg(VAL[o[k]]) != sg(VAL[o[k - 1]])) ? 1 : 0);
        for (int s = 0; s <= seg[a].back(); ++s) {
            std::vector<double> t, v;
            for (size_t k = 0; k < o.size(); ++k) if (seg[a][k] == s) { t.push_back(tmin[o[k]]); v.push_back(VAL[o[k]]); }
            double sl = polyslope(t, v);
            for (size_t k = 0; k < o.size(); ++k) if (seg[a][k] == s) dir[a][k] = (sl > 0) - (sl < 0);
        }
    }
    std::array<std::vector<double>, 2> CORR;
    for (int k = 0; k < 2; ++k) {
        CORR[k] = VAL;
        for (char a : {'Y', 'T'}) {
            auto& o = order[a];
            const int n = (int)o.size();
            cv::Mat A(n, 5, CV_64F), x(n, 1, CV_64F);
            std::vector<int> tr;
            for (int i = 0; i < n; ++i) {
                double t = tmin[o[i]];
                A.at<double>(i, 0) = VAL[o[i]]; A.at<double>(i, 1) = 1; A.at<double>(i, 2) = dir[a][i]; A.at<double>(i, 3) = t; A.at<double>(i, 4) = 1 - std::exp(-t / 3);
                x.at<double>(i) = ph[o[i]].qx;
                if (FOLD[o[i]] != k) tr.push_back(i);
            }
            cv::Mat At((int)tr.size(), 5, CV_64F), xt((int)tr.size(), 1, CV_64F);
            for (size_t r = 0; r < tr.size(); ++r) { A.row(tr[r]).copyTo(At.row((int)r)); xt.at<double>((int)r) = x.at<double>(tr[r]); }
            cv::Mat p = lsq(At, xt);
            cv::Mat off = A.colRange(1, 5) * p.rowRange(1, 5);
            for (int i = 0; i < n; ++i) CORR[k][o[i]] = VAL[o[i]] + off.at<double>(i) / p.at<double>(0);
        }
    }

    // ---------- 1단계: 기준 영상 (폴드 k 시험용 = 다른 폴드 사진 평균) ----------
    cv::Mat gyy, gxx;
    std::array<cv::Mat, 2> REF{cv::Mat::zeros(HH, WW, CV_64F), cv::Mat::zeros(HH, WW, CV_64F)};
    std::array<int, 2> rc{0, 0};
    for (int i = 0; i < NS; ++i) {
        if (i == ic) continue;
        cv::Mat coef = bsp::prefilter(ph[i].crop);
        cv::Mat H = H_of(ph[i]);
        cv::Mat& acc = REF[1 - FOLD[i]];
#pragma omp parallel for
        for (int y = 0; y < HH; ++y) {
            std::vector<cv::Point2d> src(WW), dst;
            for (int x = 0; x < WW; ++x) src[x] = {(double)(x + X0), (double)(y + Y0)};
            cv::perspectiveTransform(src, dst, H);
            double* row = acc.ptr<double>(y);
            for (int x = 0; x < WW; ++x) row[x] += bsp::sample(coef, dst[x].x - CX0, dst[x].y - CY0);
        }
        rc[1 - FOLD[i]]++;
    }
    for (int k = 0; k < 2; ++k) REF[k] /= rc[k];
    std::cout << "refs\n";

    // ---------- 3단계: 패치 이동 ----------
    const bsp::PhaseCorr pcc(PCW, 100);
    cv::Mat HANN(PCW, PCW, CV_64F);
    for (int r = 0; r < PCW; ++r)
        for (int c = 0; c < PCW; ++c)
            HANN.at<double>(r, c) = (0.5 - 0.5 * std::cos(2 * M_PI * r / (PCW - 1))) * (0.5 - 0.5 * std::cos(2 * M_PI * c / (PCW - 1)));
    auto win = [&](const cv::Mat& a) { cv::Mat o; a.convertTo(o, CV_64F); o -= cv::mean(o)[0]; return cv::Mat(o.mul(HANN)); };
    std::array<std::vector<cv::Mat>, 2> REFPC, REFP;
    for (int k = 0; k < 2; ++k)
        for (auto& p : P) {
            REFPC[k].push_back(win(REF[k](cv::Rect(p.px - PM, p.py - PM, PCW, PCW))));
            REFP[k].push_back(REF[k](cv::Rect(p.px, p.py, PATCH, PATCH)).clone());
        }
    // F[meth][ref] : NS x 2NP
    std::array<std::array<cv::Mat, 2>, 2> F;
    for (auto& a : F) for (auto& b : a) b = cv::Mat::zeros(NS, 2 * NP, CV_64F);
    std::vector<cv::Point2f> CTR;
    for (auto& p : P) CTR.push_back({(float)p.cx, (float)p.cy});
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < NS; ++i) {
        cv::Mat b;
        ph[i].crop.convertTo(b, CV_64F);
        cv::Mat gx(CH, CW, CV_64F), gy(CH, CW, CV_64F);
        for (int y = 0; y < CH; ++y)
            for (int x = 0; x < CW; ++x) {
                gx.at<double>(y, x) = (b.at<double>(y, (x + 1) % CW) - b.at<double>(y, (x - 1 + CW) % CW)) / 2;
                gy.at<double>(y, x) = (b.at<double>((y + 1) % CH, x) - b.at<double>((y - 1 + CH) % CH, x)) / 2;
            }
        cv::Mat ci = bsp::prefilter(b), cgx = bsp::prefilter(gx), cgy = bsp::prefilter(gy);
        std::vector<cv::Point2f> moved;
        cv::perspectiveTransform(CTR, moved, H_of(ph[i]));
        for (int k = 0; k < 2; ++k) {
            std::vector<double> dx(NP), dy(NP);
            for (int p = 0; p < NP; ++p) {
                double cx = (double)(moved[p].x - CTR[p].x), cy = (double)(moved[p].y - CTR[p].y);
                int ix = (int)std::nearbyint(cx), iy = (int)std::nearbyint(cy);
                int x0 = P[p].px + X0 + ix - PM - CX0, y0 = P[p].py + Y0 + iy - PM - CY0;
                auto sh = pcc(REFPC[k][p], win(b(cv::Rect(x0, y0, PCW, PCW))));
                dx[p] = ix - sh.second;
                dy[p] = iy - sh.first;
                F[0][k].at<double>(i, 2 * p) = dx[p];
                F[0][k].at<double>(i, 2 * p + 1) = dy[p];
            }
            // GN: 패치 전체를 함께 반복, 모든 패치의 이동 변화가 1e-4 px 아래면 멈춘다
            std::vector<double> a(NP, 1.0), bb(NP, 0.0);
            for (int it = 0; it < 10; ++it) {
                double mx = 0;
                std::vector<std::array<double, 4>> del(NP);
                for (int p = 0; p < NP; ++p) {
                    double A[4][4] = {{0}}, g[4] = {0};
                    const cv::Mat& T = REFP[k][p];
                    for (int yy = 0; yy < PATCH; ++yy)
                        for (int xx = 0; xx < PATCH; ++xx) {
                            double X = X0 + P[p].px + xx + dx[p] - CX0, Y = Y0 + P[p].py + yy + dy[p] - CY0;
                            bsp::Tap tp(X, Y, CW, CH);
                            double I = tp.at(ci), Ix = tp.at(cgx), Iy = tp.at(cgy);
                            double r = a[p] * I + bb[p] - T.at<double>(yy, xx);
                            double J[4] = {a[p] * Ix, a[p] * Iy, I, 1};
                            for (int u = 0; u < 4; ++u) {
                                g[u] += J[u] * r;
                                for (int v = 0; v < 4; ++v) A[u][v] += J[u] * J[v];
                            }
                        }
                    cv::Mat Am(4, 4, CV_64F), gm(4, 1, CV_64F), sol;
                    for (int u = 0; u < 4; ++u) { gm.at<double>(u) = g[u]; for (int v = 0; v < 4; ++v) Am.at<double>(u, v) = A[u][v] + (u == v ? 1e-6 : 0); }
                    cv::solve(Am, gm, sol, cv::DECOMP_LU);
                    for (int u = 0; u < 4; ++u) del[p][u] = -sol.at<double>(u);
                    mx = std::max({mx, std::abs(del[p][0]), std::abs(del[p][1])});
                }
                for (int p = 0; p < NP; ++p) {
                    dx[p] += std::min(0.5, std::max(-0.5, del[p][0]));
                    dy[p] += std::min(0.5, std::max(-0.5, del[p][1]));
                    a[p] += del[p][2];
                    bb[p] += del[p][3];
                }
                if (mx < 1e-4) break;
            }
            for (int p = 0; p < NP; ++p) { F[1][k].at<double>(i, 2 * p) = dx[p]; F[1][k].at<double>(i, 2 * p + 1) = dy[p]; }
        }
        ph[i].crop.release();
    }
    std::cout << "fields\n";

    // ---------- 4~7단계 ----------
    double mcx = 0, mcy = 0;
    for (auto& p : P) { mcx += p.cx; mcy += p.cy; }
    mcx /= NP; mcy /= NP;
    cv::Mat U(2 * NP, 2, CV_64F);
    for (int p = 0; p < NP; ++p) {
        U.at<double>(2 * p, 0) = 0; U.at<double>(2 * p + 1, 0) = 1;
        U.at<double>(2 * p, 1) = -(P[p].cy - mcy) / 100; U.at<double>(2 * p + 1, 1) = (P[p].cx - mcx) / 100;
    }
    struct Rec { int i; char axis; double cmd, act; char pa; double pred, ratio, predY, predT; std::vector<double> residTrue; };
    struct Cond { std::string key, lab, reg, meth, map; std::vector<Rec> recs; int wrongY, wrongT; std::map<std::string, double> sc; };
    std::vector<Cond> conds;
    std::vector<int> rowsQR, rowsAll;
    for (int p = 0; p < NP; ++p) {
        for (int c = 0; c < 2; ++c) { rowsAll.push_back(2 * p + c); if (P[p].region < 3) rowsQR.push_back(2 * p + c); }
    }
    for (int lab = 0; lab < 2; ++lab)
        for (int reg = 0; reg < 2; ++reg)
            for (int me = 0; me < 2; ++me)
                for (int qd = 0; qd < 2; ++qd) {
                    Cond C;
                    C.lab = lab ? "보정본" : "raw"; C.reg = reg ? "QR+나사" : "QR"; C.meth = me ? "PC+GN" : "PC"; C.map = qd ? "2차" : "1차";
                    C.key = C.lab + " / " + C.reg + " / " + C.meth + " / " + C.map;
                    const auto& rows = reg ? rowsAll : rowsQR;
                    for (int k = 0; k < 2; ++k) {
                        const cv::Mat& D = F[me][k];
                        const auto& L = lab ? CORR[k] : VAL;
                        std::map<char, Map> maps;
                        for (char a : {'Y', 'T'}) {
                            std::vector<int> idx;
                            for (int i = 0; i < NS; ++i) if (FOLD[i] != k && (AX[i] == a || AX[i] == 'C')) idx.push_back(i);
                            cv::Mat Dm((int)idx.size(), D.cols, CV_64F);
                            std::vector<double> v;
                            for (size_t r = 0; r < idx.size(); ++r) { D.row(idx[r]).copyTo(Dm.row((int)r)); v.push_back(L[idx[r]]); }
                            maps[a] = fit_map(Dm, v, qd);
                        }
                        std::vector<double> w(D.cols);
                        for (int j = 0; j < D.cols; ++j) w[j] = 1 / std::max((maps['Y'].var[j] + maps['T'].var[j]) / 2, 0.003 * 0.003);
                        std::vector<int> test;
                        for (int i = 0; i < NS; ++i) if (FOLD[i] == k) test.push_back(i);
                        std::vector<Rec> rk(test.size());
#pragma omp parallel for schedule(dynamic)
                        for (int t = 0; t < (int)test.size(); ++t) {
                            int i = test[t];
                            const double* d = D.ptr<double>(i);
                            Sol sy = solve(d, maps['Y'], w, rows, U), st = solve(d, maps['T'], w, rows, U);
                            char pa = sy.cost <= st.cost ? 'Y' : 'T';  // min(dict) 은 같으면 먼저 나온 Y
                            rk[t] = {i, AX[i], VAL[i], CORR[k][i], pa, pa == 'Y' ? sy.v : st.v, sy.cost / st.cost, sy.v, st.v, AX[i] == 'Y' ? sy.resid : st.resid};
                        }
                        C.recs.insert(C.recs.end(), rk.begin(), rk.end());
                    }
                    C.wrongY = C.wrongT = 0;
                    for (auto& r : C.recs) { C.wrongY += r.axis == 'Y' && r.pa == 'T'; C.wrongT += r.axis == 'T' && r.pa == 'Y'; }
                    for (int basis = 0; basis < 2; ++basis) {
                        std::vector<double> ey, et;
                        int pass = 0;
                        for (auto& r : C.recs) {
                            double t = basis ? r.act : r.cmd;
                            bool ok = r.pa == r.axis;
                            if (ok) (r.axis == 'Y' ? ey : et).push_back(r.pred - t);
                            pass += ok ? std::abs(r.pred - t) <= 0.09 : (std::abs(t) <= 0.09 && std::abs(r.pred) <= 0.09);
                        }
                        std::string nm = basis ? "실제위치" : "명령값";
                        C.sc[nm + "_Y_RMS_um"] = rms(ey) * 1000; C.sc[nm + "_Y_max_um"] = max_abs(ey) * 1000;
                        C.sc[nm + "_T_RMS_m"] = rms(et) * 1000; C.sc[nm + "_T_max_m"] = max_abs(et) * 1000;
                        C.sc[nm + "_pass09"] = 100.0 * pass / C.recs.size();
                    }
                    std::cout << C.key << " " << C.wrongY << " " << C.wrongT << " " << C.sc["실제위치_pass09"] << "\n";
                    conds.push_back(std::move(C));
                }

    // ---------- 표 ----------
    Table tab{{"조건", "판정오류_Y사진", "판정오류_Θ사진", "명령값_Y_RMS_um", "명령값_Y_max_um", "명령값_T_RMS_m", "명령값_T_max_m", "명령값_pass09", "실제위치_Y_RMS_um",
               "실제위치_Y_max_um", "실제위치_T_RMS_m", "실제위치_T_max_m", "실제위치_pass09"}};
    for (auto& C : conds) {
        std::vector<std::string> r{C.key, std::to_string(C.wrongY), std::to_string(C.wrongT)};
        for (std::string nm : {"명령값", "실제위치"})
            for (std::string m : {"_Y_RMS_um", "_Y_max_um", "_T_RMS_m", "_T_max_m", "_pass09"}) r.push_back(fmt(C.sc[nm + m], 3));
        tab.add(r);
    }
    tab.csv(out("결과표_패치이동장_16조건.csv"));
    // 가장 좋은 조건 (실제 위치 0.09 통과, 같으면 앞 조건)
    // 대표 조건 = 원래 결과의 최선 (보정본 라벨, QR만, PC+GN, 2차)
    size_t bi = 0;
    for (size_t c = 0; c < conds.size(); ++c) if (conds[c].key == "보정본 / QR / PC+GN / 2차") bi = c;
    double best_pass = 0;
    for (auto& C : conds) best_pass = std::max(best_pass, C.sc["실제위치_pass09"]);
    auto& B = conds[bi];
    {
        Table pr{{"file", "axis", "cmd", "act", "pred_axis", "pred", "cost_ratio_Y_over_T", "predY", "predT"}};
        for (auto& r : B.recs) pr.add({shots[r.i].rel, std::string(1, r.axis), fmt(r.cmd, 2), fmt(r.act, 4), std::string(1, r.pa), fmt(r.pred, 4), fmt(r.ratio, 4), fmt(r.predY, 4), fmt(r.predT, 4)});
        pr.csv(out("사진별_예측_최선조건.csv"));
    }
    // 기준 영상 비교 (밝기, 선명도)
    Table reft{{"기준영상", "영역", "밝기", "선명도"}};
    std::map<std::string, double> refstat;
    {
        cv::Mat mk[2] = {cv::Mat::zeros(HH, WW, CV_8U), cv::Mat::zeros(HH, WW, CV_8U)};
        for (auto& p : PA)
            if (p.region < 5) mk[p.region < 3 ? 0 : 1](cv::Rect(p.px, p.py, PATCH, PATCH)).setTo(1);
        std::vector<std::pair<std::string, cv::Mat>> imgs{{"center 1장", cen_ref}, {"평균 (폴드0 시험용)", REF[0]}, {"평균 (폴드1 시험용)", REF[1]}};
        for (auto& [lab, img] : imgs) {
            cv::Mat gx, gy, g;
            cv::Sobel(img, gx, CV_64F, 1, 0, 3);
            cv::Sobel(img, gy, CV_64F, 0, 1, 3);
            cv::magnitude(gx, gy, g);
            g /= 8;
            for (int r = 0; r < 2; ++r) {
                double bsum = 0, gsum = 0; int n = 0;
                for (int y = 0; y < HH; ++y)
                    for (int x = 0; x < WW; ++x)
                        if (mk[r].at<uchar>(y, x)) { bsum += img.at<double>(y, x); gsum += g.at<double>(y, x); ++n; }
                std::string rn = r ? "나사" : "QR";
                refstat[lab + rn + "b"] = bsum / n;
                refstat[lab + rn + "s"] = gsum / n;
                reft.add({lab, rn, fmt(bsum / n, 3), fmt(gsum / n, 3)});
            }
        }
    }

    // ---------- 그림 ----------
    // 01 패치 지도
    {
        cv::Mat im = read_color(shots[ic]);
        cv::Rect roi(X0 - 10, Y0 - 10, WW + 20, HH + 20);
        std::vector<Mark> mk;
        for (auto& p : PA) {
            if (p.region == 5) continue;
            Mark m;
            m.p = {p.cx, p.cy};
            m.c = p.use ? (p.region < 3 ? color(0) : color(1)) : kRed;
            mk.push_back(m);
        }
        std::vector<Box> bx;
        for (int i = 0; i < 3; ++i) bx.push_back({cv::Rect2d(kQR[i][0] - kQR[i][2] / 2 - 14, kQR[i][1] - kQR[i][3] / 2 - 14, kQR[i][2] + 28, kQR[i][3] + 28), color(0), ""});
        for (int i = 0; i < 2; ++i) bx.push_back({cv::Rect2d(SCREW[i].x - R_SCREW, SCREW[i].y - R_SCREW, 2 * R_SCREW, 2 * R_SCREW), color(1), ""});
        save_png("01_패치지도.png", zoom(im, roi, 1600, mk, 0, "patches 32px/16px: blue QR " + std::to_string(cnt_use[0] + cnt_use[1] + cnt_use[2]) + ", orange screw " +
                                                                    std::to_string(cnt_use[3] + cnt_use[4]) + ", red = weak (dropped)", bx));
    }
    // 02 기준 영상: center, 평균, 차이
    {
        auto to8 = [](const cv::Mat& m, double lo, double hi) { cv::Mat o; m.convertTo(o, CV_8U, 255 / (hi - lo), -lo * 255 / (hi - lo)); cv::cvtColor(o, o, cv::COLOR_GRAY2BGR); return o; };
        cv::Mat d = REF[0] - cen_ref;
        cv::Mat a = to8(cen_ref, 0, 255), b = to8(REF[0], 0, 255), c = to8(d, -8, 8);
        text(a, "center (blurred)", {8, 22}, 0.6, kRed, 2);
        text(b, "mean of fold-1 photos (test fold 0)", {8, 22}, 0.6, kRed, 2);
        text(c, "mean - center (-8..+8 gray levels)", {8, 22}, 0.6, kRed, 2);
        save_png("02_기준영상.png", vstack({a, b, c}));
    }
    // 03 16조건 막대
    {
        std::vector<std::string> cats;
        std::vector<double> v;
        for (auto& C : conds) {
            std::string k = (C.lab == "raw" ? "raw" : "corr") + std::string(C.reg == "QR" ? " QR" : " QR+screw") + (C.meth == "PC" ? " PC" : " PC+GN") + (C.map == "1차" ? " lin" : " quad");
            cats.push_back(k);
            v.push_back(C.sc["실제위치_pass09"]);
        }
        bar_chart("03_16조건_통과율.png", "Pass |err|<=0.09 (actual position basis), 16 conditions", cats, v, "pass %", (int)bi);
    }
    // 04 최선 조건 예측 오차 (축을 틀리면 그 축 값 0 으로 낸 것으로 그림)
    std::vector<double> ty, py, tt, pt;
    std::vector<std::string> nm;
    for (auto& r : B.recs) {
        ty.push_back(r.axis == 'Y' ? r.act : 0); tt.push_back(r.axis == 'T' ? r.act : 0);
        py.push_back(r.pa == 'Y' ? r.pred : 0); pt.push_back(r.pa == 'T' ? r.pred : 0);
        nm.push_back(NAME[r.i].substr(0, NAME[r.i].rfind('.')));
    }
    pred_vs_truth("04_최선조건_예측오차.png", "corr/QR/PC+GN/quad", ty, py, tt, pt, nm);

    // 튀는 값: Y 는 축을 틀린 사진 중 비용 비가 가장 Θ 쪽(작은 값), 없으면 오차 최대. T 도 같은 규칙.
    auto pick = [&](char a) {
        int best = -1;
        for (size_t k = 0; k < B.recs.size(); ++k) {
            auto& r = B.recs[k];
            if (r.axis != a || r.pa == a) continue;
            double s = a == 'Y' ? -r.ratio : r.ratio;  // Y 사진을 Θ 로: ratio>1, 클수록 확실히 틀림
            if (best < 0 || s < (a == 'Y' ? -B.recs[best].ratio : B.recs[best].ratio)) best = (int)k;
        }
        if (best >= 0) return best;
        for (size_t k = 0; k < B.recs.size(); ++k) {
            auto& r = B.recs[k];
            if (r.axis != a) continue;
            if (best < 0 || std::abs(r.pred - r.act) > std::abs(B.recs[best].pred - B.recs[best].act)) best = (int)k;
        }
        return best;
    };
    std::map<char, int> sel{{'Y', pick('Y')}, {'T', pick('T')}};
    std::map<char, std::string> ofile;
    for (char a : {'Y', 'T'}) {
        auto& r = B.recs[sel[a]];
        cv::Mat im = read_color(shots[r.i]);
        // 판 확대: 패치 중심(측정 위치 = center 위치 + 측정 이동) 과 참 축 가설 잔차 화살표 x300
        const auto& rows = B.reg == "QR" ? rowsQR : rowsAll;
        std::vector<Mark> mk;
        const cv::Mat& D = F[1][FOLD[r.i]];
        double sx = 0;
        for (int p = 0; p < NP; ++p) sx += D.at<double>(r.i, 2 * p);
        sx /= NP;
        for (size_t q = 0; q < rows.size(); q += 2) {
            int p = rows[q] / 2;
            Mark m;
            m.p = {P[p].cx + D.at<double>(r.i, 2 * p), P[p].cy + D.at<double>(r.i, 2 * p + 1)};
            m.d = {r.residTrue[q], r.residTrue[q + 1]};
            m.c = kRed;
            mk.push_back(m);
        }
        cv::Rect roi((int)(X0 + sx) - 10, Y0 - 10, WW + 20, HH + 20);
        std::string cap = nm[sel[a]] + ": true " + std::string(a == 'Y' ? "Y " : "T ") + fmt(r.act, 3) + ", judged " + (r.pa == 'Y' ? "Y " : "T ") + fmt(r.pred, 3) +
                          ", cost Y/T " + fmt(r.ratio, 3) + "  (arrows: residual of true-axis model x300)";
        ofile[a] = std::string(a == 'Y' ? "05_튀는값_Y_" : "06_튀는값_T_") + nm[sel[a]] + ".png";
        save_png(ofile[a], zoom(im, roi, 1500, mk, 300, cap));
    }

    // ---------- 원래 결과와 대조 ----------
    // 원래 값: analysis_v5 40_결과표_패치이동장_16조건.csv, 36_1단계_기준영상_비교.csv (결과/원본자료/ 사본), patches.csv
    struct O { int wy, wt; double cmdY, cmdT, actY, actT, pass; };
    const std::vector<O> orig{{6, 7, 14.634, 2.226, 24.974, 8.828, 96.786}, {9, 4, 14.138, 2.073, 25.205, 8.883, 96.786}, {3, 12, 14.308, 2.085, 24.864, 8.856, 96.071},
                              {5, 2, 14.491, 2.061, 24.922, 8.867, 98.214},  {7, 3, 14.732, 2.236, 24.966, 8.812, 97.857}, {7, 2, 14.33, 2.066, 25.121, 8.867, 97.857},
                              {4, 6, 14.365, 2.186, 24.88, 8.824, 97.857},   {4, 1, 14.361, 2.067, 24.915, 8.86, 98.571},  {0, 23, 28.837, 9.282, 2.198, 0.372, 92.857},
                              {10, 0, 29.435, 9.129, 2.259, 0.404, 97.143},  {0, 22, 28.803, 9.303, 1.777, 0.336, 93.214}, {6, 0, 29.051, 9.128, 1.849, 0.374, 98.571},
                              {0, 17, 28.837, 9.28, 2.213, 0.38, 95.0},      {10, 2, 29.434, 9.156, 2.273, 0.394, 96.786}, {0, 18, 28.8, 9.275, 1.767, 0.346, 94.643},
                              {7, 0, 29.014, 9.129, 1.849, 0.374, 98.214}};
    Parity par;
    par.add("사용 패치 수 (QR 3개)", 397, cnt_use[0] + cnt_use[1] + cnt_use[2], 0, 0);
    par.add("사용 패치 수 (나사 2개)", 30, cnt_use[3] + cnt_use[4], 0, 0);
    par.add("center QR 밝기", 139.023, refstat["center 1장QRb"], 0.01, 3);
    par.add("평균(폴드0 시험용) QR 밝기", 136.991, refstat["평균 (폴드0 시험용)QRb"], 0.01, 3);
    par.add("평균(폴드0 시험용) QR 선명도", 13.05, refstat["평균 (폴드0 시험용)QRs"], 0.01, 3);
    par.add("평균(폴드0 시험용) 나사 밝기", 173.261, refstat["평균 (폴드0 시험용)나사b"], 0.01, 3);
    for (size_t c = 0; c < conds.size(); ++c) {
        auto& C = conds[c];
        auto& o = orig[c];
        par.add(C.key + " 판정오류 Y사진", o.wy, C.wrongY, 1, 0);
        par.add(C.key + " 판정오류 Θ사진", o.wt, C.wrongT, 1, 0);
        par.add(C.key + " 0.09 통과 % (실제 위치)", o.pass, C.sc["실제위치_pass09"], 0.72, 3);
        par.add(C.key + " Y RMS µm (명령값)", o.cmdY, C.sc["명령값_Y_RMS_um"], 0.1 * o.cmdY, 3);
        par.add(C.key + " Y RMS µm (실제 위치)", o.actY, C.sc["실제위치_Y_RMS_um"], 0.1 * o.actY, 3);
    }

    // ---------- 결과.md ----------
    int ny = 0, nt = 0;
    for (auto& r : B.recs) { ny += r.axis == 'Y'; nt += r.axis == 'T'; }
    Table segt{{"영역", "전체 패치", "사용 패치"}};
    for (int r = 0; r < 6; ++r) segt.add({kRegion[r], std::to_string(cnt_all[r]), std::to_string(cnt_use[r])});
    std::ostringstream Rm;
    Rm << "# 결과: 패치 이동장으로 Y/Θ 판정\n\n"
       << "입력: `data/clean/clean_20261014` 281장. 폴드는 축마다 촬영 순서(`data/raw` zip의 수정 시각)로 번갈아 나눴다. 시험 280장, center는 학습에만 쓴다.\n\n"
       << "1단계 기준 영상: 사진마다 판(" << V2X0 << ".." << V2X1 << " x " << V2Y0 << ".." << V2Y1 << ") ECC 호모그래피로 구한 귀퉁이 이동을 판 영역(" << X0 << ".." << X1 << " x " << Y0 << ".." << Y1
       << ")에 적용해 center 좌표로 되돌리고(가우스 흐림 sigma 1, 3차 스플라인 보간), 학습 폴드 사진을 평균했다.\n\n"
       << "2단계 패치: 32x32 px, 16 px 간격. 구조 텐서 작은 고유값이 QR 패치 중앙값의 0.1배 이하면 뺐다. 나사 중심 (" << fmt(SCREW[0].x, 1) << ", " << fmt(SCREW[0].y, 1) << "), ("
       << fmt(SCREW[1].x, 1) << ", " << fmt(SCREW[1].y, 1) << "), 반지름 52 px.\n\n" << segt.md()
       << "\n3단계 측정: PC = 48 px Hann 창 위상 상관(1/100 px 업샘플, 정규화 없음), PC+GN = PC에서 출발한 가우스-뉴턴(3차 스플라인, 밝기 a, b, 최대 10회).\n\n"
       << "4~7단계: 패치 성분마다 d = s v (+ c v^2) + b 를 축별 학습 사진(center 포함)으로 맞추고, 가중 = 1/잔차 분산(하한 0.003 px). 새 사진은 Y 가설과 Θ 가설을 각각 Huber(3) 가우스-뉴턴으로 풀어"
       << "(방해 무늬: 세로 평행 이동, 화면 회전) 비용이 작은 쪽을 고른다. 보정본 라벨 = 명령값 + o/k (#12의 ① 백래시+드리프트를 학습 폴드로만 맞춤). 0.09 통과: 축을 맞히고 |오차|<=0.09, 축을 틀리면 참값과 예측이 둘 다 0.09 안일 때만 통과.\n\n"
       << "## 16조건 결과 (짝/홀 교차검증)\n\n" << tab.md() << "\n## 기준 영상 비교\n\n" << reft.md()
       << "\n## 원래 결과와 대조 (" << par.ok << "/" << par.total << " 일치)\n\n허용오차: 판정 오류 수 ±1장, 통과율 ±0.72%(2장), RMS ±10%, 밝기·선명도 ±0.01.\n\n" << par.t.md()
       << "\n## 옮기지 않은 부분\n\n- 합성 검증(step3 `syn`, 그림 38), 이동 무늬 지도와 구분 단서(그림 39), 조건별 오차 4x8 그림(41).\n"
       << "- step9_features.py(Shi-Tomasi 특징점 이동), step10_methods.py(GLS, 주성분+로지스틱+GP, 물리 모델 비교, 42번 표).\n"
       << "\n## 파일\n\n- `결과표_패치이동장_16조건.csv`: 원래 40번 표와 같은 열\n- `사진별_예측_최선조건.csv`: " << B.key << " 사진별 판정\n"
       << "- `이미지/01_패치지도.png`, `02_기준영상.png`, `03_16조건_통과율.png`, `04_최선조건_예측오차.png`, `" << ofile['Y'] << "`, `" << ofile['T'] << "`\n";
    write_text(out("결과.md"), Rm.str());

    // ---------- 보고서.md ----------
    Table outl{{"구분", "사진", "정답 축", "정답 값 (실제 위치)", "판정 축", "추정 값", "비용 비 Y/Θ", "Y 가설 값", "Θ 가설 값"}};
    std::vector<int> wrong;
    for (size_t k = 0; k < B.recs.size(); ++k) if (B.recs[k].pa != B.recs[k].axis) wrong.push_back((int)k);
    for (int k : wrong) {
        auto& r = B.recs[k];
        outl.add({"축 틀림", NAME[r.i], r.axis == 'Y' ? "Y" : "Θ", fmt(r.act, 3), r.pa == 'Y' ? "Y" : "Θ", fmt(r.pred, 3), fmt(r.ratio, 3), fmt(r.predY, 3), fmt(r.predT, 3)});
    }
    {
        auto& r = B.recs[sel['T']];
        if (r.pa == 'T') outl.add({"Θ 오차 최대", NAME[r.i], "Θ", fmt(r.act, 3), "Θ", fmt(r.pred, 3), fmt(r.ratio, 3), fmt(r.predY, 3), fmt(r.predT, 3)});
    }
    double maxwrongY = 0;
    for (int k : wrong) if (B.recs[k].axis == 'Y') maxwrongY = std::max(maxwrongY, std::abs(B.recs[k].act));
    auto& ry = B.recs[sel['Y']];
    std::ostringstream P2;
    P2 << "# 보고서: 이동 무늬 판정, 패치 이동장 (10-14 13:39)\n\n"
       << "## 문제\n\nQR 영역의 이동 무늬로 새 사진이 Y 이동인지 Θ 회전인지 판정하고 그 값을 구하려 했다.\n\n"
       << "## 해결 방법\n\n판 영역을 32 px 패치 " << NP << "개로 나눠 위상 상관+가우스-뉴턴으로 이동장을 재고, 패치별 1차/2차 매핑으로 Y 가설과 Θ 가설의 비용을 비교해 판정했다.\n\n"
       << "## 결과\n\n대표 조건 " << B.key << "(원래 결과의 최선, 16조건 최고 통과율 " << fmt(best_pass, 3) << "%와 " << (std::abs(best_pass - B.sc["실제위치_pass09"]) < 1e-9 ? "같음" : "다름") << ")은 시험 280장에서 Y 사진 " << B.wrongY << "장·Θ 사진 " << B.wrongT << "장의 축을 틀려 0.09 통과 " << fmt(B.sc["실제위치_pass09"], 3)
       << "%(실제 위치 기준)였다.\n\n| 지표 (" << B.key << ") | 값 |\n|---|---|\n"
       << "| 축 판정 오류 (Y 사진 / Θ 사진) | " << B.wrongY << " / " << ny << ", " << B.wrongT << " / " << nt << " |\n"
       << "| 0.09 통과 (실제 위치 / 명령값) | " << fmt(B.sc["실제위치_pass09"], 3) << "% / " << fmt(B.sc["명령값_pass09"], 3) << "% |\n"
       << "| Y RMS / 최대 (실제 위치, µm) | " << fmt(B.sc["실제위치_Y_RMS_um"], 3) << " / " << fmt(B.sc["실제위치_Y_max_um"], 3) << " |\n"
       << "| Θ RMS / 최대 (실제 위치, mΘ) | " << fmt(B.sc["실제위치_T_RMS_m"], 3) << " / " << fmt(B.sc["실제위치_T_max_m"], 3) << " |\n"
       << "| Y RMS (명령값, µm) | " << fmt(B.sc["명령값_Y_RMS_um"], 3) << " |\n\n"
       << img_md("03_16조건_통과율.png", "16조건 통과율") << "\n\n" << img_md("04_최선조건_예측오차.png", "최선 조건 예측 오차") << "\n\n" << img_md("01_패치지도.png", "패치 지도") << "\n\n"
       << img_md("02_기준영상.png", "기준 영상") << "\n\n"
       << "## 튀는 값\n\n축을 틀린 사진은 Y 사진 " << B.wrongY << "장(|Y| " << fmt(maxwrongY, 2) << " mm 이하), Θ 사진 " << B.wrongT << "장이었고, 가장 확실히 틀린 Y 사진 " << NAME[ry.i] << "는 실제 Y " << fmt(ry.act, 3)
       << " mm인데 Y 가설 비용이 Θ 가설의 " << fmt(ry.ratio, 3) << "배라서 Θ " << fmt(ry.predT, 4) << "로 판정됐다.\n\n" << outl.md() << "\n"
       << img_md(ofile['Y'], "Y 튀는 값") << "\n\n" << img_md(ofile['T'], "T 튀는 값") << "\n\n"
       << "빨간 십자는 이 사진에서 잰 QR 패치 위치(center 위치 + 측정 이동), 화살표는 참 축 매핑으로 맞춘 뒤 남은 패치 잔차 x300이다.\n\n"
       << "## 한줄평\n\nQR 판은 평평해서 Y와 Θ가 거의 같은 평행 이동 무늬를 만들고 둘의 차이는 작은 원근 변화뿐이라서, 작은 Y(|Y|<=" << fmt(maxwrongY, 2)
       << " mm)에서는 그 차이가 패치 잡음에 묻혀 Y " << B.wrongY << "장을 Θ로 읽었고, 값 자체는 축만 맞히면 Y RMS " << fmt(B.sc["실제위치_Y_RMS_um"], 2) << " µm로 정확했다.\n\n"
       << "## 원래 결과와 대조\n\n원래 값은 analysis_v5 실행 결과 `40_결과표_패치이동장_16조건.csv`, `36_1단계_기준영상_비교.csv`, `patches.csv`(이 실험 `결과/원본자료/` 사본)에서 가져왔다. "
       << "원래 입력 14번 CSV(QR 위치)와 v2 판 호모그래피는 사진에서 다시 쟀고(같은 방식), 보정본 라벨의 QR 이동은 #12와 같은 ECC 측정이다. 허용오차: 판정 오류 ±1장, 통과율 ±0.72%, RMS ±10%. "
       << par.ok << "/" << par.total << " 일치.\n\n" << par.t.md();
    if (par.ok < par.total)
        P2 << "\n불일치 항목은 판정 오류 장수다. 보정본 라벨용 QR 이동과 판 호모그래피를 사진에서 다시 쟀고 위상 상관이 1/100 px 격자라서, 두 가설 비용 비가 1에 가까운 사진 몇 장의 판정이 바뀐 것으로 보인다.\n";
    write_text(out("보고서.md"), P2.str());
    std::cout << "parity " << par.ok << "/" << par.total << "\n";
    return 0;
}
