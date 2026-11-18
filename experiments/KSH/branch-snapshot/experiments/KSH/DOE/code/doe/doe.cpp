// =====================================================================================
// doe.cpp — 실험계획 1~3 (설명은 doe.hpp)
// =====================================================================================
#include "doe.hpp"
#include "../yt/m1_phase_ecc.hpp"
#include "../roi/roi_opt.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>

namespace yt {
namespace doe {

namespace {

// ---------------------------------------------------------------- 평가 허용 범위 (과제의 절반 = 프로젝트 기본값)
constexpr double kTolY_um = 45.0, kTolT_deg = 0.005;
constexpr double kSpecY_um = 90.0, kSpecT_deg = 0.01;   // 과제 원래 값 (비교용)

// =====================================================================================
// 보간 커널
// =====================================================================================
enum Kern { K_LINEAR = 0, K_KEYS05, K_KEYS075, K_LANCZOS3, K_BSPLINE3, K_OMOMS3, K_COUNT, K_OPENCV_ECC = 100 };

const char* kernName(int k) {
    switch (k) {
    case K_LINEAR: return "선형 (bilinear)";
    case K_KEYS05: return "Bicubic Keys a=-0.5";
    case K_KEYS075: return "Bicubic OpenCV a=-0.75";
    case K_LANCZOS3: return "Lanczos3";
    case K_BSPLINE3: return "3차 B-스플라인";
    case K_OMOMS3: return "3차 O-MOMS";
    case K_OPENCV_ECC: return "OpenCV ECC 내장 (선형 + 5x5 가우시안)";
    default: return "?";
    }
}
const char* kernKey(int k) {
    switch (k) {
    case K_LINEAR: return "linear";
    case K_KEYS05: return "keys05";
    case K_KEYS075: return "keys075";
    case K_LANCZOS3: return "lanczos3";
    case K_BSPLINE3: return "bspline3";
    case K_OMOMS3: return "omoms3";
    case K_OPENCV_ECC: return "opencv_ecc";
    default: return "?";
    }
}
int kernRadius(int k) { return k == K_LINEAR ? 1 : (k == K_LANCZOS3 ? 3 : 2); }

double keysW(double t, double a) {
    t = std::abs(t);
    if (t < 1.0) return ((a + 2.0) * t - (a + 3.0)) * t * t + 1.0;
    if (t < 2.0) return ((a * t - 5.0 * a) * t + 8.0 * a) * t - 4.0 * a;
    return 0.0;
}
double sinc(double x) { return std::abs(x) < 1e-12 ? 1.0 : std::sin(CV_PI * x) / (CV_PI * x); }

double kernW(int k, double t) {
    const double a = std::abs(t);
    switch (k) {
    case K_LINEAR: return a < 1.0 ? 1.0 - a : 0.0;
    case K_KEYS05: return keysW(t, -0.5);
    case K_KEYS075: return keysW(t, -0.75);
    case K_LANCZOS3: return a < 3.0 ? sinc(t) * sinc(t / 3.0) : 0.0;
    case K_BSPLINE3:
        if (a < 1.0) return (4.0 - 6.0 * a * a + 3.0 * a * a * a) / 6.0;
        if (a < 2.0) return (2.0 - a) * (2.0 - a) * (2.0 - a) / 6.0;
        return 0.0;
    case K_OMOMS3:   // Blu, Thévenaz, Unser 2001
        if (a < 1.0) return 0.5 * a * a * a - a * a + a / 14.0 + 13.0 / 21.0;
        if (a < 2.0) return -a * a * a / 6.0 + a * a - 85.0 / 42.0 * a + 29.0 / 21.0;
        return 0.0;
    default: return 0.0;
    }
}
// 사전 필터가 필요한 (보간점을 지나지 않는) 커널의 극점
double kernPole(int k) {
    if (k == K_BSPLINE3) return std::sqrt(3.0) - 2.0;
    if (k == K_OMOMS3) return (std::sqrt(105.0) - 13.0) / 8.0;
    return 0.0;
}

// 1차원 재귀 사전 필터 (Unser 1993, 거울 경계) — 이득 (1 − z)(1 − 1/z)
void prefilter1D(double* c, int n, double z) {
    const double g = (1.0 - z) * (1.0 - 1.0 / z);
    for (int k = 0; k < n; ++k) c[k] *= g;
    const int horizon = std::min(n, 40);
    double zn = z, sum = c[0];
    for (int k = 1; k < horizon; ++k) { sum += zn * c[k]; zn *= z; }
    c[0] = sum;
    for (int k = 1; k < n; ++k) c[k] += z * c[k - 1];
    c[n - 1] = (z / (z * z - 1.0)) * (z * c[n - 2] + c[n - 1]);
    for (int k = n - 2; k >= 0; --k) c[k] = z * (c[k + 1] - c[k]);
}

// 영역(CV_64F)을 커널 계수로 : 보간점을 지나는 커널은 그대로, B-스플라인·O-MOMS 는 사전 필터
void toCoeffs(cv::Mat& c, int k) {
    const double z = kernPole(k);
    if (z == 0.0) return;
    for (int y = 0; y < c.rows; ++y) prefilter1D(c.ptr<double>(y), c.cols, z);
    std::vector<double> col(c.rows);
    for (int x = 0; x < c.cols; ++x) {
        for (int y = 0; y < c.rows; ++y) col[y] = c.at<double>(y, x);
        prefilter1D(col.data(), c.rows, z);
        for (int y = 0; y < c.rows; ++y) c.at<double>(y, x) = col[y];
    }
}

bool inside(const cv::Rect& r, const cv::Size& s) { return r.x >= 0 && r.y >= 0 && r.x + r.width <= s.width && r.y + r.height <= s.height; }
cv::Point iround(const cv::Point2d& p) { return cv::Point(int(std::lround(p.x)), int(std::lround(p.y))); }

// 영역 잘라 CV_64F + (σ > 0 이면) 가우시안 사전 필터
cv::Mat cropBlur(const cv::Mat& img32, const cv::Rect& r, double sigma) {
    cv::Mat c;
    img32(r).convertTo(c, CV_64F);
    if (sigma > 0) {
        const int ks = 2 * int(std::ceil(3.0 * sigma)) + 1;
        cv::GaussianBlur(c, c, cv::Size(ks, ks), sigma, sigma, cv::BORDER_REFLECT);
    }
    return c;
}

struct Fine { int kern = K_OPENCV_ECC; double sigma = 0.0; };

struct TRes { cv::Point2d p; double zncc = 0; int it = 0; bool ok = false; };

// =====================================================================================
// 평행이동 IC-GN (ZNSSD) — 기준 영역 T (기준 영상 좌표) 가 현재 영상에서 init 만큼 옮겨졌다고 보고 다듬음
// =====================================================================================
TRes icgnT(const cv::Mat& ref, const cv::Mat& cur, const cv::Rect& T, const cv::Point2d& init, const Fine& fc) {
    TRes r;
    const int R = kernRadius(fc.kern);
    const int M = 28;   // 사전 필터 경계 영향 + 반복 중 이동 여유
    // 기준 : T 주변 2 px 여유 (5점 미분)
    const cv::Rect Tr(T.x - 4, T.y - 4, T.width + 8, T.height + 8);
    if (!inside(Tr, ref.size())) return r;
    const cv::Mat F = cropBlur(ref, Tr, fc.sigma);
    // 현재 : 예상 위치 주변
    const cv::Point gi = iround(init);
    const cv::Rect Rc(T.x + gi.x - M, T.y + gi.y - M, T.width + 2 * M, T.height + 2 * M);
    if (!inside(Rc, cur.size())) return r;
    cv::Mat C = cropBlur(cur, Rc, fc.sigma);
    toCoeffs(C, fc.kern);

    const int W = T.width, H = T.height, n = W * H;
    std::vector<double> f(n), gx(n), gy(n);
    double fm = 0;
    for (int j = 0; j < H; ++j)
        for (int i = 0; i < W; ++i) {
            const int y = j + 4, x = i + 4;
            const double* row = F.ptr<double>(y);
            f[j * W + i] = row[x];
            fm += row[x];
            gx[j * W + i] = (-row[x + 2] + 8 * row[x + 1] - 8 * row[x - 1] + row[x - 2]) / 12.0;
            gy[j * W + i] = (-F.at<double>(y + 2, x) + 8 * F.at<double>(y + 1, x) - 8 * F.at<double>(y - 1, x) + F.at<double>(y - 2, x)) / 12.0;
        }
    fm /= n;
    double df = 0;
    for (int k = 0; k < n; ++k) { f[k] -= fm; df += f[k] * f[k]; }
    df = std::sqrt(df);
    if (df < 1e-9) return r;
    double h00 = 0, h01 = 0, h11 = 0;
    for (int k = 0; k < n; ++k) {
        gx[k] /= df; gy[k] /= df; f[k] /= df;
        h00 += gx[k] * gx[k]; h01 += gx[k] * gy[k]; h11 += gy[k] * gy[k];
    }
    const double det = h00 * h11 - h01 * h01;
    if (std::abs(det) < 1e-18) return r;

    cv::Point2d p = init;
    std::vector<double> g(n), tmp;
    std::vector<double> wx(2 * R), wy(2 * R);
    for (int it = 0; it < 60; ++it) {
        const double ux = std::floor(p.x), uy = std::floor(p.y);
        const double fx = p.x - ux, fy = p.y - uy;
        for (int m = 0; m < 2 * R; ++m) { wx[m] = kernW(fc.kern, fx - (m - R + 1)); wy[m] = kernW(fc.kern, fy - (m - R + 1)); }
        // 현재 영상에서 기준 화소 (T.x + i, T.y + j) + p 위치 → 계수 영상 좌표
        const int bx = T.x + int(ux) - Rc.x, by = T.y + int(uy) - Rc.y;
        if (bx - R + 1 < 0 || by - R + 1 < 0 || bx + W - 1 + R >= C.cols || by + H - 1 + R >= C.rows) return r;
        // 분리형 보간 : 가로 먼저 (필요한 행 H + 2R − 1 개), 그다음 세로
        const int rows = H + 2 * R - 1;
        tmp.assign(size_t(rows) * W, 0.0);
        for (int q = 0; q < rows; ++q) {
            const double* row = C.ptr<double>(by - R + 1 + q);
            double* t = &tmp[size_t(q) * W];
            for (int i = 0; i < W; ++i) {
                double a = 0;
                const double* src = row + bx + i - R + 1;
                for (int m = 0; m < 2 * R; ++m) a += wx[m] * src[m];
                t[i] = a;
            }
        }
        double gm = 0;
        for (int j = 0; j < H; ++j)
            for (int i = 0; i < W; ++i) {
                double s = 0;
                for (int q = 0; q < 2 * R; ++q) s += wy[q] * tmp[size_t(j + q) * W + i];
                g[j * W + i] = s;
                gm += s;
            }
        gm /= n;
        double dg = 0;
        for (int k = 0; k < n; ++k) { g[k] -= gm; dg += g[k] * g[k]; }
        dg = std::sqrt(dg);
        if (dg < 1e-9) return r;
        double b0 = 0, b1 = 0, ss = 0;
        for (int k = 0; k < n; ++k) {
            const double e = g[k] / dg - f[k];
            b0 += gx[k] * e; b1 += gy[k] * e; ss += e * e;
        }
        const double dx = (h11 * b0 - h01 * b1) / det, dy = (h00 * b1 - h01 * b0) / det;
        p.x -= dx; p.y -= dy;
        r.zncc = 1.0 - 0.5 * ss;
        r.it = it + 1;
        if (std::abs(dx) < 1e-5 && std::abs(dy) < 1e-5) break;
        if (cv::norm(p - init) > M - R - 2) return r;
    }
    r.p = p;
    r.ok = std::isfinite(p.x) && std::isfinite(p.y);
    return r;
}

// Fine Alignment(정밀 정렬) 하나 (OpenCV ECC 또는 직접 구현 IC-GN)
TRes fineAlign(const cv::Mat& ref, const cv::Mat& cur, const cv::Rect& T, const cv::Point2d& init, const Fine& fc) {
    if (fc.kern != K_OPENCV_ECC) return icgnT(ref, cur, T, init, fc);
    TRes r;
    const cv::Point gi = iround(init);
    const cv::Rect C = T + gi;
    if (!inside(T, ref.size()) || !inside(C, cur.size())) return r;
    const m1::Shift s = m1::eccTranslation(ref(T), cur(C), init - cv::Point2d(gi));
    if (!s.ok) return r;
    r.p = s.d + cv::Point2d(gi);
    r.zncc = s.cc;
    r.ok = true;
    return r;
}

// =====================================================================================
// 정렬 순서 (실험 1)
// =====================================================================================
enum Order { O_COARSE = 0, O_FINE, O_COARSE_FINE, O_FINE_COARSE, O_COUNT };
const char* orderName(int o) {
    switch (o) {
    case O_COARSE: return "Coarse only (위상 상관만)";
    case O_FINE: return "Fine only (ECC 만, 초기값 0)";
    case O_COARSE_FINE: return "Coarse -> Fine (위상 상관 후 ECC)";
    case O_FINE_COARSE: return "Fine -> Coarse (ECC 후 위상 상관)";
    default: return "?";
    }
}
const char* orderKey(int o) {
    switch (o) {
    case O_COARSE: return "coarse_only";
    case O_FINE: return "fine_only";
    case O_COARSE_FINE: return "coarse_fine";
    case O_FINE_COARSE: return "fine_coarse";
    default: return "?";
    }
}

// ROI 단위 Coarse : QR = 플레이트 Coarse Alignment 위치에서 ROI 창 위상 상관, 벽 = 2배 창 위상 상관 (M1 과 같은 창)
bool coarseRoi(const cv::Mat& ref, const cv::Mat& cur, const Region& r, const cv::Point2d& at, cv::Point2d& out) {
    const bool wide = r.group != 0;
    const int half = wide ? 2 * r.hw : r.hw;
    const cv::Rect A(r.cx - half, r.cy - half, 2 * half, 2 * half);
    const cv::Point gi = iround(at);
    const cv::Rect B = A + gi;
    if (!inside(A, ref.size()) || !inside(B, cur.size())) return false;
    out = cv::Point2d(gi) + m1::phaseShift(ref(A), cur(B));
    return true;
}

// ROI 하나의 이동 (순서 · Fine Alignment(정밀 정렬) 설정)
bool measureRoi(const cv::Mat& ref, const cv::Mat& cur, const Region& r, const cv::Point2d& plate, int order, const Fine& fc, cv::Point2d& out) {
    const cv::Rect T(r.cx - r.hw, r.cy - r.hw, 2 * r.hw, 2 * r.hw);
    switch (order) {
    case O_COARSE:
        return coarseRoi(ref, cur, r, plate, out);
    case O_FINE: {
        const TRes f = fineAlign(ref, cur, T, cv::Point2d(0, 0), fc);
        out = f.p;
        return f.ok;
    }
    case O_COARSE_FINE: {   // 현재 방법 (M1) : QR 은 플레이트 Coarse Alignment에서 바로, 벽은 2배 창 위상 상관 후
        cv::Point2d init = plate;
        if (r.group != 0 && !coarseRoi(ref, cur, r, plate, init)) return false;
        const TRes f = fineAlign(ref, cur, T, init, fc);
        if (!f.ok) return false;
        if (r.group == 0 && cv::norm(f.p - plate) > 15.0) return false;   // 이웃 QR 에 붙음
        out = f.p;
        return true;
    }
    case O_FINE_COARSE: {
        const TRes f = fineAlign(ref, cur, T, cv::Point2d(0, 0), fc);
        if (!f.ok) return false;
        return coarseRoi(ref, cur, r, f.p, out);
    }
    default: return false;
    }
}

// =====================================================================================
// 변형(variant) 목록 : 실험 1 = 순서 4 (정밀 = OpenCV ECC), 실험 2 = 보간 6 × σ 4 (Coarse -> Fine) + OpenCV ECC
// =====================================================================================
struct Variant { std::string key, name; int order; Fine fine; int exp; };

std::vector<Variant> variants() {
    std::vector<Variant> v;
    for (int o = 0; o < O_COUNT; ++o) v.push_back({std::string("e1_") + orderKey(o), orderName(o), o, Fine{K_OPENCV_ECC, 0}, 1});
    const double sig[4] = {0.0, 0.5, 1.0, 1.5};
    for (int k = 0; k < K_COUNT; ++k)
        for (double s : sig) {
            v.push_back({std::string("e2_") + kernKey(k) + fmt("_s%.1f", s), std::string(kernName(k)) + fmt(" + 가우시안 σ %.1f", s), O_COARSE_FINE, Fine{k, s}, 2});
        }
    return v;
}

const char* kRoiNames[5] = {"QR_L", "QR_C", "QR_R", "wall_W2", "wall_W3"};

// =====================================================================================
// 식 (s, d) 과 평가
// =====================================================================================
struct PhotoFeat { std::string stem; double y = 0, t = 0; int set = 1; bool first = false; double s = NAN, d = NAN; double ms = 0; };

struct ModelForm { int ordT_s = 1, ordT_d = 1; bool y2 = false; };
std::string formName(const ModelForm& f) {
    return fmt("s: Θ %d차, d: Θ %d차", f.ordT_s, f.ordT_d) + (f.y2 ? " + Y²" : "");
}

std::vector<double> basis(double y, double t, int ordT, bool y2) {
    std::vector<double> b{1.0, y};
    if (y2) b.push_back(y * y);
    double tp = 1.0;
    for (int o = 1; o <= ordT; ++o) { tp *= t; b.push_back(tp); }
    return b;
}

struct Fit { std::vector<double> cs, cd; double ss = 1, sd = 1; ModelForm f; bool ok = false; };

std::vector<double> lsq(const std::vector<std::vector<double>>& A, const std::vector<double>& b) {
    const int n = int(A.size()), k = int(A[0].size());
    cv::Mat M(n, k, CV_64F), B(n, 1, CV_64F), x;
    for (int i = 0; i < n; ++i) { for (int j = 0; j < k; ++j) M.at<double>(i, j) = A[i][j]; B.at<double>(i) = b[i]; }
    cv::solve(M, B, x, cv::DECOMP_SVD);
    return std::vector<double>(x.begin<double>(), x.end<double>());
}

Fit fitModel(const std::vector<const PhotoFeat*>& tr, const ModelForm& f) {
    Fit m;
    m.f = f;
    std::vector<std::vector<double>> As, Ad;
    std::vector<double> bs, bd;
    for (const auto* p : tr) {
        if (!std::isfinite(p->s) || !std::isfinite(p->d)) continue;
        As.push_back(basis(p->y, p->t, f.ordT_s, f.y2)); bs.push_back(p->s);
        Ad.push_back(basis(p->y, p->t, f.ordT_d, f.y2)); bd.push_back(p->d);
    }
    if (As.size() < 10) return m;
    m.cs = lsq(As, bs); m.cd = lsq(Ad, bd);
    double es = 0, ed = 0;
    for (size_t i = 0; i < As.size(); ++i) {
        double ps = 0, pd = 0;
        for (size_t j = 0; j < m.cs.size(); ++j) ps += m.cs[j] * As[i][j];
        for (size_t j = 0; j < m.cd.size(); ++j) pd += m.cd[j] * Ad[i][j];
        es += (bs[i] - ps) * (bs[i] - ps); ed += (bd[i] - pd) * (bd[i] - pd);
    }
    m.ss = std::max(1e-6, std::sqrt(es / As.size()));
    m.sd = std::max(1e-6, std::sqrt(ed / As.size()));
    m.ok = true;
    return m;
}

// 측정 (s, d) → (Y, Θ) : 가중 가우스–뉴턴 (가중 = 1/σ²)
bool invertModel(const Fit& m, double s, double d, double& y, double& t) {
    if (!m.ok || !std::isfinite(s) || !std::isfinite(d)) return false;
    y = 0; t = 0;
    const auto eval = [&](const std::vector<double>& c, int ordT, double yy, double tt, double& v, double& dy, double& dt) {
        const auto b = basis(yy, tt, ordT, m.f.y2);
        v = 0;
        for (size_t j = 0; j < c.size(); ++j) v += c[j] * b[j];
        dy = c[1] + (m.f.y2 ? 2.0 * c[2] * yy : 0.0);
        const int t0 = m.f.y2 ? 3 : 2;
        dt = 0;
        double tp = 1.0;
        for (int o = 1; o <= ordT; ++o) { dt += o * c[t0 + o - 1] * tp; tp *= tt; }
    };
    for (int it = 0; it < 30; ++it) {
        double vs, sy, st, vd, dy_, dt_;
        eval(m.cs, m.f.ordT_s, y, t, vs, sy, st);
        eval(m.cd, m.f.ordT_d, y, t, vd, dy_, dt_);
        const double ws = 1.0 / (m.ss * m.ss), wd = 1.0 / (m.sd * m.sd);
        const double rs = s - vs, rd = d - vd;
        const double a00 = ws * sy * sy + wd * dy_ * dy_, a01 = ws * sy * st + wd * dy_ * dt_, a11 = ws * st * st + wd * dt_ * dt_;
        const double b0 = ws * sy * rs + wd * dy_ * rd, b1 = ws * st * rs + wd * dt_ * rd;
        const double det = a00 * a11 - a01 * a01;
        if (std::abs(det) < 1e-18) return false;
        const double ddy = (a11 * b0 - a01 * b1) / det, ddt = (a00 * b1 - a01 * b0) / det;
        y += ddy; t += ddt;
        if (std::abs(ddy) < 1e-10 && std::abs(ddt) < 1e-11) break;
    }
    return std::isfinite(y) && std::isfinite(t);
}

// 블록 교차검증 묶음 (H5 와 같음) : Y 스윕 4 묶음, Θ 스윕 10 묶음 — 라벨 순 연속 구간
std::vector<int> blockFolds(const std::vector<const PhotoFeat*>& v) {
    std::vector<int> fold(v.size(), 0), iy, it;
    for (size_t i = 0; i < v.size(); ++i) (v[i]->t == 0 ? iy : it).push_back(int(i));
    auto assign = [&](std::vector<int>& idx, int nb, int base, bool byY) {
        std::sort(idx.begin(), idx.end(), [&](int a, int b) { return byY ? v[a]->y < v[b]->y : v[a]->t < v[b]->t; });
        for (size_t j = 0; j < idx.size(); ++j) fold[idx[j]] = base + int(j * nb / idx.size());
    };
    assign(iy, 4, 0, true);
    assign(it, 10, 4, false);
    return fold;
}

struct Pred { std::string stem; double y, t, ye, te; bool ok; int set; };
struct Score {
    double J = 0, rmsT = 0, rmsY = 0;   // 교차검증
    double Jnear = 0; int nNear = 0, passNear = 0;   // 교차검증 중 목표 근처 (|Θ| ≤ 0.2°, |Y| ≤ 0.5 mm) — 마지막 판정이 일어나는 구간
    std::vector<Pred> pred;             // 사진별 (교차검증 예측 + 2차 예측)
    int n = 0, pass = 0, passSpec = 0, fail = 0;
    int s2n[3] = {0, 0, 0}, s2pass[3] = {0, 0, 0}, s2passSpec[3] = {0, 0, 0};   // 2차 : 목표 근처 / Θ 1° / Θ 2°
    double s2rmsT[3] = {0, 0, 0}, s2rmsY[3] = {0, 0, 0};
};

// 정규화 오차는 실패·발산을 표에 담을 수 있도록 100 (허용 범위의 100 배) 에서 자름
double capN(double v) { return std::min(100.0, std::abs(v)); }

Score evaluate(const std::vector<PhotoFeat>& feats, const ModelForm& form) {
    Score sc;
    std::vector<const PhotoFeat*> T1, S2;
    for (const auto& p : feats) { if (p.set == 1 && p.first) T1.push_back(&p); if (p.set == 2) S2.push_back(&p); }
    const auto fold = blockFolds(T1);
    const int nf = *std::max_element(fold.begin(), fold.end()) + 1;
    double sj = 0, st = 0, sy = 0, sjn = 0;
    int nok = 0;
    for (int f = 0; f < nf; ++f) {
        std::vector<const PhotoFeat*> tr;
        for (size_t i = 0; i < T1.size(); ++i) if (fold[i] != f) tr.push_back(T1[i]);
        const Fit m = fitModel(tr, form);
        for (size_t i = 0; i < T1.size(); ++i) {
            if (fold[i] != f) continue;
            ++sc.n;
            double y, t;
            const bool near = std::abs(T1[i]->t) <= 0.2 + 1e-9 && std::abs(T1[i]->y) <= 0.5 + 1e-9;
            if (near) ++sc.nNear;
            if (!invertModel(m, T1[i]->s, T1[i]->d, y, t)) {
                ++sc.fail; sj += 2 * 100.0 * 100.0; if (near) sjn += 2 * 100.0 * 100.0;
                sc.pred.push_back({T1[i]->stem, T1[i]->y, T1[i]->t, NAN, NAN, false, 1});
                continue;
            }
            const double ey = (y - T1[i]->y) * 1000.0, et = t - T1[i]->t;
            sc.pred.push_back({T1[i]->stem, T1[i]->y, T1[i]->t, y, t, true, 1});
            const double jj = capN(ey / kTolY_um) * capN(ey / kTolY_um) + capN(et / kTolT_deg) * capN(et / kTolT_deg);
            sj += jj;
            if (near) { sjn += jj; if (std::abs(ey) <= kTolY_um && std::abs(et) <= kTolT_deg) ++sc.passNear; }
            st += et * et; sy += ey * ey; ++nok;
            if (std::abs(ey) <= kTolY_um && std::abs(et) <= kTolT_deg) ++sc.pass;
            if (std::abs(ey) <= kSpecY_um && std::abs(et) <= kSpecT_deg) ++sc.passSpec;
        }
    }
    sc.J = std::sqrt(sj / std::max(1, sc.n));
    sc.Jnear = std::sqrt(sjn / std::max(1, sc.nNear));
    sc.rmsT = nok ? std::sqrt(st / nok) : NAN;
    sc.rmsY = nok ? std::sqrt(sy / nok) : NAN;
    // 2차 시험 (1차 전체로 학습)
    const Fit m = fitModel(T1, form);
    double gt[3] = {0, 0, 0}, gy[3] = {0, 0, 0};
    int gn[3] = {0, 0, 0};
    for (const auto* p : S2) {
        const int g = std::abs(p->t) <= 0.2 + 1e-9 ? 0 : (std::abs(std::abs(p->t) - 1.0) < 1e-9 ? 1 : 2);
        sc.s2n[g]++;
        double y, t;
        if (!invertModel(m, p->s, p->d, y, t)) { sc.pred.push_back({p->stem, p->y, p->t, NAN, NAN, false, 2}); continue; }
        sc.pred.push_back({p->stem, p->y, p->t, y, t, true, 2});
        const double ey = (y - p->y) * 1000.0, et = t - p->t;
        gt[g] += et * et; gy[g] += ey * ey; gn[g]++;
        if (std::abs(ey) <= kTolY_um && std::abs(et) <= kTolT_deg) sc.s2pass[g]++;
        if (std::abs(ey) <= kSpecY_um && std::abs(et) <= kSpecT_deg) sc.s2passSpec[g]++;
    }
    for (int g = 0; g < 3; ++g) { sc.s2rmsT[g] = gn[g] ? std::sqrt(gt[g] / gn[g]) : NAN; sc.s2rmsY[g] = gn[g] ? std::sqrt(gy[g] / gn[g]) : NAN; }
    return sc;
}

// =====================================================================================
// 사진 측정 (모든 변형) — 캐시 : outDir/doe_features.csv
// =====================================================================================
bool measureAll(const std::vector<Sample>& ds, const std::vector<Variant>& vars, const std::string& cachePath,
                std::map<std::string, std::vector<PhotoFeat>>& out) {
    // 캐시 읽기
    {
        const CsvTable t = readCsv(cachePath);
        if (!t.empty()) {
            bool complete = true;
            std::map<std::string, std::vector<PhotoFeat>> tmp;
            for (const auto& v : vars) {
                for (const auto& s : ds) {
                    if (s.is_ref) continue;
                    const std::string key = s.stem + "|" + v.key;
                    if (!t.count(key)) { complete = false; break; }
                    PhotoFeat p;
                    p.stem = s.stem; p.y = s.y_mm; p.t = s.t_deg; p.set = s.set_id; p.first = roi::firstSession(s);
                    p.s = num(t, key, "s_px"); p.d = num(t, key, "d_px"); p.ms = num(t, key, "ms");
                    tmp[v.key].push_back(p);
                }
                if (!complete) break;
            }
            if (complete) { out = tmp; std::cout << "  측정 캐시 사용 : " << cachePath << "\n"; return true; }
        }
    }
    const Sample* r1 = findStem(ds, refStem(1));
    const Sample* r2 = findStem(ds, refStem(2));
    if (!r1 || !r2) return false;
    const cv::Mat ref1 = loadGray32(r1->path), ref2 = loadGray32(r2->path);
    std::vector<Region> rois;
    for (const char* n : kRoiNames) rois.push_back(regions()[regionIndex(n)]);
    Csv csv(cachePath);
    csv << "key" << "stem" << "variant" << "y_mm" << "t_deg" << "s_px" << "d_px" << "ms";
    for (const char* n : kRoiNames) csv << std::string(n) + "_dx";
    csv.endl();
    std::mutex mx;
    int done = 0, total = 0;
    for (const auto& s : ds) if (!s.is_ref) ++total;
    m1::resetEccFailCount();
    for (const auto& s : ds) {
        if (s.is_ref) continue;
        const cv::Mat& ref = s.set_id == 1 ? ref1 : ref2;
        const cv::Mat cur = loadGray32(s.path);
        const cv::Point2d plate = m1::coarseShift(ref, cur, cv::Point(int(kPlateCenter.x), int(kPlateCenter.y)), cv::Size(900, 400));
        std::vector<PhotoFeat> pf(vars.size());
        std::vector<std::vector<double>> dxs(vars.size(), std::vector<double>(rois.size(), NAN));
        cv::parallel_for_(cv::Range(0, int(vars.size())), [&](const cv::Range& rg) {
            for (int vi = rg.start; vi < rg.end; ++vi) {
                Timer tm;
                PhotoFeat p;
                p.stem = s.stem; p.y = s.y_mm; p.t = s.t_deg; p.set = s.set_id; p.first = roi::firstSession(s);
                bool ok = true;
                for (size_t k = 0; k < rois.size(); ++k) {
                    cv::Point2d d;
                    if (measureRoi(ref, cur, rois[k], plate, vars[vi].order, vars[vi].fine, d)) dxs[vi][k] = d.x; else ok = false;
                }
                if (ok) {
                    p.s = (dxs[vi][0] + dxs[vi][1] + dxs[vi][2]) / 3.0;
                    p.d = p.s - (dxs[vi][3] + dxs[vi][4]) / 2.0;
                }
                p.ms = tm.ms();
                pf[vi] = p;
            }
        });
        std::lock_guard<std::mutex> lk(mx);
        for (size_t vi = 0; vi < vars.size(); ++vi) {
            out[vars[vi].key].push_back(pf[vi]);
            csv << s.stem + "|" + vars[vi].key << s.stem << vars[vi].key << s.y_mm << s.t_deg << pf[vi].s << pf[vi].d << pf[vi].ms;
            for (double v : dxs[vi]) csv << v;
            csv.endl();
        }
        if (++done % 20 == 0) std::cout << "    " << done << " / " << total << " 장\n";
    }
    return true;
}

// =====================================================================================
// 합성 시험 (정답을 아는 이동)
// =====================================================================================
// 실험 1 : 큰 이동 — 넓은 영역을 푸리에 위상 회전으로 정확히 옮김
void synthLarge(const cv::Mat& ref, const std::string& outDir, Report& rep) {
    const cv::Rect big(400, 700, 3000, 1100);
    const cv::Mat B = ref(big).clone();
    const double shifts[] = {0.37, 3.62, -7.81, 15.21, 39.58, -40.27, 80.37, -80.11, 160.62};
    std::vector<Region> rois;
    for (const char* n : kRoiNames) { Region r = regions()[regionIndex(n)]; r.cx -= big.x; r.cy -= big.y; rois.push_back(r); }
    Csv csv(outDir + "/exp1_synthetic.csv");
    csv << "order" << "dx_true_px" << "dy_true_px" << "roi" << "dx_meas_px" << "err_px" << "ok";
    csv.endl();
    struct Acc { std::vector<double> e; int fail = 0, n = 0; };
    std::vector<Acc> acc(O_COUNT);
    const cv::Point pc(int(kPlateCenter.x) - big.x, int(kPlateCenter.y) - big.y);
    for (double dx : shifts) {
        const double dy = 0.21;
        const cv::Mat S = fourierShift(B, dx, dy);
        const cv::Point2d plate = m1::coarseShift(B, S, pc, cv::Size(900, 400));
        for (int o = 0; o < O_COUNT; ++o)
            for (size_t k = 0; k < rois.size(); ++k) {
                cv::Point2d d;
                const bool ok = measureRoi(B, S, rois[k], plate, o, Fine{K_OPENCV_ECC, 0}, d);
                const double e = ok ? std::abs(d.x - dx) : NAN;
                csv << orderKey(o) << dx << dy << kRoiNames[k] << (ok ? d.x : NAN) << e << int(ok);
                csv.endl();
                acc[o].n++;
                if (!ok || e > 1.0) acc[o].fail++; else acc[o].e.push_back(e);
            }
    }
    std::vector<std::vector<std::string>> rows;
    for (int o = 0; o < O_COUNT; ++o) {
        const Stat s = stat(acc[o].e);
        rows.push_back({orderName(o), fmt("%d / %d", acc[o].n - acc[o].fail, acc[o].n), acc[o].e.empty() ? "-" : fmt("%.4f", s.rms),
                        acc[o].e.empty() ? "-" : fmt("%.4f", s.maxAbs)});
    }
    rep.table({"순서", "성공 (오차 ≤ 1 px)", "성공한 것의 오차 RMS (px)", "최대 (px)"}, rows);
}

// 실험 2 : 부분 화소 이동 — ROI 마다 (dx, 0.25) px, dx = 0 ~ 1 (0.05 간격). 잡음 없는 것 = 치우침, 잡음 넣은 것 = 흔들림
void synthSubpixel(const cv::Mat& ref, const std::vector<Variant>& vars, const std::string& outDir,
                   std::map<std::string, std::pair<double, double>>& biasNoise) {
    Csv csv(outDir + "/exp2_synthetic.csv");
    csv << "variant" << "roi" << "dx_true_px" << "bias_px" << "noise_std_px";
    csv.endl();
    const int P = 64, NS = 3;
    std::mutex mx;
    std::vector<int> idx;
    for (size_t i = 0; i < vars.size(); ++i) if (vars[i].exp == 2 || vars[i].key == "e1_coarse_fine") idx.push_back(int(i));
    cv::parallel_for_(cv::Range(0, int(idx.size())), [&](const cv::Range& rg) {
        for (int ii = rg.start; ii < rg.end; ++ii) {
            const Variant& v = vars[idx[ii]];
            std::vector<double> bias, noise;
            std::ostringstream lines;
            for (const char* rn : kRoiNames) {
                const Region r = regions()[regionIndex(rn)];
                const cv::Rect box(r.cx - r.hw - P, r.cy - r.hw - P, 2 * (r.hw + P), 2 * (r.hw + P));
                const cv::Mat big = ref(box).clone();
                const Region loc{r.name, r.group, r.hw + P, r.hw + P, r.hw};
                cv::RNG rng(1234);
                for (int i = 0; i <= 20; ++i) {
                    const double dx = 0.05 * i, dy = 0.25;
                    const cv::Mat sh = fourierShift(big, dx, dy);
                    const cv::Point2d plate = m1::phaseShift(big, sh);
                    cv::Point2d d;
                    if (!measureRoi(big, sh, loc, plate, O_COARSE_FINE, v.fine, d)) continue;
                    const double b = d.x - dx;
                    double sq = 0;
                    int nn = 0;
                    for (int k = 0; k < NS; ++k) {
                        cv::Mat noisy = sh.clone(), nz(sh.size(), CV_32F);
                        rng.fill(nz, cv::RNG::NORMAL, 0.0, 0.71);
                        noisy += nz;
                        cv::Point2d dn;
                        if (measureRoi(big, noisy, loc, plate, O_COARSE_FINE, v.fine, dn)) { sq += (dn.x - d.x) * (dn.x - d.x); ++nn; }
                    }
                    const double ns = nn ? std::sqrt(sq / nn) : NAN;
                    bias.push_back(b);
                    if (nn) noise.push_back(ns);
                    lines << v.key << ',' << rn << ',' << dx << ',' << b << ',' << ns << '\n';
                }
            }
            const Stat sb = stat(bias);
            double nm = 0;
            for (double x : noise) nm += x * x;
            std::lock_guard<std::mutex> lk(mx);
            biasNoise[v.key] = {sb.rms, noise.empty() ? NAN : std::sqrt(nm / noise.size())};
            std::istringstream in(lines.str());
            std::string ln;
            while (std::getline(in, ln)) {
                std::stringstream ss(ln);
                std::string a, b, c, d, e;
                std::getline(ss, a, ','); std::getline(ss, b, ','); std::getline(ss, c, ','); std::getline(ss, d, ','); std::getline(ss, e, ',');
                csv << a << b << std::stod(c) << std::stod(d) << (e == "nan" ? NAN : std::stod(e));
                csv.endl();
            }
        }
    });
}

// 커널 자체 검사 : 정수점 재현 (사전 필터 포함), 반 칸 위치 계수 합 = 1
bool kernelSelfTest(Report& rep) {
    bool ok = true;
    std::vector<std::vector<std::string>> rows;
    cv::RNG rng(7);
    for (int k = 0; k < K_COUNT; ++k) {
        cv::Mat a(40, 40, CV_64F);
        rng.fill(a, cv::RNG::UNIFORM, 0.0, 255.0);
        cv::Mat c = a.clone();
        toCoeffs(c, k);
        const int R = kernRadius(k);
        double maxErr = 0;
        for (int y = 10; y < 30; ++y)
            for (int x = 10; x < 30; ++x) {
                double s = 0;
                for (int q = -R + 1; q <= R; ++q)
                    for (int m = -R + 1; m <= R; ++m) s += kernW(k, double(m)) * kernW(k, double(q)) * c.at<double>(y + q, x + m);
                maxErr = std::max(maxErr, std::abs(s - a.at<double>(y, x)));
            }
        double sum = 0;
        for (int m = -R + 1; m <= R; ++m) sum += kernW(k, 0.5 - m);
        const bool pass = maxErr < 1e-6 && std::abs(sum - 1.0) < (k == K_LANCZOS3 ? 2e-2 : 1e-9);
        ok = ok && pass;
        rows.push_back({kernName(k), fmt("%.1e", maxErr), fmt("%.6f", sum), pass ? "통과" : "미통과"});
    }
    rep.table({"보간 커널", "정수점 재현 최대 오차", "반 칸 위치 가중 합", "판정"}, rows);
    return ok;
}

} // namespace

// =====================================================================================
bool run(const std::vector<Sample>& ds, const std::string& outDir, const std::string& which) {
    (void)which;
    Report rep(outDir, "실험계획(DOE) — 정렬 순서 · 보간 방법 · s·d 식의 차수");
    rep.line(fmt("평가 허용 범위 : Y ±%.0f µm, Θ ±%.3f° (과제의 절반 = 프로젝트 기본값). 괄호 안은 과제 원래 범위(±90 µm, ±0.01°)의 결과.", kTolY_um, kTolT_deg));
    rep.line("선택 기준 : 1차 첫 번째 측정의 블록 교차검증 J (작을수록 좋음). 2차 결과는 보고만 하고 선택에 쓰지 않음.");
    const Sample* r1 = findStem(ds, refStem(1));
    if (!r1) { rep.check("기준 영상", false, "center 없음"); return false; }
    const cv::Mat ref = loadGray32(r1->path);

    rep.h("0. 보간 커널 구현 확인");
    rep.check("커널 구현", kernelSelfTest(rep), "정수점에서 원래 값 재현 (사전 필터 포함)");

    const auto vars = variants();
    std::map<std::string, std::vector<PhotoFeat>> feats;
    rep.h("측정 (사진 " + std::to_string(ds.size()) + "장 × 변형 " + std::to_string(vars.size()) + "가지)");
    if (!measureAll(ds, vars, outDir + "/doe_features.csv", feats)) { rep.check("측정", false, "기준 사진 없음"); return false; }
    std::map<std::string, Score> score;
    for (const auto& v : vars) score[v.key] = evaluate(feats[v.key], ModelForm{1, 1, false});
    const auto msOf = [&](const std::string& key) {
        std::vector<double> m;
        for (const auto& p : feats[key]) m.push_back(p.ms);
        return stat(m).mean;
    };

    // ---------------------------------------------------------------- 실험 1
    rep.h("실험 1 — 정렬 순서 (Fine Alignment(정밀 정렬) = OpenCV ECC, 식 = s·d 1차)");
    rep.line("1-a 합성 : 넓은 영역(3000×1100 px)을 정답을 아는 양만큼 옮긴 영상. ROI 5곳 × 이동 9가지");
    synthLarge(ref, outDir, rep);
    rep.line("1-b 실제 사진 : 1차 교차검증 + 2차 시험");
    {
        Csv c(outDir + "/exp1_summary.csv");
        c << "order" << "J" << "cv_pass" << "cv_pass_spec" << "cv_n" << "cv_fail" << "rms_t_deg" << "rms_y_um" << "s2_near_pass" << "s2_t1_pass" << "s2_t2_pass" << "ms_per_photo";
        c.endl();
        std::vector<std::vector<std::string>> rows;
        for (int o = 0; o < O_COUNT; ++o) {
            const std::string key = std::string("e1_") + orderKey(o);
            const Score& s = score[key];
            rows.push_back({orderName(o), fmt("%.3f", s.J), fmt("%d (%d) / %d", s.pass, s.passSpec, s.n), fmt("%.4f", s.rmsT), fmt("%.1f", s.rmsY),
                            fmt("%d / %d", s.s2pass[0], s.s2n[0]), fmt("%d / %d", s.s2pass[1], s.s2n[1]), fmt("%.0f", msOf(key))});
            c << orderKey(o) << s.J << s.pass << s.passSpec << s.n << s.fail << s.rmsT << s.rmsY << s.s2pass[0] << s.s2pass[1] << s.s2pass[2] << msOf(key);
            c.endl();
        }
        rep.table({"순서", "교차검증 J", "교차검증 허용 안 (과제 범위)", "Θ RMS (°)", "Y RMS (µm)", "2차 목표 근처", "2차 Θ 1°", "ROI 5곳 시간 (ms)"}, rows);
    }

    // ---------------------------------------------------------------- 실험 2
    rep.h("실험 2 — 보간 방법 × 가우시안 사전 필터 (순서 = Coarse -> Fine, 식 = s·d 1차)");
    std::map<std::string, std::pair<double, double>> bn;
    synthSubpixel(ref, vars, outDir, bn);
    std::string best;
    double bestJ = 1e9;
    {
        Csv c(outDir + "/exp2_summary.csv");
        c << "variant" << "kernel" << "sigma_px" << "bias_rms_px" << "noise_std_px" << "J" << "cv_pass" << "cv_pass_spec" << "cv_n" << "rms_t_deg" << "rms_y_um"
          << "s2_near_pass" << "s2_near_n" << "s2_t1_pass" << "s2_t1_n" << "ms_per_photo";
        c.endl();
        std::vector<std::vector<std::string>> rows;
        for (const auto& v : vars) {
            if (!(v.exp == 2 || v.key == "e1_coarse_fine")) continue;
            const Score& s = score[v.key];
            const auto b = bn[v.key];
            const std::string name = v.exp == 2 ? v.name : std::string(kernName(K_OPENCV_ECC));
            rows.push_back({name, fmt("%.4f", b.first), fmt("%.4f", b.second), fmt("%.3f", s.J), fmt("%d (%d) / %d", s.pass, s.passSpec, s.n),
                            fmt("%.4f", s.rmsT), fmt("%.1f", s.rmsY), fmt("%d / %d", s.s2pass[0], s.s2n[0]), fmt("%.0f", msOf(v.key))});
            c << v.key << (v.exp == 2 ? kernKey(v.fine.kern) : "opencv_ecc") << v.fine.sigma << b.first << b.second << s.J << s.pass << s.passSpec << s.n << s.rmsT
              << s.rmsY << s.s2pass[0] << s.s2n[0] << s.s2pass[1] << s.s2n[1] << msOf(v.key);
            c.endl();
            if (s.J < bestJ) { bestJ = s.J; best = v.key; }
        }
        rep.table({"보간 + 사전 필터", "치우침 RMS (px)", "잡음 흔들림 (px)", "교차검증 J", "교차검증 허용 안 (과제)", "Θ RMS (°)", "Y RMS (µm)", "2차 목표 근처", "ROI 5곳 (ms)"}, rows);
        rep.line("교차검증 J 가장 작은 설정 : " + best + fmt(" (J %.3f)", bestJ));
    }

    // ---------------------------------------------------------------- 실험 3
    rep.h("실험 3 — s·d 식의 차수 (측정 = 실험 2 의 최적 설정 '" + best + "', 비교용으로 OpenCV ECC 도 함께)");
    ModelForm bestForm{1, 1, false};
    double bestFormJ = 1e9;
    {
        Csv c(outDir + "/exp3_orders.csv");
        Csv pc(outDir + "/exp3_predictions.csv");
        pc << "features" << "form" << "stem" << "set" << "y_mm" << "t_deg" << "y_est_mm" << "t_est_deg" << "ok";
        pc.endl();
        c << "features" << "form" << "ordT_s" << "ordT_d" << "y2" << "n_coef" << "J" << "J_near" << "near_pass" << "near_n" << "cv_pass" << "cv_pass_spec" << "cv_n" << "rms_t_deg" << "rms_y_um"
          << "s2_near_pass" << "s2_near_n" << "s2_t1_pass" << "s2_t1_n" << "s2_t2_pass" << "s2_t2_n" << "s2_t1_rms_t" << "s2_t2_rms_t";
        c.endl();
        for (const std::string& fk : {best, std::string("e1_coarse_fine")}) {
            std::vector<std::vector<std::string>> rows;
            std::vector<ModelForm> forms;
            for (int a = 1; a <= 3; ++a)
                for (int b = 1; b <= 3; ++b) forms.push_back({a, b, false});
            forms.push_back({1, 1, true});
            for (const auto& f : forms) {
                const Score s = evaluate(feats[fk], f);
                if (fk == best && s.J < bestFormJ) { bestFormJ = s.J; bestForm = f; }
                const int nc = 2 * (2 + (f.y2 ? 1 : 0)) + f.ordT_s + f.ordT_d;
                rows.push_back({formName(f), std::to_string(nc), fmt("%.3f", s.J), fmt("%.3f", s.Jnear), fmt("%d / %d", s.passNear, s.nNear),
                                fmt("%d (%d) / %d", s.pass, s.passSpec, s.n), fmt("%.4f", s.rmsT),
                                fmt("%.1f", s.rmsY), fmt("%d / %d", s.s2pass[0], s.s2n[0]), fmt("%d / %d", s.s2pass[1], s.s2n[1]), fmt("%.4f", s.s2rmsT[2])});
                for (const auto& q : s.pred) { pc << fk << formName(f) << q.stem << q.set << q.y << q.t << q.ye << q.te << int(q.ok); pc.endl(); }
                c << fk << formName(f) << f.ordT_s << f.ordT_d << int(f.y2) << nc << s.J << s.Jnear << s.passNear << s.nNear << s.pass << s.passSpec << s.n << s.rmsT << s.rmsY << s.s2pass[0]
                  << s.s2n[0] << s.s2pass[1] << s.s2n[1] << s.s2pass[2] << s.s2n[2] << s.s2rmsT[1] << s.s2rmsT[2];
                c.endl();
            }
            rep.line("측정 설정 : " + fk);
            rep.table({"식", "계수 수", "교차검증 J", "목표 근처 J", "목표 근처 허용 안", "교차검증 허용 안 (과제)", "Θ RMS (°)", "Y RMS (µm)", "2차 목표 근처", "2차 Θ 1°", "2차 Θ 2° Θ RMS (°)"}, rows);
        }
    }
    // 최적 설정 · 최적 식의 계수 (alignment.yml 갱신용) — 특징 = c + a·Y + Σ b_k·Θ^k
    {
        rep.line("교차검증 J 가 가장 작은 식 : " + formName(bestForm) + fmt(" (J %.3f)", bestFormJ));
        std::vector<const PhotoFeat*> T1;
        for (const auto& p : feats[best]) if (p.set == 1 && p.first) T1.push_back(&p);
        for (const ModelForm& f : {ModelForm{1, 1, false}, bestForm}) {
            const Fit m = fitModel(T1, f);
            const std::string name = (f.ordT_s == 1 && f.ordT_d == 1 && !f.y2) ? "doe_best_model_linear.yml" : "doe_best_model.yml";
            cv::FileStorage fs(outDir + "/" + name, cv::FileStorage::WRITE);
            fs << "variant" << best << "form" << formName(f);
            const auto put = [&](const char* key, const std::vector<double>& c, double sg) {
                fs << key << "{" << "c_px" << c[0] << "a_px_per_mm" << c[1] << "theta_px_per_deg_k" << "[:";
                for (size_t k = 2 + (f.y2 ? 1 : 0); k < c.size(); ++k) fs << c[k];
                fs << "]" << "sigma_px" << sg << "}";
            };
            put("s", m.cs, m.ss);
            put("d", m.cd, m.sd);
        }
        rep.line("계수 → doe_best_model.yml (최적 식), doe_best_model_linear.yml (1차 식)");
    }
    rep.check("실험 완료", true, "결과 CSV : exp1_*, exp2_*, exp3_orders.csv");
    return true;
}

} // namespace doe
} // namespace yt
