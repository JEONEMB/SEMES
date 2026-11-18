// QR 격자 모서리(구조점) 찾기 + erf 경계 모델 최소제곱 맞춤.
// 원본: codex_experiments_20261014/decoded_grid_corners_20261015/build.py (QR 해독·구조점),
//       codex_experiments_20261014/subpixel_grid_corners_20261015/refine.py (경계 모델 맞춤)
// least_squares 는 scipy 1.18 의 method='trf' (bounds, tr_solver='exact', jac='2-point', x_scale=1, loss='soft_l1') 를 줄 단위로 옮겼다.
#pragma once
#include <hye/ecc_compat.hpp>
#include <hye/report.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/video/tracking.hpp>
#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <string>
#include <vector>

namespace cf {

// measurement/measure.py 의 QR 판 영역 (x0, y0, x1, y1), 해독용 자르기
const int kBoxes[3][4] = {{1552, 969, 1736, 1167}, {1809, 971, 1993, 1169}, {2067, 973, 2252, 1171}};
const int kCrop[4] = {1340, 800, 2470, 1330};

struct QRInfo {
    int qr = 0;
    std::string payload, how;
    int n = 0;
    std::array<cv::Point2d, 4> quad;  // 좌상, 우상, 우하, 좌하 (원본 좌표)
    cv::Mat grid;                     // n x n, 0 = 검정 모듈
};

// 기능 패턴(파인더 3개 + 분리대 + 타이밍) 모듈이면 true. 판 버전 1(21x21)
inline bool function_module(int x, int y, int n) {
    auto finder = [&](int x0, int y0) { return x >= x0 && x < x0 + 8 && y >= y0 && y < y0 + 8; };
    if (finder(0, 0) || finder(n - 8, 0) || finder(0, n - 8)) return true;
    return x == 6 || y == 6;
}

// center 사진에서 QR 세 개를 해독한다. build.py 처럼 자른 영역 전체에서 여러 개 찾기 -> 못 찾은 판은 판 영역을 1~4배 확대해 다시.
// OpenCV 3.4 해독기가 끝내 못 찾는 판은 이웃 판의 기능 패턴(파인더·타이밍)을 ECC affine 으로 맞춰 사각형을 옮기고, 그 사각형으로 해독한다.
inline std::vector<QRInfo> detect_qrs(const cv::Mat& ref) {
    cv::QRCodeDetector det;
    std::vector<QRInfo> found(3);
    cv::Mat crop = ref(cv::Range(kCrop[1], kCrop[3]), cv::Range(kCrop[0], kCrop[2])).clone();
    std::vector<cv::Point2f> pts;
    if (det.detectMulti(crop, pts)) {
        for (size_t i = 0; i + 3 < pts.size(); i += 4) {
            std::vector<cv::Point2f> q(pts.begin() + i, pts.begin() + i + 4);
            cv::Mat st;
            std::string t = det.decode(crop, q, st);
            if (t.empty() || st.empty()) continue;
            double gx = 0, gy = 0;
            for (auto& p : q) { gx += p.x / 4.0; gy += p.y / 4.0; }
            gx += kCrop[0]; gy += kCrop[1];
            for (int k = 0; k < 3; ++k) {
                auto b = kBoxes[k];
                if (b[0] < gx && gx < b[2] && b[1] < gy && gy < b[3] && found[k].n == 0) {
                    auto& f = found[k];
                    f.payload = t; f.how = "detectMulti"; f.n = st.rows; f.grid = st.clone();
                    for (int j = 0; j < 4; ++j) f.quad[j] = {(double)q[j].x + kCrop[0], (double)q[j].y + kCrop[1]};
                }
            }
        }
    }
    for (int k = 0; k < 3; ++k) {
        if (found[k].n) continue;
        auto b = kBoxes[k];
        int x0 = b[0] - 5, y0 = b[1] - 5;
        cv::Mat c = ref(cv::Range(y0, b[3] + 5), cv::Range(x0, b[2] + 5));
        // OpenCV 3.4 해독기는 원본 화소로 QR3 을 못 찾는다. 같은 확대를 3x3 가우시안으로 살짝 흐린 영역에도 해 본다.
        for (int t = 0; t < 8 && !found[k].n; ++t) {
            int s = t % 4 + 1;
            cv::Mat src;
            if (t >= 4) cv::GaussianBlur(c, src, cv::Size(3, 3), 0);
            else src = c.clone();
            cv::Mat r;
            cv::resize(src, r, cv::Size(), s, s);
            std::vector<cv::Point2f> q;
            cv::Mat st;
            std::string tx = det.detectAndDecode(r, q, st);
            if (!tx.empty() && q.size() == 4 && st.rows > 0) {
                auto& f = found[k];
                f.payload = tx; f.how = std::string("detectAndDecode x") + std::to_string(s) + (t >= 4 ? " (3x3 blur)" : ""); f.n = st.rows; f.grid = st.clone();
                for (int j = 0; j < 4; ++j) f.quad[j] = {(double)q[j].x / s + x0, (double)q[j].y / s + y0};
            }
        }
    }
    for (int k = 0; k < 3; ++k) {
        if (found[k].n) continue;
        int src = k == 0 ? 1 : k - 1;  // 이웃 판
        if (!found[src].n) continue;
        const auto& S = found[src];
        int n = S.n;
        // 이웃 판의 사각형 + 여백 4px 를 템플릿, 기능 패턴 모듈만 마스크
        double minx = 1e9, miny = 1e9, maxx = -1e9, maxy = -1e9;
        for (auto& p : S.quad) { minx = std::min(minx, p.x); miny = std::min(miny, p.y); maxx = std::max(maxx, p.x); maxy = std::max(maxy, p.y); }
        cv::Rect tr((int)std::floor(minx) - 4, (int)std::floor(miny) - 4, 0, 0);
        tr.width = (int)std::ceil(maxx) + 4 - tr.x;
        tr.height = (int)std::ceil(maxy) + 4 - tr.y;
        cv::Mat tpl = ref(tr).clone();
        std::vector<cv::Point2f> sq{{0, 0}, {(float)n, 0}, {(float)n, (float)n}, {0, (float)n}}, dq;
        for (auto& p : S.quad) dq.push_back(cv::Point2f((float)(p.x - tr.x), (float)(p.y - tr.y)));
        cv::Mat H = cv::getPerspectiveTransform(sq, dq);
        cv::Mat mask(tpl.size(), CV_8U, cv::Scalar(0));
        cv::Mat Hi = H.inv();
        for (int y = 0; y < tpl.rows; ++y)
            for (int x = 0; x < tpl.cols; ++x) {
                double w = Hi.at<double>(2, 0) * x + Hi.at<double>(2, 1) * y + Hi.at<double>(2, 2);
                double u = (Hi.at<double>(0, 0) * x + Hi.at<double>(0, 1) * y + Hi.at<double>(0, 2)) / w;
                double v = (Hi.at<double>(1, 0) * x + Hi.at<double>(1, 1) * y + Hi.at<double>(1, 2)) / w;
                int iu = (int)std::floor(u), iv = (int)std::floor(v);
                bool on = iu >= -1 && iv >= -1 && iu <= n && iv <= n &&
                          (iu < 0 || iv < 0 || iu >= n || iv >= n || function_module(iu, iv, n));
                // 판 밖(조용한 영역)은 바깥 1모듈만
                mask.at<uchar>(y, x) = on ? 255 : 0;
            }
        // 시작 이동 = 판 영역 원점 차이, 정수 템플릿 매칭으로 다듬는다
        auto bs = kBoxes[src], bk = kBoxes[k];
        int dx0 = bk[0] - bs[0], dy0 = bk[1] - bs[1], pad = 20;
        cv::Rect sr(tr.x + dx0 - pad, tr.y + dy0 - pad, tr.width + 2 * pad, tr.height + 2 * pad);
        cv::Mat res;
        cv::matchTemplate(ref(sr), tpl, res, cv::TM_CCORR_NORMED, mask);
        cv::Point loc;
        cv::minMaxLoc(res, nullptr, nullptr, nullptr, &loc);
        int ox = sr.x + loc.x, oy = sr.y + loc.y;
        cv::Mat target = ref(cv::Rect(ox, oy, tr.width, tr.height)).clone();
        cv::Mat W = cv::Mat::eye(2, 3, CV_32F);
        semes::ecc5::findTransformECCWithMask(tpl, target, mask, cv::noArray(), W, cv::MOTION_AFFINE,
                                              cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 200, 1e-8), 1);
        auto& f = found[k];
        f.how = "ECC affine from QR" + std::to_string(src + 1);
        f.n = n;
        std::vector<cv::Point2f> q;
        for (int j = 0; j < 4; ++j) {
            double x = dq[j].x, y = dq[j].y;
            double X = W.at<float>(0, 0) * x + W.at<float>(0, 1) * y + W.at<float>(0, 2) + ox;
            double Y = W.at<float>(1, 0) * x + W.at<float>(1, 1) * y + W.at<float>(1, 2) + oy;
            f.quad[j] = {X, Y};
            q.push_back(cv::Point2f((float)X, (float)Y));
        }
        cv::Mat st;
        f.payload = det.decode(ref, q, st);
        if (!st.empty() && st.rows == n) {
            f.grid = st.clone();
        } else {
            // 해독기가 격자를 못 내면 모듈 중심 밝기를 Otsu 로 나눈다
            cv::Mat Hq = cv::getPerspectiveTransform(sq, q);
            cv::Mat vals(n, n, CV_8U);
            for (int v = 0; v < n; ++v)
                for (int u = 0; u < n; ++u) {
                    std::vector<cv::Point2f> a{{u + 0.5f, v + 0.5f}}, b;
                    cv::perspectiveTransform(a, b, Hq);
                    vals.at<uchar>(v, u) = ref.at<uchar>((int)std::lround(b[0].y), (int)std::lround(b[0].x));
                }
            cv::threshold(vals, f.grid, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
            f.how += " + Otsu modules";
        }
    }
    for (int k = 0; k < 3; ++k) found[k].qr = k + 1;
    return found;
}

// 구조점: 2x2 이웃 모듈 중 검정 1·3개, 또는 대각으로 닿은 검정 2개인 격자 꼭짓점
struct Corner {
    int id = 0, qr = 0, gx = 0, gy = 0, black = 0;
    double x = 0, y = 0;          // 사각형 호모그래피로 투영한 시작 위치 (float32 값)
    float inv[2][2] = {{1, 0}, {0, 1}};  // 국소 축 역행렬 (float32, numpy 와 같은 정밀도)
    double patch[2][2] = {{0, 0}, {0, 0}};
};

inline std::vector<Corner> structural_corners(const std::vector<QRInfo>& qrs) {
    std::vector<Corner> out;
    for (auto& q : qrs) {
        int n = q.n;
        std::vector<cv::Point2f> sq{{0, 0}, {(float)n, 0}, {(float)n, (float)n}, {0, (float)n}}, dq;
        for (auto& p : q.quad) dq.push_back(cv::Point2f((float)p.x, (float)p.y));
        cv::Mat H = cv::getPerspectiveTransform(sq, dq);
        auto black = [&](int x, int y) { return (x < 0 || y < 0 || x >= n || y >= n) ? 0 : (q.grid.at<uchar>(y, x) == 0 ? 1 : 0); };
        for (int y = 0; y <= n; ++y)
            for (int x = 0; x <= n; ++x) {
                int a = black(x - 1, y - 1), b = black(x, y - 1), c = black(x - 1, y), d = black(x, y);
                int cnt = a + b + c + d;
                if (!(cnt == 1 || cnt == 3 || (cnt == 2 && a == d))) continue;
                Corner c0;
                c0.qr = q.qr; c0.gx = x; c0.gy = y; c0.black = cnt;
                c0.patch[0][0] = a; c0.patch[0][1] = b; c0.patch[1][0] = c; c0.patch[1][1] = d;
                std::vector<cv::Point2f> src{{(float)x, (float)y}, {x + 0.1f, (float)y}, {(float)x, y + 0.1f}}, dst;
                cv::perspectiveTransform(src, dst, H);
                c0.x = dst[0].x; c0.y = dst[0].y;
                float ax[2] = {dst[1].x - dst[0].x, dst[1].y - dst[0].y}, ay[2] = {dst[2].x - dst[0].x, dst[2].y - dst[0].y};
                float nx = std::sqrt(ax[0] * ax[0] + ax[1] * ax[1]), ny = std::sqrt(ay[0] * ay[0] + ay[1] * ay[1]);
                ax[0] /= nx; ax[1] /= nx; ay[0] /= ny; ay[1] /= ny;
                // axes = [[ax0, ay0], [ax1, ay1]] (열이 축), 역행렬
                float det = ax[0] * ay[1] - ay[0] * ax[1];
                c0.inv[0][0] = ay[1] / det; c0.inv[0][1] = -ay[0] / det;
                c0.inv[1][0] = -ax[1] / det; c0.inv[1][1] = ax[0] / det;
                c0.id = (int)out.size();
                out.push_back(c0);
            }
    }
    return out;
}

// ---- 경계 모델 ----
struct Model {
    const double* X;
    const double* Y;
    const double* I;
    int m;
    float inv[2][2];
    double patch[2][2];
    // p = [x, y, sx, sy, low, contrast]
    void residual(const double* p, double* r) const {
        const double s2 = std::sqrt(2.0);
        for (int i = 0; i < m; ++i) {
            double dx = X[i] - p[0], dy = Y[i] - p[1];
            double l0 = dx * (double)inv[0][0] + dy * (double)inv[0][1];
            double l1 = dx * (double)inv[1][0] + dy * (double)inv[1][1];
            double U = .5 * (1 + std::erf(l0 / (s2 * p[2])));
            double V = .5 * (1 + std::erf(l1 / (s2 * p[3])));
            double f = patch[0][0] * (1 - U) * (1 - V) + patch[0][1] * U * (1 - V) + patch[1][0] * (1 - U) * V + patch[1][1] * U * V;
            r[i] = p[4] + p[5] * (1 - f) - I[i];
        }
    }
};

struct FitResult {
    double p[6] = {0, 0, 0, 0, 0, 0};
    double rmse = 0;
    bool ok = false;
    int status = 0, nfev = 0;
};

namespace trf {
const double EPS = DBL_EPSILON;
using Vec = std::vector<double>;
inline double dot(const Vec& a, const Vec& b) { double s = 0; for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i]; return s; }
inline double nrm(const Vec& a) { return std::sqrt(dot(a, a)); }

// scipy.optimize.least_squares(method='trf', bounds, loss='soft_l1', jac='2-point', x_scale=1, tr_solver='exact')
inline FitResult least_squares(const Model& M, Vec x0, const Vec& lb, const Vec& ub, double f_scale, int max_nfev, double ftol, double xtol,
                               double gtol) {
    const int n = 6, m = M.m;
    auto fun = [&](const Vec& x) { Vec r(m); M.residual(x.data(), r.data()); return r; };
    // make_strictly_feasible(x0, rstep=1e-10)
    for (int i = 0; i < n; ++i) {
        double ld = x0[i] - lb[i], ud = ub[i] - x0[i];
        if (ld <= std::min(ud, 1e-10 * std::max(1.0, std::abs(lb[i])))) x0[i] = lb[i] + 1e-10 * std::max(1.0, std::abs(lb[i]));
        else if (ud <= std::min(ld, 1e-10 * std::max(1.0, std::abs(ub[i])))) x0[i] = ub[i] - 1e-10 * std::max(1.0, std::abs(ub[i]));
    }
    // 2-point 유한차분 야코비안 (bounds 쪽으로 뒤집기)
    auto jac = [&](const Vec& x, const Vec& f0) {
        cv::Mat J(m, n, CV_64F);
        const double rstep = std::sqrt(EPS);
        for (int i = 0; i < n; ++i) {
            double sgn = x[i] >= 0 ? 1.0 : -1.0;
            double h = rstep * sgn * std::max(1.0, std::abs(x[i]));
            double xt = x[i] + h;
            double lower = x[i] - lb[i], upper = ub[i] - x[i];
            bool violated = xt < lb[i] || xt > ub[i];
            bool fitting = std::abs(h) <= std::max(lower, upper);
            if (violated && fitting) h = -h;
            if (!fitting) h = upper >= lower ? upper : -lower;
            Vec x1 = x;
            x1[i] = x[i] + h;
            double dx = (x[i] + h) - x[i];
            Vec f1 = fun(x1);
            for (int r = 0; r < m; ++r) J.at<double>(r, i) = (f1[r] - f0[r]) / dx;
        }
        return J;
    };
    const double C2 = f_scale * f_scale;
    auto cost_only = [&](const Vec& f) {
        double s = 0;
        for (double v : f) { double z = (v / f_scale) * (v / f_scale); s += 2 * (std::sqrt(1 + z) - 1); }
        return 0.5 * C2 * s;
    };
    // rho 와 scale_for_robust_loss_function: J 행과 f 를 바꾼다
    auto robust = [&](cv::Mat& J, Vec& f) {
        for (int r = 0; r < m; ++r) {
            double z = (f[r] / f_scale) * (f[r] / f_scale), t = 1 + z;
            double rho1 = std::pow(t, -0.5), rho2 = -0.5 * std::pow(t, -1.5) / C2;
            double js = rho1 + 2 * rho2 * f[r] * f[r];
            if (js < EPS) js = EPS;
            js = std::sqrt(js);
            f[r] *= rho1 / js;
            for (int c = 0; c < n; ++c) J.at<double>(r, c) *= js;
        }
    };
    auto grad = [&](const cv::Mat& J, const Vec& f) {
        Vec g(n, 0.0);
        for (int c = 0; c < n; ++c) { double s = 0; for (int r = 0; r < m; ++r) s += J.at<double>(r, c) * f[r]; g[c] = s; }
        return g;
    };
    auto CL = [&](const Vec& x, const Vec& g, Vec& v, Vec& dv) {
        v.assign(n, 1.0); dv.assign(n, 0.0);
        for (int i = 0; i < n; ++i) {
            if (g[i] < 0) { v[i] = ub[i] - x[i]; dv[i] = -1; }
            if (g[i] > 0) { v[i] = x[i] - lb[i]; dv[i] = 1; }
        }
    };
    auto Jdot = [&](const cv::Mat& J, const Vec& s) {
        Vec o(m, 0.0);
        for (int r = 0; r < m; ++r) { double a = 0; for (int c = 0; c < n; ++c) a += J.at<double>(r, c) * s[c]; o[r] = a; }
        return o;
    };
    auto eval_quad = [&](const cv::Mat& Jh, const Vec& gh, const Vec& s, const Vec& diag) {
        Vec Js = Jdot(Jh, s);
        double q = dot(Js, Js);
        for (int i = 0; i < n; ++i) q += s[i] * diag[i] * s[i];
        return 0.5 * q + dot(s, gh);
    };
    auto build_q1d = [&](const cv::Mat& Jh, const Vec& g, const Vec& s, const Vec& diag, const Vec* s0, double& a, double& b, double& c) {
        Vec v = Jdot(Jh, s);
        a = dot(v, v);
        for (int i = 0; i < n; ++i) a += s[i] * diag[i] * s[i];
        a *= 0.5;
        b = dot(g, s);
        c = 0;
        if (s0) {
            Vec u = Jdot(Jh, *s0);
            b += dot(u, v);
            c = 0.5 * dot(u, u) + dot(g, *s0);
            for (int i = 0; i < n; ++i) { b += (*s0)[i] * diag[i] * s[i]; c += 0.5 * (*s0)[i] * diag[i] * (*s0)[i]; }
        }
    };
    auto min_q1d = [&](double a, double b, double lo, double hi, double c, double& tbest) {
        double t[3] = {lo, hi, 0};
        int k = 2;
        if (a != 0) { double e = -0.5 * b / a; if (lo < e && e < hi) t[k++] = e; }
        int bi = 0;
        double by = t[0] * (a * t[0] + b) + c;
        for (int i = 1; i < k; ++i) { double y = t[i] * (a * t[i] + b) + c; if (y < by) { by = y; bi = i; } }
        tbest = t[bi];
        return by;
    };
    auto step_to_bound = [&](const Vec& x, const Vec& s, std::vector<int>& hits) {
        Vec steps(n, INFINITY);
        for (int i = 0; i < n; ++i)
            if (s[i] != 0) steps[i] = std::max((lb[i] - x[i]) / s[i], (ub[i] - x[i]) / s[i]);
        double mn = *std::min_element(steps.begin(), steps.end());
        hits.assign(n, 0);
        for (int i = 0; i < n; ++i) hits[i] = steps[i] == mn ? (s[i] > 0 ? 1 : s[i] < 0 ? -1 : 0) : 0;
        return mn;
    };
    auto in_bounds = [&](const Vec& x) { for (int i = 0; i < n; ++i) if (!(x[i] >= lb[i] && x[i] <= ub[i])) return false; return true; };
    auto intersect_tr = [&](const Vec& x, const Vec& s, double Delta, double& t1, double& t2) {
        double a = dot(s, s), b = dot(x, s), c = dot(x, x) - Delta * Delta;
        double d = std::sqrt(b * b - a * c);
        double q = -(b + std::copysign(d, b));
        double u1 = q / a, u2 = c / q;
        if (u1 < u2) { t1 = u1; t2 = u2; } else { t1 = u2; t2 = u1; }
    };

    FitResult R;
    Vec x = x0;
    Vec f = fun(x);
    int nfev = 1;
    cv::Mat J = jac(x, f);
    double cost = cost_only(f);
    robust(J, f);
    Vec g = grad(J, f);
    Vec v, dv;
    CL(x, g, v, dv);
    double Delta = 0;
    for (int i = 0; i < n; ++i) Delta += x0[i] * x0[i] / v[i];
    Delta = std::sqrt(Delta);
    if (Delta == 0) Delta = 1.0;
    double alpha = 0.0;
    int status = -1;  // None
    Vec x_new;
    Vec f_new;
    double cost_new = 0;
    while (true) {
        CL(x, g, v, dv);
        double g_norm = 0;
        for (int i = 0; i < n; ++i) g_norm = std::max(g_norm, std::abs(g[i] * v[i]));
        if (g_norm < gtol) status = 1;
        if (status != -1 || nfev == max_nfev) break;
        Vec d(n), diag_h(n), g_h(n);
        for (int i = 0; i < n; ++i) { d[i] = std::sqrt(v[i]); diag_h[i] = g[i] * dv[i]; g_h[i] = d[i] * g[i]; }
        cv::Mat Ja(m + n, n, CV_64F, cv::Scalar(0));
        for (int r = 0; r < m; ++r)
            for (int c = 0; c < n; ++c) Ja.at<double>(r, c) = J.at<double>(r, c) * d[c];
        for (int i = 0; i < n; ++i) Ja.at<double>(m + i, i) = std::sqrt(diag_h[i]);
        cv::Mat J_h = Ja.rowRange(0, m).clone();
        cv::Mat U, S, Vt;
        cv::SVD::compute(Ja, S, U, Vt);
        Vec s(n), uf(n);
        for (int i = 0; i < n; ++i) {
            s[i] = S.at<double>(i);
            double a = 0;
            for (int r = 0; r < m; ++r) a += U.at<double>(r, i) * f[r];
            uf[i] = a;
        }
        double theta = std::max(0.995, 1 - g_norm);
        double actual_reduction = -1;
        while (actual_reduction <= 0 && nfev < max_nfev) {
            // solve_lsq_trust_region
            Vec p_h(n);
            {
                Vec suf(n);
                for (int i = 0; i < n; ++i) suf[i] = s[i] * uf[i];
                bool full_rank = m >= n ? s[n - 1] > EPS * m * s[0] : false;
                bool done = false;
                if (full_rank) {
                    for (int r = 0; r < n; ++r) { double a = 0; for (int i = 0; i < n; ++i) a += Vt.at<double>(i, r) * (uf[i] / s[i]); p_h[r] = -a; }
                    if (nrm(p_h) <= Delta) { alpha = 0.0; done = true; }
                }
                if (!done) {
                    auto phi_d = [&](double al, double& phi, double& dphi) {
                        double pn = 0, sd = 0;
                        for (int i = 0; i < n; ++i) { double den = s[i] * s[i] + al; pn += (suf[i] / den) * (suf[i] / den); sd += suf[i] * suf[i] / (den * den * den); }
                        pn = std::sqrt(pn);
                        phi = pn - Delta;
                        dphi = -sd / pn;
                    };
                    double au = nrm(suf) / Delta, al_lo = 0.0;
                    if (full_rank) { double ph, dp; phi_d(0.0, ph, dp); al_lo = -ph / dp; }
                    double a = alpha;
                    if (!full_rank && alpha == 0) a = std::max(0.001 * au, std::sqrt(al_lo * au));
                    for (int it = 0; it < 10; ++it) {
                        if (a < al_lo || a > au) a = std::max(0.001 * au, std::sqrt(al_lo * au));
                        double ph, dp;
                        phi_d(a, ph, dp);
                        if (ph < 0) au = a;
                        double ratio = ph / dp;
                        al_lo = std::max(al_lo, a - ratio);
                        a -= (ph + Delta) * ratio / Delta;
                        if (std::abs(ph) < 0.01 * Delta) break;
                    }
                    for (int r = 0; r < n; ++r) {
                        double acc = 0;
                        for (int i = 0; i < n; ++i) acc += Vt.at<double>(i, r) * (suf[i] / (s[i] * s[i] + a));
                        p_h[r] = -acc;
                    }
                    double pn = nrm(p_h);
                    for (auto& e : p_h) e *= Delta / pn;
                    alpha = a;
                }
            }
            Vec p(n);
            for (int i = 0; i < n; ++i) p[i] = d[i] * p_h[i];
            // select_step
            Vec step, step_h;
            double pred;
            {
                Vec xp(n);
                for (int i = 0; i < n; ++i) xp[i] = x[i] + p[i];
                if (in_bounds(xp)) {
                    step = p; step_h = p_h; pred = -eval_quad(J_h, g_h, p_h, diag_h);
                } else {
                    std::vector<int> hits;
                    double p_stride = step_to_bound(x, p, hits);
                    Vec r_h = p_h;
                    for (int i = 0; i < n; ++i) if (hits[i] != 0) r_h[i] *= -1;
                    Vec r(n);
                    for (int i = 0; i < n; ++i) r[i] = d[i] * r_h[i];
                    for (int i = 0; i < n; ++i) { p[i] *= p_stride; p_h[i] *= p_stride; }
                    Vec xob(n);
                    for (int i = 0; i < n; ++i) xob[i] = x[i] + p[i];
                    double t1, to_tr;
                    intersect_tr(p_h, r_h, Delta, t1, to_tr);
                    std::vector<int> h2;
                    double to_bound = step_to_bound(xob, r, h2);
                    double r_stride = std::min(to_bound, to_tr), rl, ru;
                    if (r_stride > 0) {
                        rl = (1 - theta) * p_stride / r_stride;
                        ru = r_stride == to_bound ? theta * to_bound : to_tr;
                    } else { rl = 0; ru = -1; }
                    double r_value;
                    if (rl <= ru) {
                        double a, b, c, tb;
                        build_q1d(J_h, g_h, r_h, diag_h, &p_h, a, b, c);
                        r_value = min_q1d(a, b, rl, ru, c, tb);
                        for (int i = 0; i < n; ++i) { r_h[i] = r_h[i] * tb + p_h[i]; r[i] = r_h[i] * d[i]; }
                    } else r_value = INFINITY;
                    for (int i = 0; i < n; ++i) { p[i] *= theta; p_h[i] *= theta; }
                    double p_value = eval_quad(J_h, g_h, p_h, diag_h);
                    Vec ag_h(n), ag(n);
                    for (int i = 0; i < n; ++i) { ag_h[i] = -g_h[i]; ag[i] = d[i] * ag_h[i]; }
                    double to_tr2 = Delta / nrm(ag_h);
                    std::vector<int> h3;
                    double tb2 = step_to_bound(x, ag, h3);
                    double ag_stride = tb2 < to_tr2 ? theta * tb2 : to_tr2;
                    double a, b, c0, tb;
                    build_q1d(J_h, g_h, ag_h, diag_h, nullptr, a, b, c0);
                    double ag_value = min_q1d(a, b, 0, ag_stride, 0, tb);
                    for (int i = 0; i < n; ++i) { ag_h[i] *= tb; ag[i] *= tb; }
                    if (p_value < r_value && p_value < ag_value) { step = p; step_h = p_h; pred = -p_value; }
                    else if (r_value < p_value && r_value < ag_value) { step = r; step_h = r_h; pred = -r_value; }
                    else { step = ag; step_h = ag_h; pred = -ag_value; }
                }
            }
            x_new.assign(n, 0);
            for (int i = 0; i < n; ++i) {
                double xv = x[i] + step[i];
                if (xv <= lb[i] && xv - lb[i] <= ub[i] - xv) xv = std::nextafter(lb[i], ub[i]);
                else if (xv >= ub[i] && ub[i] - xv <= xv - lb[i]) xv = std::nextafter(ub[i], lb[i]);
                if (xv < lb[i] || xv > ub[i]) xv = 0.5 * (lb[i] + ub[i]);
                x_new[i] = xv;
            }
            f_new = fun(x_new);
            ++nfev;
            double step_h_norm = nrm(step_h);
            bool finite = true;
            for (double e : f_new) if (!std::isfinite(e)) { finite = false; break; }
            if (!finite) { Delta = 0.25 * step_h_norm; continue; }
            cost_new = cost_only(f_new);
            actual_reduction = cost - cost_new;
            // update_tr_radius
            double ratio, Delta_new = Delta;
            if (pred > 0) ratio = actual_reduction / pred;
            else if (pred == actual_reduction && pred == 0) ratio = 1;
            else ratio = 0;
            if (ratio < 0.25) Delta_new = 0.25 * step_h_norm;
            else if (ratio > 0.75 && step_h_norm > 0.95 * Delta) Delta_new = Delta * 2.0;
            double step_norm = nrm(step);
            // check_termination
            bool ftol_ok = actual_reduction < ftol * cost && ratio > 0.25;
            bool xtol_ok = step_norm < xtol * (xtol + nrm(x));
            if (ftol_ok && xtol_ok) status = 4;
            else if (ftol_ok) status = 2;
            else if (xtol_ok) status = 3;
            if (status != -1) break;
            alpha *= Delta / Delta_new;
            Delta = Delta_new;
        }
        if (actual_reduction > 0) {
            x = x_new;
            f = f_new;
            cost = cost_new;
            J = jac(x, f);
            robust(J, f);
            g = grad(J, f);
        }
    }
    if (status == -1) status = 0;
    for (int i = 0; i < n; ++i) R.p[i] = x[i];
    R.status = status;
    R.ok = status > 0;
    R.nfev = nfev;
    return R;
}
}  // namespace trf

// numpy.percentile (linear)
inline double np_percentile(std::vector<double> v, double q) {
    std::sort(v.begin(), v.end());
    double k = (v.size() - 1) * q / 100.0;
    size_t i = (size_t)std::floor(k);
    double f = k - i;
    return i + 1 < v.size() ? v[i] + (v[i + 1] - v[i]) * f : v[i];
}

// refine.py 의 fit(xy, I, start, axes, patch)
inline FitResult fit(const std::vector<double>& X, const std::vector<double>& Y, const std::vector<double>& I, double sx, double sy,
                     const float inv[2][2], const double patch[2][2]) {
    double low = np_percentile(I, 10), high = np_percentile(I, 90), contrast = std::max(high - low, 20.0);
    std::vector<double> p0{sx, sy, 1., 1., low, contrast};
    std::vector<double> lo{sx - 3.8, sy - 3.8, .35, .35, 0, 10}, hi{sx + 3.8, sy + 3.8, 2.7, 2.7, 240, 255};
    Model M{X.data(), Y.data(), I.data(), (int)X.size()};
    for (int a = 0; a < 2; ++a) for (int b = 0; b < 2; ++b) { M.inv[a][b] = inv[a][b]; M.patch[a][b] = patch[a][b]; }
    FitResult R = trf::least_squares(M, p0, lo, hi, 6.0, 85, 1e-7, 1e-7, 1e-7);
    std::vector<double> r(X.size());
    M.residual(R.p, r.data());
    double s = 0;
    for (double e : r) s += e * e;
    R.rmse = std::sqrt(s / r.size());
    return R;
}

// 반경 radius 창(정수 중심 = rint(center)), 판 영역 [a,c)x[b,d) 안쪽 화소만
inline void window(const cv::Mat& img, double cx, double cy, int radius, double a, double b, double c, double d, std::vector<double>& X,
                   std::vector<double>& Y, std::vector<double>& I) {
    X.clear(); Y.clear(); I.clear();
    int x0 = (int)std::nearbyint(cx), y0 = (int)std::nearbyint(cy);  // 기본 반올림 모드 = 짝수 쪽 (np.rint)
    for (int yy = y0 - radius; yy <= y0 + radius; ++yy)
        for (int xx = x0 - radius; xx <= x0 + radius; ++xx) {
            if (!(xx >= a && xx < c && yy >= b && yy < d)) continue;
            X.push_back(xx); Y.push_back(yy);
            I.push_back(img.type() == CV_8U ? (double)img.at<uchar>(yy, xx) : img.at<double>(yy, xx));
        }
}

// 통과 기준 (refine.py, track_corners.py 와 같음)
inline std::string reasons(bool ok1, bool ok2, double rmse, double contrast, double stab, double dmax, double sigmax) {
    std::string r;
    auto add = [&](const char* s) { r += (r.empty() ? "" : ";") + std::string(s); };
    if (!(ok1 && ok2)) add("optimizer");
    if (rmse > 15) add("photometric_rmse");
    if (contrast < 45) add("low_contrast");
    if (stab > .65) add("window_instability");
    if (dmax > 3.6) add("search_limit");
    if (sigmax > 2.6) add("blur_limit");
    return r;
}

}  // namespace cf
