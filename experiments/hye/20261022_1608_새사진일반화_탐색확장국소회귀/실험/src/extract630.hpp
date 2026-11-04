// 고정 630 구조 모서리 측정 (원본: joint_yt_generalization_20261022/measurements/extractor_scipy.py,
// joint_center_630_20261022/reference.py).
// QR 세 영역 NCC(±130 → 실패 시 ±260 → ±512) → 모서리마다 흐린 격자 경계(erf) 6변수 모델을
// scipy.optimize.least_squares(method='trf', loss='soft_l1', f_scale=6, max_nfev=85, 2-point 미분, 경계 있음)로 맞춘다.
// least_squares 의 TRF(경계 반사, exact 신뢰영역 풀이)를 수식대로 옮겼다.
#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
#include "corners630.hpp"

namespace e630 {

constexpr int kN = 630;
constexpr double kEps = 2.220446049250313e-16;
const int kBoxes[3][4] = {{1552, 969, 1736, 1167}, {1809, 971, 1993, 1169}, {2067, 973, 2252, 1171}};

using Vec6 = std::array<double, 6>;

// numpy np.sum 과 같은 순서(8칸 펼침 pairwise, 128 이하)
inline double np_sum(const double* a, int n) {
    if (n < 8) {
        double r = 0;
        for (int i = 0; i < n; ++i) r += a[i];
        return r;
    }
    if (n <= 128) {
        double r[8];
        for (int j = 0; j < 8; ++j) r[j] = a[j];
        int i = 8;
        for (; i < n - (n % 8); i += 8)
            for (int j = 0; j < 8; ++j) r[j] += a[i + j];
        double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) res += a[i];
        return res;
    }
    int n2 = n / 2;
    n2 -= n2 % 8;
    return np_sum(a, n2) + np_sum(a + n2, n - n2);
}
inline double dot(const double* a, const double* b, int n) {
    double s = 0;
    for (int i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}
inline double nrm(const double* a, int n) { return std::sqrt(dot(a, a, n)); }

struct Problem {
    const std::vector<double>* x;
    const std::vector<double>* y;
    const std::vector<double>* I;
    double inv[4];  // float32 역행렬을 double 로
    double patch[4];
    int m() const { return (int)I->size(); }
    void residual(const double* p, double* r) const {
        const double s2 = std::sqrt(2.0);
        for (int i = 0; i < m(); ++i) {
            double dx = (*x)[i] - p[0], dy = (*y)[i] - p[1];
            double lu = dx * inv[0] + dy * inv[1], lv = dx * inv[2] + dy * inv[3];
            double U = .5 * (1 + std::erf(lu / (s2 * p[2]))), V = .5 * (1 + std::erf(lv / (s2 * p[3])));
            double F = patch[0] * (1 - U) * (1 - V) + patch[1] * U * (1 - V) + patch[2] * (1 - U) * V + patch[3] * U * V;
            r[i] = p[4] + p[5] * (1 - F) - (*I)[i];
        }
    }
};

// float32 2x2 역행렬 (LAPACK sgesv 와 같은 부분 피벗 LU, float 연산)
inline void inv2f(const float a[4], double out[4]) {
    float A00 = a[0], A01 = a[1], A10 = a[2], A11 = a[3];
    bool sw = std::fabs(A10) > std::fabs(A00);
    if (sw) { std::swap(A00, A10); std::swap(A01, A11); }
    float l = A10 * (1.0f / A00);
    float u11 = A11 - l * A01;
    // 단위행렬 각 열 (행 교환 반영)
    for (int c = 0; c < 2; ++c) {
        float b0 = (c == 0) ? 1.0f : 0.0f, b1 = (c == 1) ? 1.0f : 0.0f;
        if (sw) std::swap(b0, b1);
        float y0 = b0, y1 = b1 - l * y0;
        float x1 = y1 / u11;
        float x0 = (y0 - x1 * A01) / A00;
        out[0 * 2 + c] = x0;
        out[1 * 2 + c] = x1;
    }
}

struct LsqResult {
    Vec6 x;
    bool success = false;
    int nfev = 0;
};

// scipy.optimize._lsq.trf.trf_bounds (tr_solver='exact', x_scale=1) + soft_l1 + 2-point
class TRF {
public:
    TRF(const Problem& pr, const Vec6& lb, const Vec6& ub) : P(pr), lb(lb), ub(ub), m(pr.m()) {}

    LsqResult solve(Vec6 x0, int max_nfev, double ftol, double xtol, double gtol) {
        x0 = strictly_feasible(x0, 1e-10);
        Vec6 x = x0;
        std::vector<double> f(m), ftrue(m), J(m * 6), fnew(m);
        P.residual(x.data(), f.data());
        ftrue = f;
        int nfev = 1;
        jac(x, ftrue, J);
        double cost = robust(f, J, true);
        Vec6 g = grad(J, f);
        Vec6 v, dv;
        CL(x, g, v, dv);
        double Delta;
        {
            double t[6];
            for (int i = 0; i < 6; ++i) t[i] = x0[i] / std::sqrt(v[i]);
            Delta = nrm(t, 6);
            if (Delta == 0) Delta = 1.0;
        }
        double alpha = 0;
        int status = -100;
        const int M = m + 6;
        std::vector<double> Jh(m * 6);
        while (true) {
            CL(x, g, v, dv);
            double gn = 0;
            for (int i = 0; i < 6; ++i) gn = std::max(gn, std::abs(g[i] * v[i]));
            if (gn < gtol) status = 1;
            if (status != -100 || nfev == max_nfev) break;
            Vec6 d, diag_h, g_h;
            for (int i = 0; i < 6; ++i) {
                d[i] = std::sqrt(v[i]);
                diag_h[i] = g[i] * dv[i];
                g_h[i] = d[i] * g[i];
            }
            cv::Mat Ja(M, 6, CV_64F, cv::Scalar(0)), fa(M, 1, CV_64F, cv::Scalar(0));
            for (int i = 0; i < m; ++i) {
                fa.at<double>(i) = f[i];
                for (int k = 0; k < 6; ++k) Ja.at<double>(i, k) = Jh[i * 6 + k] = J[i * 6 + k] * d[k];
            }
            for (int k = 0; k < 6; ++k) Ja.at<double>(m + k, k) = std::sqrt(diag_h[k]);
            cv::Mat w, u, vt;
            cv::SVD::compute(Ja, w, u, vt);
            cv::Mat uf = u.t() * fa;
            double theta = std::max(0.995, 1 - gn);
            double actual = -1, cost_new = 0;
            Vec6 xnew;
            while (actual <= 0 && nfev < max_nfev) {
                Vec6 p_h;
                alpha = trust_region(w, vt, uf, Delta, alpha, p_h);
                Vec6 p;
                for (int i = 0; i < 6; ++i) p[i] = d[i] * p_h[i];
                Vec6 step, step_h;
                double pred = select_step(x, Jh, diag_h, g_h, p, p_h, d, Delta, theta, step, step_h);
                Vec6 xs;
                for (int i = 0; i < 6; ++i) xs[i] = x[i] + step[i];
                xnew = strictly_feasible(xs, 0);
                P.residual(xnew.data(), fnew.data());
                ++nfev;
                double shn = nrm(step_h.data(), 6);
                bool finite = true;
                for (double q : fnew) finite &= std::isfinite(q);
                if (!finite) { Delta = 0.25 * shn; continue; }
                cost_new = cost_only(fnew);
                actual = cost - cost_new;
                double ratio;
                double Dn = update_radius(Delta, actual, pred, shn, shn > 0.95 * Delta, ratio);
                double sn = nrm(step.data(), 6), xn = nrm(x.data(), 6);
                int t = termination(actual, cost, sn, xn, ratio, ftol, xtol);
                if (t != -100) { status = t; break; }
                alpha *= Delta / Dn;
                Delta = Dn;
            }
            if (actual > 0) {
                x = xnew;
                f = fnew;
                ftrue = f;
                cost = cost_new;
                jac(x, ftrue, J);
                robust(f, J, false);
                g = grad(J, f);
            }
        }
        if (status == -100) status = 0;
        LsqResult r;
        r.x = x;
        r.success = status > 0;
        r.nfev = nfev;
        return r;
    }

private:
    const Problem& P;
    Vec6 lb, ub;
    int m;

    Vec6 strictly_feasible(const Vec6& x, double rstep) const {
        Vec6 o = x;
        for (int i = 0; i < 6; ++i) {
            int act = 0;
            if (rstep == 0) {
                if (x[i] <= lb[i]) act = -1;
                if (x[i] >= ub[i]) act = 1;
            } else {
                double ld = x[i] - lb[i], ud = ub[i] - x[i];
                double lt = rstep * std::max(1.0, std::abs(lb[i])), ut = rstep * std::max(1.0, std::abs(ub[i]));
                if (ld <= std::min(ud, lt)) act = -1;
                if (ud <= std::min(ld, ut)) act = 1;
            }
            if (rstep == 0) {
                if (act == -1) o[i] = std::nextafter(lb[i], ub[i]);
                if (act == 1) o[i] = std::nextafter(ub[i], lb[i]);
            } else {
                if (act == -1) o[i] = lb[i] + rstep * std::max(1.0, std::abs(lb[i]));
                if (act == 1) o[i] = ub[i] - rstep * std::max(1.0, std::abs(ub[i]));
            }
            if (o[i] < lb[i] || o[i] > ub[i]) o[i] = 0.5 * (lb[i] + ub[i]);
        }
        return o;
    }
    // 2-point, 한쪽 차분, 경계 조정 (scipy _numdiff)
    void jac(const Vec6& x0, const std::vector<double>& f0, std::vector<double>& J) const {
        const double rstep = 1.4901161193847656e-08;
        std::vector<double> f1(m);
        for (int i = 0; i < 6; ++i) {
            double sgn = x0[i] >= 0 ? 1.0 : -1.0;
            double h = rstep * sgn * std::max(1.0, std::abs(x0[i]));
            double ld = x0[i] - lb[i], ud = ub[i] - x0[i];
            double xx = x0[i] + h;
            bool violated = xx < lb[i] || xx > ub[i];
            bool fitting = std::abs(h) <= std::max(ld, ud);
            if (violated && fitting) h = -h;
            if (!fitting) h = ud >= ld ? ud : -ld;
            Vec6 x = x0;
            x[i] = x0[i] + h;
            double dx = x[i] - x0[i];
            P.residual(x.data(), f1.data());
            for (int r = 0; r < m; ++r) J[r * 6 + i] = (f1[r] - f0[r]) / dx;
        }
    }
    // soft_l1, f_scale 6: J, f 를 제자리에서 바꾸고 cost 반환
    double robust(std::vector<double>& f, std::vector<double>& J, bool want_cost) const {
        const double fs = 6.0;
        std::vector<double> r0(m);
        for (int i = 0; i < m; ++i) {
            double z = (f[i] / fs) * (f[i] / fs), t = 1 + z;
            double rho0 = 2 * (std::sqrt(t) - 1) * (fs * fs);
            double rho1 = 1 / std::sqrt(t);
            double rho2 = -0.5 * std::pow(t, -1.5) / (fs * fs);
            r0[i] = rho0;
            double js = rho1 + 2 * rho2 * f[i] * f[i];
            if (js < kEps) js = kEps;
            js = std::sqrt(js);
            f[i] *= rho1 / js;
            for (int k = 0; k < 6; ++k) J[i * 6 + k] *= js;
        }
        return want_cost ? 0.5 * np_sum(r0.data(), m) : 0;
    }
    double cost_only(const std::vector<double>& f) const {
        const double fs = 6.0;
        std::vector<double> r0(m);
        for (int i = 0; i < m; ++i) {
            double z = (f[i] / fs) * (f[i] / fs);
            r0[i] = 2 * (std::sqrt(1 + z) - 1);
        }
        return 0.5 * fs * fs * np_sum(r0.data(), m);
    }
    Vec6 grad(const std::vector<double>& J, const std::vector<double>& f) const {
        Vec6 g{};
        for (int k = 0; k < 6; ++k) {
            double s = 0;
            for (int i = 0; i < m; ++i) s += J[i * 6 + k] * f[i];
            g[k] = s;
        }
        return g;
    }
    void CL(const Vec6& x, const Vec6& g, Vec6& v, Vec6& dv) const {
        for (int i = 0; i < 6; ++i) {
            v[i] = 1;
            dv[i] = 0;
            if (g[i] < 0) { v[i] = ub[i] - x[i]; dv[i] = -1; }
            if (g[i] > 0) { v[i] = x[i] - lb[i]; dv[i] = 1; }
        }
    }
    double trust_region(const cv::Mat& s, const cv::Mat& vt, const cv::Mat& uf, double Delta, double initial_alpha, Vec6& p) const {
        const int n = 6, M = m + 6;
        double sv[6], u[6], suf[6];
        for (int i = 0; i < n; ++i) { sv[i] = s.at<double>(i); u[i] = uf.at<double>(i); suf[i] = sv[i] * u[i]; }
        auto Vdot = [&](const double* c, Vec6& out) {
            for (int r = 0; r < n; ++r) {
                double a = 0;
                for (int k = 0; k < n; ++k) a += vt.at<double>(k, r) * c[k];
                out[r] = a;
            }
        };
        auto phi_d = [&](double al, double& phi, double& dphi) {
            double q[6], q3 = 0;
            for (int i = 0; i < n; ++i) {
                double den = sv[i] * sv[i] + al;
                q[i] = suf[i] / den;
                q3 += suf[i] * suf[i] / (den * den * den);
            }
            double pn = nrm(q, n);
            phi = pn - Delta;
            dphi = -q3 / pn;
        };
        bool full_rank = (M >= n) && sv[n - 1] > kEps * M * sv[0];
        if (full_rank) {
            double c[6];
            for (int i = 0; i < n; ++i) c[i] = -(u[i] / sv[i]);
            Vdot(c, p);
            if (nrm(p.data(), n) <= Delta) return 0.0;
        }
        double alpha_upper = nrm(suf, n) / Delta, alpha_lower = 0;
        if (full_rank) {
            double phi, dphi;
            phi_d(0.0, phi, dphi);
            alpha_lower = -phi / dphi;
        }
        double alpha;
        if (!full_rank && initial_alpha == 0)
            alpha = std::max(0.001 * alpha_upper, std::sqrt(alpha_lower * alpha_upper));
        else
            alpha = initial_alpha;
        for (int it = 0; it < 10; ++it) {
            if (alpha < alpha_lower || alpha > alpha_upper) alpha = std::max(0.001 * alpha_upper, std::sqrt(alpha_lower * alpha_upper));
            double phi, dphi;
            phi_d(alpha, phi, dphi);
            if (phi < 0) alpha_upper = alpha;
            double ratio = phi / dphi;
            alpha_lower = std::max(alpha_lower, alpha - ratio);
            alpha -= (phi + Delta) * ratio / Delta;
            if (std::abs(phi) < 0.01 * Delta) break;
        }
        double c[6];
        for (int i = 0; i < n; ++i) c[i] = -(suf[i] / (sv[i] * sv[i] + alpha));
        Vdot(c, p);
        double pn = nrm(p.data(), n);
        for (int i = 0; i < n; ++i) p[i] *= Delta / pn;
        return alpha;
    }
    double quad(const std::vector<double>& Jh, const Vec6& g, const Vec6& s, const Vec6& diag) const {
        std::vector<double> js(m);
        for (int i = 0; i < m; ++i) js[i] = dot(&Jh[i * 6], s.data(), 6);
        double q = dot(js.data(), js.data(), m);
        for (int i = 0; i < 6; ++i) q += s[i] * diag[i] * s[i];
        return 0.5 * q + dot(s.data(), g.data(), 6);
    }
    void quad1d(const std::vector<double>& Jh, const Vec6& g, const Vec6& s, const Vec6& diag, const Vec6* s0, double& a, double& b,
                double& c) const {
        std::vector<double> v(m), u(m);
        for (int i = 0; i < m; ++i) v[i] = dot(&Jh[i * 6], s.data(), 6);
        a = dot(v.data(), v.data(), m);
        for (int i = 0; i < 6; ++i) a += s[i] * diag[i] * s[i];
        a *= 0.5;
        b = dot(g.data(), s.data(), 6);
        c = 0;
        if (s0) {
            for (int i = 0; i < m; ++i) u[i] = dot(&Jh[i * 6], s0->data(), 6);
            b += dot(u.data(), v.data(), m);
            c = 0.5 * dot(u.data(), u.data(), m) + dot(g.data(), s0->data(), 6);
            for (int i = 0; i < 6; ++i) b += (*s0)[i] * diag[i] * s[i];
            for (int i = 0; i < 6; ++i) c += 0.5 * (*s0)[i] * diag[i] * (*s0)[i];
        }
    }
    static void min1d(double a, double b, double lo, double hi, double c, double& t, double& y) {
        double ts[3] = {lo, hi, 0};
        int n = 2;
        if (a != 0) {
            double e = -0.5 * b / a;
            if (lo < e && e < hi) ts[n++] = e;
        }
        int best = 0;
        double by = 0;
        for (int i = 0; i < n; ++i) {
            double yy = ts[i] * (a * ts[i] + b) + c;
            if (i == 0 || yy < by) { by = yy; best = i; }
        }
        t = ts[best];
        y = by;
    }
    double to_bound(const Vec6& x, const Vec6& s, Vec6* hits) const {
        double st[6], mn = std::numeric_limits<double>::infinity();
        for (int i = 0; i < 6; ++i) {
            st[i] = std::numeric_limits<double>::infinity();
            if (s[i] != 0) st[i] = std::max((lb[i] - x[i]) / s[i], (ub[i] - x[i]) / s[i]);
            mn = std::min(mn, st[i]);
        }
        if (hits)
            for (int i = 0; i < 6; ++i) (*hits)[i] = (st[i] == mn) ? (s[i] > 0 ? 1.0 : s[i] < 0 ? -1.0 : 0.0) : 0.0;
        return mn;
    }
    static void intersect(const Vec6& x, const Vec6& s, double Delta, double& t1, double& t2) {
        double a = dot(s.data(), s.data(), 6), b = dot(x.data(), s.data(), 6), c = dot(x.data(), x.data(), 6) - Delta * Delta;
        if (a == 0 || c > 0) throw std::runtime_error("intersect_trust_region");
        double d = std::sqrt(b * b - a * c);
        double q = -(b + std::copysign(d, b));
        double u1 = q / a, u2 = c / q;
        if (u1 < u2) { t1 = u1; t2 = u2; } else { t1 = u2; t2 = u1; }
    }
    double select_step(const Vec6& x, const std::vector<double>& Jh, const Vec6& diag_h, const Vec6& g_h, Vec6 p, Vec6 p_h, const Vec6& d,
                       double Delta, double theta, Vec6& step, Vec6& step_h) const {
        bool inb = true;
        for (int i = 0; i < 6; ++i) { double q = x[i] + p[i]; inb &= (q >= lb[i] && q <= ub[i]); }
        if (inb) {
            step = p; step_h = p_h;
            return -quad(Jh, g_h, p_h, diag_h);
        }
        Vec6 hits;
        double p_stride = to_bound(x, p, &hits);
        Vec6 r_h = p_h;
        for (int i = 0; i < 6; ++i) if (hits[i] != 0) r_h[i] *= -1;
        Vec6 r;
        for (int i = 0; i < 6; ++i) r[i] = d[i] * r_h[i];
        for (int i = 0; i < 6; ++i) { p[i] *= p_stride; p_h[i] *= p_stride; }
        Vec6 xb;
        for (int i = 0; i < 6; ++i) xb[i] = x[i] + p[i];
        double t1, to_tr;
        intersect(p_h, r_h, Delta, t1, to_tr);
        double tb = to_bound(xb, r, nullptr);
        double r_stride = std::min(tb, to_tr), rl, ru;
        if (r_stride > 0) {
            rl = (1 - theta) * p_stride / r_stride;
            ru = (r_stride == tb) ? theta * tb : to_tr;
        } else { rl = 0; ru = -1; }
        double r_value;
        if (rl <= ru) {
            double a, b, c;
            quad1d(Jh, g_h, r_h, diag_h, &p_h, a, b, c);
            double rs;
            min1d(a, b, rl, ru, c, rs, r_value);
            for (int i = 0; i < 6; ++i) { r_h[i] = r_h[i] * rs + p_h[i]; r[i] = r_h[i] * d[i]; }
        } else r_value = std::numeric_limits<double>::infinity();
        for (int i = 0; i < 6; ++i) { p[i] *= theta; p_h[i] *= theta; }
        double p_value = quad(Jh, g_h, p_h, diag_h);
        Vec6 ag_h, ag;
        for (int i = 0; i < 6; ++i) { ag_h[i] = -g_h[i]; ag[i] = d[i] * ag_h[i]; }
        double tt = Delta / nrm(ag_h.data(), 6);
        double tbb = to_bound(x, ag, nullptr);
        double ag_stride = tbb < tt ? theta * tbb : tt;
        double a, b, c;
        quad1d(Jh, g_h, ag_h, diag_h, nullptr, a, b, c);
        double ag_value;
        min1d(a, b, 0, ag_stride, 0, ag_stride, ag_value);
        for (int i = 0; i < 6; ++i) { ag_h[i] *= ag_stride; ag[i] *= ag_stride; }
        if (p_value < r_value && p_value < ag_value) { step = p; step_h = p_h; return -p_value; }
        if (r_value < p_value && r_value < ag_value) { step = r; step_h = r_h; return -r_value; }
        step = ag; step_h = ag_h;
        return -ag_value;
    }
    static double update_radius(double Delta, double actual, double pred, double sn, bool hit, double& ratio) {
        if (pred > 0) ratio = actual / pred;
        else if (pred == actual && actual == 0) ratio = 1;
        else ratio = 0;
        if (ratio < 0.25) Delta = 0.25 * sn;
        else if (ratio > 0.75 && hit) Delta *= 2.0;
        return Delta;
    }
    static int termination(double dF, double F, double dxn, double xn, double ratio, double ftol, double xtol) {
        bool fs = dF < ftol * F && ratio > 0.25, xs = dxn < xtol * (xtol + xn);
        if (fs && xs) return 4;
        if (fs) return 2;
        if (xs) return 3;
        return -100;
    }
};

inline double pct(std::vector<double> v, double q) {  // numpy percentile linear
    std::sort(v.begin(), v.end());
    double k = (v.size() - 1) * q / 100.0;
    size_t i = (size_t)std::floor(k);
    double f = k - i;
    return i + 1 < v.size() ? v[i] + (v[i + 1] - v[i]) * f : v[i];
}

struct Fit {
    Vec6 p;
    double rmse = 0;
    bool ok = false;
};

inline Fit fit(const std::vector<double>& xs, const std::vector<double>& ys, const std::vector<double>& I, cv::Point2d start, const c630::Corner& c) {
    double low = pct(I, 10), high = pct(I, 90), contrast = std::max(high - low, 20.0);
    Vec6 p0{start.x, start.y, 1., 1., low, contrast};
    Vec6 lo{start.x - 3.8, start.y - 3.8, .35, .35, 0, 10}, hi{start.x + 3.8, start.y + 3.8, 2.7, 2.7, 240, 255};
    Problem pr;
    pr.x = &xs; pr.y = &ys; pr.I = &I;
    // numpy: inv(axes float32) -> float32, 국소좌표 = (xy-center) @ inv.T
    inv2f(c.axes, pr.inv);
    for (int k = 0; k < 4; ++k) pr.patch[k] = c.patch[k];
    TRF t(pr, lo, hi);
    auto r = t.solve(p0, 85, 1e-7, 1e-7, 1e-7);
    Fit f;
    f.p = r.x;
    f.ok = r.success;
    std::vector<double> res(I.size());
    pr.residual(r.x.data(), res.data());
    std::vector<double> sq(res.size());
    for (size_t i = 0; i < res.size(); ++i) sq[i] = res[i] * res[i];
    f.rmse = std::sqrt(np_sum(sq.data(), (int)sq.size()) / sq.size());
    return f;
}

// joint_center_630_20261022/measurement.py 의 Numba 컴파일 LM (경계 자르기 +-1e-12, 해석 미분, 85회 평가). 같은 erf 모델.
inline Fit fit_lm(const std::vector<double>& xs, const std::vector<double>& ys, const std::vector<double>& I, cv::Point2d start, const c630::Corner& c) {
    const int n = (int)I.size();
    double inv[4];
    inv2f(c.axes, inv);
    const double* pa = c.patch;
    const double du = pa[1] - pa[0], dv = pa[2] - pa[0], cross = pa[3] - pa[1] - pa[2] + pa[0];
    auto objective = [&](const Vec6& p, std::vector<double>& r, std::vector<double>& J) {
        double cost = 0;
        for (int i = 0; i < n; ++i) {
            double dx = xs[i] - p[0], dy = ys[i] - p[1];
            double u = dx * inv[0] + dy * inv[1], v = dx * inv[2] + dy * inv[3];
            double zu = u / p[2], zv = v / p[3];
            double U = .5 * (1 + std::erf(zu / std::sqrt(2.0))), V = .5 * (1 + std::erf(zv / std::sqrt(2.0)));
            double F = pa[0] * (1 - U) * (1 - V) + pa[1] * U * (1 - V) + pa[2] * (1 - U) * V + pa[3] * U * V;
            double rr = p[4] + p[5] * (1 - F) - I[i];
            r[i] = rr;
            cost += 36 * (std::sqrt(1 + (rr / 6) * (rr / 6)) - 1);
            double gu = (du + cross * V) * std::exp(-.5 * zu * zu) / std::sqrt(2 * M_PI), gv = (dv + cross * U) * std::exp(-.5 * zv * zv) / std::sqrt(2 * M_PI);
            double* j = &J[i * 6];
            j[0] = p[5] * (gu / p[2] * inv[0] + gv / p[3] * inv[2]);
            j[1] = p[5] * (gu / p[2] * inv[1] + gv / p[3] * inv[3]);
            j[2] = p[5] * gu * u / (p[2] * p[2]);
            j[3] = p[5] * gv * v / (p[3] * p[3]);
            j[4] = 1.;
            j[5] = 1 - F;
        }
        return cost;
    };
    std::vector<double> ord(I);
    std::sort(ord.begin(), ord.end());
    auto q_ = [&](double qq) {
        double q = qq * (n - 1);
        int k = (int)q;
        return ord[k] * (1 - (q - k)) + ord[std::min(k + 1, n - 1)] * (q - k);
    };
    double low = q_(.1), high = q_(.9);
    Vec6 p{start.x, start.y, 1., 1., low, std::max(high - low, 20.)};
    Vec6 lower{start.x - 3.8, start.y - 3.8, .35, .35, 0., 10.}, upper{start.x + 3.8, start.y + 3.8, 2.7, 2.7, 240., 255.};
    const double scales[6] = {1., 1., 1., 1., 60., 120.};
    double mu = 1e-3;
    bool converged = false;
    std::vector<double> r(n), J(n * 6), nr(n), nJ(n * 6);
    double cost = objective(p, r, J);
    int evals = 1;
    for (int iter = 0; iter < 85; ++iter) {
        cv::Matx66d H = cv::Matx66d::zeros();
        cv::Vec6d g(0, 0, 0, 0, 0, 0);
        for (int i = 0; i < n; ++i) {
            double root = std::sqrt(1 + (r[i] / 6) * (r[i] / 6)), w = 1 / (root * root * root);
            for (int j = 0; j < 6; ++j) {
                double jj = J[i * 6 + j] * scales[j];
                g[j] += jj * r[i] / root;
                for (int k = 0; k < 6; ++k) H(j, k) += jj * J[i * 6 + k] * scales[k] * w;
            }
        }
        double pgmax = 0;
        for (int j = 0; j < 6; ++j) {
            double pg = g[j];
            if ((p[j] - lower[j] < 1e-9 && g[j] > 0) || (upper[j] - p[j] < 1e-9 && g[j] < 0)) pg = 0;
            pgmax = std::max(pgmax, std::abs(pg));
        }
        if (pgmax < 1e-7) { converged = true; break; }
        for (int j = 0; j < 6; ++j)
            if ((p[j] - lower[j] < 1e-8 && g[j] > 0) || (upper[j] - p[j] < 1e-8 && g[j] < 0)) {
                for (int k = 0; k < 6; ++k) { H(j, k) = 0; H(k, j) = 0; }
                H(j, j) = 1;
                g[j] = 0;
            }
        double diag[6];
        for (int j = 0; j < 6; ++j) diag[j] = std::max(H(j, j), 1e-6);
        bool accepted = false;
        for (int retry = 0; retry < 14; ++retry) {
            cv::Matx66d A = H;
            for (int j = 0; j < 6; ++j) A(j, j) += mu * diag[j];
            cv::Vec6d delta;
            cv::solve(A, -g, delta, cv::DECOMP_LU);
            Vec6 trial;
            for (int j = 0; j < 6; ++j) trial[j] = std::min(std::max(p[j] + delta[j] * scales[j], lower[j] + 1e-12), upper[j] - 1e-12);
            double ncost = objective(trial, nr, nJ);
            ++evals;
            double sn = 0, pn = 0;
            for (int j = 0; j < 6; ++j) { sn += (trial[j] - p[j]) * (trial[j] - p[j]); pn += p[j] * p[j]; }
            sn = std::sqrt(sn); pn = std::sqrt(pn);
            if (sn < 1e-7 * (1e-7 + pn)) {
                converged = true;
                if (ncost < cost) { p = trial; cost = ncost; r.swap(nr); J.swap(nJ); }
                accepted = true;
                break;
            }
            if (ncost < cost) {
                double improvement = cost - ncost;
                p = trial; cost = ncost; r.swap(nr); J.swap(nJ);
                mu = std::max(mu * .3, 1e-12);
                accepted = true;
                if (improvement < 1e-7 * std::max(cost, 1e-12) || sn < 1e-7 * (1e-7 + pn)) converged = true;
                break;
            }
            mu = std::min(mu * 10, 1e14);
            if (evals >= 85) break;
        }
        if (converged || evals >= 85 || !accepted) break;
    }
    Fit f;
    f.p = p;
    f.ok = converged;
    double s = 0;
    for (int i = 0; i < n; ++i) s += r[i] * r[i];
    f.rmse = std::sqrt(s / n);
    return f;
}

struct Measure {
    std::vector<cv::Point2d> cur;          // 측정 좌표
    std::vector<std::array<double, 5>> q;  // rmse, stability, contrast, sigma_u, sigma_v
    std::vector<unsigned char> valid;
    std::vector<int> why;                  // 무효 이유 비트: 1 optimizer 2 rmse 4 contrast 8 window 16 search 32 blur
    double ncc_min = 0;
    int pad = 0;
    cv::Point shift[3];
    bool ok = false;
};

// img, center: 8bit 회색조 전체 사진. compiled = true 면 Numba LM 변형(fit_lm)으로 맞춘다.
inline Measure measure(const cv::Mat& img, const cv::Mat& center, bool compiled = false) {
    auto fitf = compiled ? fit_lm : fit;
    Measure out;
    double scores[3];
    for (int pad : {130, 260, 512}) {
        for (int k = 0; k < 3; ++k) {
            const int* b = kBoxes[k];
            cv::Mat corr;
            cv::matchTemplate(img(cv::Range(b[1] - pad, b[3] + pad), cv::Range(b[0] - pad, b[2] + pad)), center(cv::Range(b[1], b[3]), cv::Range(b[0], b[2])), corr,
                              cv::TM_CCOEFF_NORMED);
            double mx;
            cv::Point loc;
            cv::minMaxLoc(corr, nullptr, &mx, nullptr, &loc);
            out.shift[k] = {loc.x - pad, loc.y - pad};
            scores[k] = mx;
        }
        out.pad = pad;
        if (std::min({scores[0], scores[1], scores[2]}) >= .6) break;
    }
    out.ncc_min = std::min({scores[0], scores[1], scores[2]});
    if (out.ncc_min < .6) return out;
    out.ok = true;
    out.cur.resize(kN);
    out.q.resize(kN);
    out.valid.resize(kN);
    out.why.resize(kN);
    for (int i = 0; i < kN; ++i) {
        const auto& c = c630::kCorners[i];
        cv::Point s = out.shift[c.qr - 1];
        cv::Point2d initial(c.cx + s.x, c.cy + s.y);
        const int* b = kBoxes[c.qr - 1];
        int a = b[0] + s.x, cc = b[2] + s.x, bb = b[1] + s.y, d = b[3] + s.y;
        auto window = [&](cv::Point2d e, int rad, std::vector<double>& xs, std::vector<double>& ys, std::vector<double>& I) {
            xs.clear(); ys.clear(); I.clear();
            int x0 = (int)std::nearbyint(e.x), y0 = (int)std::nearbyint(e.y);
            for (int y = y0 - rad; y <= y0 + rad; ++y)
                for (int x = x0 - rad; x <= x0 + rad; ++x)
                    if (x >= a && x < cc && y >= bb && y < d) {
                        xs.push_back(x); ys.push_back(y); I.push_back(img.at<uchar>(y, x));
                    }
        };
        std::vector<double> xs, ys, I;
        cv::Point2d est = initial;
        Fit f1;
        for (int it = 0; it < 3; ++it) {
            window(est, 4, xs, ys, I);
            f1 = fitf(xs, ys, I, initial, c);
            bool moved = std::nearbyint(f1.p[0]) != std::nearbyint(est.x) || std::nearbyint(f1.p[1]) != std::nearbyint(est.y);
            est = {f1.p[0], f1.p[1]};
            if (!moved) break;
        }
        window(est, 5, xs, ys, I);
        Fit f2 = fitf(xs, ys, I, initial, c);
        double stab = std::hypot(f1.p[0] - f2.p[0], f1.p[1] - f2.p[1]);
        int why = 0;
        if (!(f1.ok && f2.ok)) why |= 1;
        if (f1.rmse > 15) why |= 2;
        if (f1.p[5] < 45) why |= 4;
        if (stab > .65) why |= 8;
        if (std::max(std::abs(f1.p[0] - initial.x), std::abs(f1.p[1] - initial.y)) > 3.6) why |= 16;
        if (std::max(f1.p[2], f1.p[3]) > 2.6) why |= 32;
        out.cur[i] = {f1.p[0], f1.p[1]};
        out.q[i] = {f1.rmse, stab, f1.p[5], f1.p[2], f1.p[3]};
        out.valid[i] = why == 0;
        out.why[i] = why;
    }
    return out;
}

}  // namespace e630
