// scipy.optimize.least_squares(method='trf', bounds, loss='soft_l1', jac='2-point', x_scale=1) 를 그대로 옮겼다.
// 원본: scipy/optimize/_lsq/trf.py (trf_bounds), common.py, least_squares.py, _numdiff.py
// 모서리 경계 모델 적합(subpixel_grid_corners_20261015/refine.py, analysis_grid_corners_20261019/track_corners.py)에 쓴다.
#pragma once
#include <opencv2/core.hpp>
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <vector>

namespace trf {

using Vec = std::vector<double>;
using Fun = std::function<void(const Vec& x, Vec& f)>;
const double EPS = std::numeric_limits<double>::epsilon();
const double INF = std::numeric_limits<double>::infinity();

inline double norm(const Vec& a) { double s = 0; for (double v : a) s += v * v; return std::sqrt(s); }
inline double dot(const Vec& a, const Vec& b) { double s = 0; for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i]; return s; }

struct Result { Vec x; int status = 0; int nfev = 0; bool success() const { return status > 0; } };

// soft_l1 with f_scale: rho0, rho1, rho2
inline void soft_l1(const Vec& f, double fs, Vec& r0, Vec& r1, Vec& r2) {
    size_t m = f.size(); r0.resize(m); r1.resize(m); r2.resize(m);
    for (size_t i = 0; i < m; ++i) {
        double z = (f[i] / fs) * (f[i] / fs), t = 1 + z;
        r0[i] = 2 * (std::sqrt(t) - 1) * fs * fs;
        r1[i] = std::pow(t, -0.5);
        r2[i] = -0.5 * std::pow(t, -1.5) / (fs * fs);
    }
}
inline double soft_l1_cost(const Vec& f, double fs) {
    double s = 0;
    for (double v : f) { double z = (v / fs) * (v / fs); s += 2 * (std::sqrt(1 + z) - 1); }
    return 0.5 * fs * fs * s;
}

// J: m x n row-major
struct Mat { int m = 0, n = 0; Vec a; double& operator()(int i, int j) { return a[(size_t)i * n + j]; } double operator()(int i, int j) const { return a[(size_t)i * n + j]; } };

inline Mat jac2(const Fun& fun, const Vec& x0, const Vec& f0, const Vec& lb, const Vec& ub) {
    int n = (int)x0.size(), m = (int)f0.size();
    const double rstep = std::sqrt(EPS);
    Vec h(n);
    for (int i = 0; i < n; ++i) {
        double s = x0[i] >= 0 ? 1.0 : -1.0;
        h[i] = rstep * s * std::max(1.0, std::abs(x0[i]));
    }
    // _adjust_scheme_to_bounds(1-sided, num_steps=1)
    for (int i = 0; i < n; ++i) {
        double lo = x0[i] - lb[i], up = ub[i] - x0[i], x = x0[i] + h[i];
        bool violated = x < lb[i] || x > ub[i];
        bool fitting = std::abs(h[i]) <= std::max(lo, up);
        if (violated && fitting) h[i] = -h[i];
        else if (!fitting) h[i] = up >= lo ? up : -lo;
    }
    Mat J; J.m = m; J.n = n; J.a.assign((size_t)m * n, 0);
    Vec x1, f1;
    for (int i = 0; i < n; ++i) {
        x1 = x0; x1[i] = x0[i] + h[i];
        fun(x1, f1);
        double dx = (x0[i] + h[i]) - x0[i];
        for (int k = 0; k < m; ++k) J(k, i) = (f1[k] - f0[k]) / dx;
    }
    return J;
}

inline int find_active(double x, double lb, double ub, double rtol) {
    double ld = x - lb, ud = ub - x;
    double lt = rtol * std::max(1.0, std::abs(lb)), ut = rtol * std::max(1.0, std::abs(ub));
    int a = 0;
    if (ld <= std::min(ud, lt)) a = -1;
    if (ud <= std::min(ld, ut)) a = 1;
    return a;
}
inline Vec make_strictly_feasible(const Vec& x, const Vec& lb, const Vec& ub, double rstep) {
    Vec o = x;
    for (size_t i = 0; i < x.size(); ++i) {
        int a = rstep == 0 ? (x[i] <= lb[i] ? -1 : (x[i] >= ub[i] ? 1 : 0)) : find_active(x[i], lb[i], ub[i], rstep);
        if (rstep == 0) {
            if (x[i] <= lb[i]) a = -1;
            if (x[i] >= ub[i]) a = 1;
            if (a == -1) o[i] = std::nextafter(lb[i], ub[i]);
            if (a == 1) o[i] = std::nextafter(ub[i], lb[i]);
        } else {
            if (a == -1) o[i] = lb[i] + rstep * std::max(1.0, std::abs(lb[i]));
            if (a == 1) o[i] = ub[i] - rstep * std::max(1.0, std::abs(ub[i]));
        }
        if (o[i] < lb[i] || o[i] > ub[i]) o[i] = 0.5 * (lb[i] + ub[i]);
    }
    return o;
}
inline bool in_bounds(const Vec& x, const Vec& lb, const Vec& ub) {
    for (size_t i = 0; i < x.size(); ++i) if (!(x[i] >= lb[i] && x[i] <= ub[i])) return false;
    return true;
}
inline double step_size_to_bound(const Vec& x, const Vec& s, const Vec& lb, const Vec& ub, std::vector<int>* hits) {
    size_t n = x.size();
    Vec steps(n, INF);
    for (size_t i = 0; i < n; ++i)
        if (s[i] != 0) steps[i] = std::max((lb[i] - x[i]) / s[i], (ub[i] - x[i]) / s[i]);
    double mn = *std::min_element(steps.begin(), steps.end());
    if (hits) {
        hits->assign(n, 0);
        for (size_t i = 0; i < n; ++i) if (steps[i] == mn) (*hits)[i] = s[i] > 0 ? 1 : (s[i] < 0 ? -1 : 0);
    }
    return mn;
}
inline void intersect_trust_region(const Vec& x, const Vec& s, double Delta, double& t1, double& t2) {
    double a = dot(s, s), b = dot(x, s), c = dot(x, x) - Delta * Delta;
    double d = std::sqrt(b * b - a * c);
    double q = -(b + std::copysign(d, b));
    double u1 = q / a, u2 = c / q;
    if (u1 < u2) { t1 = u1; t2 = u2; } else { t1 = u2; t2 = u1; }
}
inline Vec Jdot(const Mat& J, const Vec& s) {
    Vec o(J.m, 0);
    for (int i = 0; i < J.m; ++i) { double t = 0; for (int j = 0; j < J.n; ++j) t += J(i, j) * s[j]; o[i] = t; }
    return o;
}
inline double evaluate_quadratic(const Mat& J, const Vec& g, const Vec& s, const Vec& diag) {
    Vec Js = Jdot(J, s);
    double q = dot(Js, Js);
    for (size_t i = 0; i < s.size(); ++i) q += s[i] * diag[i] * s[i];
    return 0.5 * q + dot(s, g);
}
inline void build_quadratic_1d(const Mat& J, const Vec& g, const Vec& s, const Vec& diag, const Vec* s0, double& a, double& b, double& c) {
    Vec v = Jdot(J, s);
    a = dot(v, v);
    for (size_t i = 0; i < s.size(); ++i) a += s[i] * diag[i] * s[i];
    a *= 0.5;
    b = dot(g, s);
    c = 0;
    if (s0) {
        Vec u = Jdot(J, *s0);
        b += dot(u, v);
        c = 0.5 * dot(u, u) + dot(g, *s0);
        for (size_t i = 0; i < s.size(); ++i) b += (*s0)[i] * diag[i] * s[i];
        for (size_t i = 0; i < s.size(); ++i) c += 0.5 * (*s0)[i] * diag[i] * (*s0)[i];
    }
}
inline void minimize_quadratic_1d(double a, double b, double lb, double ub, double c, double& t, double& y) {
    double ts[3] = {lb, ub, 0};
    int nt = 2;
    if (a != 0) {
        double e = -0.5 * b / a;
        if (lb < e && e < ub) ts[nt++] = e;
    }
    int best = 0;
    double yb = ts[0] * (a * ts[0] + b) + c;
    for (int i = 1; i < nt; ++i) {
        double yy = ts[i] * (a * ts[i] + b) + c;
        if (yy < yb) { yb = yy; best = i; }
    }
    t = ts[best]; y = yb;
}

inline void solve_lsq_trust_region(int n, int m, const Vec& uf, const Vec& s, const Mat& V /*n x n, columns = right vectors*/, double Delta,
                                   double initial_alpha, Vec& p, double& alpha) {
    auto phi_d = [&](double al, const Vec& suf, double& phi, double& phip) {
        double pn = 0, sp = 0;
        for (int i = 0; i < n; ++i) {
            double den = s[i] * s[i] + al;
            pn += (suf[i] / den) * (suf[i] / den);
            sp += suf[i] * suf[i] / (den * den * den);
        }
        pn = std::sqrt(pn);
        phi = pn - Delta;
        phip = -sp / pn;
    };
    Vec suf(n);
    for (int i = 0; i < n; ++i) suf[i] = s[i] * uf[i];
    bool full_rank = false;
    if (m >= n) full_rank = s[n - 1] > EPS * m * s[0];
    p.assign(n, 0);
    if (full_rank) {
        for (int i = 0; i < n; ++i) { double t = 0; for (int k = 0; k < n; ++k) t += V(i, k) * (uf[k] / s[k]); p[i] = -t; }
        if (norm(p) <= Delta) { alpha = 0.0; return; }
    }
    double alpha_upper = norm(suf) / Delta, alpha_lower = 0.0;
    if (full_rank) { double phi, phip; phi_d(0.0, suf, phi, phip); alpha_lower = -phi / phip; }
    double al;
    if (!full_rank && initial_alpha == 0) al = std::max(0.001 * alpha_upper, std::sqrt(alpha_lower * alpha_upper));
    else al = initial_alpha;
    for (int it = 0; it < 10; ++it) {
        if (al < alpha_lower || al > alpha_upper) al = std::max(0.001 * alpha_upper, std::sqrt(alpha_lower * alpha_upper));
        double phi, phip;
        phi_d(al, suf, phi, phip);
        if (phi < 0) alpha_upper = al;
        double ratio = phi / phip;
        alpha_lower = std::max(alpha_lower, al - ratio);
        al -= (phi + Delta) * ratio / Delta;
        if (std::abs(phi) < 0.01 * Delta) break;
    }
    for (int i = 0; i < n; ++i) { double t = 0; for (int k = 0; k < n; ++k) t += V(i, k) * (suf[k] / (s[k] * s[k] + al)); p[i] = -t; }
    double pn = norm(p);
    for (auto& v : p) v *= Delta / pn;
    alpha = al;
}

// least_squares(fun, x0, bounds=(lb,ub), loss='soft_l1', f_scale, max_nfev, xtol, ftol, gtol)
inline Result least_squares_soft_l1(const Fun& fun, Vec x0, const Vec& lb, const Vec& ub, double f_scale, int max_nfev, double ftol,
                                    double xtol, double gtol) {
    const int n = (int)x0.size();
    x0 = make_strictly_feasible(x0, lb, ub, 1e-10);
    Vec f0; fun(x0, f0);
    const int m = (int)f0.size();
    Mat J = jac2(fun, x0, f0, lb, ub);
    Vec x = x0, f = f0;
    int nfev = 1;
    Vec r0, r1, r2;
    auto scale_robust = [&](Mat& JJ, Vec& ff) {
        soft_l1(ff, f_scale, r0, r1, r2);
        for (int i = 0; i < m; ++i) {
            double js = r1[i] + 2 * r2[i] * ff[i] * ff[i];
            if (js < EPS) js = EPS;
            js = std::sqrt(js);
            ff[i] *= r1[i] / js;
            for (int j = 0; j < n; ++j) JJ(i, j) *= js;
        }
    };
    double cost = soft_l1_cost(f, f_scale);
    scale_robust(J, f);
    auto grad = [&](const Mat& JJ, const Vec& ff) { Vec g(n, 0); for (int i = 0; i < m; ++i) for (int j = 0; j < n; ++j) g[j] += JJ(i, j) * ff[i]; return g; };
    Vec g = grad(J, f);
    auto CL = [&](const Vec& xx, const Vec& gg, Vec& v, Vec& dv) {
        v.assign(n, 1); dv.assign(n, 0);
        for (int i = 0; i < n; ++i) {
            if (gg[i] < 0 && std::isfinite(ub[i])) { v[i] = ub[i] - xx[i]; dv[i] = -1; }
            if (gg[i] > 0 && std::isfinite(lb[i])) { v[i] = xx[i] - lb[i]; dv[i] = 1; }
        }
    };
    Vec v, dv;
    CL(x, g, v, dv);
    double Delta = 0;
    { Vec t(n); for (int i = 0; i < n; ++i) t[i] = x0[i] / std::sqrt(v[i]); Delta = norm(t); }
    if (Delta == 0) Delta = 1.0;
    double alpha = 0.0;
    int status = -100;  // None
    Vec f_aug(m + n, 0);
    while (true) {
        CL(x, g, v, dv);
        double g_norm = 0;
        for (int i = 0; i < n; ++i) g_norm = std::max(g_norm, std::abs(g[i] * v[i]));
        if (g_norm < gtol) status = 1;
        if (status != -100 || nfev == max_nfev) break;
        Vec d(n), diag_h(n), g_h(n);
        for (int i = 0; i < n; ++i) { d[i] = std::sqrt(v[i]); diag_h[i] = g[i] * dv[i]; g_h[i] = d[i] * g[i]; }
        for (int i = 0; i < m; ++i) f_aug[i] = f[i];
        for (int i = m; i < m + n; ++i) f_aug[i] = 0;
        Mat J_h; J_h.m = m; J_h.n = n; J_h.a.resize((size_t)m * n);
        cv::Mat Ja(m + n, n, CV_64F, cv::Scalar(0));
        for (int i = 0; i < m; ++i) for (int j = 0; j < n; ++j) { J_h(i, j) = J(i, j) * d[j]; Ja.at<double>(i, j) = J_h(i, j); }
        for (int j = 0; j < n; ++j) Ja.at<double>(m + j, j) = std::sqrt(diag_h[j]);
        cv::Mat W, U, Vt;
        cv::SVD::compute(Ja, W, U, Vt);
        Vec s(n), uf(n, 0);
        Mat V; V.m = n; V.n = n; V.a.resize((size_t)n * n);
        for (int k = 0; k < n; ++k) {
            s[k] = W.at<double>(k);
            double t = 0;
            for (int i = 0; i < m + n; ++i) t += U.at<double>(i, k) * f_aug[i];
            uf[k] = t;
            for (int i = 0; i < n; ++i) V(i, k) = Vt.at<double>(k, i);
        }
        double theta = std::max(0.995, 1 - g_norm);
        double actual_reduction = -1;
        Vec x_new, f_new;
        double cost_new = 0;
        while (actual_reduction <= 0 && nfev < max_nfev) {
            Vec p_h;
            solve_lsq_trust_region(n, m, uf, s, V, Delta, alpha, p_h, alpha);
            Vec p(n);
            for (int i = 0; i < n; ++i) p[i] = d[i] * p_h[i];
            // select_step
            Vec step, step_h;
            double pred;
            Vec xp(n);
            for (int i = 0; i < n; ++i) xp[i] = x[i] + p[i];
            if (in_bounds(xp, lb, ub)) {
                step = p; step_h = p_h; pred = -evaluate_quadratic(J_h, g_h, p_h, diag_h);
            } else {
                std::vector<int> hits;
                double p_stride = step_size_to_bound(x, p, lb, ub, &hits);
                Vec r_h = p_h;
                for (int i = 0; i < n; ++i) if (hits[i]) r_h[i] *= -1;
                Vec r(n);
                for (int i = 0; i < n; ++i) r[i] = d[i] * r_h[i];
                for (auto& t : p) t *= p_stride;
                for (auto& t : p_h) t *= p_stride;
                Vec x_on(n);
                for (int i = 0; i < n; ++i) x_on[i] = x[i] + p[i];
                double t1, to_tr;
                intersect_trust_region(p_h, r_h, Delta, t1, to_tr);
                double to_bound = step_size_to_bound(x_on, r, lb, ub, nullptr);
                double r_stride = std::min(to_bound, to_tr), r_l, r_u;
                if (r_stride > 0) {
                    r_l = (1 - theta) * p_stride / r_stride;
                    r_u = r_stride == to_bound ? theta * to_bound : to_tr;
                } else { r_l = 0; r_u = -1; }
                double r_value;
                if (r_l <= r_u) {
                    double a, b, c;
                    build_quadratic_1d(J_h, g_h, r_h, diag_h, &p_h, a, b, c);
                    double rs;
                    minimize_quadratic_1d(a, b, r_l, r_u, c, rs, r_value);
                    for (int i = 0; i < n; ++i) { r_h[i] = r_h[i] * rs + p_h[i]; r[i] = r_h[i] * d[i]; }
                } else r_value = INF;
                for (auto& t : p) t *= theta;
                for (auto& t : p_h) t *= theta;
                double p_value = evaluate_quadratic(J_h, g_h, p_h, diag_h);
                Vec ag_h(n), ag(n);
                for (int i = 0; i < n; ++i) { ag_h[i] = -g_h[i]; ag[i] = d[i] * ag_h[i]; }
                double to_tr2 = Delta / norm(ag_h);
                double tb2 = step_size_to_bound(x, ag, lb, ub, nullptr);
                double ag_stride = tb2 < to_tr2 ? theta * tb2 : to_tr2;
                double a, b, c;
                build_quadratic_1d(J_h, g_h, ag_h, diag_h, nullptr, a, b, c);
                double ag_value;
                minimize_quadratic_1d(a, b, 0, ag_stride, 0, ag_stride, ag_value);
                for (int i = 0; i < n; ++i) { ag_h[i] *= ag_stride; ag[i] *= ag_stride; }
                if (p_value < r_value && p_value < ag_value) { step = p; step_h = p_h; pred = -p_value; }
                else if (r_value < p_value && r_value < ag_value) { step = r; step_h = r_h; pred = -r_value; }
                else { step = ag; step_h = ag_h; pred = -ag_value; }
            }
            Vec xs(n);
            for (int i = 0; i < n; ++i) xs[i] = x[i] + step[i];
            x_new = make_strictly_feasible(xs, lb, ub, 0);
            fun(x_new, f_new);
            ++nfev;
            double step_h_norm = norm(step_h);
            bool finite = true;
            for (double t : f_new) if (!std::isfinite(t)) finite = false;
            if (!finite) { Delta = 0.25 * step_h_norm; continue; }
            cost_new = soft_l1_cost(f_new, f_scale);
            actual_reduction = cost - cost_new;
            // update_tr_radius
            double ratio;
            if (pred > 0) ratio = actual_reduction / pred;
            else if (pred == actual_reduction && pred == 0) ratio = 1;
            else ratio = 0;
            double Delta_new = Delta;
            if (ratio < 0.25) Delta_new = 0.25 * step_h_norm;
            else if (ratio > 0.75 && step_h_norm > 0.95 * Delta) Delta_new = Delta * 2.0;
            double step_norm = norm(step);
            bool ftol_ok = actual_reduction < ftol * cost && ratio > 0.25;
            bool xtol_ok = step_norm < xtol * (xtol + norm(x));
            if (ftol_ok && xtol_ok) status = 4;
            else if (ftol_ok) status = 2;
            else if (xtol_ok) status = 3;
            if (status != -100) break;
            alpha *= Delta / Delta_new;
            Delta = Delta_new;
        }
        if (actual_reduction > 0) {
            x = x_new; f = f_new; cost = cost_new;
            J = jac2(fun, x, f, lb, ub);
            scale_robust(J, f);
            g = grad(J, f);
        }
    }
    if (status == -100) status = 0;
    Result r; r.x = x; r.status = status; r.nfev = nfev;
    return r;
}

}  // namespace trf
