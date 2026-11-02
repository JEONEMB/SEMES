// Extracted verbatim from methods_points.cpp; see provenance.json.
template <class F>
std::array<double, 3> nelder_mead3(F f, std::array<double, 3> x0, const std::array<double, 3>& step,
                                   int max_iter, double tol, int* iters, bool* converged) {
    std::array<std::array<double, 3>, 4> s;
    std::array<double, 4> v;
    s[0] = x0;
    for (int i = 0; i < 3; ++i) { s[i + 1] = x0; s[i + 1][i] += step[i]; }
    for (int i = 0; i < 4; ++i) v[i] = f(s[i]);
    *converged = false;
    int it = 0;
    for (; it < max_iter; ++it) {
        std::array<int, 4> idx = {0, 1, 2, 3};
        std::sort(idx.begin(), idx.end(), [&](int a, int b) { return v[a] < v[b]; });
        auto s2 = s; auto v2 = v;
        for (int i = 0; i < 4; ++i) { s[i] = s2[idx[i]]; v[i] = v2[idx[i]]; }
        if (std::fabs(v[3] - v[0]) <= tol * (std::fabs(v[0]) + 1e-30)) { *converged = true; break; }
        std::array<double, 3> c = {0, 0, 0};
        for (int i = 0; i < 3; ++i) for (int k = 0; k < 3; ++k) c[k] += s[i][k] / 3.0;
        auto mix = [&](double a) { std::array<double, 3> r; for (int k = 0; k < 3; ++k) r[k] = c[k] + a * (s[3][k] - c[k]); return r; };
        auto xr = mix(-1.0); double fr = f(xr);
        if (fr < v[0]) {
            auto xe = mix(-2.0); double fe = f(xe);
            if (fe < fr) { s[3] = xe; v[3] = fe; } else { s[3] = xr; v[3] = fr; }
        } else if (fr < v[2]) {
            s[3] = xr; v[3] = fr;
        } else {
            auto xc = fr < v[3] ? mix(-0.5) : mix(0.5);
            double fc = f(xc);
            if (fc < std::min(fr, v[3])) { s[3] = xc; v[3] = fc; }
            else {
                for (int i = 1; i < 4; ++i) { for (int k = 0; k < 3; ++k) s[i][k] = s[0][k] + 0.5 * (s[i][k] - s[0][k]); v[i] = f(s[i]); }
            }
        }
    }
    *iters = it;
    int best = (int)(std::min_element(v.begin(), v.end()) - v.begin());
    return s[best];
}

