// geometry.cpp — 서브픽셀 측정 공통 도구 구현
//  (A507 측정 프로그램 rigpose 에서 옮겨 옴. 계산은 한 줄도 바꾸지 않음 → 같은 사진에서 같은 값)
#include "ksh/geometry.hpp"

#include <algorithm>
#include <cmath>

namespace ksh {
namespace geo {

// ======================================================================== 최소제곱
bool lstsq(const std::vector<double>& A_, int m, int n, const std::vector<double>& b_, std::vector<double>& x) {
    if (m < n) return false;
    std::vector<double> A = A_, b = b_;
    for (int k = 0; k < n; ++k) {
        double nrm = 0;
        for (int i = k; i < m; ++i) nrm += A[(size_t)i * n + k] * A[(size_t)i * n + k];
        nrm = std::sqrt(nrm);
        if (nrm < 1e-300) return false;
        double akk = A[(size_t)k * n + k];
        double alpha = akk > 0 ? -nrm : nrm;
        std::vector<double> v(m - k);
        for (int i = k; i < m; ++i) v[i - k] = A[(size_t)i * n + k];
        v[0] -= alpha;
        double vn = 0; for (double t : v) vn += t * t;
        if (vn < 1e-300) continue;
        for (int j = k; j < n; ++j) {
            double s = 0; for (int i = k; i < m; ++i) s += v[i - k] * A[(size_t)i * n + j];
            s = 2 * s / vn;
            for (int i = k; i < m; ++i) A[(size_t)i * n + j] -= s * v[i - k];
        }
        double s = 0; for (int i = k; i < m; ++i) s += v[i - k] * b[i];
        s = 2 * s / vn;
        for (int i = k; i < m; ++i) b[i] -= s * v[i - k];
    }
    x.assign(n, 0);
    for (int k = n - 1; k >= 0; --k) {
        double s = b[k];
        for (int j = k + 1; j < n; ++j) s -= A[(size_t)k * n + j] * x[j];
        double d = A[(size_t)k * n + k];
        if (std::fabs(d) < 1e-300) return false;
        x[k] = s / d;
    }
    return true;
}

// ======================================================================== 영상 도구
double bilinear(const View& im, double x, double y) {
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x > im.width - 1.001) x = im.width - 1.001;
    if (y > im.height - 1.001) y = im.height - 1.001;
    int x0 = (int)x, y0 = (int)y; double fx = x - x0, fy = y - y0;
    const uint8_t* p = im.data + (size_t)y0 * im.stride + x0;
    double a = p[0], b = p[1], c = p[im.stride], d = p[im.stride + 1];
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy;
}

// 직선을 따라 평균낸 단면. vertical=true 이면 세로 모서리(x 위치를 잼):
//   중심선 x(t) = p0 + slope*(t - tref), t 는 y. 단면 축 o 는 x 방향.

Profile line_profile(const View& im, bool vertical, double p0, double slope, double tref,
                            double t0, double t1, double range, double step) {
    Profile P;
    int n = (int)std::floor(2 * range / step) + 1;
    P.o.resize(n); P.v.assign(n, 0.0);
    for (int i = 0; i < n; ++i) P.o[i] = -range + i * step;
    int cnt = 0;
    for (int t = (int)std::ceil(t0); t < (int)t1; ++t) {
        double c = p0 + slope * (t - tref);
        for (int i = 0; i < n; ++i)
            P.v[i] += vertical ? bilinear(im, c + P.o[i], t) : bilinear(im, t, c + P.o[i]);
        ++cnt;
    }
    if (cnt) for (auto& v : P.v) v /= cnt;
    return P;
}

// 가우시안으로 다듬은 기울기의 최대점 (sign: +1 이면 어두움→밝음, o 증가 방향)
bool grad_peak(const Profile& P, int sign, double sigma, double lim, double& pos, double& peak) {
    double step = P.o[1] - P.o[0];
    int k = std::max(1, (int)std::lround(3 * sigma / step));
    std::vector<double> ker(2 * k + 1); double ks = 0;
    for (int i = -k; i <= k; ++i) { ker[i + k] = std::exp(-0.5 * (i * step / sigma) * (i * step / sigma)); ks += ker[i + k]; }
    int n = (int)P.v.size();
    std::vector<double> s(n, 0.0);
    for (int i = 0; i < n; ++i) {
        double a = 0, w = 0;
        for (int j = -k; j <= k; ++j) { int q = i + j; if (q < 0 || q >= n) continue; a += ker[j + k] * P.v[q]; w += ker[j + k]; }
        s[i] = a / w;
    }
    int best = -1; double bv = -1e30;
    std::vector<double> g(n, 0.0);
    for (int i = 1; i < n - 1; ++i) g[i] = sign * (s[i + 1] - s[i - 1]) / (2 * step);
    for (int i = k + 2; i < n - k - 2; ++i) {
        if (std::fabs(P.o[i]) > lim) continue;
        if (g[i] > bv) { bv = g[i]; best = i; }
    }
    if (best < 0 || bv <= 0) return false;
    double den = g[best - 1] - 2 * g[best] + g[best + 1];
    double d = den < 0 ? 0.5 * (g[best - 1] - g[best + 1]) / den : 0;
    pos = P.o[best] + d * step; peak = bv;
    return true;
}

// 50% 교차: 기울기 최대점 양옆 plateau 평균의 중간값을 지나는 위치
bool half_cross(const Profile& P, int sign, double o0, double plateau, double& pos, double& contrast) {
    double lo = 0, hi = 0; int nl = 0, nh = 0;
    for (size_t i = 0; i < P.o.size(); ++i) {
        double r = (P.o[i] - o0) * sign;
        if (r > -plateau - 2 && r < -plateau) { lo += P.v[i]; ++nl; }
        if (r > plateau && r < plateau + 2) { hi += P.v[i]; ++nh; }
    }
    if (!nl || !nh) return false;
    lo /= nl; hi /= nh;
    contrast = hi - lo;
    if (contrast <= 0) return false;
    double thr = 0.5 * (lo + hi);
    double bestd = 1e30; bool found = false;
    for (size_t i = 0; i + 1 < P.o.size(); ++i) {
        if (std::fabs(P.o[i] - o0) > plateau) continue;
        double a = (P.v[i] - thr) * sign, b = (P.v[i + 1] - thr) * sign;
        if (a < 0 && b >= 0) {
            double x = P.o[i] + (thr - P.v[i]) / (P.v[i + 1] - P.v[i]) * (P.o[i + 1] - P.o[i]);
            if (std::fabs(x - o0) < bestd) { bestd = std::fabs(x - o0); pos = x; found = true; }
        }
    }
    return found;
}

// 모서리 위치 측정 (한 구간 평균). 반환: 중심선 대비 오프셋

// method: 0 = 50% 교차 (기본), 1 = 기울기 최대점 (sigma 는 가우시안 폭 px). 대비는 둘 다 plateau 차이
EdgeResult measure_offset(const View& im, bool vertical, double p0, double slope, double tref,
                                 double t0, double t1, int sign, double search, double plateau,
                                 int method, double sigma) {
    EdgeResult r;
    Profile P = line_profile(im, vertical, p0, slope, tref, t0, t1, search + plateau + 3);
    double o0, pk;
    if (!grad_peak(P, sign, method == 1 ? sigma : 1.0, search, o0, pk)) return r;
    double pos, c;
    bool hc = half_cross(P, sign, o0, plateau, pos, c);
    if (method == 1) { r.off = o0; r.contrast = hc ? c : pk; }
    else if (hc) { r.off = pos; r.contrast = c; }
    else { r.off = o0; r.contrast = pk; }
    r.ok = true;
    return r;
}

// 모서리 직선 (구간 nseg 개 → 직선 맞춤). 반환: tref 에서 위치, 기울기, 잔차

EdgeLine measure_line(const View& im, bool vertical, double p0, double slope, double tref,
                             double t0, double t1, int sign, double search, double plateau, int nseg, bool fit_slope,
                             int method, double sigma) {
    EdgeLine L;
    // 1차: 전체 길이로 위치 맞추기 (넓게 찾기)
    EdgeResult e = measure_offset(im, vertical, p0, slope, tref, t0, t1, sign, search, plateau);
    if (!e.ok) return L;
    p0 += e.off;
    if (!fit_slope || nseg < 2) {
        EdgeResult e2 = measure_offset(im, vertical, p0, slope, tref, t0, t1, sign, std::min(search, plateau), plateau, method, sigma);
        if (!e2.ok) return L;
        L.ok = true; L.pos = p0 + e2.off; L.slope = slope; L.contrast = e2.contrast;
        return L;
    }
    for (int it = 0; it < 2; ++it) {
        std::vector<double> A, b; double csum = 0; int ok = 0;
        for (int s = 0; s < nseg; ++s) {
            double a0 = t0 + (t1 - t0) * s / nseg, a1 = t0 + (t1 - t0) * (s + 1) / nseg;
            EdgeResult es = measure_offset(im, vertical, p0, slope, tref, a0, a1, sign, std::min(search, plateau), plateau);
            if (!es.ok) continue;
            double tm = 0.5 * (a0 + a1);
            A.push_back(tm - tref); A.push_back(1.0);
            b.push_back(p0 + slope * (tm - tref) + es.off);
            csum += es.contrast; ++ok;
        }
        if (ok < 2) return L;
        std::vector<double> x;
        if (!lstsq(A, ok, 2, b, x)) return L;
        slope = x[0]; p0 = x[1];
        double ss = 0;
        for (int i = 0; i < ok; ++i) { double r = b[i] - (A[2 * i] * x[0] + x[1]); ss += r * r; }
        L.rms = std::sqrt(ss / ok); L.contrast = csum / ok;
    }
    L.ok = true; L.pos = p0; L.slope = slope;
    return L;
}


std::vector<double> box_smooth(const std::vector<double>& v, int r) {
    std::vector<double> o(v.size());
    for (int i = 0; i < (int)v.size(); ++i) {
        double s = 0; int n = 0;
        for (int j = std::max(0, i - r); j <= std::min((int)v.size() - 1, i + r); ++j) { s += v[j]; ++n; }
        o[i] = s / n;
    }
    return o;
}
double percentile(std::vector<double> v, double q) {
    if (v.empty()) return 0;
    size_t k = (size_t)(q * (v.size() - 1));
    std::nth_element(v.begin(), v.begin() + k, v.end());
    return v[k];
}

// 흰 QR 사각형 3개의 대략 위치 (열·행 밝기 투영, 띠 밝기에 맞춘 문턱값)

void gauss_blur(std::vector<float>& a, int w, int h, double sigma) {
    int r = std::max(1, (int)std::ceil(3 * sigma));
    std::vector<float> k(2 * r + 1); float ks = 0;
    for (int i = -r; i <= r; ++i) { k[i + r] = (float)std::exp(-0.5 * i * i / (sigma * sigma)); ks += k[i + r]; }
    for (auto& v : k) v /= ks;
    std::vector<float> t(a.size());
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            float s = 0;
            for (int j = -r; j <= r; ++j) { int xx = std::min(w - 1, std::max(0, x + j)); s += k[j + r] * a[(size_t)y * w + xx]; }
            t[(size_t)y * w + x] = s;
        }
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            float s = 0;
            for (int j = -r; j <= r; ++j) { int yy = std::min(h - 1, std::max(0, y + j)); s += k[j + r] * t[(size_t)yy * w + x]; }
            a[(size_t)y * w + x] = s;
        }
}
double bil(const std::vector<float>& a, int w, int h, double x, double y) {
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x > w - 1.001) x = w - 1.001;
    if (y > h - 1.001) y = h - 1.001;
    int x0 = (int)x, y0 = (int)y; double fx = x - x0, fy = y - y0;
    const float* p = &a[(size_t)y0 * w + x0];
    return (p[0] * (1 - fx) + p[1] * fx) * (1 - fy) + (p[w] * (1 - fx) + p[w + 1] * fx) * fy;
}
double median_of(std::vector<double> v) {
    if (v.empty()) return 0;
    std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
    return v[v.size() / 2];
}

} // namespace geo
} // namespace ksh
