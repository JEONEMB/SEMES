// 630개 QR 격자 모서리 경계 맞춤 (원본: joint_center_630_20261022/measurement.py, numba LM 을 그대로 옮김)
// 모서리마다 9x9 창에서 erf 경계 모델(위치 2, 흐림 2, 밝기 2)을 soft_l1 손실·경계 제약 LM 으로 맞추고 11x11 창으로 안정성을 본다.
#pragma once
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>
#include "fixed630.hpp"

namespace s630 {

const int kBox[3][4] = {{1552, 969, 1736, 1167}, {1809, 971, 1993, 1169}, {2067, 973, 2252, 1171}};

struct Result {
    double D[630][2];       // 현재 좌표 - 기준점 (px)
    double cur[630][2];     // 현재 좌표
    bool valid[630];
    double rmse[630];
    int pad = 0;
    double ncc[3];
    int sx[3], sy[3];
    int nvalid() const { int n = 0; for (bool v : valid) n += v; return n; }
    double mean_dx() const { double s = 0; for (auto& d : D) s += d[0]; return s / 630; }
    double mean_dy() const { double s = 0; for (auto& d : D) s += d[1]; return s / 630; }
};

struct Win {
    std::vector<double> x, y, I;
};

inline double objective(const double p[6], const Win& w, const double* inv, const double* patch, std::vector<double>& res, std::vector<std::array<double, 6>>& J) {
    const size_t n = w.I.size();
    res.resize(n);
    J.resize(n);
    double cost = 0;
    const double a = patch[0], du = patch[1] - a, dv = patch[2] - a, cross = patch[3] - patch[1] - patch[2] + a;
    const double s2 = std::sqrt(2.0), s2pi = std::sqrt(2 * M_PI);
    for (size_t i = 0; i < n; ++i) {
        const double dx = w.x[i] - p[0], dy = w.y[i] - p[1];
        const double u = dx * inv[0] + dy * inv[1], v = dx * inv[2] + dy * inv[3];
        const double zu = u / p[2], zv = v / p[3];
        const double U = .5 * (1 + std::erf(zu / s2)), V = .5 * (1 + std::erf(zv / s2));
        const double F = patch[0] * (1 - U) * (1 - V) + patch[1] * U * (1 - V) + patch[2] * (1 - U) * V + patch[3] * U * V;
        const double r = p[4] + p[5] * (1 - F) - w.I[i];
        res[i] = r;
        cost += 36 * (std::sqrt(1 + (r / 6) * (r / 6)) - 1);
        const double gu = (du + cross * V) * std::exp(-.5 * zu * zu) / s2pi, gv = (dv + cross * U) * std::exp(-.5 * zv * zv) / s2pi;
        J[i][0] = p[5] * (gu / p[2] * inv[0] + gv / p[3] * inv[2]);
        J[i][1] = p[5] * (gu / p[2] * inv[1] + gv / p[3] * inv[3]);
        J[i][2] = p[5] * gu * u / (p[2] * p[2]);
        J[i][3] = p[5] * gv * v / (p[3] * p[3]);
        J[i][4] = 1.;
        J[i][5] = 1 - F;
    }
    return cost;
}

// numpy.linalg.solve (LU, 부분 피벗)
inline void solve6(double A[6][6], double b[6], double x[6]) {
    int piv[6] = {0, 1, 2, 3, 4, 5};
    for (int k = 0; k < 6; ++k) {
        int m = k;
        for (int i = k + 1; i < 6; ++i) if (std::abs(A[i][k]) > std::abs(A[m][k])) m = i;
        if (m != k) { for (int j = 0; j < 6; ++j) std::swap(A[k][j], A[m][j]); std::swap(b[k], b[m]); std::swap(piv[k], piv[m]); }
        if (A[k][k] == 0) throw std::runtime_error("singular");
        for (int i = k + 1; i < 6; ++i) {
            const double f = A[i][k] / A[k][k];
            A[i][k] = f;
            for (int j = k + 1; j < 6; ++j) A[i][j] -= f * A[k][j];
            b[i] -= f * b[k];
        }
    }
    for (int i = 5; i >= 0; --i) {
        double s = b[i];
        for (int j = i + 1; j < 6; ++j) s -= A[i][j] * x[j];
        x[i] = s / A[i][i];
    }
}

struct Fit {
    double p[6];
    double rmse;
    bool ok;
};

inline Fit fit(const Win& w, const double start[2], const double* inv, const double* patch) {
    const size_t n = w.I.size();
    std::vector<double> ordered(w.I);
    std::sort(ordered.begin(), ordered.end());
    auto quant = [&](double f) {
        const double q = f * (n - 1);
        const int k = (int)q;
        return ordered[k] * (1 - (q - k)) + ordered[std::min<size_t>(k + 1, n - 1)] * (q - k);
    };
    const double low = quant(.1), high = quant(.9);
    double p[6] = {start[0], start[1], 1., 1., low, std::max(high - low, 20.)};
    const double lower[6] = {start[0] - 3.8, start[1] - 3.8, .35, .35, 0., 10.}, upper[6] = {start[0] + 3.8, start[1] + 3.8, 2.7, 2.7, 240., 255.};
    const double scales[6] = {1., 1., 1., 1., 60., 120.};
    double mu = 1e-3;
    bool converged = false;
    std::vector<double> r, nr;
    std::vector<std::array<double, 6>> J, nJ;
    double cost = objective(p, w, inv, patch, r, J);
    int evaluations = 1;
    auto norm = [](const double* v) { double s = 0; for (int j = 0; j < 6; ++j) s += v[j] * v[j]; return std::sqrt(s); };
    for (int iteration = 0; iteration < 85; ++iteration) {
        double H[6][6] = {}, g[6] = {};
        for (size_t i = 0; i < n; ++i) {
            const double root = std::sqrt(1 + (r[i] / 6) * (r[i] / 6)), wt = 1 / (root * root * root);
            for (int j = 0; j < 6; ++j) {
                const double jj = J[i][j] * scales[j];
                g[j] += jj * r[i] / root;
                for (int k = 0; k < 6; ++k) H[j][k] += jj * J[i][k] * scales[k] * wt;
            }
        }
        double pg[6], mx = 0;
        for (int j = 0; j < 6; ++j) {
            pg[j] = g[j];
            if ((p[j] - lower[j] < 1e-9 && g[j] > 0) || (upper[j] - p[j] < 1e-9 && g[j] < 0)) pg[j] = 0;
            mx = std::max(mx, std::abs(pg[j]));
        }
        if (mx < 1e-7) { converged = true; break; }
        for (int j = 0; j < 6; ++j)
            if ((p[j] - lower[j] < 1e-8 && g[j] > 0) || (upper[j] - p[j] < 1e-8 && g[j] < 0)) {
                for (int k = 0; k < 6; ++k) { H[j][k] = 0.; H[k][j] = 0.; }
                H[j][j] = 1.;
                g[j] = 0.;
            }
        double diag[6];
        for (int j = 0; j < 6; ++j) diag[j] = std::max(H[j][j], 1e-6);
        bool accepted = false;
        for (int retry = 0; retry < 14; ++retry) {
            double A[6][6], b[6], delta[6], trial[6], dd[6];
            for (int j = 0; j < 6; ++j) { for (int k = 0; k < 6; ++k) A[j][k] = H[j][k]; A[j][j] += mu * diag[j]; b[j] = -g[j]; }
            solve6(A, b, delta);
            for (int j = 0; j < 6; ++j) {
                delta[j] *= scales[j];
                trial[j] = std::min(std::max(p[j] + delta[j], lower[j] + 1e-12), upper[j] - 1e-12);
                dd[j] = trial[j] - p[j];
            }
            const double newcost = objective(trial, w, inv, patch, nr, nJ);
            ++evaluations;
            if (norm(dd) < 1e-7 * (1e-7 + norm(p))) {
                converged = true;
                if (newcost < cost) { std::copy(trial, trial + 6, p); cost = newcost; r.swap(nr); J.swap(nJ); }
                accepted = true;
                break;
            }
            if (newcost < cost) {
                const double improvement = cost - newcost, step = norm(dd), pn = norm(p);
                std::copy(trial, trial + 6, p);
                cost = newcost;
                r.swap(nr);
                J.swap(nJ);
                mu = std::max(mu * .3, 1e-12);
                accepted = true;
                if (improvement < 1e-7 * std::max(cost, 1e-12) || step < 1e-7 * (1e-7 + pn)) converged = true;
                break;
            }
            mu = std::min(mu * 10, 1e14);
            if (evaluations >= 85) break;
        }
        if (converged || evaluations >= 85 || !accepted) break;
    }
    Fit f;
    std::copy(p, p + 6, f.p);
    double s = 0;
    for (double v : r) s += v * v;
    f.rmse = std::sqrt(s / n);
    f.ok = converged;
    return f;
}

inline Win window(const cv::Mat& img, const double est[2], int radius, const int box[4]) {
    const int x0 = (int)std::nearbyint(est[0]), y0 = (int)std::nearbyint(est[1]);
    Win w;
    for (int y = y0 - radius; y <= y0 + radius; ++y)
        for (int x = x0 - radius; x <= x0 + radius; ++x)
            if (box[0] <= x && x < box[2] && box[1] <= y && y < box[3]) {
                w.x.push_back(x);
                w.y.push_back(y);
                w.I.push_back(img.at<uchar>(y, x));
            }
    return w;
}

// center: 기준 사진(회색), img: 현재 사진(회색)
inline Result extract(const cv::Mat& center, const cv::Mat& img) {
    Result R;
    for (int pad : {130, 260, 512}) {
        for (int k = 0; k < 3; ++k) {
            const int* b = kBox[k];
            cv::Mat c;
            cv::matchTemplate(img(cv::Range(b[1] - pad, b[3] + pad), cv::Range(b[0] - pad, b[2] + pad)), center(cv::Range(b[1], b[3]), cv::Range(b[0], b[2])), c,
                              cv::TM_CCOEFF_NORMED);
            cv::Point loc;
            cv::minMaxLoc(c, nullptr, &R.ncc[k], nullptr, &loc);
            R.sx[k] = loc.x - pad;
            R.sy[k] = loc.y - pad;
        }
        R.pad = pad;
        if (std::min({R.ncc[0], R.ncc[1], R.ncc[2]}) >= .6) break;
    }
    if (std::min({R.ncc[0], R.ncc[1], R.ncc[2]}) < .6) throw std::runtime_error("QR NCC below .6 even with expanded search");
    for (int i = 0; i < 630; ++i) {
        const int k = fixed630::kRegion[i];
        const int box[4] = {kBox[k][0] + R.sx[k], kBox[k][1] + R.sy[k], kBox[k][2] + R.sx[k], kBox[k][3] + R.sy[k]};
        const double start[2] = {fixed630::kBase[i][0] + R.sx[k], fixed630::kBase[i][1] + R.sy[k]};
        const double* inv = fixed630::kInv[i];
        const double* patch = fixed630::kPatch[i];
        double est[2] = {start[0], start[1]};
        Fit f1{};
        for (int it = 0; it < 3; ++it) {
            f1 = fit(window(img, est, 4, box), start, inv, patch);
            const bool moved = std::nearbyint(f1.p[0]) != std::nearbyint(est[0]) || std::nearbyint(f1.p[1]) != std::nearbyint(est[1]);
            est[0] = f1.p[0];
            est[1] = f1.p[1];
            if (!moved) break;
        }
        const Fit f2 = fit(window(img, est, 5, box), start, inv, patch);
        const double stability = std::hypot(f1.p[0] - f2.p[0], f1.p[1] - f2.p[1]);
        const bool bad = !(f1.ok && f2.ok) || f1.rmse > 15 || f1.p[5] < 45 || stability > .65 ||
                         std::max(std::abs(f1.p[0] - start[0]), std::abs(f1.p[1] - start[1])) > 3.6 || std::max(f1.p[2], f1.p[3]) > 2.6;
        R.valid[i] = !bad;
        R.cur[i][0] = f1.p[0];
        R.cur[i][1] = f1.p[1];
        R.D[i][0] = f1.p[0] - fixed630::kRef[i][0];
        R.D[i][1] = f1.p[1] - fixed630::kRef[i][1];
        R.rmse[i] = f1.rmse;
    }
    return R;
}

}  // namespace s630
