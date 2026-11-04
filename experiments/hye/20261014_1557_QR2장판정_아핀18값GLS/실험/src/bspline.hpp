// scipy.ndimage 의 3차 B-스플라인 (spline_filter order=3 mode='mirror' + map_coordinates order=3 prefilter=False)
// 과 skimage.registration.phase_cross_correlation(upsample_factor, normalization=None) 을 그대로 옮긴 것.
#pragma once
#include <opencv2/core.hpp>
#include <cmath>
#include <complex>
#include <vector>

namespace bsp {

// 1차원 3차 스플라인 계수 (scipy _init_causal_mirror / _init_anticausal_mirror)
inline void filter1d(double* c, int n, ptrdiff_t step) {
    if (n < 2) return;
    const double z = std::sqrt(3.0) - 2.0;
    const double gain = (1 - z) * (1 - 1 / z);  // scipy 와 같은 계산 (= 6)
    for (int i = 0; i < n; ++i) c[i * step] *= gain;
    double zi = z, zn1 = std::pow(z, n - 1);
    double c0 = c[0] + zn1 * c[(n - 1) * step];
    for (int i = 1; i < n - 1; ++i) { c0 += zi * (c[i * step] + zn1 * c[(n - 1 - i) * step]); zi *= z; }
    c[0] = c0 / (1 - zn1 * zn1);
    for (int i = 1; i < n; ++i) c[i * step] += z * c[(i - 1) * step];
    c[(n - 1) * step] = (z / (z * z - 1)) * (c[(n - 1) * step] + z * c[(n - 2) * step]);
    for (int i = n - 2; i >= 0; --i) c[i * step] = z * (c[(i + 1) * step] - c[i * step]);
}

// 2차원 계수 (CV_64F 입력 -> 새 CV_64F). scipy 는 축 0(행 방향) 다음 축 1 순서로 거른다.
inline cv::Mat prefilter(const cv::Mat& src) {
    cv::Mat c;
    src.convertTo(c, CV_64F);
    const int H = c.rows, W = c.cols;
    const ptrdiff_t st = c.step1();
    double* d = c.ptr<double>();
    for (int x = 0; x < W; ++x) filter1d(d + x, H, st);
    for (int y = 0; y < H; ++y) filter1d(d + y * st, W, 1);
    return c;
}

inline void weights(double t, double w[4]) {
    double t2 = t * t, t3 = t2 * t, u = 1 - t;
    w[0] = u * u * u / 6;
    w[1] = (4 - 6 * t2 + 3 * t3) / 6;
    w[2] = (1 + 3 * t + 3 * t2 - 3 * t3) / 6;
    w[3] = t3 / 6;
}

// 한 위치의 계수 가중 (여러 계수 영상에 같은 가중을 쓴다). mode='nearest' 처럼 범위 밖 첨자는 가장자리로 자른다.
struct Tap {
    int iy[4], ix[4];
    double wy[4], wx[4];
    Tap(double x, double y, int W, int H) {
        double fx = std::floor(x), fy = std::floor(y);
        weights(x - fx, wx);
        weights(y - fy, wy);
        for (int k = 0; k < 4; ++k) {
            ix[k] = std::min(std::max((int)fx - 1 + k, 0), W - 1);
            iy[k] = std::min(std::max((int)fy - 1 + k, 0), H - 1);
        }
    }
    double at(const cv::Mat& c) const {
        double s = 0;
        for (int a = 0; a < 4; ++a) {
            const double* r = c.ptr<double>(iy[a]);
            s += wy[a] * (wx[0] * r[ix[0]] + wx[1] * r[ix[1]] + wx[2] * r[ix[2]] + wx[3] * r[ix[3]]);
        }
        return s;
    }
};
inline double sample(const cv::Mat& c, double x, double y) { return Tap(x, y, c.cols, c.rows).at(c); }

// skimage phase_cross_correlation(reference, moving, upsample_factor=up, normalization=None).
// 반환: (행 이동, 열 이동). N x N 정사각 입력만.
struct PhaseCorr {
    int N, up, R;
    double dftshift;
    std::vector<std::complex<double>> E;  // E[u*N + c] = exp(-i2pi (u - dftshift) f_c)
    std::vector<double> f;
    explicit PhaseCorr(int n, int upsample) : N(n), up(upsample) {
        R = (int)std::ceil(up * 1.5);
        dftshift = std::trunc(R / 2.0);
        f.resize(N);
        for (int k = 0; k < N; ++k) { int kk = k < (N + 1) / 2 ? k : k - N; f[k] = kk / (double(N) * up); }
        E.resize((size_t)R * N);
        for (int u = 0; u < R; ++u)
            for (int k = 0; k < N; ++k) E[(size_t)u * N + k] = std::exp(std::complex<double>(0, -2 * M_PI * (u - dftshift) * f[k]));
    }
    std::pair<double, double> operator()(const cv::Mat& ref, const cv::Mat& mov) const {
        cv::Mat S, M, P, C;
        cv::dft(ref, S, cv::DFT_COMPLEX_OUTPUT);
        cv::dft(mov, M, cv::DFT_COMPLEX_OUTPUT);
        cv::mulSpectrums(S, M, P, 0, true);  // S * conj(M)
        cv::idft(P, C, cv::DFT_COMPLEX_OUTPUT | cv::DFT_SCALE);
        int br = 0, bc = 0;
        double best = -1;
        for (int r = 0; r < N; ++r)
            for (int c = 0; c < N; ++c) {
                auto v = C.at<cv::Vec2d>(r, c);
                double a = std::hypot(v[0], v[1]);
                if (a > best) { best = a; br = r; bc = c; }
            }
        double sr = br > N / 2 ? br - N : br, sc = bc > N / 2 ? bc - N : bc;
        sr = std::round(sr * up) / up;
        sc = std::round(sc * up) / up;
        // data = conj(P). 열 방향 먼저: T1[u][r] = sum_c K(u,c) data[r][c], K = exp(-i2pi (u - off_c) f_c)
        const double offr = dftshift - sr * up, offc = dftshift - sc * up;
        std::vector<std::complex<double>> phc(N), phr(N), T1((size_t)R * N);
        for (int k = 0; k < N; ++k) {
            phc[k] = std::exp(std::complex<double>(0, 2 * M_PI * (offc - dftshift) * f[k]));
            phr[k] = std::exp(std::complex<double>(0, 2 * M_PI * (offr - dftshift) * f[k]));
        }
        std::vector<std::complex<double>> D((size_t)N * N);
        for (int r = 0; r < N; ++r)
            for (int c = 0; c < N; ++c) {
                auto v = P.at<cv::Vec2d>(r, c);
                D[(size_t)r * N + c] = std::complex<double>(v[0], -v[1]) * phc[c];
            }
        for (int u = 0; u < R; ++u)
            for (int r = 0; r < N; ++r) {
                std::complex<double> s = 0;
                const auto* e = &E[(size_t)u * N];
                const auto* d = &D[(size_t)r * N];
                for (int c = 0; c < N; ++c) s += e[c] * d[c];
                T1[(size_t)u * N + r] = s * phr[r];
            }
        int bv = 0, bu = 0;
        best = -1;
        for (int v = 0; v < R; ++v) {
            const auto* e = &E[(size_t)v * N];
            for (int u = 0; u < R; ++u) {
                std::complex<double> s = 0;
                const auto* t = &T1[(size_t)u * N];
                for (int r = 0; r < N; ++r) s += e[r] * t[r];
                double a = std::abs(s);
                if (a > best) { best = a; bv = v; bu = u; }
            }
        }
        return {sr + (bv - dftshift) / up, sc + (bu - dftshift) / up};
    }
};

}  // namespace bsp
