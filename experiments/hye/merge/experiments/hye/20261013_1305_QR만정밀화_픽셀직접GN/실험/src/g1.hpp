// G1: QR 판 픽셀 직접 가우스-뉴턴 (analysis_v3/gn_pixel.py, run_g1.py).
// 판 귀퉁이 4개 이동(8값)으로 정해지는 원근 변환을 선형화(map = P0 + PHI d)하고, 기준 영상 픽셀을 (Θ, Y, 방해 3)이 만드는 변형으로 옮긴 위치에서
// 새 사진을 3차 B-스플라인으로 읽어 밝기 차 제곱합을 감쇠 가우스-뉴턴으로 줄인다.
// 3차 스플라인 = scipy.ndimage.spline_filter(order 3, mode mirror) + map_coordinates(order 3, prefilter False, mode nearest).
#pragma once
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace g1 {

// 1차원 3차 B-스플라인 계수 (mirror 경계, 극점 sqrt(3)-2)
inline void spline_line(double* s, int n, int stride) {
    if (n < 2) return;
    const double z = -0.267949192431122706472553658494127633;
    const double gain = (1 - z) * (1 - 1 / z);
    for (int i = 0; i < n; ++i) s[i * stride] *= gain;
    double zn = std::pow(z, n - 1), first = s[0] + zn * s[(n - 1) * stride], pw = z;
    for (int i = 1; i < n - 1; ++i) { first += pw * (s[i * stride] + zn * s[(n - 1 - i) * stride]); pw *= z; }
    s[0] = first / (1 - zn * zn);
    for (int i = 1; i < n; ++i) s[i * stride] += z * s[(i - 1) * stride];
    s[(n - 1) * stride] = (z * s[(n - 2) * stride] + s[(n - 1) * stride]) * z / (z * z - 1);
    for (int i = n - 2; i >= 0; --i) s[i * stride] = z * (s[(i + 1) * stride] - s[i * stride]);
}
inline cv::Mat spline_coef(const cv::Mat& img) {
    cv::Mat o;
    img.convertTo(o, CV_64F);
    for (int x = 0; x < o.cols; ++x) spline_line(o.ptr<double>(0) + x, o.rows, (int)o.step1());
    for (int y = 0; y < o.rows; ++y) spline_line(o.ptr<double>(y), o.cols, 1);
    return o;
}
inline void weights(double c, double* w) {
    double t = c - std::floor(c), r = 1 - t;
    w[1] = (t * t * (t - 2) * 3 + 4) / 6;
    w[2] = (r * r * (r - 2) * 3 + 4) / 6;
    w[0] = r * r * r / 6;
    w[3] = 1 - w[0] - w[1] - w[2];
}
// 계수 영상에서 (mx, my) 위치 값. 범위 밖 계수는 가장 가까운 칸 (mode nearest)
inline cv::Mat warp(const cv::Mat& coef, const cv::Mat& mx, const cv::Mat& my) {
    cv::Mat out(mx.size(), CV_32F);
    const int W = coef.cols, H = coef.rows;
    for (int y = 0; y < out.rows; ++y) {
        const float* xs = mx.ptr<float>(y);
        const float* ys = my.ptr<float>(y);
        float* o = out.ptr<float>(y);
        for (int x = 0; x < out.cols; ++x) {
            double wx[4], wy[4];
            weights(xs[x], wx);
            weights(ys[x], wy);
            int sx = (int)std::floor((double)xs[x]) - 1, sy = (int)std::floor((double)ys[x]) - 1;
            double v = 0;
            for (int r = 0; r < 4; ++r) {
                const double* row = coef.ptr<double>(std::min(std::max(sy + r, 0), H - 1));
                double a = 0;
                for (int c = 0; c < 4; ++c) a += row[std::min(std::max(sx + c, 0), W - 1)] * wx[c];
                v += a * wy[r];
            }
            o[x] = (float)v;
        }
    }
    return out;
}

struct Image { cv::Mat I, gx, gy; };

struct Engine {
    int X0, X1, Y0, Y1, HH, WW, BX0, BY0;
    static constexpr int MX = 130, MY = 40;
    std::vector<cv::Point2f> SRC;
    cv::Mat P0;          // Npix x 2 (double) 판 영역 픽셀의 전체 사진 좌표
    cv::Mat PHIx, PHIy;  // Npix x 8
    cv::Mat NUIS;        // 8 x 3: 세로 이동, 회전, 크기

    Engine(int x0, int x1, int y0, int y1) : X0(x0), X1(x1), Y0(y0), Y1(y1) {
        HH = Y1 - Y0; WW = X1 - X0; BX0 = X0 - MX; BY0 = Y0 - MY;
        SRC = {{(float)X0, (float)Y0}, {(float)X1, (float)Y0}, {(float)X1, (float)Y1}, {(float)X0, (float)Y1}};
        const int N = HH * WW;
        P0 = cv::Mat(N, 1, CV_64FC2);
        for (int y = 0; y < HH; ++y)
            for (int x = 0; x < WW; ++x) P0.at<cv::Vec2d>(y * WW + x) = {(double)(x + X0), (double)(y + Y0)};
        PHIx = cv::Mat(N, 8, CV_64F);
        PHIy = cv::Mat(N, 8, CV_64F);
        for (int k = 0; k < 8; ++k) {
            double d[8] = {0};
            d[k] = 0.5;
            cv::Mat a = exact_map(d);
            d[k] = -0.5;
            cv::Mat b = exact_map(d);
            for (int i = 0; i < N; ++i) {
                cv::Vec2d df = a.at<cv::Vec2d>(i) - b.at<cv::Vec2d>(i);
                PHIx.at<double>(i, k) = df[0];
                PHIy.at<double>(i, k) = df[1];
            }
        }
        float mx = (SRC[0].x + SRC[1].x + SRC[2].x + SRC[3].x) / 4, my = (SRC[0].y + SRC[1].y + SRC[2].y + SRC[3].y) / 4;
        NUIS = cv::Mat::zeros(8, 3, CV_64F);
        for (int c = 0; c < 4; ++c) {
            float cx = SRC[c].x - mx, cy = SRC[c].y - my;
            NUIS.at<double>(2 * c + 1, 0) = 1;
            NUIS.at<double>(2 * c, 1) = (double)(-cy / 1000.0f);
            NUIS.at<double>(2 * c + 1, 1) = (double)(cx / 1000.0f);
            NUIS.at<double>(2 * c, 2) = (double)(cx / 1000.0f);
            NUIS.at<double>(2 * c + 1, 2) = (double)(cy / 1000.0f);
        }
    }
    cv::Mat exact_map(const double* d) const {
        std::vector<cv::Point2f> dst;
        for (int c = 0; c < 4; ++c) dst.push_back({(float)(SRC[c].x + d[2 * c]), (float)(SRC[c].y + d[2 * c + 1])});
        cv::Mat H = cv::getPerspectiveTransform(SRC, dst), o;
        cv::perspectiveTransform(P0, o, H);
        return o;
    }
    double check_linear(const double* d) const {
        cv::Mat e = exact_map(d);
        double w = 0;
        for (int i = 0; i < e.rows; ++i) {
            double lx = P0.at<cv::Vec2d>(i)[0], ly = P0.at<cv::Vec2d>(i)[1];
            for (int k = 0; k < 8; ++k) { lx += PHIx.at<double>(i, k) * d[k]; ly += PHIy.at<double>(i, k) * d[k]; }
            w = std::max({w, std::abs(e.at<cv::Vec2d>(i)[0] - lx), std::abs(e.at<cv::Vec2d>(i)[1] - ly)});
        }
        return w;
    }
    void lin_map(const double* d, cv::Mat& mx, cv::Mat& my) const {
        mx.create(HH, WW, CV_32F);
        my.create(HH, WW, CV_32F);
        float* px = mx.ptr<float>();
        float* py = my.ptr<float>();
        for (int i = 0; i < HH * WW; ++i) {
            const double* fx = PHIx.ptr<double>(i);
            const double* fy = PHIy.ptr<double>(i);
            double ax = 0, ay = 0;
            for (int k = 0; k < 8; ++k) { ax += fx[k] * d[k]; ay += fy[k] * d[k]; }
            const cv::Vec2d& b = P0.at<cv::Vec2d>(i);
            px[i] = (float)(b[0] + ax - BX0);
            py[i] = (float)(b[1] + ay - BY0);
        }
    }
    // 큰 자르기 영역을 흐리고(σ1) 기울기(Sobel/8)를 구해 스플라인 계수로
    Image prepare(const cv::Mat& gray, bool grads = true) const {
        cv::Mat big;
        gray(cv::Range(BY0, Y1 + MY), cv::Range(BX0, X1 + MX)).convertTo(big, CV_32F);
        cv::GaussianBlur(big, big, cv::Size(0, 0), 1.0);
        Image im;
        im.I = spline_coef(big);
        if (grads) {
            cv::Mat gx, gy;
            cv::Sobel(big, gx, CV_32F, 1, 0, 3);
            cv::Sobel(big, gy, CV_32F, 0, 1, 3);
            im.gx = spline_coef(gx / 8);
            im.gy = spline_coef(gy / 8);
        }
        return im;
    }

    struct Fit { std::array<double, 5> p; double cost; int it; };
    // p = [Θ, Y, 세로, 회전, 크기], B = 3 x 8 ([Θ, Y, 1] → 귀퉁이 이동)
    Fit solve(const Image& img, const cv::Mat& B, std::array<double, 5> p, const cv::Mat& Tm, int iters = 60) const {
        const int k = 5, N = HH * WW;
        std::vector<cv::Mat> DX(k), DY(k);
        for (int j = 0; j < k; ++j) {
            cv::Mat v = j < 2 ? cv::Mat(B.row(j).t()) : cv::Mat(NUIS.col(j - 2));
            cv::Mat dx = PHIx * v, dy = PHIy * v;
            dx.convertTo(DX[j], CV_32F);
            dy.convertTo(DY[j], CV_32F);
        }
        auto dvec = [&](const std::array<double, 5>& q, double* d) {
            for (int c = 0; c < 8; ++c) {
                d[c] = q[0] * B.at<double>(0, c) + q[1] * B.at<double>(1, c) + B.at<double>(2, c);
                for (int j = 0; j < 3; ++j) d[c] += NUIS.at<double>(c, j) * q[2 + j];
            }
        };
        const float* T = Tm.ptr<float>();
        auto resid = [&](const std::array<double, 5>& q, double a, double b, std::vector<double>& r, cv::Mat& mx, cv::Mat& my) {
            double d[8];
            dvec(q, d);
            lin_map(d, mx, my);
            cv::Mat Iw = warp(img.I, mx, my);
            const float* w = Iw.ptr<float>();
            r.resize(N);
            double s = 0;
            for (int i = 0; i < N; ++i) { r[i] = (double)w[i] - (a * (double)T[i] + b); s += r[i] * r[i]; }
            return s / N;
        };
        double a = 1, b = 0;
        std::vector<double> r, rn;
        cv::Mat mx, my, mxn, myn;
        double c = resid(p, a, b, r, mx, my), lam = 1e-4;
        int it = 0;
        for (it = 0; it < iters; ++it) {
            cv::Mat gxw = warp(img.gx, mx, my), gyw = warp(img.gy, mx, my);
            const float* gx = gxw.ptr<float>();
            const float* gy = gyw.ptr<float>();
            double H[7][7] = {{0}}, g[7] = {0};
            for (int i = 0; i < N; ++i) {
                double A[7];
                for (int j = 0; j < k; ++j) A[j] = (double)(float)(gx[i] * DX[j].at<float>(i) + gy[i] * DY[j].at<float>(i));
                A[5] = -(double)T[i];
                A[6] = -1;
                for (int u = 0; u < 7; ++u) {
                    g[u] -= A[u] * r[i];
                    for (int v = u; v < 7; ++v) H[u][v] += A[u] * A[v];
                }
            }
            for (int u = 0; u < 7; ++u) for (int v = 0; v < u; ++v) H[u][v] = H[v][u];
            double st[7];
            while (true) {
                cv::Mat Hm(7, 7, CV_64F), gm(7, 1, CV_64F, g), sm;
                for (int u = 0; u < 7; ++u) for (int v = 0; v < 7; ++v) Hm.at<double>(u, v) = H[u][v] + (u == v ? lam * H[u][u] : 0);
                cv::solve(Hm, gm, sm, cv::DECOMP_LU);
                for (int u = 0; u < 7; ++u) st[u] = sm.at<double>(u);
                std::array<double, 5> pn;
                for (int j = 0; j < k; ++j) pn[j] = p[j] + st[j];
                double an = a + st[5], bn = b + st[6];
                double cn = resid(pn, an, bn, rn, mxn, myn);
                if (cn < c) {
                    p = pn; a = an; b = bn; std::swap(r, rn); mx = mxn.clone(); my = myn.clone(); c = cn;
                    lam = std::max(lam / 10, 1e-9);
                    break;
                }
                lam *= 10;
                if (lam > 1e6) return {p, c, it};
            }
            double mxs = 0;
            for (int j = 0; j < k; ++j) mxs = std::max(mxs, std::abs(st[j]));
            if (mxs < 1e-7) break;
        }
        return {p, c, it};
    }
    // 출발점 두 곳(가로 이동을 전부 Θ로 / 전부 Y로) 중 비용이 낮은 답
    Fit solve_multi(const Image& img, const cv::Mat& B, double m, const cv::Mat& Tm) const {
        Fit a = solve(img, B, {m / 80.0, 0, 0, 0, 0}, Tm);
        Fit b = solve(img, B, {0, m / 14.25, 0, 0, 0}, Tm);
        return b.cost < a.cost ? b : a;
    }
};

}  // namespace g1
