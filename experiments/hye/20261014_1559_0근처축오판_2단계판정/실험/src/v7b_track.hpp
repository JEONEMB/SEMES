// v7b 점 추적 (원본: analysis_v6/common.py, t11_qr_features.py, analysis_v7b/lib.py load·warp_features)
//  1) 흐린 center(σ1)에서 QR 판 모서리 고르기 (goodFeaturesToTrack, 17px 간격) -> QR 3개 위 점만
//  2) 사진마다 81x81 NCC 로 정수 이동 -> 17x17 NCC + 포물선 -> 흐린 사진에서 가우스-뉴턴 (dx, dy, 밝기 a, b)
//     가우스-뉴턴 보간은 scipy ndimage 3차 B-spline (spline_filter mirror + map_coordinates order 3) 과 같은 식
//  3) center 기준 이동 D, 판 전체 2차 다항 변형 12값(P12)으로 요약
#pragma once
#include <hye/dataset.hpp>
#include <hye/report.hpp>
#include <opencv2/imgproc.hpp>
#include <array>
#include <cmath>
#include <string>
#include <vector>

namespace v7b {

constexpr int X0 = 1420, X1 = 2392, Y0 = 948, Y1 = 1192;  // QR 판
constexpr int HP = 8, HPC = 40, SX = 110, SY = 16;
// center.bmp 의 QR 세 개 중심 (원본: 14_데이터_QR마커위치.csv center 행, 영역 나누기에만 쓰는 값)
const double kQRc[3][2] = {{1642.4546, 1067.5000}, {1900.4866, 1068.2500}, {2159.2039, 1070.5367}};
const int kScrew[2][2] = {{1492, 1063}, {2312, 1073}};

struct Pt { int x, y, region; };  // region 0 좌, 1 중, 2 우

inline cv::Mat to_f32(const cv::Mat& g) { cv::Mat f; g.convertTo(f, CV_32F); return f; }
inline cv::Mat blur1(const cv::Mat& f) { cv::Mat b; cv::GaussianBlur(f, b, cv::Size(0, 0), 1.0); return b; }

// region_of (v6 common): 라벨 영역 제외 -> 판 안 -> 나사 60px -> QR (|dx|<95, |dy|<100)
inline int region_of(int x, int y) {
    if (1740 < x && x < 2060 && 1195 < y && y < 1360) return -2;  // 라벨
    if (1840 < x && x < 2300 && 0 < y && y < 590) return -2;      // 케이블
    if (X0 <= x && x <= X1 && Y0 <= y && y <= Y1) {
        for (auto& s : kScrew) if (std::hypot(x - s[0], y - s[1]) < 60) return -3;
        for (int k = 0; k < 3; ++k) if (std::abs(x - kQRc[k][0]) < 95 && std::abs(y - kQRc[k][1]) < 100) return k;
        return -4;
    }
    return -1;
}

inline std::vector<Pt> pick_qr(const cv::Mat& cen_raw /*f32*/) {
    cv::Mat b = blur1(cen_raw);
    cv::Mat m = cv::Mat::zeros(b.size(), CV_8U);
    m(cv::Range(Y0, Y1), cv::Range(X0, X1)) = 255;
    std::vector<cv::Point2f> pts;
    cv::goodFeaturesToTrack(b, pts, 5000, 0.01, 2 * HP + 1, m, 7);
    std::vector<Pt> out;
    for (auto& p : pts) {
        int xi = (int)std::lround(p.x), yi = (int)std::lround(p.y);
        // Python round 는 짝수 쪽 반올림이지만 goodFeaturesToTrack 좌표는 정수라 차이 없음
        int r = region_of(xi, yi);
        if (r < 0) continue;  // QR 만 (나사는 v7b 에서 쓰지 않음)
        cv::Mat w = cen_raw(cv::Range(yi - HP, yi + HP + 1), cv::Range(xi - HP, xi + HP + 1));
        double mx;
        cv::minMaxLoc(w, nullptr, &mx);
        if (mx >= 250) continue;  // 포화 점 제외
        out.push_back({xi, yi, r});
    }
    return out;
}

// ---- scipy ndimage 3차 B-spline ----
inline void spline1d_mirror(double* c, int n, int stride) {
    const double z = std::sqrt(3.0) - 2.0, lam = (1 - z) * (1 - 1 / z);
    auto C = [&](int i) -> double& { return c[(size_t)i * stride]; };
    for (int i = 0; i < n; ++i) C(i) *= lam;
    double zi = z, zn1 = std::pow(z, n - 1);
    C(0) = C(0) + zn1 * C(n - 1);
    for (int i = 1; i < n - 1; ++i) { C(0) += zi * (C(i) + zn1 * C(n - 1 - i)); zi *= z; }
    C(0) /= 1 - zn1 * zn1;
    for (int i = 1; i < n; ++i) C(i) += z * C(i - 1);
    C(n - 1) = (z * C(n - 2) + C(n - 1)) * z / (z * z - 1);
    for (int i = n - 2; i >= 0; --i) C(i) = z * (C(i + 1) - C(i));
}
inline cv::Mat spline_filter(const cv::Mat& w /*f64*/) {
    cv::Mat c = w.clone();
    for (int x = 0; x < c.cols; ++x) spline1d_mirror(c.ptr<double>(0) + x, c.rows, (int)(c.step1()));  // axis 0
    for (int y = 0; y < c.rows; ++y) spline1d_mirror(c.ptr<double>(y), c.cols, 1);                   // axis 1
    return c;
}
inline void bweights(double x, int& i0, double w[4]) {
    double f = std::floor(x), t = x - f;
    i0 = (int)f - 1;
    w[0] = (1 - t) * (1 - t) * (1 - t) / 6;
    w[1] = (4 - 6 * t * t + 3 * t * t * t) / 6;
    w[2] = (1 + 3 * t + 3 * t * t - 3 * t * t * t) / 6;
    w[3] = t * t * t / 6;
}
inline double sample(const cv::Mat& c, int iy, const double* wy, int ix, const double* wx) {
    double s = 0;
    for (int a = 0; a < 4; ++a) {
        int yy = std::min(std::max(iy + a, 0), c.rows - 1);
        const double* r = c.ptr<double>(yy);
        double t = 0;
        for (int b = 0; b < 4; ++b) t += wx[b] * r[std::min(std::max(ix + b, 0), c.cols - 1)];
        s += wy[a] * t;
    }
    return s;
}

inline float parab(const float* a, int n, int stride, int i) {
    if (0 < i && i < n - 1) {
        float am = a[(i - 1) * stride], a0 = a[i * stride], ap = a[(i + 1) * stride];
        float d = am - 2 * a0 + ap;
        return (float)i + (d != 0 ? 0.5f * (am - ap) / d : 0.f);  // numpy float32 (NEP 50) 연산
    }
    return (float)i;
}

struct Tmpl { std::vector<cv::Mat> raw, blur, big; };
inline Tmpl make_templates(const cv::Mat& cen_raw, const std::vector<Pt>& F) {
    Tmpl t;
    cv::Mat cb = blur1(cen_raw);
    for (auto& p : F) {
        t.raw.push_back(cen_raw(cv::Rect(p.x - HP, p.y - HP, 2 * HP + 1, 2 * HP + 1)).clone());
        t.blur.push_back(cb(cv::Rect(p.x - HP, p.y - HP, 2 * HP + 1, 2 * HP + 1)).clone());
        t.big.push_back(cen_raw(cv::Rect(p.x - HPC, p.y - HPC, 2 * HPC + 1, 2 * HPC + 1)).clone());
    }
    return t;
}

// 한 사진의 점별 이동 (N, 2). 판 근처라 np.pad 가장자리 채우기는 쓰이지 않는다.
inline std::vector<cv::Vec2d> track(const cv::Mat& img /*f32*/, const Tmpl& T, const std::vector<Pt>& F, std::vector<double>* score = nullptr) {
    cv::Mat bimg = blur1(img);
    const int n = 2 * HP + 1, L = HP + 3 + 3;
    std::vector<cv::Vec2d> out(F.size());
    if (score) score->assign(F.size(), 0);
    for (size_t k = 0; k < F.size(); ++k) {
        const int x = F[k].x, y = F[k].y;
        // coarse_init: 81x81, 가로 ±110, 세로 ±16
        cv::Mat s = img(cv::Range(y - HPC - SY, y + HPC + SY + 1), cv::Range(x - HPC - SX, x + HPC + SX + 1)), r;
        cv::matchTemplate(s, T.big[k], r, cv::TM_CCOEFF_NORMED);
        cv::Point loc;
        cv::minMaxLoc(r, nullptr, nullptr, nullptr, &loc);
        int ox = loc.x - SX, oy = loc.y - SY;
        // track_tm: 17x17, ±4
        int x0 = x + ox - HP - 4, y0 = y + oy - HP - 4;
        cv::Mat s2 = img(cv::Rect(x0, y0, n + 8, n + 8)), r2;
        cv::matchTemplate(s2, T.raw[k], r2, cv::TM_CCOEFF_NORMED);
        double mx;
        cv::minMaxLoc(r2, nullptr, &mx, nullptr, &loc);
        if (score) (*score)[k] = mx;
        const float* rp = r2.ptr<float>(0);
        float px = parab(rp + loc.y * r2.cols, r2.cols, 1, loc.x), py = parab(rp + loc.x, r2.rows, r2.cols, loc.y);
        double d0x = (double)(px - 4.f + (float)ox), d0y = (double)(py - 4.f + (float)oy);
        // track_gn
        int cx = x + (int)std::nearbyint(d0x), cy = y + (int)std::nearbyint(d0y);
        cv::Mat win;
        bimg(cv::Range(cy - L, cy + L + 1), cv::Range(cx - L, cx + L + 1)).convertTo(win, CV_64F);
        const int W = 2 * L + 1;
        cv::Mat gx(W, W, CV_64F), gy(W, W, CV_64F);
        for (int i = 0; i < W; ++i)
            for (int j = 0; j < W; ++j) {
                gx.at<double>(i, j) = (win.at<double>(i, (j + 1) % W) - win.at<double>(i, (j - 1 + W) % W)) / 2;
                gy.at<double>(i, j) = (win.at<double>((i + 1) % W, j) - win.at<double>((i - 1 + W) % W, j)) / 2;
            }
        cv::Mat cI = spline_filter(win), cX = spline_filter(gx), cY = spline_filter(gy);
        double dx = d0x - (cx - x), dy = d0y - (cy - y), a = 1, b = 0;
        const cv::Mat& Tb = T.blur[k];
        for (int it = 0; it < 15; ++it) {
            double JtJ[4][4] = {}, Jtr[4] = {};
            for (int i = -HP; i <= HP; ++i) {
                int iy; double wy[4];
                bweights(i + L + dy, iy, wy);
                for (int j = -HP; j <= HP; ++j) {
                    int ix; double wx[4];
                    bweights(j + L + dx, ix, wx);
                    double I = sample(cI, iy, wy, ix, wx), Ix = sample(cX, iy, wy, ix, wx), Iy = sample(cY, iy, wy, ix, wx);
                    double rr = a * I + b - (double)Tb.at<float>(i + HP, j + HP);
                    double J[4] = {a * Ix, a * Iy, I, 1.0};
                    for (int p = 0; p < 4; ++p) {
                        Jtr[p] += J[p] * rr;
                        for (int q = 0; q < 4; ++q) JtJ[p][q] += J[p] * J[q];
                    }
                }
            }
            cv::Mat A(4, 4, CV_64F), B(4, 1, CV_64F), dl;
            for (int p = 0; p < 4; ++p) {
                B.at<double>(p) = Jtr[p];
                for (int q = 0; q < 4; ++q) A.at<double>(p, q) = JtJ[p][q] + (p == q ? 1e-9 : 0);
            }
            cv::solve(A, B, dl, cv::DECOMP_LU);
            double ddx = -dl.at<double>(0), ddy = -dl.at<double>(1);
            dx += std::min(std::max(ddx, -0.5), 0.5);
            dy += std::min(std::max(ddy, -0.5), 0.5);
            a += -dl.at<double>(2);
            b += -dl.at<double>(3);
            if (std::max(std::abs(ddx), std::abs(ddy)) < 1e-5) break;
        }
        out[k] = {dx + (cx - x), dy + (cy - y)};
    }
    return out;
}

// 판 2차 변형 12값: 기저 [1, x, y, x², xy, y²] (x = (X-1906)/486, y = (Y-1070)/122). 순서 b0x, b0y, b1x, b1y, ...
struct P12 {
    cv::Mat pinvB;  // 6 x N
    explicit P12(const std::vector<Pt>& F) {
        cv::Mat B((int)F.size(), 6, CV_64F);
        for (int i = 0; i < B.rows; ++i) {
            double x = (F[i].x - 1906) / 486.0, y = (F[i].y - 1070) / 122.0;
            double r[6] = {1, x, y, x * x, x * y, y * y};
            for (int j = 0; j < 6; ++j) B.at<double>(i, j) = r[j];
        }
        cv::invert(B, pinvB, cv::DECOMP_SVD);
    }
    std::vector<double> operator()(const std::vector<cv::Vec2d>& D) const {
        cv::Mat d((int)D.size(), 2, CV_64F);
        for (int i = 0; i < d.rows; ++i) { d.at<double>(i, 0) = D[i][0]; d.at<double>(i, 1) = D[i][1]; }
        cv::Mat co = pinvB * d;  // 6 x 2
        std::vector<double> f(12);
        for (int j = 0; j < 6; ++j) { f[2 * j] = co.at<double>(j, 0); f[2 * j + 1] = co.at<double>(j, 1); }
        return f;
    }
};

// 방해 무늬(세로 평행 이동, 화면 회전)를 P12 값으로 (U_for kind P12). 12 x 2
inline cv::Mat nuis_p12(const std::vector<Pt>& F, const P12& p12) {
    double mx = 0, my = 0;
    for (auto& p : F) { mx += p.x; my += p.y; }
    mx /= F.size(); my /= F.size();
    cv::Mat U(12, 2, CV_64F);
    for (int j = 0; j < 2; ++j) {
        std::vector<cv::Vec2d> D(F.size());
        for (size_t i = 0; i < F.size(); ++i) {
            double xc = (F[i].x - mx) / 100, yc = (F[i].y - my) / 100;
            D[i] = j == 0 ? cv::Vec2d(0, 0.05) : cv::Vec2d(-yc * 0.05, xc * 0.05);
        }
        auto f = p12(D);
        for (int r = 0; r < 12; ++r) U.at<double>(r, j) = f[r] / 0.05;
    }
    return U;
}

}  // namespace v7b
