// v6 특징점 추적 (원본: analysis_v6/common.py, t11_qr_features.py, analysis_v7/e4_track.py)
//   pick_qr : 흐린 center 에서 goodFeaturesToTrack -> QR 3개 + 판 나사 점
//   coarse_init : 81x81 큰 템플릿으로 넓게 찾아 정수 이동
//   track_tm : 정규화 상관 + 포물선 보간 (원본 영상)
//   track_gn : 흐린 영상에서 (dx, dy, a, b) 가우스-뉴턴. scipy 3차 스플라인(mirror 사전필터, nearest 샘플)과 같게 구현
#pragma once
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <string>
#include <vector>

namespace v6 {

// 판 영역과 QR 중심 (center.bmp). QR 중심은 analysis_v6 이 읽던 14_데이터_QR마커위치.csv 의 center 행 (영역 이름 붙이기에만 쓴다)
constexpr int X0 = 1420, X1 = 2392, Y0 = 948, Y1 = 1192;
const double kScrew[2][2] = {{1492, 1063}, {2312, 1073}};
const double kQR[3][2] = {{1642.4546, 1067.5000}, {1900.4866, 1068.2500}, {2159.2039, 1070.5367}};
constexpr int SX = 110, SY = 16, HPC = 40;

struct Feat {
    int x, y;
    int region;  // 0 좌, 1 중, 2 우 (QR), 3 판나사
    double sat;
};
inline const char* region_name(int r) {
    static const char* n[] = {"QR_L", "QR_C", "QR_R", "screw"};
    return n[r];
}

// region_of: 제외(라벨, 케이블) -> -1, 판나사 3, QR 0..2, 판 4, 판밖 5
inline int region_of(int x, int y) {
    if (1740 < x && x < 2060 && 1195 < y && y < 1360) return -1;
    if (1840 < x && x < 2300 && 0 < y && y < 590) return -1;
    if (X0 <= x && x <= X1 && Y0 <= y && y <= Y1) {
        for (auto& s : kScrew)
            if (std::hypot(x - s[0], y - s[1]) < 60) return 3;
        for (int k = 0; k < 3; ++k)
            if (std::abs(x - kQR[k][0]) < 95 && std::abs(y - kQR[k][1]) < 100) return k;
        return 4;
    }
    return 5;
}

inline cv::Mat blur(const cv::Mat& im, double s) {
    if (s <= 0) return im.clone();
    cv::Mat o;
    cv::GaussianBlur(im, o, cv::Size(0, 0), s);
    return o;
}

inline std::vector<Feat> pick_qr(const cv::Mat& cen, int hp) {
    cv::Mat b = blur(cen, 1.0);
    cv::Mat m = cv::Mat::zeros(b.size(), CV_8U);
    m(cv::Range(Y0, Y1), cv::Range(X0, X1)) = 255;
    std::vector<cv::Point2f> pts;
    cv::goodFeaturesToTrack(b, pts, 5000, 0.01, 2 * hp + 1, m, hp < 6 ? 5 : 7);
    std::vector<Feat> out;
    for (auto& p : pts) {
        int xi = (int)std::nearbyint(p.x), yi = (int)std::nearbyint(p.y);
        int r = region_of(xi, yi);
        if (!(r >= 0 && r <= 3)) continue;
        cv::Mat w = cen(cv::Range(yi - hp, yi + hp + 1), cv::Range(xi - hp, xi + hp + 1));
        double sat = (double)cv::countNonZero(w >= 250) / w.total();
        if (r < 3 && sat > 0) continue;
        if (sat > 0.6) continue;
        out.push_back({xi, yi, r, sat});
    }
    return out;
}

inline std::vector<cv::Mat> cut(const cv::Mat& img, const std::vector<Feat>& F, int hp, int off = 0) {
    std::vector<cv::Mat> o;
    for (auto& f : F) o.push_back(img(cv::Range(f.y + off - hp, f.y + off + hp + 1), cv::Range(f.x + off - hp, f.x + off + hp + 1)).clone());
    return o;
}

inline double parab(const float* a, int n, int stride, int i) {
    if (0 < i && i < n - 1) {
        double am = a[(i - 1) * stride], a0 = a[i * stride], ap = a[(i + 1) * stride];
        double d = am - 2 * a0 + ap;
        return i + (d != 0 ? 0.5 * (am - ap) / d : 0);
    }
    return i;
}

// pimg: 가장자리 복제로 HPC 만큼 넓힌 사진, big: 넓힌 center 에서 자른 81x81 템플릿
inline std::vector<cv::Point2d> coarse_init(const cv::Mat& pimg, const std::vector<cv::Mat>& big, const std::vector<Feat>& F) {
    std::vector<cv::Point2d> out(F.size());
    const int h = HPC;
    for (size_t k = 0; k < F.size(); ++k) {
        int x = F[k].x + HPC, y = F[k].y + HPC;
        int y0 = std::max(y - h - SY, 0), x0 = std::max(x - h - SX, 0);
        cv::Mat s = pimg(cv::Range(y0, std::min(y + h + SY + 1, pimg.rows)), cv::Range(x0, std::min(x + h + SX + 1, pimg.cols)));
        cv::Mat r;
        cv::matchTemplate(s, big[k], r, cv::TM_CCOEFF_NORMED);
        cv::Point loc;
        cv::minMaxLoc(r, nullptr, nullptr, nullptr, &loc);
        out[k] = {double(loc.x + x0 + h - x), double(loc.y + y0 + h - y)};
    }
    return out;
}

inline std::vector<cv::Point2d> track_tm(const cv::Mat& img, const std::vector<cv::Mat>& tm, const std::vector<Feat>& F, int hp,
                                         const std::vector<cv::Point2d>& init) {
    std::vector<cv::Point2d> out(F.size());
    for (size_t k = 0; k < F.size(); ++k) {
        int ox = (int)std::nearbyint(init[k].x), oy = (int)std::nearbyint(init[k].y), sx = 4, sy = 4;
        int x0 = F[k].x + ox - hp - sx, y0 = F[k].y + oy - hp - sy;
        cv::Mat s = img(cv::Range(y0, y0 + 2 * hp + 1 + 2 * sy), cv::Range(x0, x0 + 2 * hp + 1 + 2 * sx));
        cv::Mat r;
        cv::matchTemplate(s, tm[k], r, cv::TM_CCOEFF_NORMED);
        cv::Point loc;
        cv::minMaxLoc(r, nullptr, nullptr, nullptr, &loc);
        double px = parab(r.ptr<float>(loc.y), r.cols, 1, loc.x);
        double py = parab(r.ptr<float>(0) + loc.x, r.rows, (int)(r.step1()), loc.y);
        out[k] = {px - sx + ox, py - sy + oy};
    }
    return out;
}

// scipy.ndimage.spline_filter(order=3, mode='mirror') 한 줄
inline void spline_row(double* c, int n, int stride) {
    const double z = std::sqrt(3.0) - 2.0;
    const double gain = (1 - z) * (1 - 1 / z);
    for (int i = 0; i < n; ++i) c[i * stride] *= gain;
    double zi = z, zn1 = std::pow(z, n - 1);
    double c0 = c[0] + zn1 * c[(n - 1) * stride];
    for (int i = 1; i < n - 1; ++i) {
        c0 += zi * (c[i * stride] + zn1 * c[(n - 1 - i) * stride]);
        zi *= z;
    }
    c[0] = c0 / (1 - zn1 * zn1);
    for (int i = 1; i < n; ++i) c[i * stride] += z * c[(i - 1) * stride];
    c[(n - 1) * stride] = (z * c[(n - 2) * stride] + c[(n - 1) * stride]) * z / (z * z - 1);
    for (int i = n - 2; i >= 0; --i) c[i * stride] = z * (c[(i + 1) * stride] - c[i * stride]);
}
inline cv::Mat spline_filter(const cv::Mat& a) {
    cv::Mat c = a.clone();
    for (int x = 0; x < c.cols; ++x) spline_row(c.ptr<double>(0) + x, c.rows, (int)c.step1());
    for (int y = 0; y < c.rows; ++y) spline_row(c.ptr<double>(y), c.cols, 1);
    return c;
}
// map_coordinates(order=3, prefilter=False, mode='nearest')
inline double spline_at(const cv::Mat& c, double yy, double xx) {
    auto w = [](double f, double* o) {
        double f2 = f * f, f3 = f2 * f;
        o[0] = (1 - f) * (1 - f) * (1 - f) / 6;
        o[1] = (4 - 6 * f2 + 3 * f3) / 6;
        o[2] = (1 + 3 * f + 3 * f2 - 3 * f3) / 6;
        o[3] = f3 / 6;
    };
    double fy = std::floor(yy), fx = std::floor(xx), wy[4], wx[4];
    w(yy - fy, wy);
    w(xx - fx, wx);
    int iy = (int)fy - 1, ix = (int)fx - 1;
    double s = 0;
    for (int a = 0; a < 4; ++a) {
        int r = std::min(std::max(iy + a, 0), c.rows - 1);
        const double* p = c.ptr<double>(r);
        double t = 0;
        for (int b = 0; b < 4; ++b) t += wx[b] * p[std::min(std::max(ix + b, 0), c.cols - 1)];
        s += wy[a] * t;
    }
    return s;
}

// bimg: 흐린 사진(float32), tb: 흐린 기준에서 자른 템플릿, d0: TM 결과
inline std::vector<cv::Point2d> track_gn(const cv::Mat& bimg, const std::vector<cv::Mat>& tb, const std::vector<Feat>& F, int hp,
                                         const std::vector<cv::Point2d>& d0, int iters = 15) {
    std::vector<cv::Point2d> out(F.size());
    const int M = 3, L = hp + M + 3, N = 2 * L + 1, n = 2 * hp + 1;
    for (size_t k = 0; k < F.size(); ++k) {
        int ox = (int)std::nearbyint(d0[k].x), oy = (int)std::nearbyint(d0[k].y);
        int cx = F[k].x + ox, cy = F[k].y + oy;
        cv::Mat win;
        bimg(cv::Range(cy - L, cy + L + 1), cv::Range(cx - L, cx + L + 1)).convertTo(win, CV_64F);
        cv::Mat gx(N, N, CV_64F), gy(N, N, CV_64F);
        for (int r = 0; r < N; ++r)
            for (int c = 0; c < N; ++c) {
                gx.at<double>(r, c) = (win.at<double>(r, (c + 1) % N) - win.at<double>(r, (c - 1 + N) % N)) / 2;
                gy.at<double>(r, c) = (win.at<double>((r + 1) % N, c) - win.at<double>((r - 1 + N) % N, c)) / 2;
            }
        cv::Mat cI = spline_filter(win), cX = spline_filter(gx), cY = spline_filter(gy);
        const cv::Mat& T = tb[k];
        double dx = d0[k].x - ox, dy = d0[k].y - oy, a = 1, b = 0;
        for (int it = 0; it < iters; ++it) {
            double A[4][4] = {}, g[4] = {};
            for (int r = 0; r < n; ++r)
                for (int c = 0; c < n; ++c) {
                    double yy = r + L - hp + dy, xx = c + L - hp + dx;  // ii + L + d
                    double I = spline_at(cI, yy, xx), Ix = spline_at(cX, yy, xx), Iy = spline_at(cY, yy, xx);
                    double res = a * I + b - (double)T.at<float>(r, c);
                    double J[4] = {a * Ix, a * Iy, I, 1.0};
                    for (int p = 0; p < 4; ++p) {
                        g[p] += J[p] * res;
                        for (int q = 0; q < 4; ++q) A[p][q] += J[p] * J[q];
                    }
                }
            cv::Mat Am(4, 4, CV_64F), gm(4, 1, CV_64F), del;
            for (int p = 0; p < 4; ++p) {
                gm.at<double>(p) = g[p];
                for (int q = 0; q < 4; ++q) Am.at<double>(p, q) = A[p][q] + (p == q ? 1e-9 : 0);
            }
            cv::solve(Am, gm, del, cv::DECOMP_LU);
            double e0 = -del.at<double>(0), e1 = -del.at<double>(1);
            dx += std::min(std::max(e0, -0.5), 0.5);
            dy += std::min(std::max(e1, -0.5), 0.5);
            a += -del.at<double>(2);
            b += -del.at<double>(3);
            if (std::max(std::abs(e0), std::abs(e1)) < 1e-5) break;
        }
        out[k] = {dx + ox, dy + oy};
    }
    return out;
}

// 한 파이프라인: 템플릿 반폭 hp, 흐림 sigma
struct Pipe {
    int hp;
    double sigma;
    std::vector<cv::Mat> tr, tb;
    void setup(const cv::Mat& ref, const std::vector<Feat>& F) {
        tr = cut(ref, F, hp);
        tb = cut(blur(ref, sigma), F, hp);
    }
    std::vector<cv::Point2d> run(const cv::Mat& img, const std::vector<cv::Point2d>& c0, const std::vector<Feat>& F) const {
        auto tm = track_tm(img, tr, F, hp, c0);
        return track_gn(blur(img, sigma), tb, F, hp, tm);
    }
};

inline std::vector<cv::Mat> big_templates(const cv::Mat& ref, const std::vector<Feat>& F) {
    cv::Mat p;
    cv::copyMakeBorder(ref, p, HPC, HPC, HPC, HPC, cv::BORDER_REPLICATE);
    return cut(p, F, HPC, HPC);
}
inline std::vector<cv::Point2d> coarse(const cv::Mat& img, const std::vector<cv::Mat>& big, const std::vector<Feat>& F) {
    cv::Mat p;
    cv::copyMakeBorder(img, p, HPC, HPC, HPC, HPC, cv::BORDER_REPLICATE);
    return coarse_init(p, big, F);
}

}  // namespace v6
