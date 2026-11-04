// 10-21 4-fold 연작(#50~#55)이 함께 쓰는 측정·판정 코드.
//  - 사진 281장 목록(원래 순서: center, 그다음 파일명 정렬), 세션 이름
//  - center QR 구조 모서리 630점: QR 해독 격자 -> 경계 모델 서브픽셀(TRF soft_l1) -> 사진마다 같은 방식 추적(v1_edge630)
//  - 630점 패치 추적: QR별 NCC 대략 이동 -> LK15 -> 3차 스플라인 GN(창·흐림·강건 가중 선택)  (measurement/spread.py, refine.py)
//  - P12 공간 요약, 원점 통과 이차곡선 GLS 판정 (zero_experiment.py fit/predict, multiref experiment.py fit)
//  - sklearn StratifiedKFold(shuffle) 를 numpy MT19937 그대로 재현
#pragma once
#include <hye/dataset.hpp>
#include <hye/report.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/video/tracking.hpp>
#include <omp.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <vector>
#include "trf.hpp"

namespace yt {

// ---------------------------------------------------------------- 공통 상수 (measurement/measure.py)
const int CROP[4] = {1340, 800, 2470, 1330};
const int BOXES[3][4] = {{1552, 969, 1736, 1167}, {1809, 971, 1993, 1169}, {2067, 973, 2252, 1171}};

// ---------------------------------------------------------------- 사진 목록
struct Img {
    hye::Shot shot;
    std::string file;     // center.bmp, T+0.01.bmp ...
    char axis;            // 'C','Y','T'
    double value;         // 파일명 명령값
    std::string session;  // C, T+, T-, Y1+, Y1-, Y2+, Y2-
};

inline std::vector<Img> images_281() {
    std::vector<Img> v;
    for (auto& s : hye::load_1014()) {
        Img m;
        m.shot = s;
        m.file = s.rel.substr(s.rel.find_last_of('/') + 1);
        if (s.condition == "center") { m.axis = 'C'; m.value = 0; m.session = "C"; }
        else {
            m.axis = m.file[0];
            m.value = std::stod(m.file.substr(1, m.file.size() - 5));
            if (m.axis == 'T') m.session = "T";
            else m.session = (std::llabs(std::llround(m.value / 0.05)) % 2) ? "Y2" : "Y1";
            m.session += m.value > 0 ? "+" : "-";
        }
        v.push_back(m);
    }
    std::sort(v.begin(), v.end(), [](const Img& a, const Img& b) {
        bool ac = a.file != "center.bmp", bc = b.file != "center.bmp";
        if (ac != bc) return ac < bc;
        return a.file < b.file;
    });
    return v;
}

// 원래 측정은 모두 이 영역만 쓴다 (1130 x 530)
inline cv::Mat read_crop(const Img& m) {
    cv::Mat g = hye::read_gray(m.shot);
    return g(cv::Rect(CROP[0], CROP[1], CROP[2] - CROP[0], CROP[3] - CROP[1])).clone();
}
inline std::vector<cv::Mat> read_all_crops(const std::vector<Img>& imgs) {
    std::vector<cv::Mat> out(imgs.size());
#pragma omp parallel for schedule(dynamic) num_threads(4)
    for (int i = 0; i < (int)imgs.size(); ++i) out[i] = read_crop(imgs[i]);
    return out;
}

// ---------------------------------------------------------------- numpy 흉내
inline double np_rint(double x) { return std::nearbyint(x); }  // 짝수 반올림
inline double percentile(std::vector<double> v, double q) {
    std::sort(v.begin(), v.end());
    double k = (v.size() - 1) * q / 100.0;
    size_t i = (size_t)std::floor(k);
    double f = k - i;
    return i + 1 < v.size() ? v[i] + (v[i + 1] - v[i]) * f : v[i];
}

// ================================================================ 1. center 구조 모서리
struct Corner {
    int id, qr, gx, gy, black;
    double sx, sy;        // 격자 투영 위치 (structural_corners.csv)
    double inv[4];        // inv(axes) (float32 로 계산)
    double patch[4];      // 2x2 검정 배치 [00,01,10,11]
    double rx = 0, ry = 0;  // center 서브픽셀 맞춤 (refined_corners.csv)
};
struct QRInfo { int qr; std::string payload; cv::Mat grid; std::vector<cv::Point2f> quad; std::string how; };

// decoded_grid_corners_20261015/build.py. OpenCV 3.4 는 QR3(FP11R)을 원래 방식(여백 5, 1~4배)으로 못 찾아서
// 여백 10~60px x 1~4배에서 해독에 성공한 사각형들의 꼭짓점 평균을 쓴다(해독 격자는 첫 성공의 것).
inline std::vector<QRInfo> detect_qr(const cv::Mat& crop) {
    cv::QRCodeDetector det;
    std::map<int, QRInfo> found;
    std::vector<cv::Point2f> pts;
    if (det.detectMulti(crop, pts)) {
        for (size_t i = 0; i + 3 < pts.size(); i += 4) {
            std::vector<cv::Point2f> q(pts.begin() + i, pts.begin() + i + 4);
            cv::Mat st;
            std::string t = det.decode(crop, q, st);
            if (t.empty()) continue;
            double gx = 0, gy = 0;
            for (auto& p : q) { gx += p.x / 4; gy += p.y / 4; }
            gx += CROP[0]; gy += CROP[1];
            for (int k = 0; k < 3; ++k)
                if (BOXES[k][0] < gx && gx < BOXES[k][2] && BOXES[k][1] < gy && gy < BOXES[k][3]) {
                    QRInfo info{k + 1, t, st.clone(), {}, "detectMulti"};
                    for (auto& p : q) info.quad.push_back({p.x + CROP[0], p.y + CROP[1]});
                    found[k] = info;
                }
        }
    }
    for (int k = 0; k < 3; ++k) {
        if (found.count(k)) continue;
        // 원래 방식(여백 5px, 1~4배)부터 해 보고, 못 찾으면 여백 10~60px x 1~4배에서 해독에 성공한 사각형을 모두 모아 꼭짓점을 평균한다.
        std::vector<QRInfo> ok5, okw;
        for (int pad : {5, 10, 20, 30, 40, 60}) {
            int x0 = BOXES[k][0] - CROP[0] - pad, y0 = BOXES[k][1] - CROP[1] - pad;
            cv::Mat c = crop(cv::Range(y0, BOXES[k][3] - CROP[1] + pad), cv::Range(x0, BOXES[k][2] - CROP[0] + pad));
            for (int s = 1; s <= 4; ++s) {
                cv::Mat r;
                cv::resize(c, r, cv::Size(), s, s);
                std::vector<cv::Point2f> q;
                cv::Mat st;
                std::string t = det.detectAndDecode(r, q, st);
                if (!t.empty() && q.size() == 4) {
                    QRInfo info{k + 1, t, st.clone(), {}, "pad" + std::to_string(pad) + "_x" + std::to_string(s)};
                    for (auto& p : q) info.quad.push_back({(float)(p.x / s + x0 + CROP[0]), (float)(p.y / s + y0 + CROP[1])});
                    (pad == 5 ? ok5 : okw).push_back(info);
                    if (pad == 5) break;
                }
            }
            if (!ok5.empty()) break;
        }
        if (!ok5.empty()) { found[k] = ok5[0]; continue; }
        if (okw.empty()) continue;
        QRInfo m = okw[0];
        for (int j = 0; j < 4; ++j) {
            double sx = 0, sy = 0;
            for (auto& o : okw) { sx += o.quad[j].x; sy += o.quad[j].y; }
            m.quad[j] = {(float)(sx / okw.size()), (float)(sy / okw.size())};
        }
        m.how = "mean of " + std::to_string(okw.size()) + " (pad10-60 x1-4)";
        found[k] = m;
    }
    std::vector<QRInfo> out;
    for (auto& kv : found) out.push_back(kv.second);
    return out;
}

inline std::vector<Corner> structural_corners(const std::vector<QRInfo>& qrs) {
    std::vector<Corner> out;
    for (auto& q : qrs) {
        int n = q.grid.rows;
        cv::Mat black = cv::Mat::zeros(n + 2, n + 2, CV_8U);
        for (int y = 0; y < n; ++y) for (int x = 0; x < n; ++x) black.at<uchar>(y + 1, x + 1) = q.grid.at<uchar>(y, x) == 0;
        std::vector<cv::Point2f> src{{0, 0}, {(float)n, 0}, {(float)n, (float)n}, {0, (float)n}};
        cv::Mat H = cv::getPerspectiveTransform(src, q.quad);
        for (int y = 0; y <= n; ++y)
            for (int x = 0; x <= n; ++x) {
                int a = black.at<uchar>(y, x), b = black.at<uchar>(y, x + 1), c = black.at<uchar>(y + 1, x), d = black.at<uchar>(y + 1, x + 1);
                int cnt = a + b + c + d;
                if (!(cnt == 1 || cnt == 3 || (cnt == 2 && a == d))) continue;
                std::vector<cv::Point2f> in{{(float)x, (float)y}, {(float)(x + 0.1), (float)y}, {(float)x, (float)(y + 0.1)}}, pr;
                cv::perspectiveTransform(in, pr, H);
                float ax0 = pr[1].x - pr[0].x, ay0 = pr[1].y - pr[0].y, ax1 = pr[2].x - pr[0].x, ay1 = pr[2].y - pr[0].y;
                float n0 = std::sqrt(ax0 * ax0 + ay0 * ay0), n1 = std::sqrt(ax1 * ax1 + ay1 * ay1);
                ax0 /= n0; ay0 /= n0; ax1 /= n1; ay1 /= n1;
                // axes = [[ax0, ax1],[ay0, ay1]] (열이 축)
                float det_ = ax0 * ay1 - ax1 * ay0;
                Corner co;
                co.id = (int)out.size(); co.qr = q.qr; co.gx = x; co.gy = y; co.black = cnt;
                co.sx = pr[0].x; co.sy = pr[0].y;
                co.inv[0] = (float)(ay1 / det_); co.inv[1] = (float)(-ax1 / det_); co.inv[2] = (float)(-ay0 / det_); co.inv[3] = (float)(ax0 / det_);
                co.patch[0] = a; co.patch[1] = b; co.patch[2] = c; co.patch[3] = d;
                out.push_back(co);
            }
    }
    return out;
}

// 경계 모델: 2x2 칸이 흐린 계단으로 만나는 밝기
struct EdgeFit { double p[6]; double rmse; bool ok; };
inline EdgeFit edge_fit(const std::vector<double>& X, const std::vector<double>& Y, const std::vector<double>& I, double sx, double sy, const Corner& c) {
    double low = percentile(I, 10), high = percentile(I, 90), contrast = std::max(high - low, 20.0);
    trf::Vec p0{sx, sy, 1., 1., low, contrast};
    trf::Vec lo{sx - 3.8, sy - 3.8, .35, .35, 0, 10}, hi{sx + 3.8, sy + 3.8, 2.7, 2.7, 240, 255};
    const double r2 = std::sqrt(2.0);
    auto res = [&](const trf::Vec& p, trf::Vec& f) {
        f.resize(I.size());
        for (size_t i = 0; i < I.size(); ++i) {
            double dx = X[i] - p[0], dy = Y[i] - p[1];
            double l0 = dx * c.inv[0] + dy * c.inv[1], l1 = dx * c.inv[2] + dy * c.inv[3];
            double U = .5 * (1 + std::erf(l0 / (r2 * p[2]))), V = .5 * (1 + std::erf(l1 / (r2 * p[3])));
            double fld = c.patch[0] * (1 - U) * (1 - V) + c.patch[1] * U * (1 - V) + c.patch[2] * (1 - U) * V + c.patch[3] * U * V;
            f[i] = p[4] + p[5] * (1 - fld) - I[i];
        }
    };
    auto r = trf::least_squares_soft_l1(res, p0, lo, hi, 6, 85, 1e-7, 1e-7, 1e-7);
    EdgeFit e;
    for (int k = 0; k < 6; ++k) e.p[k] = r.x[k];
    trf::Vec f;
    res(r.x, f);
    double s = 0;
    for (double v : f) s += v * v;
    e.rmse = std::sqrt(s / f.size());
    e.ok = r.success();
    return e;
}

// 창 radius 안 픽셀 (전체 사진 좌표, crop 에서 읽음). 상자 [a,cc) x [b,d) 안만
inline void window_pixels(const cv::Mat& crop, double cx, double cy, int radius, int a, int b, int cc, int d, std::vector<double>& X,
                          std::vector<double>& Y, std::vector<double>& I) {
    X.clear(); Y.clear(); I.clear();
    int x0 = (int)np_rint(cx), y0 = (int)np_rint(cy);
    for (int yy = y0 - radius; yy <= y0 + radius; ++yy)
        for (int xx = x0 - radius; xx <= x0 + radius; ++xx) {
            if (!(xx >= a && xx < cc && yy >= b && yy < d)) continue;
            X.push_back(xx); Y.push_back(yy);
            I.push_back(crop.at<uchar>(yy - CROP[1], xx - CROP[0]));
        }
}

// subpixel_grid_corners_20261015/refine.py : 투영 위치에서 9x9 창 한 번
inline void refine_center(const cv::Mat& crop, std::vector<Corner>& cs) {
#pragma omp parallel for schedule(dynamic) num_threads(4)
    for (int i = 0; i < (int)cs.size(); ++i) {
        auto& c = cs[i];
        const int* B = BOXES[c.qr - 1];
        std::vector<double> X, Y, I;
        window_pixels(crop, c.sx, c.sy, 4, B[0], B[1], B[2], B[3], X, Y, I);
        auto f = edge_fit(X, Y, I, c.sx, c.sy, c);
        c.rx = f.p[0]; c.ry = f.p[1];
    }
}

struct Tracked { double x, y, rmse, stab, contrast; bool valid; };
// analysis_grid_corners_20261019/track_corners.py : 사진마다 QR별 정수 이동 + 창 다시 맞추기
inline std::vector<std::array<int, 2>> coarse_shift_full(const cv::Mat& img, const cv::Mat& cen) {
    std::vector<std::array<int, 2>> out;
    for (auto& B : BOXES) {
        int a = B[0] - CROP[0], b = B[1] - CROP[1], c = B[2] - CROP[0], d = B[3] - CROP[1], pad = 130;
        cv::Mat tpl = cen(cv::Range(b, d), cv::Range(a, c)), srch = img(cv::Range(b - pad, d + pad), cv::Range(a - pad, c + pad)), m;
        cv::matchTemplate(srch, tpl, m, cv::TM_CCOEFF_NORMED);
        cv::Point loc;
        cv::minMaxLoc(m, nullptr, nullptr, nullptr, &loc);
        out.push_back({loc.x - pad, loc.y - pad});
    }
    return out;
}
inline std::vector<Tracked> track_corners(const cv::Mat& img, const cv::Mat& cen, const std::vector<Corner>& cs, const std::vector<int>& use) {
    auto sh = coarse_shift_full(img, cen);
    std::vector<Tracked> out(use.size());
    for (size_t u = 0; u < use.size(); ++u) {
        const Corner& c = cs[use[u]];
        int k = c.qr - 1, dx = sh[k][0], dy = sh[k][1];
        double sx = c.rx + dx, sy = c.ry + dy;
        int a = BOXES[k][0] + dx, b = BOXES[k][1] + dy, cc = BOXES[k][2] + dx, d = BOXES[k][3] + dy;
        std::vector<double> X, Y, I;
        double ex = sx, ey = sy;
        EdgeFit f1;
        for (int it = 0; it < 3; ++it) {
            window_pixels(img, ex, ey, 4, a, b, cc, d, X, Y, I);
            f1 = edge_fit(X, Y, I, sx, sy, c);
            bool moved = np_rint(f1.p[0]) != np_rint(ex) || np_rint(f1.p[1]) != np_rint(ey);
            ex = f1.p[0]; ey = f1.p[1];
            if (!moved) break;
        }
        window_pixels(img, ex, ey, 5, a, b, cc, d, X, Y, I);
        auto f2 = edge_fit(X, Y, I, sx, sy, c);
        double stab = std::hypot(f1.p[0] - f2.p[0], f1.p[1] - f2.p[1]);
        bool valid = f1.ok && f2.ok && f1.rmse <= 15 && f1.p[5] >= 45 && stab <= .65 &&
                     std::max(std::abs(f1.p[0] - sx), std::abs(f1.p[1] - sy)) <= 3.6 && std::max(f1.p[2], f1.p[3]) <= 2.6;
        out[u] = {f1.p[0], f1.p[1], f1.rmse, stab, f1.p[5], valid};
    }
    return out;
}

// ================================================================ 2. 3차 B-스플라인 (scipy.ndimage 와 같게)
namespace spl {
const double Z = std::sqrt(3.0) - 2.0;
// mode: 0 = mirror, 1 = reflect(nearest 의 패딩 뒤 필터)
inline void filter1d(double* c, int n, int stride, int mode) {
    const double z = Z, lam = (1 - z) * (1 - 1 / z);
    for (int i = 0; i < n; ++i) c[i * stride] *= lam;
    if (mode == 0) {
        double zi = z, zn1 = std::pow(z, n - 1), c0 = c[0] + zn1 * c[(n - 1) * stride];
        for (int i = 1; i < n - 1; ++i) { c0 += zi * (c[i * stride] + zn1 * c[(n - 1 - i) * stride]); zi *= z; }
        c[0] = c0 / (1 - zn1 * zn1);
    } else {
        double zi = z, zn = std::pow(z, n), orig = c[0], c0 = c[0] + zn * c[(n - 1) * stride];
        for (int i = 1; i < n; ++i) { c0 += zi * (c[i * stride] + zn * c[(n - 1 - i) * stride]); zi *= z; }
        c[0] = c0 * z / (1 - zn * zn) + orig;
    }
    for (int i = 1; i < n; ++i) c[i * stride] += z * c[(i - 1) * stride];
    if (mode == 0) c[(n - 1) * stride] = (z * c[(n - 2) * stride] + c[(n - 1) * stride]) * z / (z * z - 1);
    else c[(n - 1) * stride] *= z / (z - 1);
    for (int i = n - 2; i >= 0; --i) c[i * stride] = z * (c[(i + 1) * stride] - c[i * stride]);
}
inline cv::Mat filter2d(const cv::Mat& in, int mode) {
    cv::Mat c = in.clone();
    for (int x = 0; x < c.cols; ++x) filter1d(c.ptr<double>(0) + x, c.rows, (int)(c.step1()), mode);
    for (int y = 0; y < c.rows; ++y) filter1d(c.ptr<double>(y), c.cols, 1, mode);
    return c;
}
struct Coef {
    cv::Mat c;
    int pad = 0;
};
// map_coordinates(..., prefilter=False, mode='nearest') 에 쓸 계수: spline_filter(mode='mirror')
inline Coef coef_mirror(const cv::Mat& img64) { return {filter2d(img64, 0), 0}; }
// map_coordinates(..., prefilter=True, mode='nearest'): 12px 가장자리 복제 뒤 reflect 필터
inline Coef coef_nearest(const cv::Mat& img64) {
    cv::Mat p;
    cv::copyMakeBorder(img64, p, 12, 12, 12, 12, cv::BORDER_REPLICATE);
    return {filter2d(p, 1), 12};
}
inline void w4(double t, int& s, double w[4]) {
    double f = std::floor(t);
    s = (int)f - 1;
    double u = t - f;  // 0..1
    w[0] = (1 - u) * (1 - u) * (1 - u) / 6;
    w[1] = 2.0 / 3 - u * u + u * u * u / 2;
    double v = 1 - u;
    w[2] = 2.0 / 3 - v * v + v * v * v / 2;
    w[3] = u * u * u / 6;
}
inline double at(const Coef& C, double y, double x) {
    y += C.pad; x += C.pad;
    int sy, sx;
    double wy[4], wx[4];
    w4(y, sy, wy);
    w4(x, sx, wx);
    const int R = C.c.rows, Cc = C.c.cols;
    double v = 0;
    for (int a = 0; a < 4; ++a) {
        int yy = std::min(std::max(sy + a, 0), R - 1);
        const double* row = C.c.ptr<double>(yy);
        double t = 0;
        for (int b = 0; b < 4; ++b) {
            int xx = std::min(std::max(sx + b, 0), Cc - 1);
            t += row[xx] * wx[b];
        }
        v += t * wy[a];
    }
    return v;
}
}  // namespace spl

inline cv::Mat np_gradient(const cv::Mat& f, int axis) {
    cv::Mat g(f.size(), CV_64F);
    int R = f.rows, C = f.cols;
    for (int y = 0; y < R; ++y)
        for (int x = 0; x < C; ++x) {
            double v;
            if (axis == 0) v = y == 0 ? f.at<double>(1, x) - f.at<double>(0, x) : (y == R - 1 ? f.at<double>(R - 1, x) - f.at<double>(R - 2, x) : (f.at<double>(y + 1, x) - f.at<double>(y - 1, x)) / 2);
            else v = x == 0 ? f.at<double>(y, 1) - f.at<double>(y, 0) : (x == C - 1 ? f.at<double>(y, C - 1) - f.at<double>(y, C - 2) : (f.at<double>(y, x + 1) - f.at<double>(y, x - 1)) / 2);
            g.at<double>(y, x) = v;
        }
    return g;
}

// ================================================================ 3. GN 정밀 추적 (measurement/refine.py Refiner + robust_gn RobustRefiner)
struct GNOut { std::vector<double> dx, dy; std::vector<std::array<double, 4>> q; std::vector<bool> valid; };

struct Refiner {
    int window = 17;
    double sigma = .8;
    int robust = 0;  // 0 없음, 1 huber, 2 cauchy
    double delta = 6;
    int n = 0, P = 0;
    std::vector<double> px, py;            // 점 (crop 상자 안 좌표)
    std::vector<int> off;                  // 창 오프셋 (P)
    std::vector<double> tmpl;              // n x P
    std::vector<std::array<double, 4>> pinv_rows;  // n x 4 x P 를 [n][k][p] 로
    std::vector<double> pinv;              // n*4*P
    std::vector<double> A;                 // robust: n x P x 4 (scale 나눈 값)
    std::vector<std::array<double, 4>> scale;

    Refiner() = default;
    Refiner(const cv::Mat& ref8, const std::vector<double>& x, const std::vector<double>& y, int win, double sig, int rob = 0, double del = 6)
        : window(win), sigma(sig), robust(rob), delta(del), px(x), py(y) {
        n = (int)x.size();
        int lo = -(window / 2), hi = window / 2;  // 홀수 창: -w//2+1 .. w//2
        std::vector<int> oy, ox;
        for (int a = lo; a <= hi; ++a) for (int b = lo; b <= hi; ++b) { oy.push_back(a); ox.push_back(b); }
        P = (int)oy.size();
        cv::Mat r64, rb;
        ref8.convertTo(r64, CV_64F);
        cv::GaussianBlur(r64, rb, cv::Size(0, 0), sigma);
        cv::Mat gy = np_gradient(rb, 0), gx = np_gradient(rb, 1);
        auto Ct = spl::coef_nearest(rb), Cx = spl::coef_nearest(gx), Cy = spl::coef_nearest(gy);
        tmpl.assign((size_t)n * P, 0);
        pinv.assign((size_t)n * 4 * P, 0);
        if (robust) { A.assign((size_t)n * P * 4, 0); scale.resize(n); }
        oyv = oy; oxv = ox;
        for (int i = 0; i < n; ++i) {
            std::vector<double> D((size_t)P * 4);
            for (int p = 0; p < P; ++p) {
                double cy = py[i] + oy[p], cx = px[i] + ox[p];
                double tv = spl::at(Ct, cy, cx), txv = spl::at(Cx, cy, cx), tyv = spl::at(Cy, cy, cx);
                tmpl[(size_t)i * P + p] = tv;
                D[p * 4 + 0] = txv; D[p * 4 + 1] = tyv; D[p * 4 + 2] = tv; D[p * 4 + 3] = 1;
            }
            // pinv = (D^T D)^-1 D^T
            cv::Mat Dm(P, 4, CV_64F, D.data()), G = Dm.t() * Dm, Gi;
            cv::invert(G, Gi, cv::DECOMP_SVD);
            cv::Mat Pi = Gi * Dm.t();
            for (int k = 0; k < 4; ++k) for (int p = 0; p < P; ++p) pinv[((size_t)i * 4 + k) * P + p] = Pi.at<double>(k, p);
            if (robust) {
                std::array<double, 4> sc;
                for (int k = 0; k < 4; ++k) {
                    double s = 0;
                    for (int p = 0; p < P; ++p) s += D[p * 4 + k] * D[p * 4 + k];
                    sc[k] = std::max(std::sqrt(s / P), 1e-8);
                }
                scale[i] = sc;
                for (int p = 0; p < P; ++p) for (int k = 0; k < 4; ++k) A[((size_t)i * P + p) * 4 + k] = D[p * 4 + k] / sc[k];
            }
        }
    }
    std::vector<int> oyv, oxv;

    // cur8: 대상 상자 crop, (dx0, dy0): LK 초기값(float32)
    GNOut run(const cv::Mat& cur8, const std::vector<float>& ix, const std::vector<float>& iy) const {
        cv::Mat c64, cb;
        cur8.convertTo(c64, CV_64F);
        cv::GaussianBlur(c64, cb, cv::Size(0, 0), sigma);
        auto C = spl::coef_mirror(cb);
        std::vector<double> dx(n), dy(n), vals((size_t)n * P), pars((size_t)n * 4), sx(n), sy(n);
        for (int i = 0; i < n; ++i) { dx[i] = ix[i]; dy[i] = iy[i]; }
        for (int it = 0; it < 12; ++it) {
            double mx = 0;
            for (int i = 0; i < n; ++i) {
                double* v = &vals[(size_t)i * P];
                for (int p = 0; p < P; ++p) v[p] = spl::at(C, py[i] + oyv[p] + dy[i], px[i] + oxv[p] + dx[i]);
                double pr[4];
                for (int k = 0; k < 4; ++k) {
                    const double* w = &pinv[((size_t)i * 4 + k) * P];
                    double s = 0;
                    for (int p = 0; p < P; ++p) s += w[p] * v[p];
                    pr[k] = s;
                }
                if (robust) {
                    const auto& sc = scale[i];
                    for (int rep = 0; rep < 2; ++rep) {
                        double G[16] = {0}, rhs[4] = {0};
                        for (int p = 0; p < P; ++p) {
                            const double* a = &A[((size_t)i * P + p) * 4];
                            double fit = 0;
                            for (int k = 0; k < 4; ++k) fit += a[k] * (pr[k] * sc[k]);
                            double rr = std::abs(v[p] - fit) / delta;
                            double w = robust == 1 ? std::min(1.0, 1 / std::max(rr, 1e-12)) : 1 / (1 + rr * rr);
                            for (int k = 0; k < 4; ++k) {
                                rhs[k] += a[k] * w * v[p];
                                for (int j = 0; j < 4; ++j) G[k * 4 + j] += a[k] * w * a[j];
                            }
                        }
                        for (int k = 0; k < 4; ++k) G[k * 4 + k] += 1e-9;
                        cv::Mat Gm(4, 4, CV_64F, G), R(4, 1, CV_64F, rhs), X;
                        cv::solve(Gm, R, X, cv::DECOMP_LU);
                        for (int k = 0; k < 4; ++k) pr[k] = X.at<double>(k) / sc[k];
                    }
                }
                for (int k = 0; k < 4; ++k) pars[(size_t)i * 4 + k] = pr[k];
                double den = std::max(pr[2], .25);
                double s0 = std::min(std::max(pr[0] / den, -.5), .5), s1 = std::min(std::max(pr[1] / den, -.5), .5);
                dx[i] -= s0; dy[i] -= s1;
                sx[i] = s0; sy[i] = s1;
                mx = std::max(mx, std::hypot(s0, s1));
            }
            if (mx < 2e-5) break;
        }
        GNOut o;
        o.dx = dx; o.dy = dy; o.q.resize(n); o.valid.resize(n);
        for (int i = 0; i < n; ++i) {
            const double* v = &vals[(size_t)i * P];
            const double* t = &tmpl[(size_t)i * P];
            double g = pars[(size_t)i * 4 + 2], b = pars[(size_t)i * 4 + 3], s = 0;
            for (int p = 0; p < P; ++p) { double r = v[p] - g * t[p] - b; s += r * r; }
            double stepn = std::hypot(sx[i], sy[i]);
            o.q[i] = {std::sqrt(s / P), stepn, g, b};
            bool fin = std::isfinite(dx[i]) && std::isfinite(dy[i]);
            o.valid[i] = fin && std::hypot(dx[i] - ix[i], dy[i] - iy[i]) < 1.5 && stepn < .02;
        }
        return o;
    }
};

// ================================================================ 4. 630점 패치 추적 (measurement/spread.py SpreadPairExtractor)
struct Tracks {
    // [img][pt]
    std::vector<std::vector<float>> Dx, Dy;
    std::vector<std::vector<std::array<float, 5>>> Q;
    std::vector<std::vector<char>> V;
};
struct Extractor {
    cv::Mat ref;
    std::vector<double> gx, gy;  // 전체 좌표 (float32 값)
    std::vector<int> region;
    std::vector<std::vector<int>> members;
    std::vector<cv::Mat> lk_templates;
    std::vector<Refiner> models;
    std::vector<std::vector<float>> lpx, lpy;
    int box[3][4];

    Extractor(const cv::Mat& ref_crop, const std::vector<double>& X, const std::vector<double>& Y, int window = 17, double sigma = .8, int robust = 0,
              double delta = 6)
        : ref(ref_crop), gx(X), gy(Y) {
        const double cen[3] = {1644, 1901, 2159};
        region.resize(X.size());
        members.assign(3, {});
        for (size_t i = 0; i < X.size(); ++i) {
            int b = 0;
            for (int k = 1; k < 3; ++k) if (std::abs(X[i] - cen[k]) < std::abs(X[i] - cen[b])) b = k;
            region[i] = b;
            members[b].push_back((int)i);
        }
        for (int k = 0; k < 3; ++k) {
            int x0 = BOXES[k][0] - CROP[0], y0 = BOXES[k][1] - CROP[1], x1 = BOXES[k][2] - CROP[0], y1 = BOXES[k][3] - CROP[1];
            box[k][0] = x0; box[k][1] = y0; box[k][2] = x1; box[k][3] = y1;
            cv::Mat t = ref(cv::Range(y0, y1), cv::Range(x0, x1)).clone(), tb;
            std::vector<float> lx, ly;
            std::vector<double> dxs, dys;
            for (int i : members[k]) {
                // points = (features - CROP) float32, local = points - [x0,y0] float32
                float p0 = (float)(gx[i] - CROP[0]), p1 = (float)(gy[i] - CROP[1]);
                lx.push_back(p0 - (float)x0); ly.push_back(p1 - (float)y0);
                dxs.push_back(lx.back()); dys.push_back(ly.back());
            }
            models.emplace_back(t, dxs, dys, window, sigma, robust, delta);
            cv::GaussianBlur(t, tb, cv::Size(0, 0), .8);
            lk_templates.push_back(tb);
            lpx.push_back(lx); lpy.push_back(ly);
        }
    }

    // measure.py coarse(include_screws=False)
    std::vector<std::array<int, 2>> coarse(const cv::Mat& cur) const {
        std::vector<std::array<int, 2>> out;
        for (int k = 0; k < 3; ++k) {
            int x0 = box[k][0], y0 = box[k][1], x1 = box[k][2], y1 = box[k][3], sx = 110, sy = 22;
            cv::Mat t = ref(cv::Range(y0, y1), cv::Range(x0, x1)), s = cur(cv::Range(y0 - sy, y1 + sy), cv::Range(x0 - sx, x1 + sx)), c;
            cv::matchTemplate(s, t, c, cv::TM_CCOEFF_NORMED);
            cv::Point pos;
            cv::minMaxLoc(c, nullptr, nullptr, nullptr, &pos);
            out.push_back({pos.x - sx, pos.y - sy});
        }
        return out;
    }

    void extract(const cv::Mat& cur, std::vector<float>& Dx, std::vector<float>& Dy, std::vector<std::array<float, 5>>& Q, std::vector<char>& V) const {
        size_t N = gx.size();
        Dx.assign(N, 0); Dy.assign(N, 0); Q.assign(N, {}); V.assign(N, 0);
        auto init = coarse(cur);
        for (int k = 0; k < 3; ++k) {
            int dx = init[k][0], dy = init[k][1];
            cv::Mat target = cur(cv::Range(box[k][1] + dy, box[k][3] + dy), cv::Range(box[k][0] + dx, box[k][2] + dx)).clone(), tb;
            cv::GaussianBlur(target, tb, cv::Size(0, 0), .8);
            const auto& lx = lpx[k];
            const auto& ly = lpy[k];
            size_t n = lx.size();
            std::vector<cv::Point2f> p0(n), p1(n), p2;
            for (size_t i = 0; i < n; ++i) p0[i] = {lx[i], ly[i]};
            p1 = p0;
            std::vector<uchar> s1, s2;
            std::vector<float> e1, e2;
            cv::TermCriteria crit(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 60, 0.0001);
            cv::calcOpticalFlowPyrLK(lk_templates[k], tb, p0, p1, s1, e1, cv::Size(15, 15), 0, crit, cv::OPTFLOW_USE_INITIAL_FLOW, 1e-5);
            p2 = p0;
            cv::calcOpticalFlowPyrLK(tb, lk_templates[k], p1, p2, s2, e2, cv::Size(15, 15), 0, crit, cv::OPTFLOW_USE_INITIAL_FLOW, 1e-5);
            std::vector<float> ddx(n), ddy(n), fb(n);
            std::vector<char> lv(n);
            for (size_t i = 0; i < n; ++i) {
                float ex = p2[i].x - p0[i].x, ey = p2[i].y - p0[i].y;
                fb[i] = std::sqrt(ex * ex + ey * ey);
                ddx[i] = p1[i].x - p0[i].x; ddy[i] = p1[i].y - p0[i].y;
                lv[i] = s1[i] && s2[i] && fb[i] < .5f && std::isfinite(ddx[i]) && std::isfinite(ddy[i]);
            }
            auto g = models[k].run(target, ddx, ddy);
            for (size_t j = 0; j < n; ++j) {
                int i = members[k][j];
                Dx[i] = (float)g.dx[j] + (float)dx;
                Dy[i] = (float)g.dy[j] + (float)dy;
                Q[i] = {(float)g.q[j][0], (float)g.q[j][1], (float)g.q[j][2], (float)g.q[j][3], fb[j]};
                V[i] = lv[j] && g.valid[j];
            }
        }
    }

    Tracks run_all(const std::vector<cv::Mat>& crops) const {
        Tracks t;
        size_t M = crops.size();
        t.Dx.resize(M); t.Dy.resize(M); t.Q.resize(M); t.V.resize(M);
#pragma omp parallel for schedule(dynamic) num_threads(4)
        for (int i = 0; i < (int)M; ++i) extract(crops[i], t.Dx[i], t.Dy[i], t.Q[i], t.V[i]);
        return t;
    }
};

// ================================================================ 5. P12 공간 요약 (center_patch_v2 run.py spatial)
using Feat = std::vector<std::array<double, 12>>;
inline cv::Mat p12_basis(const std::vector<double>& X, const std::vector<double>& Y) {
    cv::Mat B((int)X.size(), 6, CV_64F);
    for (int i = 0; i < (int)X.size(); ++i) {
        double u = (X[i] - 1906) / 486, v = (Y[i] - 1070) / 122;
        double r[6] = {1, u, v, u * u, u * v, v * v};
        for (int k = 0; k < 6; ++k) B.at<double>(i, k) = r[k];
    }
    return B;
}
// D: [img][pt] (중심 뺀 값), w: 없으면 균등
inline Feat spatial(const std::vector<double>& X, const std::vector<double>& Y, const std::vector<std::vector<double>>& Dx,
                    const std::vector<std::vector<double>>& Dy, const std::vector<std::vector<double>>* W = nullptr) {
    cv::Mat B = p12_basis(X, Y);
    Feat F(Dx.size());
    int n = (int)X.size();
    for (size_t i = 0; i < Dx.size(); ++i) {
        cv::Mat Bw = B.clone(), R(n, 2, CV_64F);
        for (int p = 0; p < n; ++p) {
            double sw = W ? std::sqrt((*W)[i][p]) : 1.0;
            if (W) for (int k = 0; k < 6; ++k) Bw.at<double>(p, k) *= sw;
            R.at<double>(p, 0) = Dx[i][p] * sw;
            R.at<double>(p, 1) = Dy[i][p] * sw;
        }
        cv::Mat C;
        cv::solve(Bw, R, C, cv::DECOMP_QR);
        for (int k = 0; k < 6; ++k) { F[i][2 * k] = C.at<double>(k, 0); F[i][2 * k + 1] = C.at<double>(k, 1); }
    }
    return F;
}

// ================================================================ 6. GLS 판정
struct Branch { std::string name; char axis; std::array<double, 12> base{}, s{}, c{}; };
struct Model { std::vector<Branch> br; cv::Mat W; };  // W 12x12

// Z.fit 의 곡선 + (multiref) 공분산 축소. shrink = .001 이면 Z.fit 과 같은 W 방향(분모 상수 차이는 판정에 영향 없음)
inline Model fit(const std::vector<Img>& m, const Feat& X, const std::vector<int>& tr, bool sessions, bool quad, double shrink) {
    Model M;
    std::vector<std::array<double, 12>> R;
    for (char a : {'Y', 'T'}) {
        std::vector<int> ii;
        for (int i : tr) if (m[i].axis == a) ii.push_back(i);
        std::vector<std::string> ss;
        for (int i : ii) ss.push_back(m[i].session);
        std::vector<std::string> uniq = ss;
        std::sort(uniq.begin(), uniq.end());
        uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
        int nb = quad ? 2 : 1, ncol = nb + (sessions ? (int)uniq.size() : 0);
        cv::Mat A((int)ii.size(), ncol, CV_64F, cv::Scalar(0)), Zm((int)ii.size(), 12, CV_64F);
        for (int r = 0; r < (int)ii.size(); ++r) {
            double v = m[ii[r]].value;
            A.at<double>(r, 0) = v;
            if (quad) A.at<double>(r, 1) = v * v;
            if (sessions) for (size_t u = 0; u < uniq.size(); ++u) A.at<double>(r, nb + (int)u) = ss[r] == uniq[u];
            for (int k = 0; k < 12; ++k) Zm.at<double>(r, k) = X[ii[r]][k];
        }
        cv::Mat co;
        cv::solve(A, Zm, co, cv::DECOMP_SVD);
        cv::Mat res = Zm - A * co;
        for (int r = 0; r < res.rows; ++r) { std::array<double, 12> t; for (int k = 0; k < 12; ++k) t[k] = res.at<double>(r, k); R.push_back(t); }
        Branch proto;
        proto.axis = a;
        for (int k = 0; k < 12; ++k) { proto.s[k] = co.at<double>(0, k); proto.c[k] = quad ? co.at<double>(1, k) : 0; }
        if (sessions) {
            for (size_t u = 0; u < uniq.size(); ++u) {
                Branch b = proto;
                b.name = uniq[u];
                for (int k = 0; k < 12; ++k) b.base[k] = co.at<double>(nb + (int)u, k);
                M.br.push_back(b);
            }
        } else {
            proto.name = std::string(1, a);
            M.br.push_back(proto);
        }
    }
    cv::Mat S(12, 12, CV_64F, cv::Scalar(0));
    for (auto& r : R) for (int a = 0; a < 12; ++a) for (int b = 0; b < 12; ++b) S.at<double>(a, b) += r[a] * r[b];
    S /= std::max((double)R.size() - 16, 1.0);
    double sd[12];
    for (int k = 0; k < 12; ++k) sd[k] = std::sqrt(std::max(S.at<double>(k, k), 1e-14));
    cv::Mat C(12, 12, CV_64F);
    for (int a = 0; a < 12; ++a) for (int b = 0; b < 12; ++b) C.at<double>(a, b) = (1 - shrink) * S.at<double>(a, b) / sd[a] / sd[b] + (a == b ? shrink : 0);
    cv::Mat Ci = C.inv(cv::DECOMP_SVD);
    Ci = (Ci + Ci.t()) * 0.5;
    // cholesky lower L, W = L^T / sd
    cv::Mat L(12, 12, CV_64F, cv::Scalar(0));
    for (int i = 0; i < 12; ++i)
        for (int j = 0; j <= i; ++j) {
            double s = Ci.at<double>(i, j);
            for (int k = 0; k < j; ++k) s -= L.at<double>(i, k) * L.at<double>(j, k);
            if (i == j) L.at<double>(i, i) = std::sqrt(s);
            else L.at<double>(i, j) = s / L.at<double>(j, j);
        }
    M.W = cv::Mat(12, 12, CV_64F);
    for (int a = 0; a < 12; ++a) for (int b = 0; b < 12; ++b) M.W.at<double>(a, b) = L.at<double>(b, a) / sd[b];
    return M;
}

struct Pred { int idx; char pred_axis; double pred; std::string branch; double cost; };
// Z.predict. mode sessions: 가지 부호 제약
inline std::vector<Pred> predict(const Model& M, const Feat& X, const std::vector<int>& te, bool sessions) {
    std::vector<Pred> out;
    auto mul = [&](const std::array<double, 12>& x, double* o) {
        for (int a = 0; a < 12; ++a) { double s = 0; for (int b = 0; b < 12; ++b) s += M.W.at<double>(a, b) * x[b]; o[a] = s; }
    };
    std::vector<std::array<double, 12>> SW(M.br.size()), CW(M.br.size());
    for (size_t b = 0; b < M.br.size(); ++b) { mul(M.br[b].s, SW[b].data()); mul(M.br[b].c, CW[b].data()); }
    for (int i : te) {
        int best = -1;
        double bc = std::numeric_limits<double>::infinity(), bv = 0;
        for (size_t b = 0; b < M.br.size(); ++b) {
            std::array<double, 12> d;
            for (int k = 0; k < 12; ++k) d[k] = X[i][k] - M.br[b].base[k];
            double z[12];
            mul(d, z);
            const auto& sw = SW[b];
            const auto& cw = CW[b];
            double num = 0, den = 0;
            for (int k = 0; k < 12; ++k) { num += z[k] * sw[k]; den += sw[k] * sw[k]; }
            double v = num / den;
            for (int it = 0; it < 8; ++it) {
                double a1 = 0, a2 = 0;
                for (int k = 0; k < 12; ++k) {
                    double J = sw[k] + 2 * v * cw[k], r = z[k] - v * sw[k] - v * v * cw[k];
                    a1 += J * r; a2 += J * J;
                }
                v += a1 / std::max(a2, 1e-20);
            }
            double cost = 0;
            for (int k = 0; k < 12; ++k) { double r = z[k] - v * sw[k] - v * v * cw[k]; cost += r * r; }
            if (sessions) {
                double sg = M.br[b].name.back() == '+' ? 1 : -1;
                if (!(v * sg >= 0)) cost = std::numeric_limits<double>::infinity();
            }
            if (best < 0 || cost < bc) { best = (int)b; bc = cost; bv = v; }
        }
        out.push_back({i, M.br[best].axis, bv, M.br[best].name, bc});
    }
    return out;
}

// enrich / stats (multiref experiment.py)
struct Row { int idx; char axis; double value; char pred_axis; double pred; bool correct; double eY, eT; bool pass; std::string branch; double cost; };
inline double sgn(double v) { return (v > 0) - (v < 0); }
inline std::vector<Row> enrich(const std::vector<Img>& m, const std::vector<Pred>& ps) {
    std::vector<Row> o;
    for (auto& p : ps) {
        const Img& g = m[p.idx];
        Row r{p.idx, g.axis, g.value, p.pred_axis, p.pred, false, 0, 0, false, p.branch, p.cost};
        r.correct = g.axis == p.pred_axis && sgn(g.value) == sgn(p.pred);
        r.eY = std::abs((g.axis == 'Y' ? g.value : 0) - (p.pred_axis == 'Y' ? p.pred : 0));
        r.eT = std::abs((g.axis == 'T' ? g.value : 0) - (p.pred_axis == 'T' ? p.pred : 0));
        r.pass = r.correct && r.eY <= .09 && r.eT <= .09;
        o.push_back(r);
    }
    return o;
}
struct Stats { int n = 0, Yc = 0, Yn = 0, Tc = 0, Tn = 0, fail = 0; double Ymax = 0, Tmax = 0; };
inline Stats stats(const std::vector<Row>& rs) {
    Stats s;
    for (auto& r : rs) {
        ++s.n;
        if (r.axis == 'Y') { ++s.Yn; s.Yc += r.correct; }
        if (r.axis == 'T') { ++s.Tn; s.Tc += r.correct; }
        s.fail += !r.pass;
        s.Ymax = std::max(s.Ymax, r.eY);
        s.Tmax = std::max(s.Tmax, r.eT);
    }
    return s;
}

// 튀는 값: Y 사진 중 max(eY,eT) 가 가장 큰 것, T 사진 중 가장 큰 것
inline void pick_outliers(const std::vector<Row>& rs, size_t& wy, size_t& wt) {
    double by = -1, bt = -1;
    wy = wt = 0;
    for (size_t i = 0; i < rs.size(); ++i) {
        double e = std::max(rs[i].eY, rs[i].eT);
        if (rs[i].axis == 'Y' && e > by) { by = e; wy = i; }
        if (rs[i].axis == 'T' && e > bt) { bt = e; wt = i; }
    }
}

// ================================================================ 7. StratifiedKFold(shuffle=True, random_state) = numpy RandomState(MT19937)
inline uint32_t random_interval(std::mt19937& g, uint32_t mx) {
    if (mx == 0) return 0;
    uint32_t mask = mx;
    mask |= mask >> 1; mask |= mask >> 2; mask |= mask >> 4; mask |= mask >> 8; mask |= mask >> 16;
    uint32_t v;
    while ((v = (g() & mask)) > mx) {}
    return v;
}
// y: 클래스 문자열(ids 순서). 반환: fold 별 (train 위치, test 위치) — 위치는 y 의 인덱스
inline std::vector<std::pair<std::vector<int>, std::vector<int>>> stratified_kfold(const std::vector<std::string>& y, int k, uint32_t seed) {
    std::mt19937 g(seed);
    int N = (int)y.size();
    std::vector<std::string> uniq = y;
    std::sort(uniq.begin(), uniq.end());
    uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
    // 처음 나온 순서로 클래스 번호
    std::vector<int> first(uniq.size(), N);
    for (int i = 0; i < N; ++i) {
        int u = (int)(std::lower_bound(uniq.begin(), uniq.end(), y[i]) - uniq.begin());
        first[u] = std::min(first[u], i);
    }
    std::vector<int> order(uniq.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return first[a] < first[b]; });
    std::vector<int> perm(uniq.size());
    for (size_t r = 0; r < order.size(); ++r) perm[order[r]] = (int)r;
    std::vector<int> enc(N);
    for (int i = 0; i < N; ++i) enc[i] = perm[std::lower_bound(uniq.begin(), uniq.end(), y[i]) - uniq.begin()];
    int nc = (int)uniq.size();
    std::vector<int> ys = enc;
    std::sort(ys.begin(), ys.end());
    std::vector<std::vector<int>> alloc(k, std::vector<int>(nc, 0));
    for (int i = 0; i < k; ++i) for (int j = i; j < N; j += k) alloc[i][ys[j]]++;
    std::vector<int> test_fold(N);
    for (int c = 0; c < nc; ++c) {
        std::vector<int> f;
        for (int i = 0; i < k; ++i) for (int r = 0; r < alloc[i][c]; ++r) f.push_back(i);
        for (int i = (int)f.size() - 1; i >= 1; --i) { uint32_t j = random_interval(g, (uint32_t)i); std::swap(f[i], f[j]); }
        int t = 0;
        for (int i = 0; i < N; ++i) if (enc[i] == c) test_fold[i] = f[t++];
    }
    std::vector<std::pair<std::vector<int>, std::vector<int>>> out(k);
    for (int i = 0; i < N; ++i) for (int f = 0; f < k; ++f) (test_fold[i] == f ? out[f].second : out[f].first).push_back(i);
    return out;
}

}  // namespace yt
