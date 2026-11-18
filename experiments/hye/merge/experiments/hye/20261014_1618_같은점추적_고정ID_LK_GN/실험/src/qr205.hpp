// QR 고정 ID 점 측정 (원본: codex_experiments_20261014/measurement/measure.py, refine.py, spread.py)
//  - 기준 center 의 판 영역(CROP)만 읽는다. QR 상자마다 Shi-Tomasi 점 + cornerSubPix
//  - 사진마다: QR 전체 NCC 로 정수 이동 -> QR 상자를 잘라 LK(15) -> GN17 (흐림 0.8, 3차 B-spline, 밝기 gain/offset)
//  - GN 보간은 scipy ndimage 와 같은 식: 템플릿은 map_coordinates(prefilter, mode nearest), 현재 사진은 spline_filter(mirror)
#pragma once
#include <hye/dataset.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <array>
#include <cmath>
#include <vector>

namespace qr {

const int CROP[4] = {1340, 800, 2470, 1330};
const int BOXES[3][4] = {{1552, 969, 1736, 1167}, {1809, 971, 1993, 1169}, {2067, 973, 2252, 1171}};

inline cv::Mat read_crop(const hye::Shot& s) {
    cv::Mat g = hye::read_gray(s);
    return g(cv::Range(CROP[1], CROP[3]), cv::Range(CROP[0], CROP[2])).clone();
}

// ---- scipy 3차 B-spline ----
enum class Bd { Mirror, Reflect };  // scipy mode 'mirror' / 'nearest'(=reflect 식)
inline void spline1d(double* c, int n, int stride, Bd bd) {
    const double z = std::sqrt(3.0) - 2.0, lam = (1 - z) * (1 - 1 / z);
    auto C = [&](int i) -> double& { return c[(size_t)i * stride]; };
    for (int i = 0; i < n; ++i) C(i) *= lam;
    if (bd == Bd::Mirror) {
        double zi = z, zn1 = std::pow(z, n - 1);
        C(0) = C(0) + zn1 * C(n - 1);
        for (int i = 1; i < n - 1; ++i) { C(0) += zi * (C(i) + zn1 * C(n - 1 - i)); zi *= z; }
        C(0) /= 1 - zn1 * zn1;
    } else {
        double zi = z, zn = std::pow(z, n), c0 = C(0);
        C(0) = C(0) + zn * C(n - 1);
        for (int i = 1; i < n; ++i) { C(0) += zi * (C(i) + zn * C(n - 1 - i)); zi *= z; }
        C(0) *= z / (1 - zn * zn);
        C(0) += c0;
    }
    for (int i = 1; i < n; ++i) C(i) += z * C(i - 1);
    if (bd == Bd::Mirror) C(n - 1) = (z * C(n - 2) + C(n - 1)) * z / (z * z - 1);
    else C(n - 1) *= z / (z - 1);
    for (int i = n - 2; i >= 0; --i) C(i) = z * (C(i + 1) - C(i));
}
inline cv::Mat spline_filter(const cv::Mat& w /*f64*/, Bd bd) {
    cv::Mat c = w.clone();
    for (int x = 0; x < c.cols; ++x) spline1d(c.ptr<double>(0) + x, c.rows, (int)c.step1(), bd);
    for (int y = 0; y < c.rows; ++y) spline1d(c.ptr<double>(y), c.cols, 1, bd);
    return c;
}
struct BW { int i0; double w[4]; };
inline BW bweights(double x) {
    BW b;
    double f = std::floor(x), t = x - f;
    b.i0 = (int)f - 1;
    b.w[0] = (1 - t) * (1 - t) * (1 - t) / 6;
    b.w[1] = (4 - 6 * t * t + 3 * t * t * t) / 6;
    b.w[2] = (1 + 3 * t + 3 * t * t - 3 * t * t * t) / 6;
    b.w[3] = t * t * t / 6;
    return b;
}
inline double sample(const cv::Mat& c, const BW& y, const BW& x) {
    double s = 0;
    for (int a = 0; a < 4; ++a) {
        const double* r = c.ptr<double>(std::min(std::max(y.i0 + a, 0), c.rows - 1));
        double t = 0;
        for (int b = 0; b < 4; ++b) t += x.w[b] * r[std::min(std::max(x.i0 + b, 0), c.cols - 1)];
        s += y.w[a] * t;
    }
    return s;
}
inline cv::Mat np_gradient(const cv::Mat& f, int axis) {  // np.gradient (가운데 차분, 가장자리 한쪽)
    cv::Mat g(f.size(), CV_64F);
    for (int y = 0; y < f.rows; ++y)
        for (int x = 0; x < f.cols; ++x) {
            if (axis == 1) {
                int n = f.cols;
                g.at<double>(y, x) = x == 0 ? f.at<double>(y, 1) - f.at<double>(y, 0)
                                   : x == n - 1 ? f.at<double>(y, n - 1) - f.at<double>(y, n - 2)
                                                : (f.at<double>(y, x + 1) - f.at<double>(y, x - 1)) / 2;
            } else {
                int n = f.rows;
                g.at<double>(y, x) = y == 0 ? f.at<double>(1, x) - f.at<double>(0, x)
                                   : y == n - 1 ? f.at<double>(n - 1, x) - f.at<double>(n - 2, x)
                                                : (f.at<double>(y + 1, x) - f.at<double>(y - 1, x)) / 2;
            }
        }
    return g;
}
inline cv::Mat blur64(const cv::Mat& u8, double s) { cv::Mat f, b; u8.convertTo(f, CV_64F); cv::GaussianBlur(f, b, cv::Size(0, 0), s); return b; }
inline cv::Mat blur8(const cv::Mat& u8, double s) { cv::Mat b; cv::GaussianBlur(u8, b, cv::Size(0, 0), s); return b; }

// 역합성 GN: 점마다 (이동, gain, offset). 템플릿 기울기 고정, pinv 미리 계산
struct Refiner {
    int window = 17;
    double sigma = 0.8;
    std::vector<cv::Point2f> pts;
    std::vector<cv::Mat> tmpl, pinv;  // 점마다 (window², 1), (4, window²)
    Refiner() = default;
    Refiner(const cv::Mat& ref_u8, const std::vector<cv::Point2f>& p, int win = 17, double sg = 0.8) : window(win), sigma(sg), pts(p) {
        cv::Mat ref = blur64(ref_u8, sigma);
        cv::Mat cR = spline_filter(ref, Bd::Reflect), gy = np_gradient(ref, 0), gx = np_gradient(ref, 1);
        cv::Mat cX = spline_filter(gx, Bd::Reflect), cY = spline_filter(gy, Bd::Reflect);
        const int h = window / 2, n = window * window;
        for (auto& q : pts) {
            cv::Mat T(n, 1, CV_64F), Dm(n, 4, CV_64F);
            int k = 0;
            for (int yy = -h; yy <= h; ++yy) {
                BW wy = bweights((double)q.y + yy);
                for (int xx = -h; xx <= h; ++xx, ++k) {
                    BW wx = bweights((double)q.x + xx);
                    double t = sample(cR, wy, wx);
                    T.at<double>(k) = t;
                    Dm.at<double>(k, 0) = sample(cX, wy, wx);
                    Dm.at<double>(k, 1) = sample(cY, wy, wx);
                    Dm.at<double>(k, 2) = t;
                    Dm.at<double>(k, 3) = 1;
                }
            }
            cv::Mat P;
            cv::invert(Dm, P, cv::DECOMP_SVD);
            tmpl.push_back(T);
            pinv.push_back(P);
        }
    }
    // 반환: 이동(float32 로 반올림), 마지막 step, 유효
    void run(const cv::Mat& cur_u8, const std::vector<cv::Point2d>& init, std::vector<cv::Point2d>& d, std::vector<double>& last, std::vector<bool>& valid,
             int* iters = nullptr) const {
        cv::Mat coeff = spline_filter(blur64(cur_u8, sigma), Bd::Mirror);
        const int h = window / 2, n = window * window, N = (int)pts.size();
        d = init;
        std::vector<cv::Point2d> step(N);
        cv::Mat vals(n, 1, CV_64F);
        int it = 0;
        for (; it < 12; ++it) {
            double mx = 0;
            for (int p = 0; p < N; ++p) {
                int k = 0;
                for (int yy = -h; yy <= h; ++yy) {
                    BW wy = bweights((double)pts[p].y + yy + d[p].y);
                    for (int xx = -h; xx <= h; ++xx, ++k) vals.at<double>(k) = sample(coeff, wy, bweights((double)pts[p].x + xx + d[p].x));
                }
                cv::Mat pr = pinv[p] * vals;
                double g = std::max(pr.at<double>(2), 0.25);
                double sx = std::min(std::max(pr.at<double>(0) / g, -0.5), 0.5), sy = std::min(std::max(pr.at<double>(1) / g, -0.5), 0.5);
                step[p] = {sx, sy};
                mx = std::max(mx, std::hypot(sx, sy));
            }
            for (int p = 0; p < N; ++p) d[p] -= step[p];
            if (mx < 2e-5) { ++it; break; }
        }
        if (iters) *iters = it;
        last.assign(N, 0);
        valid.assign(N, false);
        for (int p = 0; p < N; ++p) {
            last[p] = std::hypot(step[p].x, step[p].y);
            valid[p] = std::isfinite(d[p].x) && std::isfinite(d[p].y) && std::hypot(d[p].x - init[p].x, d[p].y - init[p].y) < 1.5 && last[p] < 0.02;
            d[p] = {(double)(float)d[p].x, (double)(float)d[p].y};
        }
    }
};

// LK 왕복 (measure.track). 반환 이동, 유효, FB 오차
inline void lk_track(const cv::Mat& ref, const cv::Mat& cur, const std::vector<cv::Point2f>& p, const std::vector<cv::Point2f>& init, int window,
                     std::vector<cv::Point2d>& d, std::vector<bool>& valid, std::vector<double>* fb_out = nullptr, std::vector<float>* err_out = nullptr) {
    cv::TermCriteria cr(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 60, 0.0001);
    std::vector<cv::Point2f> p1(p.size()), p2 = p;
    for (size_t i = 0; i < p.size(); ++i) p1[i] = p[i] + init[i];
    std::vector<uchar> s1, s2;
    std::vector<float> e1, e2;
    cv::calcOpticalFlowPyrLK(ref, cur, p, p1, s1, e1, cv::Size(window, window), 0, cr, cv::OPTFLOW_USE_INITIAL_FLOW, 1e-5);
    cv::calcOpticalFlowPyrLK(cur, ref, p1, p2, s2, e2, cv::Size(window, window), 0, cr, cv::OPTFLOW_USE_INITIAL_FLOW, 1e-5);
    d.resize(p.size());
    valid.resize(p.size());
    if (fb_out) fb_out->resize(p.size());
    for (size_t i = 0; i < p.size(); ++i) {
        double fb = std::hypot((double)p2[i].x - p[i].x, (double)p2[i].y - p[i].y);
        d[i] = {(double)(p1[i].x - p[i].x), (double)(p1[i].y - p[i].y)};
        valid[i] = s1[i] && s2[i] && fb < 0.5 && std::isfinite(d[i].x) && std::isfinite(d[i].y);
        if (fb_out) (*fb_out)[i] = fb;
    }
    if (err_out) *err_out = e1;
}

// QR 전체 NCC 정수 이동 (가로 ±110, 세로 ±22)
inline std::array<cv::Point, 3> coarse(const cv::Mat& ref, const cv::Mat& cur, double* score = nullptr) {
    std::array<cv::Point, 3> out;
    for (int k = 0; k < 3; ++k) {
        int x0 = BOXES[k][0] - CROP[0], y0 = BOXES[k][1] - CROP[1], x1 = BOXES[k][2] - CROP[0], y1 = BOXES[k][3] - CROP[1];
        cv::Mat r;
        cv::matchTemplate(cur(cv::Range(y0 - 22, y1 + 22), cv::Range(x0 - 110, x1 + 110)), ref(cv::Range(y0, y1), cv::Range(x0, x1)), r, cv::TM_CCOEFF_NORMED);
        double v;
        cv::Point pos;
        cv::minMaxLoc(r, nullptr, &v, nullptr, &pos);
        out[k] = {pos.x - 110, pos.y - 22};
        if (score) score[k] = v;
    }
    return out;
}

// 넓게 퍼진 QR 점 (spread.select). 반환 crop 좌표
inline void select_spread(const cv::Mat& ref, int min_distance, std::vector<cv::Point2f>& pts, std::vector<int>& regions) {
    cv::Mat sm = blur8(ref, 0.65);
    for (int k = 0; k < 3; ++k) {
        const int* b = BOXES[k];
        cv::Mat mask = cv::Mat::zeros(ref.size(), CV_8U);
        mask(cv::Range(b[1] + 16 - CROP[1], b[3] - 16 - CROP[1]), cv::Range(b[0] + 16 - CROP[0], b[2] - 16 - CROP[0])) = 255;
        std::vector<cv::Point2f> p;
        cv::goodFeaturesToTrack(sm, p, 600, 0.02, min_distance, mask, 5);
        cv::cornerSubPix(sm, p, cv::Size(2, 2), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 30, 0.001));
        for (auto& q : p) {
            double gx = (double)q.x + CROP[0], gy = (double)q.y + CROP[1];
            if (!(b[0] + 15 <= gx && gx < b[2] - 15 && b[1] + 15 <= gy && gy < b[3] - 15)) continue;
            pts.push_back(q);
            regions.push_back(k);
        }
    }
}

// SpreadPairExtractor: QR 상자마다 잘라서 LK15 -> GN17. D 는 crop 이동(px), float32 반올림
struct Spread {
    cv::Mat ref;
    std::vector<cv::Point2f> points;  // crop 좌표
    std::vector<int> regions;
    Refiner models[3];
    cv::Mat tblur[3];
    std::vector<cv::Point2f> local[3];
    explicit Spread(const cv::Mat& ref_crop, int min_distance = 14) : ref(ref_crop) {
        select_spread(ref, min_distance, points, regions);
        for (int k = 0; k < 3; ++k) {
            int x0 = BOXES[k][0] - CROP[0], y0 = BOXES[k][1] - CROP[1], x1 = BOXES[k][2] - CROP[0], y1 = BOXES[k][3] - CROP[1];
            cv::Mat t = ref(cv::Range(y0, y1), cv::Range(x0, x1)).clone();
            for (size_t i = 0; i < points.size(); ++i)
                if (regions[i] == k) local[k].push_back(points[i] - cv::Point2f((float)x0, (float)y0));
            models[k] = Refiner(t, local[k], 17, 0.8);
            tblur[k] = blur8(t, 0.8);
        }
    }
    std::vector<cv::Point2d> global_points() const {
        std::vector<cv::Point2d> g;
        for (auto& p : points) g.push_back({(double)p.x + CROP[0], (double)p.y + CROP[1]});
        return g;
    }
    struct Out { std::vector<cv::Point2d> D; std::vector<bool> valid; std::array<cv::Point, 3> init; };
    Out extract(const cv::Mat& cur) const {
        Out o;
        o.init = coarse(ref, cur);
        o.D.assign(points.size(), {0, 0});
        o.valid.assign(points.size(), false);
        for (int k = 0; k < 3; ++k) {
            int x0 = BOXES[k][0] - CROP[0], y0 = BOXES[k][1] - CROP[1], x1 = BOXES[k][2] - CROP[0], y1 = BOXES[k][3] - CROP[1];
            int dx = o.init[k].x, dy = o.init[k].y;
            cv::Mat target = cur(cv::Range(y0 + dy, y1 + dy), cv::Range(x0 + dx, x1 + dx)).clone();
            std::vector<cv::Point2d> d, gd;
            std::vector<bool> lv, gv;
            std::vector<double> last;
            lk_track(tblur[k], blur8(target, 0.8), local[k], std::vector<cv::Point2f>(local[k].size(), {0, 0}), 15, d, lv);
            models[k].run(target, d, gd, last, gv);
            size_t j = 0;
            for (size_t i = 0; i < points.size(); ++i)
                if (regions[i] == k) {
                    o.D[i] = {(double)(float)(gd[j].x + dx), (double)(float)(gd[j].y + dy)};
                    o.valid[i] = lv[j] && gv[j];
                    ++j;
                }
        }
        return o;
    }
};

// 판 2차 변형 12값 (zero_experiment.load): 기저 [1, x, y, x², xy, y²], 순서 b0x, b0y, ...
struct P12 {
    cv::Mat B, pinvB;
    explicit P12(const std::vector<cv::Point2d>& g) {
        B.create((int)g.size(), 6, CV_64F);
        for (int i = 0; i < B.rows; ++i) {
            double x = (g[i].x - 1906) / 486, y = (g[i].y - 1070) / 122;
            double r[6] = {1, x, y, x * x, x * y, y * y};
            for (int j = 0; j < 6; ++j) B.at<double>(i, j) = r[j];
        }
        cv::invert(B, pinvB, cv::DECOMP_SVD);
    }
    std::vector<double> operator()(const std::vector<cv::Point2d>& D) const {
        cv::Mat d((int)D.size(), 2, CV_64F);
        for (int i = 0; i < d.rows; ++i) { d.at<double>(i, 0) = D[i].x; d.at<double>(i, 1) = D[i].y; }
        cv::Mat co = pinvB * d;
        std::vector<double> f(12);
        for (int j = 0; j < 6; ++j) { f[2 * j] = co.at<double>(j, 0); f[2 * j + 1] = co.at<double>(j, 1); }
        return f;
    }
    // 12값 -> 점 이동장
    std::vector<cv::Point2d> field(const std::vector<double>& f) const {
        std::vector<cv::Point2d> o(B.rows);
        for (int i = 0; i < B.rows; ++i) {
            double x = 0, y = 0;
            for (int j = 0; j < 6; ++j) { x += B.at<double>(i, j) * f[2 * j]; y += B.at<double>(i, j) * f[2 * j + 1]; }
            o[i] = {x, y};
        }
        return o;
    }
};

}  // namespace qr
