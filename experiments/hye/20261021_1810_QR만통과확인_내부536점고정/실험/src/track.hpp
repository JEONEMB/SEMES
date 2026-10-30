// QR 패치 추적: 전체 QR NCC → LK15 → GN (measurement/spread.py SpreadPairExtractor, refine.py Refiner,
// forward_gn_20261021/measure_forward.py ForwardRefiner). QR 상자 하나씩 잘라서 그 안의 픽셀만 쓴다.
#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <cmath>
#include <memory>
#include <vector>
#include "bspline.hpp"
#include "corners.hpp"

namespace g09 {

// 의사역행렬 (P x k 설계 → k x P), numpy pinv 처럼 SVD 로
inline cv::Mat pinv(const cv::Mat& A) {
    cv::Mat out;
    cv::invert(A, out, cv::DECOMP_SVD);
    return out;
}

struct GnOut {
    std::vector<cv::Point2d> d;
    std::vector<std::array<double, 4>> q;  // 광도 RMSE, 마지막 걸음, 이득, 오프셋
    std::vector<bool> valid;
};

// 공통: 창 좌표 (점마다 window x window)
struct WindowCoords {
    int window = 17, P = 0;
    std::vector<double> cy, cx;  // n*P
    void build(const std::vector<cv::Point2f>& pts, int w) {
        window = w;
        const int h = w / 2;  // -w//2+1 .. w//2
        const int lo = -(w / 2 + 1) + 1;  // python -w//2+1 (floor)
        P = w * w;
        cy.resize(pts.size() * P);
        cx.resize(pts.size() * P);
        for (size_t i = 0; i < pts.size(); ++i) {
            int k = 0;
            for (int yy = lo; yy <= h; ++yy)
                for (int xx = lo; xx <= h; ++xx, ++k) {
                    cy[i * P + k] = (double)pts[i].y + yy;
                    cx[i * P + k] = (double)pts[i].x + xx;
                }
        }
    }
};

inline cv::Mat blur64(const cv::Mat& u8, double sigma) {
    cv::Mat f, b;
    u8.convertTo(f, CV_64F);
    cv::GaussianBlur(f, b, cv::Size(0, 0), sigma);
    return b;
}

// refine.py Refiner: 고정 템플릿 기울기의 역방향 GN, 이득·오프셋 해석 소거
struct Refiner {
    std::vector<cv::Point2f> pts;
    WindowCoords wc;
    double sigma = .8;
    std::vector<double> T;          // n*P 템플릿
    std::vector<cv::Mat> pinverse;  // 점마다 4 x P
    Refiner() = default;
    Refiner(const cv::Mat& ref_u8, const std::vector<cv::Point2f>& points, int window, double s = .8) { init(ref_u8, points, window, s); }
    void init(const cv::Mat& ref_u8, const std::vector<cv::Point2f>& points, int window, double s) {
        pts = points;
        sigma = s;
        wc.build(points, window);
        cv::Mat ref = blur64(ref_u8, s);
        Spline sr = Spline::prefiltered_nearest(ref);
        cv::Mat gy, gx;
        np_gradient(ref, gy, gx);
        Spline sx = Spline::prefiltered_nearest(gx), sy = Spline::prefiltered_nearest(gy);
        const int P = wc.P;
        T.resize(points.size() * P);
        pinverse.resize(points.size());
        for (size_t i = 0; i < points.size(); ++i) {
            cv::Mat D(P, 4, CV_64F);
            for (int k = 0; k < P; ++k) {
                double y = wc.cy[i * P + k], x = wc.cx[i * P + k];
                T[i * P + k] = sr.sample(y, x);
                D.at<double>(k, 0) = sx.sample(y, x);
                D.at<double>(k, 1) = sy.sample(y, x);
                D.at<double>(k, 2) = T[i * P + k];
                D.at<double>(k, 3) = 1.0;
            }
            pinverse[i] = pinv(D);
        }
    }
    GnOut run(const cv::Mat& cur_u8, const std::vector<cv::Point2d>& initial) const {
        cv::Mat cur = blur64(cur_u8, sigma);
        Spline sc = Spline::from_coeff(spline_filter(cur));
        const size_t n = pts.size();
        const int P = wc.P;
        std::vector<cv::Point2d> d(initial);
        std::vector<double> values(n * P);
        std::vector<std::array<double, 4>> pars(n);
        std::vector<cv::Point2d> step(n);
        for (int it = 0; it < 12; ++it) {
            double mx = 0;
            for (size_t i = 0; i < n; ++i) {
                const double* pv = pinverse[i].ptr<double>();
                double a[4] = {0, 0, 0, 0};
                for (int k = 0; k < P; ++k) {
                    double v = sc.sample(wc.cy[i * P + k] + d[i].y, wc.cx[i * P + k] + d[i].x);
                    values[i * P + k] = v;
                }
                for (int r = 0; r < 4; ++r) {
                    double s = 0;
                    for (int k = 0; k < P; ++k) s += pv[r * P + k] * values[i * P + k];
                    a[r] = s;
                }
                pars[i] = {a[0], a[1], a[2], a[3]};
                double den = std::max(a[2], .25);
                double sx = std::min(std::max(a[0] / den, -.5), .5), sy = std::min(std::max(a[1] / den, -.5), .5);
                step[i] = {sx, sy};
                d[i].x -= sx;
                d[i].y -= sy;
                mx = std::max(mx, std::hypot(sx, sy));
            }
            if (mx < 2e-5) break;
        }
        GnOut o;
        o.d = d;
        o.q.resize(n);
        o.valid.resize(n);
        for (size_t i = 0; i < n; ++i) {
            double ss = 0;
            for (int k = 0; k < P; ++k) {
                double r = values[i * P + k] - pars[i][2] * T[i * P + k] - pars[i][3];
                ss += r * r;
            }
            double sn = std::hypot(step[i].x, step[i].y);
            o.q[i] = {std::sqrt(ss / P), sn, pars[i][2], pars[i][3]};
            o.valid[i] = std::isfinite(d[i].x) && std::isfinite(d[i].y) && std::hypot(d[i].x - initial[i].x, d[i].y - initial[i].y) < 1.5 && sn < .02;
        }
        return o;
    }
};

// measure_forward.py ForwardRefiner: 현재 영상 spline 미분(±0.01px 중앙차분) 순방향 GN, symmetric 이면 기준 기울기와 평균
struct ForwardRefiner {
    std::vector<cv::Point2f> pts;
    WindowCoords wc;
    bool symmetric = false;
    std::vector<double> T;           // n*P
    std::vector<cv::Mat> photinv;    // 2 x P
    std::vector<double> g0x, g0y;    // 기준 영상 기울기
    void init(const cv::Mat& ref_u8, const std::vector<cv::Point2f>& points, int window, bool sym) {
        pts = points;
        symmetric = sym;
        wc.build(points, window);
        cv::Mat ref = blur64(ref_u8, .8);
        Spline sr = Spline::prefiltered_nearest(ref);
        const int P = wc.P;
        const size_t n = points.size();
        T.resize(n * P);
        photinv.resize(n);
        for (size_t i = 0; i < n; ++i) {
            cv::Mat D(P, 2, CV_64F);
            for (int k = 0; k < P; ++k) {
                T[i * P + k] = sr.sample(wc.cy[i * P + k], wc.cx[i * P + k]);
                D.at<double>(k, 0) = T[i * P + k];
                D.at<double>(k, 1) = 1.0;
            }
            photinv[i] = pinv(D);
        }
        Spline c = Spline::from_coeff(spline_filter(ref));
        g0x.resize(n * P);
        g0y.resize(n * P);
        for (size_t i = 0; i < n * P; ++i) {
            g0x[i] = (c.sample(wc.cy[i], wc.cx[i] + .01) - c.sample(wc.cy[i], wc.cx[i] - .01)) / .02;
            g0y[i] = (c.sample(wc.cy[i] + .01, wc.cx[i]) - c.sample(wc.cy[i] - .01, wc.cx[i])) / .02;
        }
    }
    GnOut run(const cv::Mat& cur_u8, const std::vector<cv::Point2d>& initial) const {
        Spline c = Spline::from_coeff(spline_filter(blur64(cur_u8, .8)));
        const size_t n = pts.size();
        const int P = wc.P;
        std::vector<cv::Point2d> d(initial);
        std::vector<double> resid(n * P);
        std::vector<std::array<double, 2>> photo(n);
        std::vector<cv::Point2d> step(n);
        std::vector<double> vals(P), gx(P), gy(P);
        for (int it = 0; it < 20; ++it) {
            double mx = 0;
            for (size_t i = 0; i < n; ++i) {
                for (int k = 0; k < P; ++k) {
                    double y = wc.cy[i * P + k] + d[i].y, x = wc.cx[i * P + k] + d[i].x;
                    vals[k] = c.sample(y, x);
                    gx[k] = (c.sample(y, x + .01) - c.sample(y, x - .01)) / .02;
                    gy[k] = (c.sample(y + .01, x) - c.sample(y - .01, x)) / .02;
                }
                const double* pi = photinv[i].ptr<double>();
                double gain = 0, off = 0;
                for (int k = 0; k < P; ++k) { gain += pi[k] * vals[k]; off += pi[P + k] * vals[k]; }
                photo[i] = {gain, off};
                for (int k = 0; k < P; ++k) resid[i * P + k] = vals[k] - gain * T[i * P + k] - off;
                if (symmetric)
                    for (int k = 0; k < P; ++k) {
                        gx[k] = (gx[k] + gain * g0x[i * P + k]) / 2;
                        gy[k] = (gy[k] + gain * g0y[i * P + k]) / 2;
                    }
                // A = [gx, gy, -T, -1], 열 RMS 로 나눔
                double sc[4] = {0, 0, 0, 0};
                for (int k = 0; k < P; ++k) {
                    double t = T[i * P + k];
                    sc[0] += gx[k] * gx[k]; sc[1] += gy[k] * gy[k]; sc[2] += t * t; sc[3] += 1;
                }
                for (int j = 0; j < 4; ++j) sc[j] = std::max(std::sqrt(sc[j] / P), 1e-8);
                double G[4][4] = {}, rhs[4] = {};
                for (int k = 0; k < P; ++k) {
                    double a[4] = {gx[k] / sc[0], gy[k] / sc[1], -T[i * P + k] / sc[2], -1.0 / sc[3]};
                    double r = resid[i * P + k];
                    for (int u = 0; u < 4; ++u) {
                        rhs[u] += a[u] * r;
                        for (int v = 0; v < 4; ++v) G[u][v] += a[u] * a[v];
                    }
                }
                for (int u = 0; u < 4; ++u) G[u][u] += 1e-9;
                solve_n<4>(G, rhs);
                double sx = std::min(std::max(rhs[0] / sc[0], -.5), .5), sy = std::min(std::max(rhs[1] / sc[1], -.5), .5);
                step[i] = {sx, sy};
                d[i].x -= sx;
                d[i].y -= sy;
                mx = std::max(mx, std::hypot(sx, sy));
            }
            if (mx < 2e-5) break;
        }
        GnOut o;
        o.d = d;
        o.q.resize(n);
        o.valid.resize(n);
        for (size_t i = 0; i < n; ++i) {
            double ss = 0;
            for (int k = 0; k < P; ++k) ss += resid[i * P + k] * resid[i * P + k];
            double sn = std::hypot(step[i].x, step[i].y);
            o.q[i] = {std::sqrt(ss / P), sn, photo[i][0], photo[i][1]};
            o.valid[i] = std::isfinite(d[i].x) && std::isfinite(d[i].y) && std::hypot(d[i].x - initial[i].x, d[i].y - initial[i].y) < 1.5 && sn < .02;
        }
        return o;
    }
};

// 사진 읽기: 회색 전체 → CROP 영역만 남김
inline cv::Mat crop_of(const cv::Mat& full) { return full(cv::Rect(CROP[0], CROP[1], CROP[2] - CROP[0], CROP[3] - CROP[1])).clone(); }

struct TrackResult {
    std::vector<cv::Point2d> D;  // 점별 이동 (float32 로 저장되던 값)
    std::vector<bool> valid;
    std::vector<std::array<double, 5>> quality;  // GN 4개 + LK fb
    int shift[3][2];
    double score[3];
};

// spread.py SpreadPairExtractor (points 는 원래 좌표, regions 0..2)
struct SpreadExtractor {
    cv::Mat ref;  // CROP 된 center
    std::vector<cv::Point2d> points;
    std::vector<int> regions;
    std::vector<int> idx[3];                 // 영역별 점 색인
    std::vector<cv::Point2f> local[3];       // 영역 상자 기준 float32 좌표
    cv::Mat templ_blur[3];
    int box[3][4];
    int kind = 0;  // 0 inverse Refiner, 1 Forward
    std::unique_ptr<Refiner> inv[3];
    std::unique_ptr<ForwardRefiner> fwd[3];

    void init(const cv::Mat& center_full, const std::vector<cv::Point2d>& pts_full, const std::vector<int>& regs, int kind_, int window, bool symmetric) {
        ref = crop_of(center_full);
        points = pts_full;
        regions = regs;
        kind = kind_;
        for (int k = 0; k < 3; ++k) {
            idx[k].clear();
            local[k].clear();
            box[k][0] = BOXES[k][0] - CROP[0]; box[k][2] = BOXES[k][2] - CROP[0];
            box[k][1] = BOXES[k][1] - CROP[1]; box[k][3] = BOXES[k][3] - CROP[1];
            for (size_t i = 0; i < points.size(); ++i)
                if (regions[i] == k) {
                    idx[k].push_back((int)i);
                    float px = (float)(points[i].x - CROP[0]), py = (float)(points[i].y - CROP[1]);
                    local[k].push_back({px - (float)box[k][0], py - (float)box[k][1]});
                }
            cv::Mat t = ref(cv::Rect(box[k][0], box[k][1], box[k][2] - box[k][0], box[k][3] - box[k][1])).clone();
            cv::GaussianBlur(t, templ_blur[k], cv::Size(0, 0), .8);
            if (kind == 0) inv[k].reset(new Refiner(t, local[k], window, .8));
            else { fwd[k].reset(new ForwardRefiner()); fwd[k]->init(t, local[k], window, symmetric); }
        }
    }

    TrackResult extract(const cv::Mat& cur_full) const {
        cv::Mat cur = crop_of(cur_full);
        TrackResult r;
        r.D.assign(points.size(), {0, 0});
        r.valid.assign(points.size(), false);
        r.quality.assign(points.size(), {0, 0, 0, 0, 0});
        // coarse(include_screws=False): ±110 x ±22
        for (int k = 0; k < 3; ++k) {
            const int sx = 110, sy = 22;
            cv::Mat tpl = ref(cv::Rect(box[k][0], box[k][1], box[k][2] - box[k][0], box[k][3] - box[k][1]));
            cv::Mat search = cur(cv::Rect(box[k][0] - sx, box[k][1] - sy, box[k][2] - box[k][0] + 2 * sx, box[k][3] - box[k][1] + 2 * sy));
            cv::Mat m;
            cv::matchTemplate(search, tpl, m, cv::TM_CCOEFF_NORMED);
            double mx;
            cv::Point loc;
            cv::minMaxLoc(m, nullptr, &mx, nullptr, &loc);
            r.shift[k][0] = loc.x - sx;
            r.shift[k][1] = loc.y - sy;
            r.score[k] = mx;
        }
        for (int k = 0; k < 3; ++k) {
            const int dx = r.shift[k][0], dy = r.shift[k][1];
            cv::Mat target = cur(cv::Rect(box[k][0] + dx, box[k][1] + dy, box[k][2] - box[k][0], box[k][3] - box[k][1])).clone();
            cv::Mat tb;
            cv::GaussianBlur(target, tb, cv::Size(0, 0), .8);
            const auto& p = local[k];
            std::vector<cv::Point2f> p1(p), p2(p);
            std::vector<uchar> s1, s2;
            std::vector<float> e1, e2;
            cv::TermCriteria crit(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 60, 0.0001);
            cv::calcOpticalFlowPyrLK(templ_blur[k], tb, p, p1, s1, e1, cv::Size(15, 15), 0, crit, cv::OPTFLOW_USE_INITIAL_FLOW, 1e-5);
            p2 = p;
            cv::calcOpticalFlowPyrLK(tb, templ_blur[k], p1, p2, s2, e2, cv::Size(15, 15), 0, crit, cv::OPTFLOW_USE_INITIAL_FLOW, 1e-5);
            std::vector<cv::Point2d> init(p.size());
            std::vector<double> fb(p.size());
            std::vector<bool> lv(p.size());
            for (size_t i = 0; i < p.size(); ++i) {
                float ddx = p1[i].x - p[i].x, ddy = p1[i].y - p[i].y;  // float32 d
                init[i] = {ddx, ddy};
                float fx = p2[i].x - p[i].x, fy = p2[i].y - p[i].y;
                fb[i] = std::sqrt((double)fx * fx + (double)fy * fy);
                lv[i] = s1[i] && s2[i] && fb[i] < .5 && std::isfinite(ddx) && std::isfinite(ddy);
            }
            GnOut g = kind == 0 ? inv[k]->run(target, init) : fwd[k]->run(target, init);
            for (size_t i = 0; i < p.size(); ++i) {
                int j = idx[k][i];
                // gd(float32) + initial(float32) -> float32
                float gxv = (float)g.d[i].x, gyv = (float)g.d[i].y;
                r.D[j] = {(double)(float)(gxv + (float)dx), (double)(float)(gyv + (float)dy)};
                r.quality[j] = {g.q[i][0], g.q[i][1], g.q[i][2], g.q[i][3], fb[i]};
                r.valid[j] = lv[i] && g.valid[i];
            }
        }
        return r;
    }
};

// P12: 630점 공간 이차 회귀 계수 (6 x 2) → 12 특징 [c0x,c0y,c1x,c1y,...]
struct P12 {
    cv::Mat pinvB;  // 6 x n
    void init(const std::vector<cv::Point2d>& pts) {
        cv::Mat B((int)pts.size(), 6, CV_64F);
        for (int i = 0; i < (int)pts.size(); ++i) {
            double u = (pts[i].x - 1906) / 486, v = (pts[i].y - 1070) / 122;
            double row[6] = {1, u, v, u * u, u * v, v * v};
            for (int j = 0; j < 6; ++j) B.at<double>(i, j) = row[j];
        }
        cv::invert(B, pinvB, cv::DECOMP_SVD);
    }
    std::vector<double> apply(const std::vector<cv::Point2d>& D) const {
        std::vector<double> X(12, 0.0);
        for (int a = 0; a < 6; ++a) {
            double sx = 0, sy = 0;
            const double* r = pinvB.ptr<double>(a);
            for (size_t i = 0; i < D.size(); ++i) { sx += r[i] * D[i].x; sy += r[i] * D[i].y; }
            X[a * 2] = sx;
            X[a * 2 + 1] = sy;
        }
        return X;
    }
};

}  // namespace g09
