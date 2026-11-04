// 10-21 4-fold 실험군(#56~#61) 공통 측정: center 630 구조점 패치를 QR별로 잘라 LK15 시작 -> GN(3차 B-스플라인 보간, 밝기 gain/offset) 추적.
// 원본: codex_experiments_20261014/measurement/{measure.py, refine.py, spread.py, extract_pair.py},
//       patch_windows_20261021/measure.py (창·흐림 변경), robust_gn_20261021/measure_robust.py (Huber IRLS),
//       center_patch_v2_20261021/run.py (spatial P12, 품질 가중).
// scipy.ndimage 의 spline_filter(mirror/nearest) 와 map_coordinates(order=3) 를 수식 그대로 옮겼다 (scratch 검증 1e-13).
#pragma once
#include <hye/dataset.hpp>
#include <hye/report.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <array>
#include <cmath>
#include <string>
#include <vector>
#include "points630.hpp"

namespace g8 {

constexpr int kCX = 1340, kCY = 800, kCW = 1130, kCH = 530;  // measure.py CROP = (1340, 800, 2470, 1330)
const int kBox[3][4] = {{1552, 969, 1736, 1167}, {1809, 971, 1993, 1169}, {2067, 973, 2252, 1171}};  // x0 y0 x1 y1
const int kScrew[2][2] = {{1492, 1063}, {2312, 1073}};

inline cv::Mat read_crop(const hye::Shot& s) {
    cv::Mat m = hye::read_gray(s);
    return m(cv::Rect(kCX, kCY, kCW, kCH)).clone();
}

// ---- scipy.ndimage 3차 B-스플라인 ----
inline void spline_line(double* c, int n, int st, bool reflect) {
    if (n == 1) return;
    const double z = std::sqrt(3.0) - 2.0, lam = (1 - z) * (1 - 1 / z);
    for (int i = 0; i < n; ++i) c[i * st] *= lam;
    if (!reflect) {
        const double zn1 = std::pow(z, n - 1);
        double c0 = c[0] + zn1 * c[(n - 1) * st], zi = z;
        for (int i = 1; i < n - 1; ++i) { c0 += zi * (c[i * st] + zn1 * c[(n - 1 - i) * st]); zi *= z; }
        c[0] = c0 / (1 - zn1 * zn1);
    } else {
        const double zn = std::pow(z, n), c00 = c[0];
        double c0 = c[0] + zn * c[(n - 1) * st], zi = z;
        for (int i = 1; i < n; ++i) { c0 += zi * (c[i * st] + zn * c[(n - 1 - i) * st]); zi *= z; }
        c[0] = c0 * z / (1 - zn * zn) + c00;
    }
    for (int i = 1; i < n; ++i) c[i * st] += z * c[(i - 1) * st];
    if (!reflect) c[(n - 1) * st] = (z * c[(n - 2) * st] + c[(n - 1) * st]) * z / (z * z - 1);
    else c[(n - 1) * st] *= z / (z - 1);
    for (int i = n - 2; i >= 0; --i) c[i * st] = z * (c[(i + 1) * st] - c[i * st]);
}
// spline_filter(a, 3, mode): axis 0 다음 axis 1
inline cv::Mat spline_filter(const cv::Mat& a64, bool reflect) {
    cv::Mat c = a64.clone();
    CV_Assert(c.type() == CV_64F && c.isContinuous());
    double* p = c.ptr<double>();
    for (int x = 0; x < c.cols; ++x) spline_line(p + x, c.rows, c.cols, reflect);
    for (int y = 0; y < c.rows; ++y) spline_line(p + (size_t)y * c.cols, c.cols, 1, reflect);
    return c;
}
// map_coordinates(..., mode='nearest') 에 쓰는 사전 필터: 가장자리 12 px 복제 후 nearest(=reflect 초기값) 필터
struct PaddedSpline {
    cv::Mat c;
    int pad = 12;
    explicit PaddedSpline(const cv::Mat& a64) {
        cv::Mat p;
        cv::copyMakeBorder(a64, p, 12, 12, 12, 12, cv::BORDER_REPLICATE);
        c = spline_filter(p, true);
    }
};
inline void bweights(double t, double w[4]) {
    double y = t, z = 1 - t;
    w[0] = z * z * z / 6;
    w[1] = (y * y * (y - 2) * 3 + 4) / 6;
    w[2] = (z * z * (z - 2) * 3 + 4) / 6;
    w[3] = y * y * y / 6;
}
// prefilter=False, mode='nearest' (계수 배열 인덱스를 가장자리로 고정)
inline double spline_at(const cv::Mat& c, double y, double x) {
    double fy = std::floor(y), fx = std::floor(x), wy[4], wx[4];
    bweights(y - fy, wy);
    bweights(x - fx, wx);
    int iy0 = (int)fy - 1, ix0 = (int)fx - 1, ix[4];
    for (int j = 0; j < 4; ++j) ix[j] = std::min(std::max(ix0 + j, 0), c.cols - 1);
    double s = 0;
    for (int i = 0; i < 4; ++i) {
        const double* row = c.ptr<double>(std::min(std::max(iy0 + i, 0), c.rows - 1));
        double r = 0;
        for (int j = 0; j < 4; ++j) r += wx[j] * row[ix[j]];
        s += wy[i] * r;
    }
    return s;
}

inline cv::Mat blur64(const cv::Mat& u8, double sigma) {
    cv::Mat f, b;
    u8.convertTo(f, CV_64F);
    cv::GaussianBlur(f, b, cv::Size(0, 0), sigma);
    return b;
}

// ---- GN 정밀화 (refine.py Refiner / measure_robust.py RobustRefiner) ----
struct GNOut {
    std::vector<cv::Point2f> d;
    std::vector<std::array<float, 4>> q;  // rmse, last step, gain, offset
    std::vector<char> valid;
};
class Refiner {
public:
    int window;
    double sigma;
    int robust = 0;  // 0 일반, 1 huber
    double delta = 6;
    std::vector<cv::Point2f> pts;  // 국소 좌표 (float32)
    std::vector<int> oy, ox;
    std::vector<std::vector<double>> tmpl;       // [n][P]
    std::vector<std::vector<double>> pinv;       // [n][4*P]
    std::vector<std::vector<double>> A;          // [n][P*4] 정규화 설계 (robust)
    std::vector<std::array<double, 4>> scale;    // robust
    Refiner(const cv::Mat& ref_u8, const std::vector<cv::Point2f>& points, int win, double sig, int robust_kind = 0, double dlt = 6)
        : window(win), sigma(sig), robust(robust_kind), delta(dlt), pts(points) {
        cv::Mat refb = blur64(ref_u8, sigma);
        cv::Mat gx(refb.size(), CV_64F), gy(refb.size(), CV_64F);
        const int H = refb.rows, W = refb.cols;
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                const double* r = refb.ptr<double>(y);
                gx.at<double>(y, x) = x == 0 ? r[1] - r[0] : x == W - 1 ? r[W - 1] - r[W - 2] : (r[x + 1] - r[x - 1]) / 2;
                gy.at<double>(y, x) = y == 0 ? refb.at<double>(1, x) - refb.at<double>(0, x)
                                    : y == H - 1 ? refb.at<double>(H - 1, x) - refb.at<double>(H - 2, x)
                                                 : (refb.at<double>(y + 1, x) - refb.at<double>(y - 1, x)) / 2;
            }
        PaddedSpline sr(refb), sx(gx), sy(gy);
        const int a = -((window + 1) / 2) + 1;  // np.mgrid[-window//2+1 : window//2+1] (홀수 창)
        for (int yy = a; yy < window / 2 + 1; ++yy)
            for (int xx = a; xx < window / 2 + 1; ++xx) { oy.push_back(yy); ox.push_back(xx); }
        const int P = (int)oy.size(), n = (int)pts.size();
        tmpl.assign(n, std::vector<double>(P));
        pinv.assign(n, std::vector<double>(4 * P));
        if (robust) { A.assign(n, std::vector<double>(P * 4)); scale.resize(n); }
        for (int i = 0; i < n; ++i) {
            cv::Mat D(P, 4, CV_64F);
            for (int p = 0; p < P; ++p) {
                double cy = (double)pts[i].y + oy[p] + 12, cx = (double)pts[i].x + ox[p] + 12;
                double t = spline_at(sr.c, cy, cx);
                tmpl[i][p] = t;
                D.at<double>(p, 0) = spline_at(sx.c, cy, cx);
                D.at<double>(p, 1) = spline_at(sy.c, cy, cx);
                D.at<double>(p, 2) = t;
                D.at<double>(p, 3) = 1;
            }
            cv::Mat Pi;
            cv::invert(D, Pi, cv::DECOMP_SVD);  // np.linalg.pinv
            for (int k = 0; k < 4; ++k)
                for (int p = 0; p < P; ++p) pinv[i][k * P + p] = Pi.at<double>(k, p);
            if (robust) {
                for (int k = 0; k < 4; ++k) {
                    double s = 0;
                    for (int p = 0; p < P; ++p) s += D.at<double>(p, k) * D.at<double>(p, k);
                    scale[i][k] = std::max(std::sqrt(s / P), 1e-8);
                }
                for (int p = 0; p < P; ++p)
                    for (int k = 0; k < 4; ++k) A[i][p * 4 + k] = D.at<double>(p, k) / scale[i][k];
            }
        }
    }
    GNOut run(const cv::Mat& cur_u8, const std::vector<cv::Point2f>& initial) const {
        cv::Mat coeff = spline_filter(blur64(cur_u8, sigma), false);
        const int P = (int)oy.size(), n = (int)pts.size();
        std::vector<double> dx(n), dy(n), sx(n), sy(n);
        std::vector<std::array<double, 4>> pars(n);
        std::vector<std::vector<double>> vals(n, std::vector<double>(P));
        for (int i = 0; i < n; ++i) { dx[i] = initial[i].x; dy[i] = initial[i].y; }
        for (int it = 0; it < 12; ++it) {
            double mx = 0;
            for (int i = 0; i < n; ++i) {
                auto& v = vals[i];
                for (int p = 0; p < P; ++p) v[p] = spline_at(coeff, (double)pts[i].y + oy[p] + dy[i], (double)pts[i].x + ox[p] + dx[i]);
                std::array<double, 4> pr{0, 0, 0, 0};
                for (int k = 0; k < 4; ++k) {
                    const double* pk = &pinv[i][k * P];
                    double s = 0;
                    for (int p = 0; p < P; ++p) s += pk[p] * v[p];
                    pr[k] = s;
                }
                if (robust) {
                    for (int rep = 0; rep < 2; ++rep) {
                        double G[16] = {0}, rhs[4] = {0};
                        for (int p = 0; p < P; ++p) {
                            const double* a = &A[i][p * 4];
                            double fit = 0;
                            for (int k = 0; k < 4; ++k) fit += a[k] * (pr[k] * scale[i][k]);
                            double rr = std::abs(v[p] - fit) / delta;
                            double w = std::min(1.0, 1 / std::max(rr, 1e-12));
                            for (int r = 0; r < 4; ++r) {
                                rhs[r] += a[r] * w * v[p];
                                for (int c = 0; c < 4; ++c) G[r * 4 + c] += a[r] * w * a[c];
                            }
                        }
                        cv::Mat Gm(4, 4, CV_64F, G), Rm(4, 1, CV_64F, rhs), X;
                        Gm += cv::Mat::eye(4, 4, CV_64F) * 1e-9;
                        cv::solve(Gm, Rm, X, cv::DECOMP_LU);
                        for (int k = 0; k < 4; ++k) pr[k] = X.at<double>(k) / scale[i][k];
                    }
                }
                pars[i] = pr;
                double g = std::max(pr[2], 0.25);
                double stx = std::min(std::max(pr[0] / g, -0.5), 0.5), sty = std::min(std::max(pr[1] / g, -0.5), 0.5);
                dx[i] -= stx;
                dy[i] -= sty;
                sx[i] = stx;
                sy[i] = sty;
                mx = std::max(mx, std::sqrt(stx * stx + sty * sty));
            }
            if (mx < 2e-5) break;
        }
        GNOut o;
        o.d.resize(n);
        o.q.resize(n);
        o.valid.resize(n);
        for (int i = 0; i < n; ++i) {
            double s = 0;
            for (int p = 0; p < P; ++p) {
                double r = vals[i][p] - pars[i][2] * tmpl[i][p] - pars[i][3];
                s += r * r;
            }
            o.d[i] = cv::Point2f((float)dx[i], (float)dy[i]);
            double step = std::sqrt(sx[i] * sx[i] + sy[i] * sy[i]);
            o.q[i] = {(float)std::sqrt(s / P), (float)step, (float)pars[i][2], (float)pars[i][3]};
            double ddx = dx[i] - initial[i].x, ddy = dy[i] - initial[i].y;
            o.valid[i] = std::isfinite(dx[i]) && std::isfinite(dy[i]) && std::sqrt(ddx * ddx + ddy * ddy) < 1.5 && o.q[i][1] < 0.02f;
        }
        return o;
    }
};

// 전체 QR 템플릿 매칭으로 정수 이동 (measure.py coarse)
inline std::vector<cv::Point> coarse(const cv::Mat& ref, const cv::Mat& cur, bool screws) {
    std::vector<cv::Point> out;
    for (int k = 0; k < (screws ? 5 : 3); ++k) {
        int x0, y0, x1, y1;
        if (k < 3) { x0 = kBox[k][0]; y0 = kBox[k][1]; x1 = kBox[k][2]; y1 = kBox[k][3]; }
        else { int x = kScrew[k - 3][0], y = kScrew[k - 3][1]; x0 = x - 31; y0 = y - 31; x1 = x + 32; y1 = y + 32; }
        x0 -= kCX; x1 -= kCX; y0 -= kCY; y1 -= kCY;
        const int sx = 110, sy = 22;
        cv::Mat t = ref(cv::Range(y0, y1), cv::Range(x0, x1)), s = cur(cv::Range(y0 - sy, y1 + sy), cv::Range(x0 - sx, x1 + sx)), c;
        cv::matchTemplate(s, t, c, cv::TM_CCOEFF_NORMED);
        double v;
        cv::Point pos;
        cv::minMaxLoc(c, nullptr, &v, nullptr, &pos);
        out.push_back({pos.x - sx, pos.y - sy});
    }
    return out;
}

struct LKOut {
    std::vector<cv::Point2f> d;
    std::vector<float> fb;
    std::vector<char> valid;
};
// measure.py track: 단일 층 LK 정방향 + 역방향
inline LKOut track(const cv::Mat& ref, const cv::Mat& cur, const std::vector<cv::Point2f>& p, const std::vector<cv::Point2f>& init, int win) {
    cv::TermCriteria crit(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 60, 0.0001);
    std::vector<cv::Point2f> p1(p.size()), p2 = p;
    for (size_t i = 0; i < p.size(); ++i) p1[i] = p[i] + init[i];
    std::vector<uchar> s1, s2;
    std::vector<float> e1, e2;
    cv::calcOpticalFlowPyrLK(ref, cur, p, p1, s1, e1, cv::Size(win, win), 0, crit, cv::OPTFLOW_USE_INITIAL_FLOW, 1e-5);
    cv::calcOpticalFlowPyrLK(cur, ref, p1, p2, s2, e2, cv::Size(win, win), 0, crit, cv::OPTFLOW_USE_INITIAL_FLOW, 1e-5);
    LKOut o;
    for (size_t i = 0; i < p.size(); ++i) {
        float fx = p2[i].x - p[i].x, fy = p2[i].y - p[i].y;
        float fb = (float)std::sqrt((double)fx * fx + (double)fy * fy);
        cv::Point2f d = p1[i] - p[i];
        o.d.push_back(d);
        o.fb.push_back(fb);
        o.valid.push_back(s1[i] && s2[i] && fb < 0.5f && std::isfinite(d.x) && std::isfinite(d.y));
    }
    return o;
}

// 한 장 측정 결과 (630 점)
struct Track {
    std::vector<cv::Point2f> D;           // center 기준 원 좌표 이동 (float32, 원본과 같게)
    std::vector<std::array<float, 5>> q;  // rmse, last step, gain, offset, LK FB
    std::vector<char> valid;
};

// spread.py SpreadPairExtractor: QR마다 잘라서 LK15(흐림 .8) 시작 -> GN(window, sigma)
struct QRExtractor {
    cv::Mat ref;
    std::vector<int> region;  // 630 점의 QR 0..2
    std::vector<std::vector<int>> members;
    std::vector<Refiner> models;
    std::vector<cv::Mat> lk_tmpl;
    std::vector<std::vector<cv::Point2f>> lp;
    QRExtractor(const cv::Mat& center_crop, int window = 17, double sigma = 0.8, int robust = 0, double delta = 6) : ref(center_crop) {
        members.resize(3);
        for (int i = 0; i < kN630; ++i) {
            double x = kPoints630[i].x;
            const double c[3] = {1644, 1901, 2159};
            int best = 0;
            for (int k = 1; k < 3; ++k) if (std::abs(x - c[k]) < std::abs(x - c[best])) best = k;
            region.push_back(best);
            members[best].push_back(i);
        }
        for (int k = 0; k < 3; ++k) {
            int x0 = kBox[k][0] - kCX, y0 = kBox[k][1] - kCY, x1 = kBox[k][2] - kCX, y1 = kBox[k][3] - kCY;
            cv::Mat t = ref(cv::Range(y0, y1), cv::Range(x0, x1)).clone();
            std::vector<cv::Point2f> pts;
            for (int i : members[k]) {
                float px = (float)(kPoints630[i].x - kCX), py = (float)(kPoints630[i].y - kCY);
                pts.push_back(cv::Point2f(px - (float)x0, py - (float)y0));
            }
            models.emplace_back(t, pts, window, sigma, robust, delta);
            cv::Mat tb;
            cv::GaussianBlur(t, tb, cv::Size(0, 0), 0.8);
            lk_tmpl.push_back(tb);
            lp.push_back(pts);
        }
    }
    Track extract(const cv::Mat& cur) const {
        auto init = coarse(ref, cur, false);
        Track o;
        o.D.resize(kN630);
        o.q.resize(kN630);
        o.valid.resize(kN630);
        for (int k = 0; k < 3; ++k) {
            int x0 = kBox[k][0] - kCX, y0 = kBox[k][1] - kCY, x1 = kBox[k][2] - kCX, y1 = kBox[k][3] - kCY;
            int dx = init[k].x, dy = init[k].y;
            cv::Mat target = cur(cv::Range(y0 + dy, y1 + dy), cv::Range(x0 + dx, x1 + dx)).clone(), tb;
            cv::GaussianBlur(target, tb, cv::Size(0, 0), 0.8);
            std::vector<cv::Point2f> zero(lp[k].size(), cv::Point2f(0, 0));
            LKOut l = track(lk_tmpl[k], tb, lp[k], zero, 15);
            GNOut g = models[k].run(target, l.d);
            for (size_t j = 0; j < members[k].size(); ++j) {
                int i = members[k][j];
                o.D[i] = cv::Point2f(g.d[j].x + (float)dx, g.d[j].y + (float)dy);
                o.q[i] = {g.q[j][0], g.q[j][1], g.q[j][2], g.q[j][3], l.fb[j]};
                o.valid[i] = l.valid[j] && g.valid[j];
            }
        }
        return o;
    }
};

// ---- P12: 630 점 이동을 [1,u,v,u^2,uv,v^2] 두 방향 계수 12개로 요약 (center_patch_v2 run.py spatial) ----
inline cv::Mat p12_basis(const std::vector<cv::Point2d>& pts) {
    cv::Mat B((int)pts.size(), 6, CV_64F);
    for (int i = 0; i < B.rows; ++i) {
        double u = (pts[i].x - 1906) / 486, v = (pts[i].y - 1070) / 122;
        double r[6] = {1, u, v, u * u, u * v, v * v};
        for (int k = 0; k < 6; ++k) B.at<double>(i, k) = r[k];
    }
    return B;
}
inline std::vector<cv::Point2d> points630() {
    std::vector<cv::Point2d> p;
    for (int i = 0; i < kN630; ++i) p.push_back({kPoints630[i].x, kPoints630[i].y});
    return p;
}
// D: [N][630] (center 를 뺀 값), w: 가중(빈 값이면 균등) -> N x 12
inline cv::Mat spatial(const std::vector<cv::Point2d>& pts, const std::vector<std::vector<cv::Point2d>>& D,
                       const std::vector<std::vector<double>>* w = nullptr) {
    cv::Mat B = p12_basis(pts), Bp;
    cv::invert(B, Bp, cv::DECOMP_SVD);
    cv::Mat X((int)D.size(), 12, CV_64F);
    for (int n = 0; n < (int)D.size(); ++n) {
        cv::Mat d((int)pts.size(), 2, CV_64F), c;
        for (int i = 0; i < d.rows; ++i) { d.at<double>(i, 0) = D[n][i].x; d.at<double>(i, 1) = D[n][i].y; }
        if (!w) c = Bp * d;
        else {
            cv::Mat Bw = B.clone(), dw = d.clone();
            for (int i = 0; i < d.rows; ++i) {
                double s = std::sqrt((*w)[n][i]);
                Bw.row(i) *= s;
                dw.row(i) *= s;
            }
            cv::solve(Bw, dw, c, cv::DECOMP_SVD);
        }
        for (int k = 0; k < 6; ++k) { X.at<double>(n, 2 * k) = c.at<double>(k, 0); X.at<double>(n, 2 * k + 1) = c.at<double>(k, 1); }
    }
    return X;
}

// ---- 나사 39점 (measure.py feature_set 의 regions 3,4 + extract_pair.py PairExtractor gn17, include_screws) ----
struct ScrewSet {
    std::vector<cv::Point2f> pts;  // crop 좌표
    std::vector<int> region;       // 3, 4
};
inline ScrewSet screw_features(const cv::Mat& ref) {
    cv::Mat sm;
    cv::GaussianBlur(ref, sm, cv::Size(0, 0), 0.65);
    ScrewSet s;
    for (int k = 3; k < 5; ++k) {
        cv::Mat mask = cv::Mat::zeros(ref.size(), CV_8U);
        cv::circle(mask, cv::Point(kScrew[k - 3][0] - kCX, kScrew[k - 3][1] - kCY), 21, 255, -1);
        std::vector<cv::Point2f> p;
        cv::goodFeaturesToTrack(sm, p, 60, 0.01, 4, mask, 5);
        cv::cornerSubPix(sm, p, cv::Size(2, 2), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 30, 0.001));
        for (auto& q : p) { s.pts.push_back(q); s.region.push_back(k); }
    }
    return s;
}
struct ScrewExtractor {
    cv::Mat ref, ref_blur;
    ScrewSet f;
    Refiner model;
    explicit ScrewExtractor(const cv::Mat& center_crop, const ScrewSet& s)
        : ref(center_crop), f(s), model(center_crop, s.pts, 17, 0.8) {
        cv::GaussianBlur(ref, ref_blur, cv::Size(0, 0), 0.8);
    }
    Track extract(const cv::Mat& cur) const {
        auto init = coarse(ref, cur, true);
        cv::Mat cb;
        cv::GaussianBlur(cur, cb, cv::Size(0, 0), 0.8);
        std::vector<cv::Point2f> ini;
        for (int r : f.region) ini.push_back(cv::Point2f((float)init[r].x, (float)init[r].y));
        LKOut l = track(ref_blur, cb, f.pts, ini, 31);
        GNOut g = model.run(cur, l.d);
        Track o;
        o.D = g.d;
        for (size_t i = 0; i < g.d.size(); ++i) {
            o.q.push_back({g.q[i][0], g.q[i][1], g.q[i][2], g.q[i][3], l.fb[i]});
            o.valid.push_back(l.valid[i] && g.valid[i]);
        }
        return o;
    }
};

}  // namespace g8
