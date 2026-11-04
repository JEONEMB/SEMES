// codex measurement/measure.py, refine.py, spread.py 의 C++ 이식.
// center 사진에서 QR 안쪽 특징점을 고르고, 현재 사진에서 그 점을 LK 또는 GN17(3차 스플라인, gain/offset 제거)로 추적한다.
// scipy.ndimage 의 spline_filter / map_coordinates(order=3) 경계 규칙을 그대로 따른다.
#pragma once
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/core.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <type_traits>
#include <vector>

namespace trk {

// 3840x2160 사진에서 쓰는 영역 (measure.py)
const int CX0 = 1340, CY0 = 800, CX1 = 2470, CY1 = 1330;
const std::array<cv::Vec4i, 3> BOXES{{{1552, 969, 1736, 1167}, {1809, 971, 1993, 1169}, {2067, 973, 2252, 1171}}};  // x0,y0,x1,y1

inline cv::Mat crop(const cv::Mat& gray) { return gray(cv::Range(CY0, CY1), cv::Range(CX0, CX1)).clone(); }

struct Features {
    std::vector<cv::Point2f> pts;  // crop 좌표
    std::vector<int> region;       // 0,1,2 = QR 좌/중/우
    std::vector<float> strength;
    std::vector<cv::Point2f> global() const {
        std::vector<cv::Point2f> g;
        for (auto& p : pts) g.push_back(p + cv::Point2f((float)CX0, (float)CY0));
        return g;
    }
};

// measure.py feature_set (QR 부분만) / spread.py select
//   margin: 검출 마스크 침식, keep: 유지 경계, maxn, quality, min_distance
inline Features select_features(const cv::Mat& ref, int margin, int keep, int maxn, double quality, double min_dist) {
    Features f;
    cv::Mat sm, eig;
    cv::GaussianBlur(ref, sm, cv::Size(0, 0), 0.65);
    cv::cornerMinEigenVal(sm, eig, 5);
    for (int k = 0; k < 3; ++k) {
        auto b = BOXES[k];
        cv::Mat mask = cv::Mat::zeros(ref.size(), CV_8U);
        mask(cv::Range(b[1] + margin - CY0, b[3] - margin - CY0), cv::Range(b[0] + margin - CX0, b[2] - margin - CX0)).setTo(255);
        std::vector<cv::Point2f> p;
        cv::goodFeaturesToTrack(sm, p, maxn, quality, min_dist, mask, 5, false, 0.04);
        cv::cornerSubPix(sm, p, cv::Size(2, 2), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 30, 0.001));
        for (auto& q : p) {
            float gx = q.x + (float)CX0, gy = q.y + (float)CY0;
            if (!(b[0] + keep <= gx && gx < b[2] - keep && b[1] + keep <= gy && gy < b[3] - keep)) continue;
            f.pts.push_back(q);
            f.region.push_back(k);
            f.strength.push_back(eig.at<float>((int)std::nearbyint((double)q.y), (int)std::nearbyint((double)q.x)));
        }
    }
    return f;
}
inline Features features928(const cv::Mat& ref) { return select_features(ref, 28, 27, 600, 0.005, 4); }
inline Features features_spread(const cv::Mat& ref, int min_dist) { return select_features(ref, 16, 15, 600, 0.02, min_dist); }

// 전체 QR NCC 정수 이동 (measure.py coarse)
inline std::array<cv::Point, 3> coarse(const cv::Mat& ref, const cv::Mat& cur, std::array<float, 3>* score = nullptr) {
    std::array<cv::Point, 3> out;
    for (int k = 0; k < 3; ++k) {
        auto b = BOXES[k];
        int x0 = b[0] - CX0, x1 = b[2] - CX0, y0 = b[1] - CY0, y1 = b[3] - CY0, sx = 110, sy = 22;
        cv::Mat tpl = ref(cv::Range(y0, y1), cv::Range(x0, x1));
        cv::Mat search = cur(cv::Range(y0 - sy, y1 + sy), cv::Range(x0 - sx, x1 + sx));
        cv::Mat c;
        cv::matchTemplate(search, tpl, c, cv::TM_CCOEFF_NORMED);
        double v;
        cv::Point pos;
        cv::minMaxLoc(c, nullptr, &v, nullptr, &pos);
        out[k] = {pos.x - sx, pos.y - sy};
        if (score) (*score)[k] = (float)v;
    }
    return out;
}

struct Track {
    std::vector<cv::Point2f> d;  // float32 이동
    std::vector<float> fb, err;
    std::vector<char> valid;
};

// measure.py track: 단일 층 LK, 초기 이동 init, 순/역방향 FB
inline Track lk_track(const cv::Mat& ref, const cv::Mat& cur, const std::vector<cv::Point2f>& p, const std::vector<cv::Point2f>& init, int window) {
    cv::TermCriteria crit(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 60, 0.0001);
    std::vector<cv::Point2f> p1(p.size()), p2 = p;
    for (size_t i = 0; i < p.size(); ++i) p1[i] = p[i] + init[i];
    std::vector<uchar> s1, s2;
    std::vector<float> e1, e2;
    cv::calcOpticalFlowPyrLK(ref, cur, p, p1, s1, e1, cv::Size(window, window), 0, crit, cv::OPTFLOW_USE_INITIAL_FLOW, 1e-5);
    cv::calcOpticalFlowPyrLK(cur, ref, p1, p2, s2, e2, cv::Size(window, window), 0, crit, cv::OPTFLOW_USE_INITIAL_FLOW, 1e-5);
    Track t;
    for (size_t i = 0; i < p.size(); ++i) {
        cv::Point2f dd = p1[i] - p[i];
        float fb = (float)std::hypot((double)(p2[i].x - p[i].x), (double)(p2[i].y - p[i].y));
        t.d.push_back(dd);
        t.fb.push_back(fb);
        t.err.push_back(e1[i]);
        t.valid.push_back(s1[i] && s2[i] && fb < 0.5f && std::isfinite(dd.x) && std::isfinite(dd.y));
    }
    return t;
}

// ---- scipy.ndimage 3차 B-스플라인 ----
namespace spl {
const double Z = std::sqrt(3.0) - 2.0;
inline void filt_mirror(double* c, int n, int s) {
    auto C = [&](int i) -> double& { return c[(size_t)i * s]; };
    double zi = Z, zn1 = std::pow(Z, n - 1), c0 = C(0) + zn1 * C(n - 1);
    for (int i = 1; i < n - 1; ++i) { c0 += zi * (C(i) + zn1 * C(n - 1 - i)); zi *= Z; }
    C(0) = c0 / (1 - zn1 * zn1);
    for (int i = 1; i < n; ++i) C(i) += Z * C(i - 1);
    C(n - 1) = (Z * C(n - 2) + C(n - 1)) * Z / (Z * Z - 1);
    for (int i = n - 2; i >= 0; --i) C(i) = Z * (C(i + 1) - C(i));
}
inline void filt_reflect(double* c, int n, int s) {
    auto C = [&](int i) -> double& { return c[(size_t)i * s]; };
    double zi = Z, zn = std::pow(Z, n), c00 = C(0), c0 = C(0) + zn * C(n - 1);
    for (int i = 1; i < n; ++i) { c0 += zi * (C(i) + zn * C(n - 1 - i)); zi *= Z; }
    C(0) = c0 * Z / (1 - zn * zn) + c00;
    for (int i = 1; i < n; ++i) C(i) += Z * C(i - 1);
    C(n - 1) *= Z / (Z - 1);
    for (int i = n - 2; i >= 0; --i) C(i) = Z * (C(i + 1) - C(i));
}
// spline_filter(order=3): 축 0, 축 1 순서. reflect=true 는 mode nearest/reflect 경계
inline cv::Mat filter(const cv::Mat& in, bool reflect) {
    cv::Mat c;
    in.convertTo(c, CV_64F);
    c = c.clone();
    const double gain = (1 - Z) * (1 - 1 / Z);
    double* p = c.ptr<double>();
    int R = c.rows, W = c.cols;
    for (int x = 0; x < W; ++x) {
        for (int y = 0; y < R; ++y) p[(size_t)y * W + x] *= gain;
        reflect ? filt_reflect(p + x, R, W) : filt_mirror(p + x, R, W);
    }
    for (int y = 0; y < R; ++y) {
        for (int x = 0; x < W; ++x) p[(size_t)y * W + x] *= gain;
        reflect ? filt_reflect(p + (size_t)y * W, W, 1) : filt_mirror(p + (size_t)y * W, W, 1);
    }
    return c;
}
// map_coordinates(mode='nearest') 의 prefilter: 12 px edge 패딩 후 reflect 필터. off = 12
inline cv::Mat filter_nearest(const cv::Mat& in) {
    cv::Mat pad;
    cv::copyMakeBorder(in, pad, 12, 12, 12, 12, cv::BORDER_REPLICATE);
    return filter(pad, true);
}
inline void weights(double x, double w[4], int& start) {
    double f = std::floor(x), y = x - f, z = 1.0 - y;
    start = (int)f - 1;
    w[1] = (y * y * (y - 2.0) * 3.0 + 4.0) / 6.0;
    w[2] = (z * z * (z - 2.0) * 3.0 + 4.0) / 6.0;
    w[3] = y * y * y / 6.0;
    w[0] = 1.0 - w[1] - w[2] - w[3];
}
// 계수 c 에서 (row, col) 값. 경계는 nearest (안쪽 점만 쓴다)
inline double eval(const cv::Mat& c, double row, double col, int off = 0) {
    double wr[4], wc[4];
    int sr, sc;
    weights(row + off, wr, sr);
    weights(col + off, wc, sc);
    double v = 0;
    for (int a = 0; a < 4; ++a) {
        int r = std::min(std::max(sr + a, 0), c.rows - 1);
        const double* q = c.ptr<double>(r);
        double t = 0;
        for (int b = 0; b < 4; ++b) t += wc[b] * q[std::min(std::max(sc + b, 0), c.cols - 1)];
        v += wr[a] * t;
    }
    return v;
}
}  // namespace spl

inline cv::Mat np_gradient(const cv::Mat& f, int axis) {  // np.gradient, 끝은 한쪽 차분
    cv::Mat g(f.size(), CV_64F);
    int R = f.rows, W = f.cols;
    for (int y = 0; y < R; ++y)
        for (int x = 0; x < W; ++x) {
            double v;
            if (axis == 0) v = y == 0 ? f.at<double>(1, x) - f.at<double>(0, x) : y == R - 1 ? f.at<double>(R - 1, x) - f.at<double>(R - 2, x)
                                                                                         : (f.at<double>(y + 1, x) - f.at<double>(y - 1, x)) / 2;
            else v = x == 0 ? f.at<double>(y, 1) - f.at<double>(y, 0) : x == W - 1 ? f.at<double>(y, W - 1) - f.at<double>(y, W - 2)
                                                                                  : (f.at<double>(y, x + 1) - f.at<double>(y, x - 1)) / 2;
            g.at<double>(y, x) = v;
        }
    return g;
}

// refine.py Refiner: 역합성 GN. 창 window, 흐림 sigma. 선택적으로 픽셀 가중(역분산)을 넣을 수 있다.
struct Refiner {
    int window = 17;
    double sigma = 0.8;
    std::vector<cv::Point2f> pts;
    std::vector<double> oy, ox;          // 창 안 상대 좌표 (행 우선)
    std::vector<cv::Mat> pinv;           // 점마다 4 x K
    std::vector<cv::Mat> design;         // 점마다 K x 4 [tx, ty, 템플릿, 1] (가중 전)
    std::vector<std::vector<double>> tpl;  // 점마다 템플릿 K
    std::vector<std::vector<double>> wpx;  // 빈 값이면 가중 없음

    Refiner() = default;
    // weight_fn: 템플릿 밝기 -> 가중 (없으면 nullptr)
    template <class WF = std::nullptr_t>
    Refiner(const cv::Mat& ref_u8, const std::vector<cv::Point2f>& points, int win = 17, double sig = 0.8, WF weight_fn = nullptr)
        : window(win), sigma(sig), pts(points) {
        cv::Mat ref;
        ref_u8.convertTo(ref, CV_64F);
        cv::GaussianBlur(ref, ref, cv::Size(0, 0), sigma);
        const int lo = (int)std::floor(-window / 2.0) + 1;  // python -window//2+1 (창 17 -> -8..8)
        for (int a = 0; a < window; ++a)
            for (int b = 0; b < window; ++b) { oy.push_back(lo + a); ox.push_back(lo + b); }
        cv::Mat cr = spl::filter_nearest(ref), gy = spl::filter_nearest(np_gradient(ref, 0)), gx = spl::filter_nearest(np_gradient(ref, 1));
        const int K = (int)oy.size();
        for (auto& p : pts) {
            cv::Mat A(K, 4, CV_64F);
            std::vector<double> t(K);
            for (int k = 0; k < K; ++k) {
                double r = (double)p.y + oy[k], c = (double)p.x + ox[k];
                t[k] = spl::eval(cr, r, c, 12);
                A.at<double>(k, 0) = spl::eval(gx, r, c, 12);
                A.at<double>(k, 1) = spl::eval(gy, r, c, 12);
                A.at<double>(k, 2) = t[k];
                A.at<double>(k, 3) = 1;
            }
            std::vector<double> w;
            if constexpr (!std::is_same<WF, std::nullptr_t>::value) {
                w.resize(K);
                for (int k = 0; k < K; ++k) {
                    w[k] = weight_fn(t[k]);
                    double s = std::sqrt(w[k]);
                    for (int j = 0; j < 4; ++j) A.at<double>(k, j) *= s;
                }
            }
            if (w.empty()) design.push_back(A.clone());
            cv::Mat P;
            cv::invert(A, P, cv::DECOMP_SVD);
            if (!w.empty())
                for (int k = 0; k < K; ++k)
                    for (int j = 0; j < 4; ++j) P.at<double>(j, k) *= std::sqrt(w[k]);
            pinv.push_back(P);
            tpl.push_back(t);
            wpx.push_back(w);
        }
    }

    struct Out {
        std::vector<cv::Point2f> d;
        std::vector<double> rmse, last, gain, offset;
        std::vector<char> valid;
    };
    // 현재 QR 조각의 스플라인 계수 (흐림 sigma 후 scipy spline_filter, mirror)
    cv::Mat coefficients(const cv::Mat& cur_u8) const {
        cv::Mat cur;
        cur_u8.convertTo(cur, CV_64F);
        cv::GaussianBlur(cur, cur, cv::Size(0, 0), sigma);
        return spl::filter(cur, false);
    }
    Out run(const cv::Mat& cur_u8, const std::vector<cv::Point2f>& initial) const { return run_coef(coefficients(cur_u8), initial, pinv); }
    // 계수와 점별 4 x K 해 행렬(pv)로 GN. initial 은 double 로 받아도 된다
    template <class PT>
    Out run_coef(const cv::Mat& coef, const std::vector<PT>& initial, const std::vector<cv::Mat>& pv) const {
        const size_t n = pts.size();
        const int K = (int)oy.size();
        std::vector<double> dx(n), dy(n), sx(n, 0), sy(n, 0);
        std::vector<std::array<double, 4>> pars(n);
        std::vector<std::vector<double>> vals(n, std::vector<double>(K));
        for (size_t i = 0; i < n; ++i) { dx[i] = initial[i].x; dy[i] = initial[i].y; }
        for (int it = 0; it < 12; ++it) {
            double mx = 0;
            for (size_t i = 0; i < n; ++i) {
                const double* P = pv[i].ptr<double>();
                std::array<double, 4> pr{0, 0, 0, 0};
                for (int k = 0; k < K; ++k) {
                    double v = spl::eval(coef, (double)pts[i].y + oy[k] + dy[i], (double)pts[i].x + ox[k] + dx[i]);
                    vals[i][k] = v;
                    for (int j = 0; j < 4; ++j) pr[j] += P[(size_t)j * K + k] * v;
                }
                pars[i] = pr;
                double g = std::max(pr[2], 0.25);
                double a = std::min(std::max(pr[0] / g, -0.5), 0.5), b = std::min(std::max(pr[1] / g, -0.5), 0.5);
                sx[i] = a; sy[i] = b;
                dx[i] -= a; dy[i] -= b;
                mx = std::max(mx, std::hypot(a, b));
            }
            if (mx < 2e-5) break;
        }
        Out o;
        for (size_t i = 0; i < n; ++i) {
            double s = 0;
            for (int k = 0; k < K; ++k) {
                double r = vals[i][k] - pars[i][2] * tpl[i][k] - pars[i][3];
                s += r * r;
            }
            o.d.push_back(cv::Point2f((float)dx[i], (float)dy[i]));
            o.rmse.push_back(std::sqrt(s / K));
            o.last.push_back(std::hypot(sx[i], sy[i]));
            o.gain.push_back(pars[i][2]);
            o.offset.push_back(pars[i][3]);
            double mv = std::hypot(dx[i] - initial[i].x, dy[i] - initial[i].y);
            o.valid.push_back(std::isfinite(dx[i]) && std::isfinite(dy[i]) && mv < 1.5 && o.last.back() < 0.02);
        }
        return o;
    }
};

// spread.py SpreadPairExtractor: QR 패널마다 따로 잘라 LK15 초기화 후 GN17
struct Spread {
    Features f;
    std::array<Refiner, 3> model;
    std::array<cv::Mat, 3> tplblur;
    std::array<std::vector<cv::Point2f>, 3> local;
    std::array<std::vector<int>, 3> idx;
    std::array<cv::Vec4i, 3> box;  // crop 좌표 x0,y0,x1,y1
    cv::Mat ref;

    Spread() = default;
    template <class WF = std::nullptr_t>
    Spread(const cv::Mat& ref_crop, const Features& feats, WF wf = nullptr) : f(feats), ref(ref_crop) {
        for (int k = 0; k < 3; ++k) {
            auto b = BOXES[k];
            box[k] = {b[0] - CX0, b[1] - CY0, b[2] - CX0, b[3] - CY0};
            cv::Mat t = ref(cv::Range(box[k][1], box[k][3]), cv::Range(box[k][0], box[k][2])).clone();
            for (size_t i = 0; i < f.pts.size(); ++i)
                if (f.region[i] == k) {
                    idx[k].push_back((int)i);
                    local[k].push_back(f.pts[i] - cv::Point2f((float)box[k][0], (float)box[k][1]));
                }
            model[k] = Refiner(t, local[k], 17, 0.8, wf);
            cv::GaussianBlur(t, tplblur[k], cv::Size(0, 0), 0.8);
        }
    }
    struct Out {
        std::vector<cv::Point2f> D;
        std::vector<char> valid;
        std::vector<double> rmse;
        std::array<cv::Point, 3> init;
    };
    Out extract(const cv::Mat& cur) const {
        Out o;
        o.D.assign(f.pts.size(), {0, 0});
        o.valid.assign(f.pts.size(), 0);
        o.rmse.assign(f.pts.size(), 0);
        o.init = coarse(ref, cur);
        for (int k = 0; k < 3; ++k) {
            int dx = o.init[k].x, dy = o.init[k].y;
            cv::Mat target = cur(cv::Range(box[k][1] + dy, box[k][3] + dy), cv::Range(box[k][0] + dx, box[k][2] + dx)).clone();
            cv::Mat tb;
            cv::GaussianBlur(target, tb, cv::Size(0, 0), 0.8);
            std::vector<cv::Point2f> zero(local[k].size(), {0, 0});
            Track lt = lk_track(tplblur[k], tb, local[k], zero, 15);
            auto g = model[k].run(target, lt.d);
            for (size_t j = 0; j < idx[k].size(); ++j) {
                int i = idx[k][j];
                o.D[i] = g.d[j] + cv::Point2f((float)dx, (float)dy);
                o.valid[i] = lt.valid[j] && g.valid[j];
                o.rmse[i] = g.rmse[j];
            }
        }
        return o;
    }
};

}  // namespace trk
