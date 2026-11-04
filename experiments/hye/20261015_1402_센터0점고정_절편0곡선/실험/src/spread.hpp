// codex gn17_spread14 측정과 6후보 0점 모델 (원본: codex_experiments_20261014/measurement/measure.py, refine.py, spread.py,
// zero_experiment.py, infer.py)
//   측정: QR 판 crop -> QR 세 상자 전체 NCC(±110x22) 정수 이동 -> LK(15px, σ0.8) -> GN17(3차 스플라인, 밝기 이득·오프셋)
//   요약: 점 이동을 판 좌표의 2차식(6항)으로 맞춘 12값
//   모델: 축마다 [v, v², 구간 원-핫] 최소제곱 -> 잔차 상관으로 백색화 W -> 후보마다 v 를 가우스-뉴턴으로 풀고 비용 최소 선택
#pragma once
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <string>
#include <vector>

namespace cx {

const int CROP[4] = {1340, 800, 2470, 1330};
const int BOXES[3][4] = {{1552, 969, 1736, 1167}, {1809, 971, 1993, 1169}, {2067, 973, 2252, 1171}};

inline cv::Mat crop(const cv::Mat& gray) { return gray(cv::Range(CROP[1], CROP[3]), cv::Range(CROP[0], CROP[2])).clone(); }

// ---- scipy.ndimage 3차 B-스플라인 ----
inline void spline_row(double* c, int n, int stride, bool reflect) {
    const double z = std::sqrt(3.0) - 2.0;
    const double gain = (1 - z) * (1 - 1 / z);
    for (int i = 0; i < n; ++i) c[i * stride] *= gain;
    if (reflect) {  // mode 'nearest' / 'reflect' 초기값
        double zi = z, zn = std::pow(z, n), c0 = c[0];
        double s = c[0] + zn * c[(n - 1) * stride];
        for (int i = 1; i < n; ++i) {
            s += zi * (c[i * stride] + zn * c[(n - 1 - i) * stride]);
            zi *= z;
        }
        s *= z / (1 - zn * zn);
        c[0] = s + c0;
    } else {  // mode 'mirror'
        double zi = z, zn1 = std::pow(z, n - 1);
        double s = c[0] + zn1 * c[(n - 1) * stride];
        for (int i = 1; i < n - 1; ++i) {
            s += zi * (c[i * stride] + zn1 * c[(n - 1 - i) * stride]);
            zi *= z;
        }
        c[0] = s / (1 - zn1 * zn1);
    }
    for (int i = 1; i < n; ++i) c[i * stride] += z * c[(i - 1) * stride];
    if (reflect) c[(n - 1) * stride] *= z / (z - 1);
    else c[(n - 1) * stride] = (z * c[(n - 2) * stride] + c[(n - 1) * stride]) * z / (z * z - 1);
    for (int i = n - 2; i >= 0; --i) c[i * stride] = z * (c[(i + 1) * stride] - c[i * stride]);
}
inline cv::Mat spline_filter(const cv::Mat& a, bool reflect) {
    cv::Mat c = a.clone();
    for (int x = 0; x < c.cols; ++x) spline_row(c.ptr<double>(0) + x, c.rows, (int)c.step1(), reflect);
    for (int y = 0; y < c.rows; ++y) spline_row(c.ptr<double>(y), c.cols, 1, reflect);
    return c;
}
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
        const double* p = c.ptr<double>(std::min(std::max(iy + a, 0), c.rows - 1));
        double t = 0;
        for (int b = 0; b < 4; ++b) t += wx[b] * p[std::min(std::max(ix + b, 0), c.cols - 1)];
        s += wy[a] * t;
    }
    return s;
}
// map_coordinates(order=3, mode='nearest', prefilter=True): 가장자리 12px 복제 후 사전필터
struct NearestSpline {
    cv::Mat c;
    explicit NearestSpline(const cv::Mat& a) {
        cv::Mat p;
        cv::copyMakeBorder(a, p, 12, 12, 12, 12, cv::BORDER_REPLICATE);
        c = spline_filter(p, true);
    }
    double at(double y, double x) const { return spline_at(c, y + 12, x + 12); }
};
inline cv::Mat np_gradient(const cv::Mat& a, int axis) {
    cv::Mat g(a.size(), CV_64F);
    for (int r = 0; r < a.rows; ++r)
        for (int c = 0; c < a.cols; ++c) {
            if (axis == 1) {
                int n = a.cols;
                g.at<double>(r, c) = c == 0 ? a.at<double>(r, 1) - a.at<double>(r, 0)
                                   : c == n - 1 ? a.at<double>(r, n - 1) - a.at<double>(r, n - 2)
                                                : (a.at<double>(r, c + 1) - a.at<double>(r, c - 1)) / 2;
            } else {
                int n = a.rows;
                g.at<double>(r, c) = r == 0 ? a.at<double>(1, c) - a.at<double>(0, c)
                                   : r == n - 1 ? a.at<double>(n - 1, c) - a.at<double>(n - 2, c)
                                                : (a.at<double>(r + 1, c) - a.at<double>(r - 1, c)) / 2;
            }
        }
    return g;
}

// ---- GN17 정밀화 (refine.Refiner) ----
struct Refiner {
    std::vector<cv::Point2f> pts;
    int window = 17;
    double sigma = 0.8;
    std::vector<cv::Mat> pinv;    // 점마다 4 x 289
    std::vector<cv::Mat> tmpl;    // 점마다 289 x 1
    std::vector<cv::Point2d> off; // 창 안 상대 좌표 (dx, dy)
    Refiner() = default;
    Refiner(const cv::Mat& ref8, const std::vector<cv::Point2f>& p, int win = 17, double sg = 0.8) : pts(p), window(win), sigma(sg) {
        cv::Mat r64, ref;
        ref8.convertTo(r64, CV_64F);
        cv::GaussianBlur(r64, ref, cv::Size(0, 0), sigma);
        int lo = -window / 2, hi = window / 2;  // mgrid[-8:9]
        for (int yy = lo; yy <= hi; ++yy)
            for (int xx = lo; xx <= hi; ++xx) off.push_back({(double)xx, (double)yy});
        NearestSpline sI(ref), sX(np_gradient(ref, 1)), sY(np_gradient(ref, 0));
        for (auto& q : pts) {
            int n = (int)off.size();
            cv::Mat D(n, 4, CV_64F), T(n, 1, CV_64F);
            for (int k = 0; k < n; ++k) {
                double y = (double)q.y + off[k].y, x = (double)q.x + off[k].x;
                double t = sI.at(y, x);
                D.at<double>(k, 0) = sX.at(y, x);
                D.at<double>(k, 1) = sY.at(y, x);
                D.at<double>(k, 2) = t;
                D.at<double>(k, 3) = 1;
                T.at<double>(k) = t;
            }
            cv::Mat P;
            cv::invert(D, P, cv::DECOMP_SVD);
            pinv.push_back(P);
            tmpl.push_back(T);
        }
    }
    // cur8: 현재 사진에서 같은 크기로 자른 상자, init: LK 결과 (float32)
    void run(const cv::Mat& cur8, const std::vector<cv::Point2f>& init, std::vector<cv::Point2f>& out, std::vector<std::array<float, 4>>& q,
             std::vector<bool>& valid) const {
        cv::Mat c64, cb;
        cur8.convertTo(c64, CV_64F);
        cv::GaussianBlur(c64, cb, cv::Size(0, 0), sigma);
        cv::Mat coeff = spline_filter(cb, false);
        size_t N = pts.size();
        int n = (int)off.size();
        std::vector<cv::Point2d> d(N), step(N);
        for (size_t i = 0; i < N; ++i) d[i] = {(double)init[i].x, (double)init[i].y};
        std::vector<cv::Mat> vals(N), pars(N);
        for (int it = 0; it < 12; ++it) {
            double mx = 0;
            for (size_t i = 0; i < N; ++i) {
                cv::Mat v(n, 1, CV_64F);
                for (int k = 0; k < n; ++k)
                    v.at<double>(k) = spline_at(coeff, (double)pts[i].y + off[k].y + d[i].y, (double)pts[i].x + off[k].x + d[i].x);
                vals[i] = v;
                pars[i] = pinv[i] * v;
                double g = std::max(pars[i].at<double>(2), 0.25);
                double sx = std::min(std::max(pars[i].at<double>(0) / g, -0.5), 0.5);
                double sy = std::min(std::max(pars[i].at<double>(1) / g, -0.5), 0.5);
                step[i] = {sx, sy};
                d[i] -= step[i];
                mx = std::max(mx, std::hypot(sx, sy));
            }
            if (mx < 2e-5) break;
        }
        out.resize(N);
        q.resize(N);
        valid.resize(N);
        for (size_t i = 0; i < N; ++i) {
            double a = pars[i].at<double>(2), b = pars[i].at<double>(3), ss = 0;
            for (int k = 0; k < n; ++k) {
                double r = vals[i].at<double>(k) - a * tmpl[i].at<double>(k) - b;
                ss += r * r;
            }
            out[i] = cv::Point2f((float)d[i].x, (float)d[i].y);
            double ls = std::hypot(step[i].x, step[i].y);
            q[i] = {(float)std::sqrt(ss / n), (float)ls, (float)a, (float)b};
            valid[i] = std::isfinite(d[i].x) && std::isfinite(d[i].y) && std::hypot(d[i].x - init[i].x, d[i].y - init[i].y) < 1.5 && ls < 0.02;
        }
    }
};

// ---- LK 왕복 (measure.track) ----
inline void lk_track(const cv::Mat& ref, const cv::Mat& cur, const std::vector<cv::Point2f>& p, int window, std::vector<cv::Point2f>& d,
                     std::vector<bool>& valid, std::vector<float>& fb) {
    cv::TermCriteria crit(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 60, 0.0001);
    std::vector<cv::Point2f> p1 = p, p2 = p;
    std::vector<uchar> s1, s2;
    std::vector<float> e1, e2;
    cv::calcOpticalFlowPyrLK(ref, cur, p, p1, s1, e1, cv::Size(window, window), 0, crit, cv::OPTFLOW_USE_INITIAL_FLOW, 1e-5);
    cv::calcOpticalFlowPyrLK(cur, ref, p1, p2, s2, e2, cv::Size(window, window), 0, crit, cv::OPTFLOW_USE_INITIAL_FLOW, 1e-5);
    d.resize(p.size());
    valid.resize(p.size());
    fb.resize(p.size());
    for (size_t i = 0; i < p.size(); ++i) {
        float fx = p2[i].x - p[i].x, fy = p2[i].y - p[i].y;
        fb[i] = std::sqrt(fx * fx + fy * fy);
        d[i] = p1[i] - p[i];
        valid[i] = s1[i] > 0 && s2[i] > 0 && fb[i] < 0.5f && std::isfinite(d[i].x) && std::isfinite(d[i].y);
    }
}

// ---- QR 상자 전체 NCC (measure.coarse, 나사 제외) ----
inline std::vector<cv::Point> coarse(const cv::Mat& ref, const cv::Mat& cur, std::vector<double>* score = nullptr) {
    std::vector<cv::Point> out;
    for (int k = 0; k < 3; ++k) {
        int x0 = BOXES[k][0] - CROP[0], y0 = BOXES[k][1] - CROP[1], x1 = BOXES[k][2] - CROP[0], y1 = BOXES[k][3] - CROP[1];
        const int sx = 110, sy = 22;
        cv::Mat t = ref(cv::Range(y0, y1), cv::Range(x0, x1)), s = cur(cv::Range(y0 - sy, y1 + sy), cv::Range(x0 - sx, x1 + sx)), r;
        cv::matchTemplate(s, t, r, cv::TM_CCOEFF_NORMED);
        double v;
        cv::Point pos;
        cv::minMaxLoc(r, nullptr, &v, nullptr, &pos);
        out.push_back({pos.x - sx, pos.y - sy});
        if (score) score->push_back(v);
    }
    return out;
}

// ---- 특징점 고르기 (spread.select) ----
struct Points {
    std::vector<cv::Point2f> p;  // crop 좌표
    std::vector<int> region;
    std::vector<float> strength;
};
inline Points select(const cv::Mat& ref, int min_distance) {
    cv::Mat sm, eig;
    cv::GaussianBlur(ref, sm, cv::Size(0, 0), 0.65);
    cv::cornerMinEigenVal(sm, eig, 5);
    Points out;
    for (int k = 0; k < 3; ++k) {
        int x0 = BOXES[k][0], y0 = BOXES[k][1], x1 = BOXES[k][2], y1 = BOXES[k][3];
        cv::Mat mask = cv::Mat::zeros(ref.size(), CV_8U);
        const int m = 16;
        mask(cv::Range(y0 + m - CROP[1], y1 - m - CROP[1]), cv::Range(x0 + m - CROP[0], x1 - m - CROP[0])) = 255;
        std::vector<cv::Point2f> p;
        cv::goodFeaturesToTrack(sm, p, 600, 0.02, min_distance, mask, 5);
        cv::cornerSubPix(sm, p, cv::Size(2, 2), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 30, 0.001));
        for (auto& q : p) {
            double gx = (double)q.x + CROP[0], gy = (double)q.y + CROP[1];
            if (!(x0 + 15 <= gx && gx < x1 - 15 && y0 + 15 <= gy && gy < y1 - 15)) continue;
            out.p.push_back(q);
            out.region.push_back(k);
            out.strength.push_back(eig.at<float>((int)std::nearbyint(q.y), (int)std::nearbyint(q.x)));
        }
    }
    return out;
}

// ---- 사진 한 쌍 추출기 (spread.SpreadPairExtractor) ----
struct Extractor {
    cv::Mat ref;
    Points P;
    std::vector<Refiner> models;
    std::vector<cv::Mat> blurred;
    std::vector<std::vector<cv::Point2f>> local;
    std::vector<std::vector<int>> index;
    std::vector<cv::Rect> boxes;
    explicit Extractor(const cv::Mat& ref_crop, int min_distance = 14) : ref(ref_crop) {
        P = select(ref, min_distance);
        for (int k = 0; k < 3; ++k) {
            int x0 = BOXES[k][0] - CROP[0], x1 = BOXES[k][2] - CROP[0], y0 = BOXES[k][1] - CROP[1], y1 = BOXES[k][3] - CROP[1];
            cv::Mat t = ref(cv::Range(y0, y1), cv::Range(x0, x1)).clone(), tb;
            std::vector<cv::Point2f> pts;
            std::vector<int> ix;
            for (size_t i = 0; i < P.p.size(); ++i)
                if (P.region[i] == k) { pts.push_back(P.p[i] - cv::Point2f((float)x0, (float)y0)); ix.push_back((int)i); }
            models.emplace_back(t, pts, 17, 0.8);
            cv::GaussianBlur(t, tb, cv::Size(0, 0), 0.8);
            blurred.push_back(tb);
            local.push_back(pts);
            index.push_back(ix);
            boxes.push_back(cv::Rect(x0, y0, x1 - x0, y1 - y0));
        }
    }
    struct Result {
        std::vector<cv::Point2f> D;
        std::vector<bool> valid;
        std::vector<std::array<float, 5>> quality;
        std::vector<cv::Point> init;
    };
    Result extract(const cv::Mat& cur) const {
        Result r;
        r.init = coarse(ref, cur);
        size_t N = P.p.size();
        r.D.assign(N, {0, 0});
        r.valid.assign(N, false);
        r.quality.assign(N, {});
        for (int k = 0; k < 3; ++k) {
            auto& b = boxes[k];
            int dx = r.init[k].x, dy = r.init[k].y;
            cv::Mat target = cur(cv::Rect(b.x + dx, b.y + dy, b.width, b.height)).clone(), tb;
            cv::GaussianBlur(target, tb, cv::Size(0, 0), 0.8);
            std::vector<cv::Point2f> d, gd;
            std::vector<bool> lv, gv;
            std::vector<float> fb;
            lk_track(blurred[k], tb, local[k], 15, d, lv, fb);
            std::vector<std::array<float, 4>> gq;
            models[k].run(target, d, gd, gq, gv);
            for (size_t j = 0; j < index[k].size(); ++j) {
                int i = index[k][j];
                r.D[i] = gd[j] + cv::Point2f((float)dx, (float)dy);
                r.quality[i] = {gq[j][0], gq[j][1], gq[j][2], gq[j][3], fb[j]};
                r.valid[i] = lv[j] && gv[j];
            }
        }
        return r;
    }
    cv::Point2d global(size_t i) const { return {(double)P.p[i].x + CROP[0], (double)P.p[i].y + CROP[1]}; }
};

// ---- 판 좌표 2차식 12값 (zero_experiment.load) ----
// pts: 전체 좌표, d: 점 이동. 반환 X[2k + c] (k = 1, x, y, x², xy, y² 계수, c = dx/dy)
inline std::vector<double> p12(const std::vector<cv::Point2d>& pts, const std::vector<cv::Point2d>& d, const std::vector<bool>* use = nullptr) {
    std::vector<int> ix;
    for (size_t i = 0; i < pts.size(); ++i) if (!use || (*use)[i]) ix.push_back((int)i);
    cv::Mat B((int)ix.size(), 6, CV_64F), Y((int)ix.size(), 2, CV_64F);
    for (int r = 0; r < (int)ix.size(); ++r) {
        double x = (pts[ix[r]].x - 1906) / 486, y = (pts[ix[r]].y - 1070) / 122;
        double row[6] = {1, x, y, x * x, x * y, y * y};
        for (int c = 0; c < 6; ++c) B.at<double>(r, c) = row[c];
        Y.at<double>(r, 0) = d[ix[r]].x;
        Y.at<double>(r, 1) = d[ix[r]].y;
    }
    cv::Mat C;
    cv::solve(B, Y, C, cv::DECOMP_SVD);
    std::vector<double> X(12);
    for (int k = 0; k < 6; ++k) { X[2 * k] = C.at<double>(k, 0); X[2 * k + 1] = C.at<double>(k, 1); }
    return X;
}

// ---- 사진 목록 (zero_experiment.manifest) ----
struct Item {
    std::string file;  // T+0.01.bmp
    char axis;         // 'C', 'T', 'Y'
    double value;
    std::string session;
};
inline std::string session_of(char axis, double v) {
    if (axis == 'C') return "C";
    if (axis == 'T') return v > 0 ? "T+" : "T-";
    return std::string((std::llround(std::abs(v / 0.05)) % 2) ? "Y2" : "Y1") + (v > 0 ? "+" : "-");
}

// ---- 0점 모델 (zero_experiment.fit / predict) ----
struct Branch {
    char axis;
    std::vector<double> base, s, c;
};
struct Model {
    std::vector<std::pair<std::string, Branch>> branches;  // 원래 dict 순서
    cv::Mat W;                                              // 12 x 12
};
// mode: "inferred_session" (구간 원-핫), "center" (절편 0), "free" (절편 하나)
inline Model fit(const std::vector<Item>& m, const std::vector<std::vector<double>>& X, const std::vector<int>& tr, const std::string& mode, bool quad) {
    Model out;
    const int P = 12;
    std::vector<cv::Mat> res;
    int rank = 0;
    bool use_s = mode == "inferred_session" || mode == "oracle_zero";
    for (char a : {'Y', 'T'}) {
        std::vector<int> ii;
        for (int i : tr) if (m[i].axis == a) ii.push_back(i);
        std::vector<std::string> ses;
        for (int i : ii) ses.push_back(m[i].session);
        std::sort(ses.begin(), ses.end());
        ses.erase(std::unique(ses.begin(), ses.end()), ses.end());
        int nb = quad ? 2 : 1, ncol = nb + (use_s ? (int)ses.size() : mode == "free" ? 1 : 0);
        cv::Mat A((int)ii.size(), ncol, CV_64F, cv::Scalar(0)), Z((int)ii.size(), P, CV_64F);
        for (int r = 0; r < (int)ii.size(); ++r) {
            double v = m[ii[r]].value;
            A.at<double>(r, 0) = v;
            if (quad) A.at<double>(r, 1) = v * v;
            if (use_s) {
                int j = (int)(std::find(ses.begin(), ses.end(), m[ii[r]].session) - ses.begin());
                A.at<double>(r, nb + j) = 1;
            } else if (mode == "free") A.at<double>(r, nb) = 1;
            for (int c = 0; c < P; ++c) Z.at<double>(r, c) = X[ii[r]][c];
        }
        cv::Mat co;
        cv::solve(A, Z, co, cv::DECOMP_SVD);
        cv::Mat w;
        cv::SVD::compute(A, w);
        double tol = w.at<double>(0) * std::max(A.rows, A.cols) * std::numeric_limits<double>::epsilon();
        for (int k = 0; k < w.rows; ++k) rank += w.at<double>(k) > tol;
        res.push_back(Z - A * co);
        auto row = [&](int k) { std::vector<double> v(P); for (int c = 0; c < P; ++c) v[c] = co.at<double>(k, c); return v; };
        std::vector<double> s = row(0), c = quad ? row(1) : std::vector<double>(P, 0.0);
        if (use_s)
            for (int j = 0; j < (int)ses.size(); ++j) out.branches.push_back({ses[j], {a, row(nb + j), s, c}});
        else out.branches.push_back({std::string(1, a), {a, mode == "free" ? row(ncol - 1) : std::vector<double>(P, 0.0), s, c}});
    }
    cv::Mat R;
    cv::vconcat(res, R);
    cv::Mat S = R.t() * R / std::max(R.rows - rank, 1);
    std::vector<double> sd(P);
    for (int i = 0; i < P; ++i) sd[i] = std::sqrt(std::max(S.at<double>(i, i), 1e-14));
    cv::Mat C(P, P, CV_64F);
    for (int i = 0; i < P; ++i)
        for (int j = 0; j < P; ++j) C.at<double>(i, j) = 0.999 * S.at<double>(i, j) / sd[i] / sd[j] + (i == j ? 0.001 : 0);
    cv::Mat Ci = C.inv(cv::DECOMP_LU);
    // np.linalg.cholesky(inv(C)) = L (아래 삼각). W = L^T / std (열마다)
    cv::Mat L = cv::Mat::zeros(P, P, CV_64F);
    for (int i = 0; i < P; ++i)
        for (int j = 0; j <= i; ++j) {
            double s = Ci.at<double>(i, j);
            for (int k = 0; k < j; ++k) s -= L.at<double>(i, k) * L.at<double>(j, k);
            L.at<double>(i, j) = i == j ? std::sqrt(s) : s / L.at<double>(j, j);
        }
    out.W = L.t();
    for (int i = 0; i < P; ++i)
        for (int j = 0; j < P; ++j) out.W.at<double>(i, j) /= sd[j];
    return out;
}

struct Pred {
    char axis;
    double value, cost, second_cost;  // second_cost: 다른 축 최소 비용
    std::string branch;
    std::vector<std::pair<std::string, std::pair<double, double>>> all;  // 후보별 (v, cost)
};
inline Pred predict_one(const std::vector<double>& x, const Model& md, bool sign_rule) {
    const int P = 12;
    Pred best{};
    best.cost = std::numeric_limits<double>::infinity();
    std::map<char, double> axis_best{{'Y', std::numeric_limits<double>::infinity()}, {'T', std::numeric_limits<double>::infinity()}};
    for (auto& [name, b] : md.branches) {
        std::vector<double> z(P, 0), sw(P, 0), cw(P, 0);
        for (int i = 0; i < P; ++i)
            for (int j = 0; j < P; ++j) {
                double w = md.W.at<double>(i, j);
                z[i] += w * (x[j] - b.base[j]);
                sw[i] += w * b.s[j];
                cw[i] += w * b.c[j];
            }
        double zs = 0, ss = 0;
        for (int i = 0; i < P; ++i) { zs += z[i] * sw[i]; ss += sw[i] * sw[i]; }
        double v = zs / ss;
        for (int it = 0; it < 8; ++it) {
            double jr = 0, jj = 0;
            for (int i = 0; i < P; ++i) {
                double J = sw[i] + 2 * v * cw[i], r = z[i] - v * sw[i] - v * v * cw[i];
                jr += J * r;
                jj += J * J;
            }
            v += jr / std::max(jj, 1e-20);
        }
        double cost = 0;
        for (int i = 0; i < P; ++i) { double r = z[i] - v * sw[i] - v * v * cw[i]; cost += r * r; }
        if (sign_rule && !(v * (name.back() == '+' ? 1 : -1) >= 0)) cost = std::numeric_limits<double>::infinity();
        best.all.push_back({name, {v, cost}});
        axis_best[b.axis] = std::min(axis_best[b.axis], cost);
        if (cost < best.cost) { best.cost = cost; best.value = v; best.axis = b.axis; best.branch = name; }
    }
    best.second_cost = axis_best[best.axis == 'Y' ? 'T' : 'Y'];
    return best;
}

}  // namespace cx
