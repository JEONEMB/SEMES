// 630 구조 모서리: center.bmp 에서 QR 해독 격자 → 모서리 위치 → erf 경계 모델 맞춤.
//   원래 경로: decoded_grid_corners_20261015/build.py (QR 해독, 격자 꼭짓점)
//            → subpixel_grid_corners_20261015/refine.py (9x9/11x11 경계 맞춤)
//            → analysis_grid_corners_20261019/track_corners.py (창 다시 맞추기 최대 3번, center 행 = 기준 좌표)
//   경계 맞춤 계산기는 motion_speed_630_20261022/compiled.py 의 해석미분 감쇠 LM 을 그대로 옮겼다.
//   (원래 center 기준 좌표는 SciPy least_squares TRF 로 맞췄다. 같은 목적함수의 다른 계산기다.)
#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace g09 {

const int BOXES[3][4] = {{1552, 969, 1736, 1167}, {1809, 971, 1993, 1169}, {2067, 973, 2252, 1171}};
const int CROP[4] = {1340, 800, 2470, 1330};

struct Corner {
    int id = 0, qr = 1, gx = 0, gy = 0, black = 0;
    double sx = 0, sy = 0;      // 격자 투영 위치 (structural_corners)
    double cx = 0, cy = 0;      // refine.py 맞춤 위치 (track_corners 시작점)
    double px = 0, py = 0;      // track_corners center 맞춤 위치 (기준 좌표)
    double inv[2][2] = {{1, 0}, {0, 1}};  // 국소 축 역행렬 (float32 로 계산)
    double patch[4] = {0, 0, 0, 0};       // 2x2 검정 배치 [00,01,10,11]
    bool ref_valid = true;
};

struct QrInfo {
    std::string payload;
    int modules = 0;
    std::array<cv::Point2f, 4> quad;
    int margin = 0, scale = 0;  // 0 = 전체 crop 다중 검출로 찾음
};

// ---- erf 경계 모델 + 해석미분 감쇠 LM (compiled.py objective / fit_compiled) ----
struct FitResult {
    double p[6];
    double rmse;
    bool ok;
};

inline double objective(const double* p, const std::vector<double>& xs, const std::vector<double>& ys, const std::vector<double>& I,
                        const double inv[2][2], const double* patch, std::vector<double>& res, std::vector<double>& J) {
    const size_t n = I.size();
    res.resize(n);
    J.resize(n * 6);
    double cost = 0;
    const double a = patch[0], du = patch[1] - a, dv = patch[2] - a, cross = patch[3] - patch[1] - patch[2] + a;
    const double s2 = std::sqrt(2.0), s2pi = std::sqrt(2 * M_PI);
    for (size_t i = 0; i < n; ++i) {
        double dx = xs[i] - p[0], dy = ys[i] - p[1];
        double u = dx * inv[0][0] + dy * inv[0][1], v = dx * inv[1][0] + dy * inv[1][1];
        double zu = u / p[2], zv = v / p[3];
        double U = .5 * (1 + std::erf(zu / s2)), V = .5 * (1 + std::erf(zv / s2));
        double F = patch[0] * (1 - U) * (1 - V) + patch[1] * U * (1 - V) + patch[2] * (1 - U) * V + patch[3] * U * V;
        double r = p[4] + p[5] * (1 - F) - I[i];
        res[i] = r;
        cost += 36 * (std::sqrt(1 + (r / 6) * (r / 6)) - 1);
        double gu = (du + cross * V) * std::exp(-.5 * zu * zu) / s2pi, gv = (dv + cross * U) * std::exp(-.5 * zv * zv) / s2pi;
        double* j = &J[i * 6];
        j[0] = p[5] * (gu / p[2] * inv[0][0] + gv / p[3] * inv[1][0]);
        j[1] = p[5] * (gu / p[2] * inv[0][1] + gv / p[3] * inv[1][1]);
        j[2] = p[5] * gu * u / (p[2] * p[2]);
        j[3] = p[5] * gv * v / (p[3] * p[3]);
        j[4] = 1.;
        j[5] = 1 - F;
    }
    return cost;
}

// 부분 피벗 가우스 소거 (LAPACK gesv 와 같은 방식)
template <int N>
inline bool solve_n(double A[N][N], double* b) {
    int piv[N];
    for (int k = 0; k < N; ++k) {
        int m = k;
        for (int i = k + 1; i < N; ++i) if (std::abs(A[i][k]) > std::abs(A[m][k])) m = i;
        piv[k] = m;
        if (A[m][k] == 0) return false;
        if (m != k) { for (int j = 0; j < N; ++j) std::swap(A[k][j], A[m][j]); std::swap(b[k], b[m]); }
        for (int i = k + 1; i < N; ++i) {
            double f = A[i][k] / A[k][k];
            A[i][k] = f;
            for (int j = k + 1; j < N; ++j) A[i][j] -= f * A[k][j];
            b[i] -= f * b[k];
        }
    }
    for (int i = N - 1; i >= 0; --i) {
        double s = b[i];
        for (int j = i + 1; j < N; ++j) s -= A[i][j] * b[j];
        b[i] = s / A[i][i];
    }
    (void)piv;
    return true;
}

inline FitResult fit_compiled(const std::vector<double>& xs, const std::vector<double>& ys, const std::vector<double>& I, double sx0, double sy0,
                              const double inv[2][2], const double* patch) {
    const int n = (int)I.size();
    std::vector<double> ordered(I);
    std::sort(ordered.begin(), ordered.end());
    auto pct = [&](double q) {
        q = q * (n - 1);
        int k = (int)q;
        return ordered[k] * (1 - (q - k)) + ordered[std::min(k + 1, n - 1)] * (q - k);
    };
    double low = pct(.1), high = pct(.9);
    double p[6] = {sx0, sy0, 1., 1., low, std::max(high - low, 20.)};
    const double lower[6] = {sx0 - 3.8, sy0 - 3.8, .35, .35, 0., 10.}, upper[6] = {sx0 + 3.8, sy0 + 3.8, 2.7, 2.7, 240., 255.};
    const double scales[6] = {1., 1., 1., 1., 60., 120.};
    double mu = 1e-3;
    bool converged = false;
    std::vector<double> r, J, nr, nJ;
    double cost = objective(p, xs, ys, I, inv, patch, r, J);
    int evaluations = 1;
    for (int iteration = 0; iteration < 85; ++iteration) {
        double H[6][6] = {}, g[6] = {};
        for (int i = 0; i < n; ++i) {
            double root = std::sqrt(1 + (r[i] / 6) * (r[i] / 6)), w = 1 / (root * root * root);
            for (int j = 0; j < 6; ++j) {
                double jj = J[i * 6 + j] * scales[j];
                g[j] += jj * r[i] / root;
                for (int k = 0; k < 6; ++k) H[j][k] += jj * J[i * 6 + k] * scales[k] * w;
            }
        }
        double pgmax = 0;
        for (int j = 0; j < 6; ++j) {
            double pg = g[j];
            if ((p[j] - lower[j] < 1e-9 && g[j] > 0) || (upper[j] - p[j] < 1e-9 && g[j] < 0)) pg = 0;
            pgmax = std::max(pgmax, std::abs(pg));
        }
        if (pgmax < 1e-7) { converged = true; break; }
        for (int j = 0; j < 6; ++j)
            if ((p[j] - lower[j] < 1e-8 && g[j] > 0) || (upper[j] - p[j] < 1e-8 && g[j] < 0)) {
                for (int k = 0; k < 6; ++k) { H[j][k] = 0; H[k][j] = 0; }
                H[j][j] = 1.;
                g[j] = 0.;
            }
        double diagonal[6];
        for (int j = 0; j < 6; ++j) diagonal[j] = std::max(H[j][j], 1e-6);
        bool accepted = false;
        for (int retry = 0; retry < 14; ++retry) {
            double A[6][6], delta[6];
            for (int j = 0; j < 6; ++j) { for (int k = 0; k < 6; ++k) A[j][k] = H[j][k]; delta[j] = -g[j]; }
            for (int j = 0; j < 6; ++j) A[j][j] += mu * diagonal[j];
            solve_n<6>(A, delta);
            double trial[6], dn = 0, pn = 0;
            for (int j = 0; j < 6; ++j) {
                trial[j] = std::min(std::max(p[j] + delta[j] * scales[j], lower[j] + 1e-12), upper[j] - 1e-12);
                dn += (trial[j] - p[j]) * (trial[j] - p[j]);
                pn += p[j] * p[j];
            }
            dn = std::sqrt(dn);
            pn = std::sqrt(pn);
            double newcost = objective(trial, xs, ys, I, inv, patch, nr, nJ);
            ++evaluations;
            if (dn < 1e-7 * (1e-7 + pn)) {
                converged = true;
                if (newcost < cost) { std::copy(trial, trial + 6, p); cost = newcost; r.swap(nr); J.swap(nJ); }
                accepted = true;
                break;
            }
            if (newcost < cost) {
                double improvement = cost - newcost;
                std::copy(trial, trial + 6, p);
                cost = newcost;
                r.swap(nr);
                J.swap(nJ);
                mu = std::max(mu * .3, 1e-12);
                accepted = true;
                if (improvement < 1e-7 * std::max(cost, 1e-12) || dn < 1e-7 * (1e-7 + pn)) converged = true;
                break;
            }
            mu = std::min(mu * 10, 1e14);
            if (evaluations >= 85) break;
        }
        if (converged || evaluations >= 85 || !accepted) break;
    }
    double ss = 0;
    for (double v : r) ss += v * v;
    FitResult out;
    std::copy(p, p + 6, out.p);
    out.rmse = std::sqrt(ss / n);
    out.ok = converged;
    return out;
}

// np.rint (짝수 반올림)
inline int rint_i(double v) { return (int)std::nearbyint(v); }

// 창: 중심 rint, 반경 r, 상자 [x0,x1)x[y0,y1) 안 픽셀만
inline void window(const cv::Mat& img, double ex, double ey, int radius, const int box[4], std::vector<double>& xs, std::vector<double>& ys,
                   std::vector<double>& I) {
    xs.clear(); ys.clear(); I.clear();
    int x0 = rint_i(ex), y0 = rint_i(ey);
    for (int y = y0 - radius; y <= y0 + radius; ++y)
        for (int x = x0 - radius; x <= x0 + radius; ++x)
            if (box[0] <= x && x < box[2] && box[1] <= y && y < box[3]) {
                xs.push_back(x); ys.push_back(y); I.push_back(img.at<uchar>(y, x));
            }
}

struct CornerFit {
    double x = 0, y = 0;
    double rmse = 0, stability = 0, contrast = 0, sigx = 0, sigy = 0;
    bool valid = false;
    unsigned flags = 0;  // bit0 optimizer, 1 rmse, 2 contrast, 3 window, 4 search, 5 blur
};

// track_corners.run / compiled.run_corners 의 한 점
inline CornerFit fit_corner(const cv::Mat& img, double startx, double starty, const int box[4], const double inv[2][2], const double* patch) {
    std::vector<double> xs, ys, I;
    double ex = startx, ey = starty;
    FitResult f1{};
    for (int it = 0; it < 3; ++it) {
        window(img, ex, ey, 4, box, xs, ys, I);
        f1 = fit_compiled(xs, ys, I, startx, starty, inv, patch);
        bool moved = rint_i(f1.p[0]) != rint_i(ex) || rint_i(f1.p[1]) != rint_i(ey);
        ex = f1.p[0];
        ey = f1.p[1];
        if (!moved) break;
    }
    window(img, ex, ey, 5, box, xs, ys, I);
    FitResult f2 = fit_compiled(xs, ys, I, startx, starty, inv, patch);
    CornerFit c;
    c.x = f1.p[0];
    c.y = f1.p[1];
    c.rmse = f1.rmse;
    c.stability = std::hypot(f1.p[0] - f2.p[0], f1.p[1] - f2.p[1]);
    c.contrast = f1.p[5];
    c.sigx = f1.p[2];
    c.sigy = f1.p[3];
    if (!(f1.ok && f2.ok)) c.flags |= 1;
    if (c.rmse > 15) c.flags |= 2;
    if (c.contrast < 45) c.flags |= 4;
    if (c.stability > .65) c.flags |= 8;
    if (std::max(std::abs(c.x - startx), std::abs(c.y - starty)) > 3.6) c.flags |= 16;
    if (std::max(c.sigx, c.sigy) > 2.6) c.flags |= 32;
    c.valid = c.flags == 0;
    return c;
}

// ---- center 에서 630점 만들기 ----
struct Reference {
    std::vector<QrInfo> qr;
    std::vector<Corner> all;      // 755 구조 꼭짓점 (대각 접촉 포함)
    std::vector<Corner> c630;     // 대각 접촉(검정 2개) 제외 630점, id 순
};

inline Reference build_reference(const cv::Mat& center) {
    Reference R;
    cv::Mat crop = center(cv::Rect(CROP[0], CROP[1], CROP[2] - CROP[0], CROP[3] - CROP[1])).clone();
    cv::QRCodeDetector det;
    std::vector<cv::String> texts;
    std::vector<cv::Point2f> quads;
    std::vector<cv::Mat> grids;
    struct Found { bool ok = false; int margin = 0, scale = 0; std::string t; std::array<cv::Point2f, 4> q; cv::Mat g; };
    Found found[3];
    if (det.detectAndDecodeMulti(crop, texts, quads, grids)) {
        for (size_t i = 0; i < texts.size(); ++i) {
            if (texts[i].empty() || i >= grids.size() || grids[i].empty()) continue;
            double gx = 0, gy = 0;
            for (int j = 0; j < 4; ++j) { gx += quads[i * 4 + j].x / 4; gy += quads[i * 4 + j].y / 4; }
            gx += CROP[0]; gy += CROP[1];
            for (int k = 0; k < 3; ++k)
                if (BOXES[k][0] < gx && gx < BOXES[k][2] && BOXES[k][1] < gy && gy < BOXES[k][3]) {
                    found[k].ok = true; found[k].t = texts[i]; found[k].g = grids[i].clone();
                    for (int j = 0; j < 4; ++j) found[k].q[j] = quads[i * 4 + j] + cv::Point2f((float)CROP[0], (float)CROP[1]);
                }
        }
    }
    for (int k = 0; k < 3; ++k) {
        if (found[k].ok) continue;
        // 원래는 여백 5, 배율 1..4. OpenCV 3.4 검출기는 QR3 를 그 조건에서 못 찾아 여백을 15, 30 으로 넓혀 이어서 찾는다.
        for (int M : {5, 15, 30}) {
        if (found[k].ok) break;
        int x0 = BOXES[k][0] - CROP[0] - M, y0 = BOXES[k][1] - CROP[1] - M;
        cv::Mat sub = crop(cv::Rect(x0, y0, BOXES[k][2] - CROP[0] + M - x0, BOXES[k][3] - CROP[1] + M - y0));
        for (int s = 1; s <= 4; ++s) {
            cv::Mat big, pts, g;
            cv::resize(sub, big, cv::Size(), s, s);
            std::string t = det.detectAndDecode(big, pts, g);
            if (!t.empty()) {
                found[k].ok = true; found[k].t = t; found[k].g = g.clone();
                pts = pts.reshape(2, 4);
                for (int j = 0; j < 4; ++j) {
                    cv::Point2f q = pts.at<cv::Point2f>(j);
                    found[k].q[j] = cv::Point2f(q.x / s + x0 + CROP[0], q.y / s + y0 + CROP[1]);
                }
                found[k].margin = M;
                found[k].scale = s;
                break;
            }
        }
        }
        if (!found[k].ok) throw std::runtime_error("QR 해독 실패: " + std::to_string(k + 1));
    }
    int id = 0;
    for (int k = 0; k < 3; ++k) {
        const cv::Mat& g = found[k].g;
        int n = g.rows;
        QrInfo info;
        info.payload = found[k].t;
        info.modules = n;
        info.quad = found[k].q;
        info.margin = found[k].margin;
        info.scale = found[k].scale;
        R.qr.push_back(info);
        auto blackat = [&](int y, int x) -> int {  // np.pad(black,1) 의 [y][x]
            if (y < 1 || x < 1 || y > n || x > n) return 0;
            return g.at<uchar>(y - 1, x - 1) == 0 ? 1 : 0;
        };
        cv::Point2f src[4] = {{0, 0}, {(float)n, 0}, {(float)n, (float)n}, {0, (float)n}};
        cv::Point2f dst[4] = {found[k].q[0], found[k].q[1], found[k].q[2], found[k].q[3]};
        cv::Mat H = cv::getPerspectiveTransform(src, dst);
        for (int y = 0; y <= n; ++y)
            for (int x = 0; x <= n; ++x) {
                int p00 = blackat(y, x), p01 = blackat(y, x + 1), p10 = blackat(y + 1, x), p11 = blackat(y + 1, x + 1);
                int count = p00 + p01 + p10 + p11;
                if (!(count == 1 || count == 3 || (count == 2 && p00 == p11))) continue;
                Corner c;
                c.id = id++;
                c.qr = k + 1;
                c.gx = x;
                c.gy = y;
                c.black = count;
                c.patch[0] = p00; c.patch[1] = p01; c.patch[2] = p10; c.patch[3] = p11;
                std::vector<cv::Point2f> in = {{(float)x, (float)y}, {(float)x + .1f, (float)y}, {(float)x, (float)y + .1f}}, outp;
                cv::perspectiveTransform(in, outp, H);
                c.sx = outp[0].x;
                c.sy = outp[0].y;
                float ax[2][2] = {{outp[1].x - outp[0].x, outp[2].x - outp[0].x}, {outp[1].y - outp[0].y, outp[2].y - outp[0].y}};
                for (int col = 0; col < 2; ++col) {
                    float nn = std::sqrt(ax[0][col] * ax[0][col] + ax[1][col] * ax[1][col]);
                    ax[0][col] /= nn; ax[1][col] /= nn;
                }
                float det2 = ax[0][0] * ax[1][1] - ax[0][1] * ax[1][0];
                c.inv[0][0] = (float)(ax[1][1] / det2); c.inv[0][1] = (float)(-ax[0][1] / det2);
                c.inv[1][0] = (float)(-ax[1][0] / det2); c.inv[1][1] = (float)(ax[0][0] / det2);
                R.all.push_back(c);
            }
    }
    // refine.py: 투영 위치에서 9x9 (와 11x11) 맞춤, 9x9 결과를 시작점으로
#pragma omp parallel for schedule(dynamic, 8)
    for (int i = 0; i < (int)R.all.size(); ++i) {
        Corner& c = R.all[i];
        std::vector<double> xs, ys, I;
        window(center, c.sx, c.sy, 4, BOXES[c.qr - 1], xs, ys, I);
        FitResult f = fit_compiled(xs, ys, I, c.sx, c.sy, c.inv, c.patch);
        c.cx = f.p[0];
        c.cy = f.p[1];
    }
    // track_corners.py center 행: 이동 0, 창 다시 맞추기
#pragma omp parallel for schedule(dynamic, 8)
    for (int i = 0; i < (int)R.all.size(); ++i) {
        Corner& c = R.all[i];
        CornerFit f = fit_corner(center, c.cx, c.cy, BOXES[c.qr - 1], c.inv, c.patch);
        c.px = f.x;
        c.py = f.y;
        c.ref_valid = f.valid;
    }
    for (auto& c : R.all) if (c.black != 2) R.c630.push_back(c);
    return R;
}

// QR 상자별 정수 이동 (TM_CCOEFF_NORMED, 원래 좌표). pad 130/260/512
inline void ncc_shift(const cv::Mat& img, const cv::Mat& center, int pad, int shift[3][2], double score[3]) {
    for (int k = 0; k < 3; ++k) {
        const int* b = BOXES[k];
        cv::Mat tpl = center(cv::Rect(b[0], b[1], b[2] - b[0], b[3] - b[1]));
        cv::Mat search = img(cv::Rect(b[0] - pad, b[1] - pad, b[2] - b[0] + 2 * pad, b[3] - b[1] + 2 * pad));
        cv::Mat m;
        cv::matchTemplate(search, tpl, m, cv::TM_CCOEFF_NORMED);
        double mx;
        cv::Point loc;
        cv::minMaxLoc(m, nullptr, &mx, nullptr, &loc);
        shift[k][0] = loc.x - pad;
        shift[k][1] = loc.y - pad;
        score[k] = mx;
    }
}

}  // namespace g09
