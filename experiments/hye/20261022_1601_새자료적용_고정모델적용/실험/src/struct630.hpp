// 고정 630점 구조 모서리 측정 (원본 세 단계를 한 파일로 옮김)
//   1. decoded_grid_corners_20261015/build.py : center.bmp 의 QR 3개를 디코딩해 21x21 모듈 격자와 사각형(quad)을 얻고,
//      격자 꼭짓점 중 검정 이웃이 1·3칸 또는 대각선 2칸인 755점을 구조 모서리로 정한다.
//   2. subpixel_grid_corners_20261015/refine.py : center 에서 흐린 경계 모델로 755점을 서브픽셀로 맞춘다.
//   3. analysis_grid_corners_20261019/track_corners.py : 사진마다 QR 템플릿 매칭(±130px)으로 대략 이동을 구하고
//      9x9 창(최대 3번 다시 맞춤) + 11x11 확인 창으로 같은 모델을 맞춘다.
//   이 실험들은 대각선(검정 2칸이 꼭짓점만 맞닿음) 125점을 뺀 630점만 쓰므로 630점만 추적한다.
// 밝기 모델: I = low + contrast (1 - B),  B = 2x2 검정 배치의 쌍선형 혼합, U = Φ(local_u/σx), V = Φ(local_v/σy),
//   local = inv(axes) (xy - c), axes = QR 사각형 호모그래피로 구한 격자 방향 단위벡터.
// 최적화: scipy least_squares(TRF, soft_l1 f=6, 범위, max_nfev 85, tol 1e-7) 대신 같은 비용의 범위 제한 LM.
//   수렴점은 같고 종료 지점 차이(1e-4 px 수준)만 남는다.
#pragma once
#include <hye/report.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <array>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace s630 {

const std::array<cv::Vec4i, 3> kBoxes{{{1552, 969, 1736, 1167}, {1809, 971, 1993, 1169}, {2067, 973, 2252, 1171}}};  // x0,y0,x1,y1
const cv::Vec4i kCrop{1340, 800, 2470, 1330};

inline cv::Mat read_gray(const std::string& path) {
    std::ifstream in(hye::fs::u8path(path), std::ios::binary);
    std::vector<uchar> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    cv::Mat img = cv::imdecode(bytes, cv::IMREAD_GRAYSCALE);
    if (img.empty()) throw std::runtime_error("읽기 실패: " + path);
    return img;
}

struct QrInfo {
    int qr = 0, modules = 0;
    std::string payload;
    std::array<cv::Point2f, 4> quad;
    cv::Mat black;  // (n+2)x(n+2) 0/1, 바깥 1칸 흰색 패딩
    std::string note = "multi";  // 어떻게 찾았는지
};

struct Corner {
    int id = 0, qr = 0, gx = 0, gy = 0, nblack = 0;
    double ix = 0, iy = 0;  // 격자에서 투영한 처음 위치
    double cx = 0, cy = 0;  // center 에서 맞춘 위치
    double ainv[4] = {1, 0, 0, 1};
    double patch[4] = {0, 0, 0, 0};  // TL, TR, BL, BR 검정 여부
    bool center_valid = false;
};

// ---- 1. QR 디코딩과 구조 모서리 ----
inline std::vector<QrInfo> decode_qr(const cv::Mat& full) {
    cv::Mat ref = full(cv::Range(kCrop[1], kCrop[3]), cv::Range(kCrop[0], kCrop[2])).clone();
    cv::QRCodeDetector det;
    std::vector<std::string> text;
    std::vector<cv::Point2f> pts;
    std::vector<cv::Mat> grids;
    std::vector<QrInfo> found(3);
    std::vector<bool> have(3, false);
    if (det.detectAndDecodeMulti(ref, text, pts, grids)) {
        for (size_t i = 0; i < text.size(); ++i) {
            if (text[i].empty()) continue;
            cv::Point2f c(0, 0);
            for (int k = 0; k < 4; ++k) c += pts[i * 4 + k];
            c *= 0.25f;
            const double gx = c.x + kCrop[0], gy = c.y + kCrop[1];
            for (int k = 0; k < 3; ++k) {
                auto b = kBoxes[k];
                if (b[0] < gx && gx < b[2] && b[1] < gy && gy < b[3] && !have[k]) {
                    have[k] = true;
                    found[k].payload = text[i];
                    for (int j = 0; j < 4; ++j) found[k].quad[j] = pts[i * 4 + j] + cv::Point2f((float)kCrop[0], (float)kCrop[1]);
                    found[k].black = grids[i].clone();
                }
            }
        }
    }
    // 원래(cv2 5.0)는 여백 5px 확대 1~4배에서 QR3 을 찾았다. OpenCV 3.4 는 여백 5px 에서 못 찾아 여백을 15, 30px 로 넓혀 다시 찾는다.
    for (int k = 0; k < 3; ++k) {
        if (have[k]) continue;
        auto b = kBoxes[k];
        for (int m : {5, 15, 30}) {
            if (have[k]) break;
            const int x0 = b[0] - kCrop[0] - m, y0 = b[1] - kCrop[1] - m;
            cv::Mat crop = ref(cv::Range(y0, b[3] - kCrop[1] + m), cv::Range(x0, b[2] - kCrop[0] + m));
            for (int s = 1; s <= 4 && !have[k]; ++s) {
                cv::Mat big, grid;
                cv::resize(crop, big, cv::Size(), s, s);
                std::vector<cv::Point2f> q;
                std::string t = det.detectAndDecode(big, q, grid);
                if (!t.empty() && q.size() == 4) {
                    have[k] = true;
                    found[k].payload = t;
                    found[k].note = "margin " + std::to_string(m) + "px, scale " + std::to_string(s);
                    for (int j = 0; j < 4; ++j) found[k].quad[j] = q[j] / (float)s + cv::Point2f((float)(x0 + kCrop[0]), (float)(y0 + kCrop[1]));
                    found[k].black = grid.clone();
                }
            }
        }
        if (!have[k]) throw std::runtime_error("QR 디코딩 실패 " + std::to_string(k + 1));
    }
    for (int k = 0; k < 3; ++k) {
        auto& f = found[k];
        f.qr = k + 1;
        cv::Mat g = f.black;
        f.modules = g.rows;
        cv::Mat b = cv::Mat::zeros(g.rows + 2, g.cols + 2, CV_8U);
        for (int y = 0; y < g.rows; ++y)
            for (int x = 0; x < g.cols; ++x) b.at<uchar>(y + 1, x + 1) = g.at<uchar>(y, x) == 0 ? 1 : 0;
        f.black = b;
    }
    return found;
}

// 모든 구조 모서리(755). 대각선 포함, id = 만든 순서
inline std::vector<Corner> structural_corners(const std::vector<QrInfo>& qrs) {
    std::vector<Corner> out;
    for (auto& f : qrs) {
        const int n = f.modules;
        std::vector<cv::Point2f> src{{0, 0}, {(float)n, 0}, {(float)n, (float)n}, {0, (float)n}};
        std::vector<cv::Point2f> dst(f.quad.begin(), f.quad.end());
        cv::Mat H = cv::getPerspectiveTransform(src, dst);
        for (int y = 0; y <= n; ++y)
            for (int x = 0; x <= n; ++x) {
                const int tl = f.black.at<uchar>(y, x), tr = f.black.at<uchar>(y, x + 1), bl = f.black.at<uchar>(y + 1, x), br = f.black.at<uchar>(y + 1, x + 1);
                const int count = tl + tr + bl + br;
                if (!(count == 1 || count == 3 || (count == 2 && tl == br))) continue;
                Corner c;
                c.id = (int)out.size();
                c.qr = f.qr;
                c.gx = x;
                c.gy = y;
                c.nblack = count;
                c.patch[0] = tl; c.patch[1] = tr; c.patch[2] = bl; c.patch[3] = br;
                std::vector<cv::Point2f> in{{(float)x, (float)y}, {x + 0.1f, (float)y}, {(float)x, y + 0.1f}}, pr;
                cv::perspectiveTransform(in, pr, H);
                c.ix = pr[0].x;
                c.iy = pr[0].y;
                // axes 열 = 격자 u, v 방향 단위벡터 (float32, 원래와 같게)
                float a00 = pr[1].x - pr[0].x, a10 = pr[1].y - pr[0].y, a01 = pr[2].x - pr[0].x, a11 = pr[2].y - pr[0].y;
                float n0 = std::sqrt(a00 * a00 + a10 * a10), n1 = std::sqrt(a01 * a01 + a11 * a11);
                a00 /= n0; a10 /= n0; a01 /= n1; a11 /= n1;
                const double det = (double)a00 * a11 - (double)a01 * a10;
                c.ainv[0] = a11 / det; c.ainv[1] = -a01 / det; c.ainv[2] = -a10 / det; c.ainv[3] = a00 / det;
                out.push_back(c);
            }
    }
    return out;
}

// ---- 2·3. 흐린 경계 모델 맞춤 ----
struct Fit {
    double p[6];
    double rmse;
    bool ok;
};

inline double pct(std::vector<double> v, double q) {  // numpy 선형 보간 백분위
    std::sort(v.begin(), v.end());
    double k = (v.size() - 1) * q;
    size_t i = (size_t)std::floor(k);
    double f = k - i;
    return i + 1 < v.size() ? v[i] + (v[i + 1] - v[i]) * f : v[i];
}

inline Fit fit_model(const std::vector<double>& X, const std::vector<double>& Y, const std::vector<double>& I, double sx, double sy, const Corner& c) {
    const int n = (int)I.size();
    Fit fr{};
    if (n < 7) { fr.ok = false; fr.rmse = hye::kNaN; for (auto& v : fr.p) v = hye::kNaN; return fr; }
    const double low = pct(I, 0.10), high = pct(I, 0.90);
    double p[6] = {sx, sy, 1.0, 1.0, low, std::max(high - low, 20.0)};
    const double lo[6] = {sx - 3.8, sy - 3.8, 0.35, 0.35, 0, 10};
    const double hi[6] = {sx + 3.8, sy + 3.8, 2.7, 2.7, 240, 255};
    for (int a = 0; a < 6; ++a) p[a] = std::min(hi[a], std::max(lo[a], p[a]));
    const double tl = c.patch[0], tr = c.patch[1], bl = c.patch[2], br = c.patch[3];
    const double A00 = c.ainv[0], A01 = c.ainv[1], A10 = c.ainv[2], A11 = c.ainv[3];
    const double fS = 6.0, kInvSqrt2 = 0.70710678118654752, kInvSqrt2Pi = 0.39894228040143268;
    auto resid = [&](const double* q, std::vector<double>& r, cv::Mat* J) {
        for (int k = 0; k < n; ++k) {
            const double ex = X[k] - q[0], ey = Y[k] - q[1];
            const double a = A00 * ex + A01 * ey, b = A10 * ex + A11 * ey;
            const double U = 0.5 * (1 + std::erf(a / q[2] * kInvSqrt2)), V = 0.5 * (1 + std::erf(b / q[3] * kInvSqrt2));
            const double B = tl * (1 - U) * (1 - V) + tr * U * (1 - V) + bl * (1 - U) * V + br * U * V;
            r[k] = q[4] + q[5] * (1 - B) - I[k];
            if (!J) continue;
            const double gU = kInvSqrt2Pi / q[2] * std::exp(-0.5 * a * a / (q[2] * q[2]));
            const double gV = kInvSqrt2Pi / q[3] * std::exp(-0.5 * b * b / (q[3] * q[3]));
            const double dBdU = (tr - tl) * (1 - V) + (br - bl) * V, dBdV = (bl - tl) * (1 - U) + (br - tr) * U;
            double* row = J->ptr<double>(k);
            row[0] = q[5] * (dBdU * gU * A00 + dBdV * gV * A10);
            row[1] = q[5] * (dBdU * gU * A01 + dBdV * gV * A11);
            row[2] = q[5] * dBdU * gU * a / q[2];
            row[3] = q[5] * dBdV * gV * b / q[3];
            row[4] = 1;
            row[5] = 1 - B;
        }
    };
    auto cost = [&](const std::vector<double>& r) {
        double s = 0;
        for (double v : r) s += 2 * fS * fS * (std::sqrt(1 + (v / fS) * (v / fS)) - 1);
        return s;
    };
    std::vector<double> r(n), rn(n);
    cv::Mat J(n, 6, CV_64F);
    resid(p, r, &J);
    int nfev = 1;
    double lambda = 1e-3, cur = cost(r);
    bool conv = false;
    while (!conv && nfev < 85) {
        cv::Matx66d A = cv::Matx66d::zeros();
        cv::Vec6d g = cv::Vec6d::all(0);
        for (int k = 0; k < n; ++k) {
            const double w = 1.0 / std::sqrt(1 + (r[k] / fS) * (r[k] / fS));
            const double* row = J.ptr<double>(k);
            for (int a = 0; a < 6; ++a) {
                g[a] += w * row[a] * r[k];
                for (int b = a; b < 6; ++b) A(a, b) += w * row[a] * row[b];
            }
        }
        for (int a = 0; a < 6; ++a) for (int b = 0; b < a; ++b) A(a, b) = A(b, a);
        double gmax = 0;
        for (int a = 0; a < 6; ++a) {
            const bool atLo = p[a] <= lo[a] + 1e-12 && g[a] > 0, atHi = p[a] >= hi[a] - 1e-12 && g[a] < 0;
            if (atLo || atHi) {
                for (int b = 0; b < 6; ++b) { A(a, b) = 0; A(b, a) = 0; }
                A(a, a) = 1;
                g[a] = 0;
            }
            gmax = std::max(gmax, std::abs(g[a]));
        }
        if (gmax < 1e-10) { conv = true; break; }
        bool improved = false;
        while (!improved && nfev < 85) {
            cv::Matx66d Ad = A;
            for (int a = 0; a < 6; ++a) Ad(a, a) *= (1 + lambda);
            cv::Vec6d step;
            cv::solve(Ad, -g, step, cv::DECOMP_SVD);
            double q[6], ch2 = 0, nrm2 = 0;
            for (int a = 0; a < 6; ++a) {
                q[a] = std::min(hi[a], std::max(lo[a], p[a] + step[a]));
                ch2 += (q[a] - p[a]) * (q[a] - p[a]);
                nrm2 += p[a] * p[a];
            }
            resid(q, rn, nullptr);
            ++nfev;
            const double nc = cost(rn);
            if (nc <= cur) {
                improved = true;
                const double rel = (cur - nc) / std::max(cur, 1e-300);
                std::copy(q, q + 6, p);
                cur = nc;
                lambda = std::max(lambda * 0.3, 1e-9);
                resid(p, r, &J);
                if (std::sqrt(ch2) < 1e-7 * (1e-7 + std::sqrt(nrm2)) || rel < 1e-7) conv = true;
            } else {
                lambda *= 10;
                if (lambda > 1e12) { conv = true; break; }
            }
        }
    }
    std::copy(p, p + 6, fr.p);
    double ss = 0;
    for (double v : r) ss += v * v;
    fr.rmse = std::sqrt(ss / n);
    fr.ok = conv;
    return fr;
}

// 창: rint(center) 중심 (2r+1)^2, box(x0,y0,x1,y1) 안 픽셀만
inline Fit fit_at(const cv::Mat& img, cv::Point2d center, int radius, cv::Vec4d box, double sx, double sy, const Corner& c) {
    const int x0 = (int)std::nearbyint(center.x), y0 = (int)std::nearbyint(center.y);
    std::vector<double> X, Y, I;
    for (int y = y0 - radius; y <= y0 + radius; ++y)
        for (int x = x0 - radius; x <= x0 + radius; ++x) {
            if (!(x >= box[0] && x < box[2] && y >= box[1] && y < box[3])) continue;
            if (x < 0 || y < 0 || x >= img.cols || y >= img.rows) continue;
            X.push_back(x); Y.push_back(y); I.push_back(img.at<uchar>(y, x));
        }
    return fit_model(X, Y, I, sx, sy, c);
}

struct Obs {
    double x = hye::kNaN, y = hye::kNaN, rmse = hye::kNaN, stab = hye::kNaN, contrast = hye::kNaN, sigma = hye::kNaN;
    bool valid = false;
};

inline bool judge(const Fit& f1, const Fit& f2, double sx, double sy, Obs& o) {
    o.x = f1.p[0]; o.y = f1.p[1]; o.rmse = f1.rmse; o.contrast = f1.p[5]; o.sigma = std::max(f1.p[2], f1.p[3]);
    o.stab = std::hypot(f1.p[0] - f2.p[0], f1.p[1] - f2.p[1]);
    bool bad = !(f1.ok && f2.ok) || f1.rmse > 15 || f1.p[5] < 45 || o.stab > .65 || std::max(std::abs(f1.p[0] - sx), std::abs(f1.p[1] - sy)) > 3.6 ||
               o.sigma > 2.6;
    o.valid = !bad;
    return o.valid;
}

// refine.py: center 에서 처음 위치 그대로 창 4, 5
inline void refine_center(const cv::Mat& center, std::vector<Corner>& cs) {
#pragma omp parallel for schedule(dynamic) num_threads(4)
    for (int i = 0; i < (int)cs.size(); ++i) {
        auto& c = cs[i];
        auto b = kBoxes[c.qr - 1];
        cv::Vec4d box(b[0], b[1], b[2], b[3]);
        Fit f1 = fit_at(center, {c.ix, c.iy}, 4, box, c.ix, c.iy, c), f2 = fit_at(center, {c.ix, c.iy}, 5, box, c.ix, c.iy, c);
        Obs o;
        c.center_valid = judge(f1, f2, c.ix, c.iy, o);
        c.cx = f1.p[0];
        c.cy = f1.p[1];
    }
}

struct Track {
    std::vector<Obs> obs;
    std::array<cv::Point, 3> shift;
    std::array<double, 3> score;
};

// track_corners.py: 대략 이동(±pad) 후 창 다시 맞추기
inline Track track(const cv::Mat& img, const cv::Mat& center, const std::vector<Corner>& cs, int pad = 130) {
    Track t;
    for (int k = 0; k < 3; ++k) {
        auto b = kBoxes[k];
        cv::Mat tpl = center(cv::Range(b[1], b[3]), cv::Range(b[0], b[2]));
        cv::Mat s = img(cv::Range(b[1] - pad, b[3] + pad), cv::Range(b[0] - pad, b[2] + pad)), res;
        cv::matchTemplate(s, tpl, res, cv::TM_CCOEFF_NORMED);
        double mx;
        cv::Point loc;
        cv::minMaxLoc(res, nullptr, &mx, nullptr, &loc);
        t.shift[k] = {loc.x - pad, loc.y - pad};
        t.score[k] = mx;
    }
    t.obs.resize(cs.size());
    for (size_t i = 0; i < cs.size(); ++i) {
        auto& c = cs[i];
        auto sh = t.shift[c.qr - 1];
        auto b = kBoxes[c.qr - 1];
        cv::Vec4d box(b[0] + sh.x, b[1] + sh.y, b[2] + sh.x, b[3] + sh.y);
        const double sx = c.cx + sh.x, sy = c.cy + sh.y;
        cv::Point2d est(sx, sy);
        Fit f1{};
        for (int k = 0; k < 3; ++k) {
            f1 = fit_at(img, est, 4, box, sx, sy, c);
            bool moved = std::nearbyint(f1.p[0]) != std::nearbyint(est.x) || std::nearbyint(f1.p[1]) != std::nearbyint(est.y);
            est = {f1.p[0], f1.p[1]};
            if (!moved) break;
        }
        Fit f2 = fit_at(img, est, 5, box, sx, sy, c);
        judge(f1, f2, sx, sy, t.obs[i]);
    }
    return t;
}

// 630점 (대각선 제외) 추출
inline std::vector<Corner> non_diagonal(const std::vector<Corner>& all) {
    std::vector<Corner> o;
    for (auto& c : all) if (c.nblack != 2) o.push_back(c);
    return o;
}

// P12: 점 변위장을 [1,u,v,u²,uv,v²] 로 최소제곱 (u=(x-1906)/486, v=(y-1070)/122) -> (6 x 2) 를 행 우선 12개
inline std::vector<double> p12(const std::vector<cv::Point2d>& pts, const std::vector<cv::Point2d>& d) {
    const int n = (int)pts.size();
    cv::Mat B(n, 6, CV_64F), D(n, 2, CV_64F), C;
    for (int i = 0; i < n; ++i) {
        double u = (pts[i].x - 1906) / 486, v = (pts[i].y - 1070) / 122;
        double row[6] = {1, u, v, u * u, u * v, v * v};
        for (int j = 0; j < 6; ++j) B.at<double>(i, j) = row[j];
        D.at<double>(i, 0) = d[i].x;
        D.at<double>(i, 1) = d[i].y;
    }
    cv::solve(B, D, C, cv::DECOMP_SVD);
    std::vector<double> f(12);
    for (int j = 0; j < 6; ++j) { f[2 * j] = C.at<double>(j, 0); f[2 * j + 1] = C.at<double>(j, 1); }
    return f;
}

}  // namespace s630
