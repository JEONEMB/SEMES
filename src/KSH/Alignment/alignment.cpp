// =====================================================================================
// alignment.cpp — Y·Θ 추정 구현
//
//  ① Coarse Alignment (대략 정렬) : 플레이트 주변 창의 위상 상관(FFT, 해닝 창). 수십 px 이동을 한 번에, 밝기 변화에 강함
//  ② Fine Alignment (정밀 정렬)   : ROI 마다 평행이동 IC-GN (ZNSSD, 가우스–뉴턴 반복) + 3차 O-MOMS 보간
//                                   배경 ROI 는 2배 창 위상 상관으로 한 번 더 맞춘 뒤 정밀 정렬
//                                   (설정으로 OpenCV findTransformECC = 선형 보간 + 5×5 가우시안 으로 바꿀 수 있음)
//  ③ 특징 : s = QR 3개 평균 dx,  d = s − 배경 ROI 평균 dx  (가까운 플레이트와 먼 하단 문의 시차)
//  ④ 추정 : s = c_s + a_s·Y + Σ b_s,k·Θ^k ,  d = c_d + a_d·Y + Σ b_d,k·Θ^k  을 동시에 만족하는 (Y, Θ)
//          가중 가우스–뉴턴 (가중 = 1/σ²). 출발점 = 1차 항만 쓴 두 직선의 교점
//  선택 근거 : QR_Image 연구 프로젝트의 실험계획(DOE) — 정렬 순서 · 보간 방법 · 식의 차수를 1차 블록 교차검증으로 비교
// =====================================================================================
#include "alignment.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>

namespace {

// ---------------------------------------------------------------- 설정 (alignment.yml)
struct RoiCfg {
    std::string name;
    int cx_px = 0, cy_px = 0, hw_px = 0;   // 기준 사진 좌표의 중심, 반폭
};
struct FeatureModel {                      // 특징 = c + a·Y + Σ_k b_k·Θ^k
    double c_px = 0.0, a_px_per_mm = 0.0;
    std::vector<double> b_px_per_deg_k;    // [b1, b2, b3 ...]
    double sigma_px = 1.0;                 // 학습 잔차 σ (가중치)
};
enum Kernel { K_LINEAR, K_KEYS05, K_KEYS075, K_LANCZOS3, K_BSPLINE3, K_OMOMS3 };
struct Config {
    bool loaded = false;
    std::string reference_path;
    cv::Size image_size_px;
    cv::Point coarse_center_px;
    cv::Size coarse_window_px;
    int wide_window_scale = 2;              // 배경 ROI 위상 상관 창 = ROI 의 이 배수
    double qr_max_dev_px = 15.0;            // QR 이 Coarse 결과에서 이만큼 벗어나면 이웃 QR 에 붙은 것으로 보고 실패
    bool fine_opencv_ecc = false;           // true = OpenCV findTransformECC, false = IC-GN
    Kernel fine_kernel = K_OMOMS3;
    double fine_gauss_sigma_px = 0.0;
    int fine_max_iter = 60;
    double fine_eps_px = 1e-5;
    int ecc_max_iter = 100;
    double ecc_eps = 1e-7;
    std::vector<RoiCfg> qr;
    std::vector<RoiCfg> bg;
    FeatureModel s, d;
};

Config g_cfg;
cv::Mat g_ref32;            // 기준 사진 (CV_32F 회색)
AlignmentInfo g_info;

double nan_value() { return std::numeric_limits<double>::quiet_NaN(); }

bool inside(const cv::Rect &r, const cv::Size &s) {
    return r.x >= 0 && r.y >= 0 && r.x + r.width <= s.width && r.y + r.height <= s.height;
}
cv::Point round_pt(const cv::Point2d &p) { return cv::Point(int(std::lround(p.x)), int(std::lround(p.y))); }

cv::Mat to_gray32(const cv::Mat &src) {
    cv::Mat gray, out;
    if (src.channels() == 3) cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    else if (src.channels() == 4) cv::cvtColor(src, gray, cv::COLOR_BGRA2GRAY);
    else gray = src;
    gray.convertTo(out, CV_32F);
    return out;
}

std::vector<RoiCfg> read_rois(const cv::FileNode &n) {
    std::vector<RoiCfg> v;
    for (auto it = n.begin(); it != n.end(); ++it) {
        RoiCfg r;
        (*it)["name"] >> r.name;
        r.cx_px = int((*it)["cx_px"]);
        r.cy_px = int((*it)["cy_px"]);
        r.hw_px = int((*it)["hw_px"]);
        v.push_back(r);
    }
    return v;
}
FeatureModel read_model(const cv::FileNode &n) {
    FeatureModel m;
    m.c_px = double(n["c_px"]);
    m.a_px_per_mm = double(n["a_px_per_mm"]);
    const cv::FileNode k = n["theta_px_per_deg_k"];
    if (k.isSeq()) for (auto it = k.begin(); it != k.end(); ++it) m.b_px_per_deg_k.push_back(double(*it));
    else m.b_px_per_deg_k.push_back(double(n["b_px_per_deg"]));   // 이전 형식 (1차식)
    if (!n["sigma_px"].empty()) m.sigma_px = double(n["sigma_px"]);
    return m;
}
bool read_kernel(const std::string &s, Kernel &k) {
    if (s == "linear") k = K_LINEAR;
    else if (s == "keys05") k = K_KEYS05;
    else if (s == "keys075") k = K_KEYS075;
    else if (s == "lanczos3") k = K_LANCZOS3;
    else if (s == "bspline3") k = K_BSPLINE3;
    else if (s == "omoms3") k = K_OMOMS3;
    else return false;
    return true;
}

// ---------------------------------------------------------------- ① 위상 상관
// b 가 a 대비 이동한 양 (px). cv::phaseCorrelate 는 창 함수를 입력에 제자리로 곱하므로 복사본을 넘김
cv::Point2d phase_shift(const cv::Mat &a, const cv::Mat &b) {
    cv::Mat win;
    cv::createHanningWindow(win, a.size(), CV_32F);
    cv::Mat a2 = a.clone(), b2 = b.clone();
    return cv::phaseCorrelate(a2, b2, win);
}

// ---------------------------------------------------------------- ② 보간 커널 (실험계획 2 와 같은 구현)
int kernel_radius(Kernel k) { return k == K_LINEAR ? 1 : (k == K_LANCZOS3 ? 3 : 2); }
double keys_w(double t, double a) {
    t = std::abs(t);
    if (t < 1.0) return ((a + 2.0) * t - (a + 3.0)) * t * t + 1.0;
    if (t < 2.0) return ((a * t - 5.0 * a) * t + 8.0 * a) * t - 4.0 * a;
    return 0.0;
}
double sinc(double x) { return std::abs(x) < 1e-12 ? 1.0 : std::sin(CV_PI * x) / (CV_PI * x); }
double kernel_w(Kernel k, double t) {
    const double a = std::abs(t);
    switch (k) {
    case K_LINEAR: return a < 1.0 ? 1.0 - a : 0.0;
    case K_KEYS05: return keys_w(t, -0.5);
    case K_KEYS075: return keys_w(t, -0.75);
    case K_LANCZOS3: return a < 3.0 ? sinc(t) * sinc(t / 3.0) : 0.0;
    case K_BSPLINE3:
        if (a < 1.0) return (4.0 - 6.0 * a * a + 3.0 * a * a * a) / 6.0;
        if (a < 2.0) return (2.0 - a) * (2.0 - a) * (2.0 - a) / 6.0;
        return 0.0;
    case K_OMOMS3:   // Blu, Thévenaz, Unser 2001 — 같은 지지 폭(4 탭)의 3차 커널 중 근사 오차가 가장 작음
        if (a < 1.0) return 0.5 * a * a * a - a * a + a / 14.0 + 13.0 / 21.0;
        if (a < 2.0) return -a * a * a / 6.0 + a * a - 85.0 / 42.0 * a + 29.0 / 21.0;
        return 0.0;
    }
    return 0.0;
}
double kernel_pole(Kernel k) {   // 보간점을 지나지 않는 커널은 사전 필터(재귀)로 계수를 구함
    if (k == K_BSPLINE3) return std::sqrt(3.0) - 2.0;
    if (k == K_OMOMS3) return (std::sqrt(105.0) - 13.0) / 8.0;
    return 0.0;
}
void prefilter_1d(double *c, int n, double z) {   // Unser 1993, 거울 경계
    const double g = (1.0 - z) * (1.0 - 1.0 / z);
    for (int k = 0; k < n; ++k) c[k] *= g;
    const int horizon = std::min(n, 40);
    double zn = z, sum = c[0];
    for (int k = 1; k < horizon; ++k) { sum += zn * c[k]; zn *= z; }
    c[0] = sum;
    for (int k = 1; k < n; ++k) c[k] += z * c[k - 1];
    c[n - 1] = (z / (z * z - 1.0)) * (z * c[n - 2] + c[n - 1]);
    for (int k = n - 2; k >= 0; --k) c[k] = z * (c[k + 1] - c[k]);
}
void to_coeffs(cv::Mat &c, Kernel k) {
    const double z = kernel_pole(k);
    if (z == 0.0) return;
    for (int y = 0; y < c.rows; ++y) prefilter_1d(c.ptr<double>(y), c.cols, z);
    std::vector<double> col(c.rows);
    for (int x = 0; x < c.cols; ++x) {
        for (int y = 0; y < c.rows; ++y) col[y] = c.at<double>(y, x);
        prefilter_1d(col.data(), c.rows, z);
        for (int y = 0; y < c.rows; ++y) c.at<double>(y, x) = col[y];
    }
}
cv::Mat crop_blur(const cv::Mat &img32, const cv::Rect &r, double sigma) {
    cv::Mat c;
    img32(r).convertTo(c, CV_64F);
    if (sigma > 0) {
        const int ks = 2 * int(std::ceil(3.0 * sigma)) + 1;
        cv::GaussianBlur(c, c, cv::Size(ks, ks), sigma, sigma, cv::BORDER_REFLECT);
    }
    return c;
}

// ---------------------------------------------------------------- ② 평행이동 IC-GN (ZNSSD, Pan 2013)
// 기준 영역 T (기준 좌표) 가 현재 영상에서 init 만큼 옮겨졌다고 보고 다듬음. 반환 = 이동 (px), cc = ZNCC
bool icgn_translation(const cv::Mat &ref, const cv::Mat &cur, const cv::Rect &T, const cv::Point2d &init, cv::Point2d &out, double &cc) {
    const Kernel K = g_cfg.fine_kernel;
    const int R = kernel_radius(K), M = 28;
    const cv::Rect Tr(T.x - 4, T.y - 4, T.width + 8, T.height + 8);
    if (!inside(Tr, ref.size())) return false;
    const cv::Mat F = crop_blur(ref, Tr, g_cfg.fine_gauss_sigma_px);
    const cv::Point gi = round_pt(init);
    const cv::Rect Rc(T.x + gi.x - M, T.y + gi.y - M, T.width + 2 * M, T.height + 2 * M);
    if (!inside(Rc, cur.size())) return false;
    cv::Mat C = crop_blur(cur, Rc, g_cfg.fine_gauss_sigma_px);
    to_coeffs(C, K);

    const int W = T.width, H = T.height, n = W * H;
    std::vector<double> f(n), gx(n), gy(n);
    double fm = 0;
    for (int j = 0; j < H; ++j)
        for (int i = 0; i < W; ++i) {
            const int y = j + 4, x = i + 4;
            const double *row = F.ptr<double>(y);
            f[j * W + i] = row[x];
            fm += row[x];
            gx[j * W + i] = (-row[x + 2] + 8 * row[x + 1] - 8 * row[x - 1] + row[x - 2]) / 12.0;   // 5점 미분
            gy[j * W + i] = (-F.at<double>(y + 2, x) + 8 * F.at<double>(y + 1, x) - 8 * F.at<double>(y - 1, x) + F.at<double>(y - 2, x)) / 12.0;
        }
    fm /= n;
    double df = 0;
    for (int k = 0; k < n; ++k) { f[k] -= fm; df += f[k] * f[k]; }
    df = std::sqrt(df);
    if (df < 1e-9) return false;
    double h00 = 0, h01 = 0, h11 = 0;
    for (int k = 0; k < n; ++k) {
        gx[k] /= df; gy[k] /= df; f[k] /= df;
        h00 += gx[k] * gx[k]; h01 += gx[k] * gy[k]; h11 += gy[k] * gy[k];
    }
    const double det = h00 * h11 - h01 * h01;
    if (std::abs(det) < 1e-18) return false;

    cv::Point2d p = init;
    std::vector<double> g(n), tmp, wx(2 * R), wy(2 * R);
    for (int it = 0; it < g_cfg.fine_max_iter; ++it) {
        const double ux = std::floor(p.x), uy = std::floor(p.y);
        for (int m = 0; m < 2 * R; ++m) { wx[m] = kernel_w(K, (p.x - ux) - (m - R + 1)); wy[m] = kernel_w(K, (p.y - uy) - (m - R + 1)); }
        const int bx = T.x + int(ux) - Rc.x, by = T.y + int(uy) - Rc.y;
        if (bx - R + 1 < 0 || by - R + 1 < 0 || bx + W - 1 + R >= C.cols || by + H - 1 + R >= C.rows) return false;
        // 분리형 보간 : 가로 먼저, 그다음 세로
        const int rows = H + 2 * R - 1;
        tmp.assign(size_t(rows) * W, 0.0);
        for (int q = 0; q < rows; ++q) {
            const double *src = C.ptr<double>(by - R + 1 + q) + bx - R + 1;
            double *t = &tmp[size_t(q) * W];
            for (int i = 0; i < W; ++i) {
                double a = 0;
                for (int m = 0; m < 2 * R; ++m) a += wx[m] * src[i + m];
                t[i] = a;
            }
        }
        double gm = 0;
        for (int j = 0; j < H; ++j)
            for (int i = 0; i < W; ++i) {
                double s = 0;
                for (int q = 0; q < 2 * R; ++q) s += wy[q] * tmp[size_t(j + q) * W + i];
                g[j * W + i] = s;
                gm += s;
            }
        gm /= n;
        double dg = 0;
        for (int k = 0; k < n; ++k) { g[k] -= gm; dg += g[k] * g[k]; }
        dg = std::sqrt(dg);
        if (dg < 1e-9) return false;
        double b0 = 0, b1 = 0, ss = 0;
        for (int k = 0; k < n; ++k) {
            const double e = g[k] / dg - f[k];
            b0 += gx[k] * e; b1 += gy[k] * e; ss += e * e;
        }
        const double dx = (h11 * b0 - h01 * b1) / det, dy = (h00 * b1 - h01 * b0) / det;
        p.x -= dx; p.y -= dy;   // 역합성 갱신
        cc = 1.0 - 0.5 * ss;
        if (std::abs(dx) < g_cfg.fine_eps_px && std::abs(dy) < g_cfg.fine_eps_px) break;
        if (cv::norm(p - init) > M - R - 2) return false;
    }
    out = p;
    return std::isfinite(p.x) && std::isfinite(p.y);
}

// ---------------------------------------------------------------- ② OpenCV ECC (비교·이전 방식)
// OpenCV 디버그 라이브러리는 예외 전에 오류 문구를 콘솔에 찍음 → ECC 동안만 문구 출력을 막음 (미수렴 = 정상적인 측정 실패 경로)
int quiet_error(int, const char *, const char *, const char *, int, void *) { return 0; }

bool ecc_translation(const cv::Mat &ref, const cv::Mat &cur, const cv::Rect &T, const cv::Point2d &init, cv::Point2d &out, double &cc) {
    const cv::Point gi = round_pt(init);
    const cv::Rect C = T + gi;
    if (!inside(T, ref.size()) || !inside(C, cur.size())) return false;
    cv::Mat warp = cv::Mat::eye(2, 3, CV_32F);
    warp.at<float>(0, 2) = float(init.x - gi.x);
    warp.at<float>(1, 2) = float(init.y - gi.y);
    void *prev_data = nullptr;
    const cv::ErrorCallback prev = cv::redirectError(quiet_error, nullptr, &prev_data);
    try {
        cc = cv::findTransformECC(ref(T), cur(C), warp, cv::MOTION_TRANSLATION,
                                  cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, g_cfg.ecc_max_iter, g_cfg.ecc_eps),
                                  cv::noArray());
    } catch (const cv::Exception &) {
        cv::redirectError(prev, prev_data);
        return false;
    }
    cv::redirectError(prev, prev_data);
    out = cv::Point2d(warp.at<float>(0, 2) + gi.x, warp.at<float>(1, 2) + gi.y);
    return true;
}

// ROI 하나의 이동. wide = 넓은 창 위상 상관을 먼저 (배경). QR 은 반복 무늬라 넓은 창을 쓰면 이웃 QR 에 붙을 수 있어 바로 Fine
AlignmentRoi measure_roi(const cv::Mat &ref, const cv::Mat &cur, const RoiCfg &r, const cv::Point2d &guess_px, bool wide) {
    AlignmentRoi m;
    m.name = r.name;
    m.ref_rect_px = cv::Rect(r.cx_px - r.hw_px, r.cy_px - r.hw_px, 2 * r.hw_px, 2 * r.hw_px);
    cv::Point2d g = guess_px;
    if (wide) {
        const int half_px = g_cfg.wide_window_scale * r.hw_px;
        const cv::Rect A(r.cx_px - half_px, r.cy_px - half_px, 2 * half_px, 2 * half_px);
        const cv::Rect B = A + round_pt(guess_px);
        if (!inside(A, ref.size()) || !inside(B, cur.size())) return m;
        g = cv::Point2d(round_pt(guess_px)) + phase_shift(ref(A), cur(B));
    }
    const bool ok = g_cfg.fine_opencv_ecc ? ecc_translation(ref, cur, m.ref_rect_px, g, m.shift_px, m.ecc_cc)
                                          : icgn_translation(ref, cur, m.ref_rect_px, g, m.shift_px, m.ecc_cc);
    m.ok = ok;
    return m;
}

// ---------------------------------------------------------------- ④ (s, d) → (Y, Θ)
void eval_model(const FeatureModel &m, double y, double t, double &v, double &dv_dy, double &dv_dt) {
    v = m.c_px + m.a_px_per_mm * y;
    dv_dy = m.a_px_per_mm;
    dv_dt = 0;
    double tp = 1.0;   // Θ^(k-1)
    for (size_t k = 0; k < m.b_px_per_deg_k.size(); ++k) {
        dv_dt += double(k + 1) * m.b_px_per_deg_k[k] * tp;
        tp *= t;
        v += m.b_px_per_deg_k[k] * tp;
    }
}

bool invert_model(double s, double d, double &y, double &t) {
    const FeatureModel &S = g_cfg.s, &D = g_cfg.d;
    // 출발점 : 1차 항만 쓴 두 직선의 교점
    const double det = S.a_px_per_mm * D.b_px_per_deg_k[0] - D.a_px_per_mm * S.b_px_per_deg_k[0];
    if (std::abs(det) < 1e-12) return false;
    const double rs = s - S.c_px, rd = d - D.c_px;
    y = (rs * D.b_px_per_deg_k[0] - rd * S.b_px_per_deg_k[0]) / det;
    t = (S.a_px_per_mm * rd - D.a_px_per_mm * rs) / det;
    const double ws = 1.0 / (S.sigma_px * S.sigma_px), wd = 1.0 / (D.sigma_px * D.sigma_px);
    for (int it = 0; it < 30; ++it) {   // 가중 가우스–뉴턴
        double vs, sy, st, vd, dy, dt;
        eval_model(S, y, t, vs, sy, st);
        eval_model(D, y, t, vd, dy, dt);
        const double es = s - vs, ed = d - vd;
        const double a00 = ws * sy * sy + wd * dy * dy, a01 = ws * sy * st + wd * dy * dt, a11 = ws * st * st + wd * dt * dt;
        const double b0 = ws * sy * es + wd * dy * ed, b1 = ws * st * es + wd * dt * ed;
        const double dd = a00 * a11 - a01 * a01;
        if (std::abs(dd) < 1e-18) return false;
        const double ddy = (a11 * b0 - a01 * b1) / dd, ddt = (a00 * b1 - a01 * b0) / dd;
        y += ddy; t += ddt;
        if (std::abs(ddy) < 1e-10 && std::abs(ddt) < 1e-11) break;
    }
    return std::isfinite(y) && std::isfinite(t);
}

bool ensure_ready() {
    if (g_cfg.loaded && !g_ref32.empty()) return true;
    if (!g_cfg.loaded && !alignment_init()) return false;
    return !g_ref32.empty();
}

} // namespace

// =====================================================================================
bool alignment_init(const std::string &config_path) {
    g_cfg = Config();
    g_ref32.release();
    cv::FileStorage fs;
    try {
        if (!fs.open(config_path, cv::FileStorage::READ)) {
            std::cerr << "[alignment] 설정 파일을 열 수 없음 : " << config_path << "\n";
            return false;
        }
    } catch (const cv::Exception &) {
        std::cerr << "[alignment] 설정 파일 형식 오류 : " << config_path << "\n";
        return false;
    }
    Config c;
    fs["reference_path"] >> c.reference_path;
    c.image_size_px = cv::Size(int(fs["image_width_px"]), int(fs["image_height_px"]));
    const cv::FileNode co = fs["coarse"];
    c.coarse_center_px = cv::Point(int(co["center_x_px"]), int(co["center_y_px"]));
    c.coarse_window_px = cv::Size(int(co["width_px"]), int(co["height_px"]));
    c.wide_window_scale = int(fs["wide_window_scale"]);
    c.qr_max_dev_px = double(fs["qr_max_dev_px"]);
    const cv::FileNode fi = fs["fine"];
    if (!fi.empty()) {
        std::string method, kernel;
        fi["method"] >> method;
        fi["kernel"] >> kernel;
        c.fine_opencv_ecc = method == "opencv_ecc";
        if (!c.fine_opencv_ecc && !read_kernel(kernel, c.fine_kernel)) {
            std::cerr << "[alignment] fine.kernel 을 알 수 없음 : " << kernel << "\n";
            return false;
        }
        c.fine_gauss_sigma_px = double(fi["gauss_sigma_px"]);
        if (!fi["max_iter"].empty()) c.fine_max_iter = int(fi["max_iter"]);
        if (!fi["eps_px"].empty()) c.fine_eps_px = double(fi["eps_px"]);
    } else {
        c.fine_opencv_ecc = true;   // 이전 형식 설정 파일 (fine 항목 없음) = OpenCV ECC
    }
    if (!fs["ecc"].empty()) {
        c.ecc_max_iter = int(fs["ecc"]["max_iter"]);
        c.ecc_eps = double(fs["ecc"]["eps"]);
    }
    c.qr = read_rois(fs["qr_rois"]);
    c.bg = read_rois(fs["background_rois"]);
    c.s = read_model(fs["model"]["s"]);
    c.d = read_model(fs["model"]["d"]);
    if (c.qr.empty() || c.bg.empty() || c.coarse_window_px.area() == 0 || c.wide_window_scale <= 0 || c.s.b_px_per_deg_k.empty() ||
        c.d.b_px_per_deg_k.empty() || c.s.sigma_px <= 0 || c.d.sigma_px <= 0) {
        std::cerr << "[alignment] 설정 항목 누락 : " << config_path << "\n";
        return false;
    }
    g_cfg = c;
    g_cfg.loaded = true;
    // 기준 사진 경로는 설정 파일 위치 기준 상대 경로도 허용
    std::string ref_path = c.reference_path;
    cv::Mat ref = cv::imread(ref_path, cv::IMREAD_GRAYSCALE);
    if (ref.empty()) {
        const size_t p = config_path.find_last_of("/\\");
        if (p != std::string::npos) ref = cv::imread(config_path.substr(0, p + 1) + ref_path, cv::IMREAD_GRAYSCALE);
    }
    if (ref.empty()) {
        std::cerr << "[alignment] 기준 사진을 읽을 수 없음 : " << ref_path << "\n";
        return true;   // 설정은 읽음. alignment_set_reference 로 기준을 따로 줄 수 있음
    }
    return alignment_set_reference(ref);
}

bool alignment_set_reference(const cv::Mat &reference) {
    if (reference.empty()) return false;
    if (g_cfg.image_size_px.area() > 0 && reference.size() != g_cfg.image_size_px) {
        std::cerr << "[alignment] 기준 사진 크기 " << reference.cols << "×" << reference.rows << " ≠ 설정 "
                  << g_cfg.image_size_px.width << "×" << g_cfg.image_size_px.height << "\n";
        return false;
    }
    g_ref32 = to_gray32(reference);
    return true;
}

const AlignmentInfo &alignment_last_info() { return g_info; }

// =====================================================================================
Alignment_t alignment(cv::Mat &src)
{
    Alignment_t result;
    result.y = nan_value();
    result.t = nan_value();

    const auto t0 = std::chrono::steady_clock::now();
    g_info = AlignmentInfo();
    auto finish = [&](const std::string &msg) {
        g_info.message = msg;
        g_info.elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        return result;
    };

    if (!ensure_ready()) return finish("설정 또는 기준 사진 없음");
    if (src.empty()) return finish("입력 사진이 비어 있음");
    if (src.size() != g_ref32.size()) return finish("입력 사진 크기가 기준 사진과 다름");
    const cv::Mat cur = to_gray32(src);

    // ① Coarse Alignment : 플레이트 주변 창
    const cv::Rect W(g_cfg.coarse_center_px.x - g_cfg.coarse_window_px.width / 2,
                     g_cfg.coarse_center_px.y - g_cfg.coarse_window_px.height / 2,
                     g_cfg.coarse_window_px.width, g_cfg.coarse_window_px.height);
    if (!inside(W, cur.size())) return finish("Coarse Alignment 창이 사진 밖");
    g_info.coarse_px = phase_shift(g_ref32(W), cur(W));

    // ② Fine Alignment : QR (바로 Fine) + 배경 ROI (넓은 창 위상 상관 → Fine)
    double qr_sum_px = 0.0, bg_sum_px = 0.0;
    bool all_ok = true;
    for (const auto &r : g_cfg.qr) {
        AlignmentRoi m = measure_roi(g_ref32, cur, r, g_info.coarse_px, false);
        if (m.ok && cv::norm(m.shift_px - g_info.coarse_px) > g_cfg.qr_max_dev_px) m.ok = false;
        all_ok = all_ok && m.ok;
        qr_sum_px += m.shift_px.x;
        g_info.rois.push_back(m);
    }
    for (const auto &r : g_cfg.bg) {
        const AlignmentRoi m = measure_roi(g_ref32, cur, r, g_info.coarse_px, true);
        all_ok = all_ok && m.ok;
        bg_sum_px += m.shift_px.x;
        g_info.rois.push_back(m);
    }
    if (!all_ok) return finish("ROI 측정 실패 (Fine Alignment 미수렴 또는 사진 밖)");

    // ③ 특징
    g_info.s_px = qr_sum_px / double(g_cfg.qr.size());
    g_info.d_px = g_info.s_px - bg_sum_px / double(g_cfg.bg.size());

    // ④ 두 식을 동시에 만족하는 (Y, Θ)
    double y = 0, t = 0;
    if (!invert_model(g_info.s_px, g_info.d_px, y, t)) return finish("모델 계수가 Y·Θ 를 구별하지 못함");
    result.y = y;
    result.t = t;
    g_info.ok = true;
    return finish("");
}
