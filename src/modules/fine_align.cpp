// fine_align.cpp — ② Fine Alignment 구현 (IC-GN + 보간 커널, OpenCV ECC)
#include "ksh/fine_align.hpp"

#include "ksh/image_util.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <cmath>
#include <vector>

namespace ksh {

namespace {

// 영역을 잘라 CV_64F 로, σ > 0 이면 가우시안 사전 필터
cv::Mat crop_blur64(const cv::Mat &img32, const cv::Rect &rect, double sigma_px) {
    cv::Mat c;
    img32(rect).convertTo(c, CV_64F);
    if (sigma_px > 0) {
        const int ksize = 2 * int(std::ceil(3.0 * sigma_px)) + 1;
        cv::GaussianBlur(c, c, cv::Size(ksize, ksize), sigma_px, sigma_px, cv::BORDER_REFLECT);
    }
    return c;
}

// OpenCV 디버그 라이브러리는 예외 전에 오류 문구를 콘솔에 찍음 → ECC 동안만 막음 (미수렴 = 정상적인 측정 실패 경로)
int quiet_error(int, const char *, const char *, const char *, int, void *) { return 0; }

} // namespace

bool icgn_translation(const cv::Mat &ref32, const cv::Mat &cur32, const cv::Rect &T, const cv::Point2d &init_px,
                      const FineParams &params, FineResult &out) {
    const Kernel K = params.kernel;
    const int R = kernel_radius(K), M = params.search_margin_px;
    const int edge_px = 4;   // 5점 미분에 필요한 여백
    const cv::Rect Tr(T.x - edge_px, T.y - edge_px, T.width + 2 * edge_px, T.height + 2 * edge_px);
    if (!rect_inside(Tr, ref32.size())) return false;
    const cv::Mat F = crop_blur64(ref32, Tr, params.gauss_sigma_px);
    const cv::Point gi = round_point(init_px);
    const cv::Rect Rc(T.x + gi.x - M, T.y + gi.y - M, T.width + 2 * M, T.height + 2 * M);
    if (!rect_inside(Rc, cur32.size())) return false;
    cv::Mat C = crop_blur64(cur32, Rc, params.gauss_sigma_px);
    to_kernel_coeffs(C, K);

    // 기준 영역 : 평균을 빼고 크기로 나눔 (ZNSSD) + 5점 미분 기울기 → 헤시안은 반복 동안 고정 (역합성)
    const int W = T.width, H = T.height, n = W * H;
    std::vector<double> f(n), gx(n), gy(n);
    double fm = 0;
    for (int j = 0; j < H; ++j)
        for (int i = 0; i < W; ++i) {
            const int y = j + edge_px, x = i + edge_px;
            const double *row = F.ptr<double>(y);
            f[j * W + i] = row[x];
            fm += row[x];
            gx[j * W + i] = (-row[x + 2] + 8 * row[x + 1] - 8 * row[x - 1] + row[x - 2]) / 12.0;
            gy[j * W + i] = (-F.at<double>(y + 2, x) + 8 * F.at<double>(y + 1, x) - 8 * F.at<double>(y - 1, x) + F.at<double>(y - 2, x)) / 12.0;
        }
    fm /= n;
    double df = 0;
    for (int k = 0; k < n; ++k) { f[k] -= fm; df += f[k] * f[k]; }
    df = std::sqrt(df);
    if (df < 1e-9) return false;   // 민무늬 영역
    double h00 = 0, h01 = 0, h11 = 0;
    for (int k = 0; k < n; ++k) {
        gx[k] /= df; gy[k] /= df; f[k] /= df;
        h00 += gx[k] * gx[k]; h01 += gx[k] * gy[k]; h11 += gy[k] * gy[k];
    }
    const double det = h00 * h11 - h01 * h01;
    if (std::abs(det) < 1e-18) return false;

    cv::Point2d p = init_px;
    double zncc = 0.0;
    bool converged = false;
    int iterations = 0;
    std::vector<double> g(n), tmp, wx(2 * R), wy(2 * R);
    for (int it = 0; it < params.max_iter; ++it) {
        iterations = it + 1;
        const double ux = std::floor(p.x), uy = std::floor(p.y);
        for (int m = 0; m < 2 * R; ++m) {
            wx[m] = kernel_weight(K, (p.x - ux) - (m - R + 1));
            wy[m] = kernel_weight(K, (p.y - uy) - (m - R + 1));
        }
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
        zncc = 1.0 - 0.5 * ss;
        if (std::abs(dx) < params.eps_px && std::abs(dy) < params.eps_px) { converged = true; break; }
        if (cv::norm(p - init_px) > M - R - 2) return false;   // 잘라 온 범위를 벗어남
    }
    if (!std::isfinite(p.x) || !std::isfinite(p.y)) return false;
    out.shift_px = p;
    out.zncc = zncc;
    out.converged = converged;
    out.iterations = iterations;
    return true;   // 수렴 여부 판정은 fine_alignment 에서 (값은 진단용으로 남김)
}

bool ecc_translation(const cv::Mat &ref32, const cv::Mat &cur32, const cv::Rect &T, const cv::Point2d &init_px,
                     const FineParams &params, FineResult &out) {
    const cv::Point gi = round_point(init_px);
    const cv::Rect C = T + gi;
    if (!rect_inside(T, ref32.size()) || !rect_inside(C, cur32.size())) return false;
    cv::Mat warp = cv::Mat::eye(2, 3, CV_32F);
    warp.at<float>(0, 2) = float(init_px.x - gi.x);
    warp.at<float>(1, 2) = float(init_px.y - gi.y);
    void *prev_data = nullptr;
    const cv::ErrorCallback prev = cv::redirectError(quiet_error, nullptr, &prev_data);
    double cc = 0.0;
    try {
        cc = cv::findTransformECC(ref32(T), cur32(C), warp, cv::MOTION_TRANSLATION,
                                  cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, params.ecc_max_iter, params.ecc_eps),
                                  cv::noArray());
    } catch (const cv::Exception &) {
        cv::redirectError(prev, prev_data);
        return false;
    }
    cv::redirectError(prev, prev_data);
    out.shift_px = cv::Point2d(warp.at<float>(0, 2) + gi.x, warp.at<float>(1, 2) + gi.y);
    out.zncc = cc;
    out.converged = true;   // findTransformECC 는 최대 반복 도달 여부를 알려 주지 않음 (이전 방식, 비교용) — 상관 하한으로만 거름
    return true;
}

bool fine_alignment(const cv::Mat &ref32, const cv::Mat &cur32, const cv::Rect &ref_rect_px, const cv::Point2d &init_px,
                    const FineParams &params, FineResult &out) {
    out = FineResult();
    const bool done = params.method == FineMethod::opencv_ecc ? ecc_translation(ref32, cur32, ref_rect_px, init_px, params, out)
                                                              : icgn_translation(ref32, cur32, ref_rect_px, init_px, params, out);
    // 계산이 끝났어도 미수렴이거나 상관이 낮으면 측정 실패 (엉뚱한 곳에 맞은 값을 s·d 에 넣지 않음)
    return done && out.converged && out.zncc >= params.min_zncc;
}

} // namespace ksh
