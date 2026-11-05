// fine_align.cpp — ② Fine Alignment 구현 (IC-GN + 보간 커널)
#include "ksh/fine_align.hpp"

#include "ksh/image_util.hpp"

#include <opencv2/imgproc.hpp>

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

} // namespace

// ② IC-GN (역합성 가우스–뉴턴) 평행이동 정합 — Pan, Li, Tong 2013
//
//  목표 : 기준 영역 f(x) 와 현재 영상을 p = (u, v) 만큼 옮긴 g(x + p) 가 가장 잘 겹치는 p 찾기
//  비교 : ZNSSD  C(p) = Σ [ ĝ(x) − f̂(x) ]²,  f̂ = (f − f̄)/Δf, ĝ = (g − ḡ)/Δg  (= 2(1 − ZNCC))
//         평균을 빼고 크기로 나누므로 밝기 더하기·곱하기 변화에 영향받지 않음
//  갱신 : 기준 쪽에서 1차 테일러 전개 → H·Δp = Σ ∇f̂ᵀ (ĝ − f̂),  H = Σ ∇f̂ ∇f̂ᵀ (2×2),  p ← p − Δp
//         H 와 ∇f̂ 는 기준 영역만으로 정해지므로 반복 전에 한 번만 계산 (역합성의 장점 = 빠름)
//  보간 : 화소 사이 값 g(x + p) 는 설정 커널(O-MOMS3)로 분리형 보간 (현재 영상은 미리 커널 계수로 바꿔 둠)
//
//  T        : 기준 사진에서의 ROI 사각형
//  init_px  : 출발 이동 (Coarse 또는 넓은 창 위상 상관 결과). 정답에서 약 ±0.5 px 안이어야 안정적으로 수렴
//  반환 false = 계산 불가 (영역이 사진 밖, 민무늬, 탐색 범위 초과, 수치 오류). 수렴 여부는 out.converged
bool icgn_translation(const cv::Mat &ref32, const cv::Mat &cur32, const cv::Rect &T, const cv::Point2d &init_px,
                      const FineParams &params, FineResult &out) {
    const Kernel K = params.kernel;
    const int R = kernel_radius(K), M = params.search_margin_px;   // R = 커널 반경(탭 2R), M = 출발점 주변 탐색 여유 (px)
    const int edge_px = 4;   // 5점 미분에 필요한 여백
    // 기준 영역 F : ROI 에 미분용 여백을 붙여 자름
    const cv::Rect Tr(T.x - edge_px, T.y - edge_px, T.width + 2 * edge_px, T.height + 2 * edge_px);
    if (!rect_inside(Tr, ref32.size())) return false;
    const cv::Mat F = crop_blur64(ref32, Tr, params.gauss_sigma_px);
    // 현재 영역 C : 출발점(정수로 반올림) 기준으로 ROI ± M 을 잘라 커널 계수로 바꿔 둠 (반복 중에는 여기서만 보간)
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
            // 5점 중앙 차분 f'(x) ≈ [−f(x+2) + 8f(x+1) − 8f(x−1) + f(x−2)] / 12  (오차 O(h⁴), 3점 차분보다 정확)
            gx[j * W + i] = (-row[x + 2] + 8 * row[x + 1] - 8 * row[x - 1] + row[x - 2]) / 12.0;
            gy[j * W + i] = (-F.at<double>(y + 2, x) + 8 * F.at<double>(y + 1, x) - 8 * F.at<double>(y - 1, x) + F.at<double>(y - 2, x)) / 12.0;
        }
    // f̂ = (f − 평균) / 크기, 기울기도 같은 크기로 나눔 → 헤시안 H = Σ ∇f̂ ∇f̂ᵀ
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
    if (std::abs(det) < 1e-18) return false;   // 헤시안이 특이 = 한 방향 질감만 있어 이동을 정할 수 없음

    cv::Point2d p = init_px;
    double zncc = 0.0;
    bool converged = false;
    int iterations = 0;
    std::vector<double> g(n), tmp, wx(2 * R), wy(2 * R);   // g = 보간한 현재 영역, wx·wy = 가로·세로 커널 가중치
    for (int it = 0; it < params.max_iter; ++it) {
        iterations = it + 1;
        // 이동 p 의 정수부(ux, uy)와 소수부로 나누고, 소수부에 대한 커널 가중치 2R 개를 미리 계산
        const double ux = std::floor(p.x), uy = std::floor(p.y);
        for (int m = 0; m < 2 * R; ++m) {
            wx[m] = kernel_weight(K, (p.x - ux) - (m - R + 1));
            wy[m] = kernel_weight(K, (p.y - uy) - (m - R + 1));
        }
        // C 안에서 보간을 시작할 위치. 커널이 C 밖을 읽어야 하면 탐색 범위를 넘은 것
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
        // ĝ = (g − 평균) / 크기
        gm /= n;
        double dg = 0;
        for (int k = 0; k < n; ++k) { g[k] -= gm; dg += g[k] * g[k]; }
        dg = std::sqrt(dg);
        if (dg < 1e-9) return false;
        // 오른쪽 항 b = Σ ∇f̂ (ĝ − f̂),  ss = Σ (ĝ − f̂)² (ZNSSD)
        double b0 = 0, b1 = 0, ss = 0;
        for (int k = 0; k < n; ++k) {
            const double e = g[k] / dg - f[k];
            b0 += gx[k] * e; b1 += gy[k] * e; ss += e * e;
        }
        // H·Δp = b 를 2×2 크라메르 공식으로 풂
        const double dx = (h11 * b0 - h01 * b1) / det, dy = (h00 * b1 - h01 * b0) / det;
        p.x -= dx; p.y -= dy;   // 역합성 갱신
        zncc = 1.0 - 0.5 * ss;   // ZNSSD = 2(1 − ZNCC) 관계로 상관 계산
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

// ② Fine Alignment 판정 : IC-GN 을 돌리고, 미수렴이거나 상관이 하한(min_zncc)보다 낮으면 실패
bool fine_alignment(const cv::Mat &ref32, const cv::Mat &cur32, const cv::Rect &ref_rect_px, const cv::Point2d &init_px,
                    const FineParams &params, FineResult &out) {
    out = FineResult();
    const bool done = icgn_translation(ref32, cur32, ref_rect_px, init_px, params, out);
    // 계산이 끝났어도 미수렴이거나 상관이 낮으면 측정 실패 (엉뚱한 곳에 맞은 값을 s·d 에 넣지 않음)
    return done && out.converged && out.zncc >= params.min_zncc;
}

} // namespace ksh
