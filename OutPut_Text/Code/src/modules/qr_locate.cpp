// qr_locate.cpp — ① QR 3개 위치 구현 (설정 파일의 절대 좌표 박스, 기준 사진 없음)
#include "ksh/qr_locate.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>

namespace ksh {

namespace {

// 1차원 밝기 단면에서 가장 센 변(밝기 변화)의 서브픽셀 위치
//  g[i] = (p[i+1] − p[i]) × sign 의 최대점 i 를 이웃 세 값의 포물선 꼭짓점으로 보정 → i + 0.5 + off
//  sign = +1 : 어두움 → 밝음 (왼쪽·위 변),  −1 : 밝음 → 어두움 (오른쪽·아래 변)
//  gauss = true : 세 값의 로그로 포물선 (흐린 변의 기울기 봉우리는 가우시안 모양이라 치우침이 작음)
//          false : 값 그대로 포물선 (2026-11-16 Python 검토판과 같은 계산)
double subpix_edge(const std::vector<double> &p, int sign, bool gauss) {
    const int n = int(p.size()) - 1;
    if (n < 1) return -1.0;
    std::vector<double> g(n);
    for (int i = 0; i < n; ++i) g[i] = (p[i + 1] - p[i]) * sign;
    const int i = int(std::max_element(g.begin(), g.end()) - g.begin());
    double off = 0.0;
    if (i > 0 && i < n - 1) {
        const double a = g[i - 1], b = g[i], c = g[i + 1];
        if (gauss && a > 0 && b > 0 && c > 0) {
            const double la = std::log(a), lb = std::log(b), lc = std::log(c), den = la - 2.0 * lb + lc;
            off = den != 0.0 ? 0.5 * (la - lc) / den : 0.0;
        } else {
            const double den = a - 2.0 * b + c;
            off = den != 0.0 ? 0.5 * (a - c) / den : 0.0;
        }
    }
    return i + 0.5 + off;
}

// 중앙값 (짝수 개면 가운데 둘의 평균)
double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    const size_t h = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + h, v.end());
    double m = v[h];
    if (v.size() % 2 == 0) m = 0.5 * (m + *std::max_element(v.begin(), v.begin() + h));
    return m;
}

// 최소제곱 직선 v = p0·t + p1
void fit_line(const std::vector<double> &t, const std::vector<double> &v, const std::vector<char> &use, double &p0, double &p1) {
    double n = 0, st = 0, sv = 0, stt = 0, stv = 0;
    for (size_t i = 0; i < t.size(); ++i) {
        if (!use[i]) continue;
        n += 1; st += t[i]; sv += v[i]; stt += t[i] * t[i]; stv += t[i] * v[i];
    }
    const double den = n * stt - st * st;
    p0 = den != 0.0 ? (n * stv - st * sv) / den : 0.0;
    p1 = n > 0 ? (sv - p0 * st) / n : 0.0;
}

// 강건 직선 : 맞춤 → 잔차 3σ(σ = 1.4826 × 중앙 절대편차) 밖 제외, 3회
bool robust_line(const std::vector<double> &t, const std::vector<double> &v, double &p0, double &p1) {
    if (t.size() < 5) return false;
    std::vector<char> use(t.size(), 1);
    for (int it = 0; it < 3; ++it) {
        fit_line(t, v, use, p0, p1);
        std::vector<double> r(t.size()), ar;
        for (size_t i = 0; i < t.size(); ++i) {
            r[i] = v[i] - (p0 * t[i] + p1);
            if (use[i]) ar.push_back(std::abs(r[i]));
        }
        const double s = 1.4826 * median(ar) + 1e-6;
        for (size_t i = 0; i < t.size(); ++i) use[i] = std::abs(r[i]) < 3.0 * s;
    }
    return std::count(use.begin(), use.end(), 1) >= 5;
}

// 행 r0 ~ r1−1 의 열 평균 (r0 ≤ r < r1)
std::vector<double> col_mean(const cv::Mat &f, int r0, int r1) {
    std::vector<double> m(f.cols, 0.0);
    for (int r = r0; r < r1; ++r) {
        const float *p = f.ptr<float>(r);
        for (int c = 0; c < f.cols; ++c) m[c] += p[c];
    }
    for (double &x : m) x /= double(r1 - r0);
    return m;
}
std::vector<double> row_mean(const cv::Mat &f, int c0, int c1) {
    std::vector<double> m(f.rows, 0.0);
    for (int r = 0; r < f.rows; ++r) {
        const float *p = f.ptr<float>(r);
        for (int c = c0; c < c1; ++c) m[r] += p[c];
        m[r] /= double(c1 - c0);
    }
    return m;
}

} // namespace

// ①-1 대략 위치 : QR 사이 어두운 틈 → 설정 좌표 대비 가로 이동
bool coarse_qr_shift(const cv::Mat &gray8, const QrParams &P, double &dx_px, std::string &message) {
    if (P.boxes.size() < 2) { message = "qr_rois 가 2개 미만"; return false; }
    std::vector<BoxSpec> qs = P.boxes;
    std::sort(qs.begin(), qs.end(), [](const BoxSpec &a, const BoxSpec &b) { return a.cx_px < b.cx_px; });
    const int hw = qs.front().hw_px;
    double cy_sum = 0;
    for (const auto &q : qs) cy_sum += q.cy_px;
    const int cy = int(cy_sum / double(qs.size()));
    const int x0 = qs.front().cx_px - hw - P.coarse_range_px, x1 = qs.back().cx_px + hw + P.coarse_range_px;
    const cv::Rect band_rect(x0, cy - hw / 2, x1 - x0, hw / 2 * 2);
    if (band_rect.x < 0 || band_rect.y < 0 || band_rect.br().x > gray8.cols || band_rect.br().y > gray8.rows) {
        message = "대략 위치 탐색 띠가 사진 밖 (coarse_range_px · qr_rois 확인)";
        return false;
    }
    cv::Mat band;
    gray8(band_rect).convertTo(band, CV_32F);
    cv::morphologyEx(band, band, cv::MORPH_CLOSE, cv::Mat::ones(P.close_kernel_px, P.close_kernel_px, CV_8U));
    cv::GaussianBlur(band, band, cv::Size(0, 0), 2.0);
    const std::vector<double> prof = col_mean(band, 0, band.rows);
    const double thr = P.gap_dark_frac * median(prof);
    std::vector<double> gaps;
    int start = -1;
    for (int i = 0; i <= int(prof.size()); ++i) {
        const bool dark = i < int(prof.size()) && prof[i] < thr;
        if (dark && start < 0) start = i;
        if (!dark && start >= 0) {
            const int len = i - start;
            if (len >= P.gap_min_px && len <= P.gap_max_px) gaps.push_back(x0 + (start + i) / 2.0);
            start = -1;
        }
    }
    std::vector<double> expv;
    for (size_t i = 0; i + 1 < qs.size(); ++i) expv.push_back((qs[i].cx_px + qs[i + 1].cx_px) / 2.0);
    double best_off = 0, best_err = 1e30;
    bool found = false;
    for (size_t i = 0; i + expv.size() <= gaps.size(); ++i) {
        double off = 0;
        for (size_t j = 0; j < expv.size(); ++j) off += gaps[i + j] - expv[j];
        off /= double(expv.size());
        double err = 0;
        for (size_t j = 0; j < expv.size(); ++j) err = std::max(err, std::abs(gaps[i + j] - expv[j] - off));
        if (err < best_err) { best_err = err; best_off = off; found = true; }
    }
    if (!found || best_err > P.coarse_max_spread_px) {
        message = "대략 위치 : QR 사이 틈을 찾지 못함 (" + std::to_string(gaps.size()) + "개)";
        return false;
    }
    if (std::abs(best_off) > P.coarse_range_px) {   // 창 가장자리에 걸린 틈 — 허용 범위 밖 이동
        message = "대략 위치 : 이동 " + std::to_string(int(best_off)) + " px 가 허용 범위 ±" + std::to_string(P.coarse_range_px) + " px 밖";
        return false;
    }
    dx_px = best_off;
    return true;
}

// ①-2 사각형 하나의 중심 (4변 서브픽셀 → 강건 직선 → 중간선 교점)
bool qr_square_center(const cv::Mat &gray8, const BoxSpec &q, const QrParams &P, cv::Point2d &center_px, std::string &message) {
    const int hw = q.hw_px, m = P.search_px;
    const cv::Rect rr(q.cx_px - hw - m, q.cy_px - hw - m, 2 * (hw + m), 2 * (hw + m));
    if (rr.x < 0 || rr.y < 0 || rr.br().x > gray8.cols || rr.br().y > gray8.rows) { message = q.name + " 박스가 사진 밖"; return false; }
    cv::Mat roi;
    gray8(rr).convertTo(roi, CV_32F);
    cv::GaussianBlur(roi, roi, cv::Size(0, 0), P.blur_sigma_px);
    const int H = roi.rows, W = roi.cols;
    const bool G = P.edge_gauss_peak;
    // 대략 변 : 가운데 띠의 열·행 평균 단면
    const std::vector<double> colm = col_mean(roi, H / 2 - hw / 2, H / 2 + hw / 2), rowm = row_mean(roi, W / 2 - hw / 2, W / 2 + hw / 2);
    const double L = subpix_edge(std::vector<double>(colm.begin(), colm.begin() + W / 2), +1, G);
    const double R = W / 2 + subpix_edge(std::vector<double>(colm.begin() + W / 2, colm.end()), -1, G);
    const double T = subpix_edge(std::vector<double>(rowm.begin(), rowm.begin() + H / 2), +1, G);
    const double B = H / 2 + subpix_edge(std::vector<double>(rowm.begin() + H / 2, rowm.end()), -1, G);
    const int w = P.edge_half_window_px, in = P.edge_inset_px;
    const int iL = int(L), iR = int(R), iT = int(T), iB = int(B);
    if (iR - iL < 4 * in || iB - iT < 4 * in || iL - w < 0 || iT - w < 0 || iR + w + 1 > W || iB + w + 1 > H) {
        message = q.name + " 사각형 변을 찾지 못함 (가림·조명)";
        return false;
    }
    // 행마다 좌·우 변, 열마다 위·아래 변
    std::vector<double> ys, lx, rx, xs, ty, by;
    for (int y = iT + in; y < iB - in; ++y) {
        const float *p = roi.ptr<float>(y);
        std::vector<double> a(p + iL - w, p + iL + w + 1), b(p + iR - w, p + iR + w + 1);
        ys.push_back(y);
        lx.push_back(iL - w + subpix_edge(a, +1, G));
        rx.push_back(iR - w + subpix_edge(b, -1, G));
    }
    for (int x = iL + in; x < iR - in; ++x) {
        std::vector<double> a(2 * w + 1), b(2 * w + 1);
        for (int k = 0; k <= 2 * w; ++k) { a[k] = roi.at<float>(iT - w + k, x); b[k] = roi.at<float>(iB - w + k, x); }
        xs.push_back(x);
        ty.push_back(iT - w + subpix_edge(a, +1, G));
        by.push_back(iB - w + subpix_edge(b, -1, G));
    }
    double l0, l1, r0, r1, t0, t1, b0, b1;
    if (!robust_line(ys, lx, l0, l1) || !robust_line(ys, rx, r0, r1) || !robust_line(xs, ty, t0, t1) || !robust_line(xs, by, b0, b1)) {
        message = q.name + " 변 직선 맞춤 실패";
        return false;
    }
    // 위·아래 중간선과 좌·우 중간선의 교점 (2회 교대)
    const double xm = (L + R) / 2.0;
    double yc = ((t0 * xm + t1) + (b0 * xm + b1)) / 2.0;
    const double xc = ((l0 * yc + l1) + (r0 * yc + r1)) / 2.0;
    yc = ((t0 * xc + t1) + (b0 * xc + b1)) / 2.0;
    center_px = cv::Point2d(rr.x + xc, rr.y + yc);
    return true;
}

// ① 전체
QrResult locate_qr(const cv::Mat &gray8, const QrParams &P) {
    QrResult out;
    if (gray8.empty() || gray8.type() != CV_8UC1) { out.message = "입력은 8bit 회색 영상이어야 함"; return out; }
    if (!coarse_qr_shift(gray8, P, out.coarse_dx_px, out.message)) return out;
    double sx = 0, sy = 0;
    for (const auto &b : P.boxes) {
        BoxSpec q = b;
        q.cx_px = int(std::nearbyint(b.cx_px + out.coarse_dx_px));   // 짝수 반올림 (검토판과 같은 규칙)
        out.boxes_px.push_back(cv::Rect(q.cx_px - q.hw_px, q.cy_px - q.hw_px, 2 * q.hw_px, 2 * q.hw_px));
        cv::Point2d c;
        if (!qr_square_center(gray8, q, P, c, out.message)) return out;
        out.centers_px.push_back(c);
        sx += c.x; sy += c.y;
    }
    out.center_px = cv::Point2d(sx / P.boxes.size(), sy / P.boxes.size());
    out.ok = true;
    return out;
}

} // namespace ksh
