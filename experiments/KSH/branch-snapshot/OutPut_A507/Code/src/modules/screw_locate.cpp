// screw_locate.cpp — ② 나사 ROI 위치 추정 + 머리 테두리 원 맞춤 구현
//  (A507 측정 프로그램 rigpose 에서 옮겨 옴. 계산은 바꾸지 않음 → 같은 사진에서 같은 값)
#include "ksh/screw_locate.hpp"

#include "ksh/geometry.hpp"

#include <algorithm>
#include <cmath>

namespace ksh {

using namespace geo;

namespace {

struct ScrewFit { bool ok = false; double x = 0, y = 0, r = 0, res = 0, contrast = 0; };

// (xp, yp) 를 중심으로 한 ROI 에서 가장 어두운 둥근 덩어리를 찾고, 머리 바깥 테두리를 방사선 rays 개로 잡아 원을 맞춘다
ScrewFit screw_fit(const View& im, double xp, double yp, const ScrewParams& P) {
    ScrewFit R;
    const int HW = P.roi_half_width_px, HH = P.roi_half_height_px;
    double rmax = P.rmax_px, frac = P.thr_frac, sig = P.blur_sigma_px;
    int nray = P.rays;
    int x0 = (int)std::lround(xp) - HW, y0 = (int)std::lround(yp) - HH, w = 2 * HW, h = 2 * HH;
    if (x0 < 0 || y0 < 0 || x0 + w > im.width || y0 + h > im.height) return R;
    std::vector<float> raw((size_t)w * h);
    for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) raw[(size_t)y * w + x] = im.at(x0 + x, y0 + y);
    std::vector<float> big = raw, bl = raw;
    gauss_blur(big, w, h, 8.0);
    gauss_blur(bl, w, h, sig);
    double cx = 0, cy = 0; float mn = 1e9f;
    for (int y = 40; y < h - 40; ++y) for (int x = 40; x < w - 40; ++x) if (big[(size_t)y * w + x] < mn) { mn = big[(size_t)y * w + x]; cx = x; cy = y; }
    const double step = 0.25;
    int nr = (int)((rmax + 12) / step);
    std::vector<double> prof(nr);
    for (int it = 0; it < 3; ++it) {
        std::vector<std::vector<double>> P(nray, std::vector<double>(nr));
        std::vector<double> bgv, cov;
        for (int k = 0; k < nray; ++k) {
            double a = 2 * kPi * k / nray, ca = std::cos(a), sa = std::sin(a);
            for (int i = 0; i < nr; ++i) {
                double r = i * step;
                P[k][i] = bil(bl, w, h, cx + r * ca, cy + r * sa);
                if (r > 44 && r < 58) bgv.push_back(P[k][i]);
                if (r < 14) cov.push_back(P[k][i]);
            }
        }
        double bg = median_of(bgv), core = median_of(cov);
        double thr = bg - frac * (bg - core);
        R.contrast = bg - core;
        std::vector<double> px, py;
        for (int k = 0; k < nray; ++k) {
            double a = 2 * kPi * k / nray;
            int last = -1;
            for (int i = 0; i + 1 < nr; ++i) {
                double r = i * step;
                if (r <= 15 || r >= rmax) continue;
                if (P[k][i] < thr && P[k][i + 1] >= thr) last = i;
            }
            if (last < 0) continue;
            double r = last * step + (thr - P[k][last]) / (P[k][last + 1] - P[k][last]) * step;
            px.push_back(cx + r * std::cos(a)); py.push_back(cy + r * std::sin(a));
        }
        if ((int)px.size() < nray / 2) return R;
        std::vector<char> in(px.size(), 1);
        double ccx = cx, ccy = cy, rad = 0, sd = 0; int nin = 0;
        for (int rep = 0; rep < 3; ++rep) {
            std::vector<double> A, b; nin = 0;
            for (size_t j = 0; j < px.size(); ++j) if (in[j]) { A.push_back(2 * px[j]); A.push_back(2 * py[j]); A.push_back(1); b.push_back(px[j] * px[j] + py[j] * py[j]); ++nin; }
            std::vector<double> c;
            if (nin < 10 || !lstsq(A, nin, 3, b, c)) return R;
            ccx = c[0]; ccy = c[1]; rad = std::sqrt(std::max(0.0, c[2] + c[0] * c[0] + c[1] * c[1]));
            std::vector<double> d(px.size()), ad;
            for (size_t j = 0; j < px.size(); ++j) { d[j] = std::hypot(px[j] - ccx, py[j] - ccy) - rad; if (in[j]) ad.push_back(std::fabs(d[j])); }
            double s = 1.4826 * median_of(ad) + 1e-6;
            double ss = 0; int n = 0;
            for (size_t j = 0; j < px.size(); ++j) { in[j] = std::fabs(d[j]) < 3 * s; if (in[j]) { ss += d[j] * d[j]; ++n; } }
            sd = n ? std::sqrt(ss / n) : 0;
        }
        cx = ccx; cy = ccy; R.r = rad; R.res = sd;
    }
    R.x = x0 + cx; R.y = y0 + cy; R.ok = true;
    return R;
}


} // namespace

// ROI 중심 추정 : 센터 사진의 나사 위치를 이번 프레임의 QR 이동량만큼 옮김
cv::Point2d predict_screw_center(const ScrewSpec &spec, const ScrewParams &P, const cv::Point2d &qr_center_px) {
    return cv::Point2d(spec.ref_center_px.x + P.follow_x * (qr_center_px.x - P.qr_ref_px.x),
                       spec.ref_center_px.y + P.follow_y * (qr_center_px.y - P.qr_ref_px.y));
}

ScrewResult locate_screw(const cv::Mat &gray8, const ScrewSpec &spec, const ScrewParams &P, const cv::Point2d &qr_center_px) {
    ScrewResult r;
    r.name = spec.name;
    const cv::Point2d c = predict_screw_center(spec, P, qr_center_px);
    r.roi_px = cv::Rect(int(std::lround(c.x)) - P.roi_half_width_px, int(std::lround(c.y)) - P.roi_half_height_px,
                        2 * P.roi_half_width_px, 2 * P.roi_half_height_px);
    if (gray8.empty() || gray8.type() != CV_8UC1) { r.message = "8bit 회색 영상이 아님"; return r; }
    const ScrewFit f = screw_fit(view(gray8), c.x, c.y, P);
    if (!f.ok) { r.message = spec.name + " 측정 실패 (ROI 가 사진 밖이거나 테두리를 못 잡음)"; return r; }
    r.center_px = cv::Point2d(f.x, f.y);
    r.radius_px = f.r;
    r.fit_rms_px = f.res;
    r.contrast = f.contrast;
    if (f.contrast < P.min_contrast) { r.message = spec.name + " 대비 부족 (가림·조명 확인)"; return r; }
    r.ok = true;
    return r;
}

} // namespace ksh
