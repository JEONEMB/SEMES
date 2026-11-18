// qr_locate.cpp — ① QR 3개 위치 찾기 구현
//  (A507 측정 프로그램 rigpose 에서 옮겨 옴. 계산은 바꾸지 않음 → 같은 사진에서 같은 값)
#include "ksh/qr_locate.hpp"

#include "ksh/geometry.hpp"

#include <algorithm>
#include <cmath>

namespace ksh {

using namespace geo;

namespace {

struct Box { int x = 0, y = 0, w = 0, h = 0, area = 0; };

bool find_strip(const View& im, const QrParams& P, Box& out, std::string& err) {
    const int S = 4;
    int x0 = 0, y0 = 0, x1 = im.width, y1 = im.height;
    if (P.search_roi.area() > 0) { x0 = P.search_roi.x; y0 = P.search_roi.y; x1 = P.search_roi.x + P.search_roi.width; y1 = P.search_roi.y + P.search_roi.height; }
    x0 = std::max(0, x0); y0 = std::max(0, y0); x1 = std::min(im.width, x1); y1 = std::min(im.height, y1);
    int W = (x1 - x0) / S, H = (y1 - y0) / S;
    if (W < 10 || H < 10) { err = "search_roi 가 너무 작음"; return false; }
    double thr = P.strip_dark_thr;
    std::vector<uint8_t> m((size_t)W * H);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            int s = 0;
            for (int dy = 0; dy < S; ++dy) { const uint8_t* p = im.data + (size_t)(y0 + y * S + dy) * im.stride + x0 + x * S; for (int dx = 0; dx < S; ++dx) s += p[dx]; }
            m[(size_t)y * W + x] = (s < thr * S * S) ? 1 : 0;
        }
    std::vector<int> stack; Box best; bool any = false;
    double minw = P.strip_min_w_px / S, amin = P.strip_aspect_min, amax = P.strip_aspect_max;
    for (int sy = 0; sy < H; ++sy)
        for (int sx = 0; sx < W; ++sx) {
            if (m[(size_t)sy * W + sx] != 1) continue;
            int bx0 = sx, bx1 = sx, by0 = sy, by1 = sy, area = 0;
            stack.clear(); stack.push_back(sy * W + sx); m[(size_t)sy * W + sx] = 2;
            while (!stack.empty()) {
                int p = stack.back(); stack.pop_back();
                int px = p % W, py = p / W; ++area;
                bx0 = std::min(bx0, px); bx1 = std::max(bx1, px); by0 = std::min(by0, py); by1 = std::max(by1, py);
                const int nx[4] = {px - 1, px + 1, px, px}, ny[4] = {py, py, py - 1, py + 1};
                for (int k = 0; k < 4; ++k) {
                    if (nx[k] < 0 || ny[k] < 0 || nx[k] >= W || ny[k] >= H) continue;
                    uint8_t& q = m[(size_t)ny[k] * W + nx[k]];
                    if (q == 1) { q = 2; stack.push_back(ny[k] * W + nx[k]); }
                }
            }
            int bw = bx1 - bx0 + 1, bh = by1 - by0 + 1;
            double asp = (double)bw / bh;
            if (bw >= minw && asp >= amin && asp <= amax && area > best.area) {
                best = {x0 + bx0 * S, y0 + by0 * S, bw * S, bh * S, area}; any = true;
            }
        }
    if (!any) { err = "QR 띠를 찾지 못함"; return false; }
    out = best;
    return true;
}

// 흰 QR 사각형 3개의 대략 위치 (열·행 밝기 투영)

bool find_squares(const View& im, const QrParams& P, const Box& st, std::vector<Box>& sq, std::string& err) {
    double frac = P.thr_frac;       // 어두움 + frac × (밝음 − 어두움)
    int sm = P.proj_smooth_px;
    int ya = st.y + st.h / 4, yb = st.y + st.h * 3 / 4;
    std::vector<double> col(st.w, 0.0);
    for (int x = 0; x < st.w; ++x) {
        double s = 0;
        for (int y = ya; y < yb; ++y) s += im.at(st.x + x, y);
        col[x] = s / (yb - ya);
    }
    col = box_smooth(col, sm);
    double dark = percentile(col, 0.10), bright = percentile(col, 0.97);
    double cthr = dark + frac * (bright - dark);
    std::vector<std::pair<int, int>> runs;
    int gap_merge = P.gap_merge_px;
    for (int x = 0; x < st.w;) {
        if (col[x] < cthr) { ++x; continue; }
        int a = x; while (x < st.w && col[x] >= cthr) ++x;
        if (!runs.empty() && a - runs.back().second < gap_merge) runs.back().second = x;
        else runs.push_back({a, x});
    }
    double smin = P.size_min_px, smax = P.size_max_px;
    sq.clear();
    for (auto& r : runs) {
        int w = r.second - r.first;
        if (w < smin || w > smax) continue;
        int ys0 = std::max(0, st.y - 20), ys1 = std::min(im.height, st.y + st.h + 20);
        std::vector<double> row(ys1 - ys0, 0.0);
        for (int y = ys0; y < ys1; ++y) { double s = 0; for (int x = r.first; x < r.second; ++x) s += im.at(st.x + x, y); row[y - ys0] = s / w; }
        row = box_smooth(row, sm);
        int c = (st.y + st.h / 2) - ys0, a = c, b = c;
        while (a > 0 && row[a - 1] >= cthr) --a;
        while (b < (int)row.size() - 1 && row[b + 1] >= cthr) ++b;
        int h = b - a + 1;
        if (h < smin || h > smax) continue;
        sq.push_back({st.x + r.first, ys0 + a, w, h, w * h});
    }
    if (sq.size() != 3) { err = "QR 사각형 " + std::to_string(sq.size()) + "개 (3개 필요)"; return false; }
    return true;
}

// 사각형 4변 → 꼭짓점 → 중심
bool square_center(const View& im, const QrParams& P, const Box& b, double& cx, double& cy, double& top_ang, double& rms) {
    double m = P.edge_margin, search = P.edge_search_px, plat = P.edge_plateau_px;
    int nseg = P.edge_segments;
    double xm = b.x + b.w * 0.5, ym = b.y + b.h * 0.5;
    // 가로변: 위(어두움→밝음, 아래로 +), 아래(밝음→어두움)
    EdgeLine T = measure_line(im, false, b.y, 0, xm, b.x + b.w * m, b.x + b.w * (1 - m), +1, search, plat, nseg, true);
    EdgeLine B = measure_line(im, false, b.y + b.h, 0, xm, b.x + b.w * m, b.x + b.w * (1 - m), -1, search, plat, nseg, true);
    EdgeLine Lf = measure_line(im, true, b.x, 0, ym, b.y + b.h * m, b.y + b.h * (1 - m), +1, search, plat, nseg, true);
    EdgeLine R = measure_line(im, true, b.x + b.w, 0, ym, b.y + b.h * m, b.y + b.h * (1 - m), -1, search, plat, nseg, true);
    if (!T.ok || !B.ok || !Lf.ok || !R.ok) return false;
    // 가로변 y = Y0 + s (x - xm), 세로변 x = X0 + s (y - ym)
    auto inter = [&](const EdgeLine& v, const EdgeLine& h, double& x, double& y) {
        // x = v.pos + v.s (y - ym); y = h.pos + h.s (x - xm)
        double den = 1 - v.slope * h.slope;
        y = (h.pos + h.slope * (v.pos - xm - v.slope * ym)) / den;
        x = v.pos + v.slope * (y - ym);
    };
    double px[4], py[4];
    inter(Lf, T, px[0], py[0]); inter(R, T, px[1], py[1]); inter(R, B, px[2], py[2]); inter(Lf, B, px[3], py[3]);
    cx = (px[0] + px[1] + px[2] + px[3]) / 4; cy = (py[0] + py[1] + py[2] + py[3]) / 4;
    top_ang = std::atan(0.5 * (T.slope + B.slope)) * 180.0 / kPi;
    rms = (T.rms + B.rms + Lf.rms + R.rms) / 4;
    return true;
}



} // namespace

// ① 1/4 축소 영상에서 어두운 QR 띠 → 띠 안 흰 사각형 3개(열·행 밝기 투영) → 사각형마다 4변 서브픽셀 직선 → 꼭짓점 → 중심
QrResult locate_qr(const cv::Mat &gray8, const QrParams &P) {
    QrResult r;
    if (gray8.empty() || gray8.type() != CV_8UC1) { r.message = "8bit 회색 영상이 아님"; return r; }
    const View im = view(gray8);
    std::string err;
    Box st;
    if (!find_strip(im, P, st, err)) { r.message = err; return r; }
    r.strip_px = cv::Rect(st.x, st.y, st.w, st.h);
    std::vector<Box> sq;
    if (!find_squares(im, P, st, sq, err)) { r.message = err; return r; }
    double ang = 0, rms = 0;
    for (int i = 0; i < 3; ++i) {
        double cx, cy, a, e;
        if (!square_center(im, P, sq[i], cx, cy, a, e)) { r.message = "QR 사각형 모서리 측정 실패"; return r; }
        r.squares_px[i] = cv::Rect(sq[i].x, sq[i].y, sq[i].w, sq[i].h);
        r.centers_px[i] = cv::Point2d(cx, cy);
        ang += a / 3; rms += e / 3;
    }
    r.center_px = (r.centers_px[0] + r.centers_px[1] + r.centers_px[2]) / 3.0;
    r.edge_fit_rms_px = rms;
    r.ok = true;
    return r;
}

} // namespace ksh
