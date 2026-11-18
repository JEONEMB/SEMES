// hole_locate.cpp — ② 뒷벽 홀 중심 구현 (방사 대칭 중심)
#include "ksh/hole_locate.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>

namespace ksh {

// Parthasarathy (2012) 방사 대칭 중심
//  ① 대각선 방향 밝기 차 dI/du, dI/dv (2×2 Roberts, 화소 사이 격자점) → 3×3 평균
//  ② 격자점마다 기울기 방향 직선 (기울기 m, 절편 b)
//  ③ 각 직선까지 거리² 를 |∇I|² / (중심 후보까지 거리) 로 가중한 합이 최소인 점 (닫힌 해)
bool radial_symmetry_center(const cv::Mat &roi32, double &xc, double &yc, int robust_iter, double max_line_dist_px) {
    cv::Mat I;
    roi32.convertTo(I, CV_64F);
    const int Ny = I.rows, Nx = I.cols;
    if (Ny < 4 || Nx < 4) return false;
    cv::Mat du = I(cv::Rect(1, 0, Nx - 1, Ny - 1)) - I(cv::Rect(0, 1, Nx - 1, Ny - 1));
    cv::Mat dv = I(cv::Rect(0, 0, Nx - 1, Ny - 1)) - I(cv::Rect(1, 1, Nx - 1, Ny - 1));
    const cv::Mat k = cv::Mat::ones(3, 3, CV_64F) / 9.0;
    cv::filter2D(du, du, -1, k);
    cv::filter2D(dv, dv, -1, k);
    const int H = Ny - 1, W = Nx - 1;
    cv::Mat mag2 = du.mul(du) + dv.mul(dv), mm(H, W, CV_64F), bb(H, W, CV_64F);
    double s2 = 0, sx = 0, sy = 0;
    for (int r = 0; r < H; ++r)
        for (int c = 0; c < W; ++c) {
            const double u = du.at<double>(r, c), v = dv.at<double>(r, c), xm = c + 0.5, ym = r + 0.5;
            double m = -(v + u) / (u - v + 1e-12);
            if (!std::isfinite(m)) m = 0.0;
            mm.at<double>(r, c) = m;
            bb.at<double>(r, c) = ym - m * xm;
            const double g = mag2.at<double>(r, c);
            s2 += g; sx += g * xm; sy += g * ym;
        }
    if (s2 <= 0) return false;
    const double xcen = sx / s2, ycen = sy / s2;
    // 가중 최소제곱 (중심 후보까지 거리로 나눈 |∇I|² 가중). robust_iter > 0 이면 중심에서 max_line_dist_px 보다 먼 기울기 직선
    //  (먼지·이물·다른 모서리의 기울기)을 빼고 다시 풂 — 둥근 홀의 기울기 직선은 모두 중심 근처를 지남
    cv::Mat use(H, W, CV_8U, cv::Scalar(1));
    double cx0 = xcen, cy0 = ycen;
    for (int pass = 0; pass <= robust_iter; ++pass) {
        double sw = 0, smmw = 0, smw = 0, smbw = 0, sbw = 0;
        for (int r = 0; r < H; ++r)
            for (int c = 0; c < W; ++c) {
                if (!use.at<uchar>(r, c)) continue;
                const double xm = c + 0.5, ym = r + 0.5, m = mm.at<double>(r, c), b = bb.at<double>(r, c);
                const double w = mag2.at<double>(r, c) / std::sqrt((xm - cx0) * (xm - cx0) + (ym - cy0) * (ym - cy0) + 1e-9);
                const double wm = w / (m * m + 1.0);
                sw += wm; smmw += m * m * wm; smw += m * wm; smbw += m * b * wm; sbw += b * wm;
            }
        const double det = smw * smw - smmw * sw;
        if (det == 0.0 || !std::isfinite(det)) return false;
        xc = (smbw * sw - smw * sbw) / det;
        yc = (smbw * smw - smmw * sbw) / det;
        if (pass == robust_iter) break;
        // 다음 회 : 가중 중심을 새 중심으로, 먼 직선 제외 (직선 y = m·x + b 와 점 사이 거리)
        cx0 = xc;
        cy0 = yc;
        for (int r = 0; r < H; ++r)
            for (int c = 0; c < W; ++c) {
                const double m = mm.at<double>(r, c), b = bb.at<double>(r, c);
                use.at<uchar>(r, c) = std::abs(m * xc - yc + b) / std::sqrt(m * m + 1.0) <= max_line_dist_px;
            }
    }
    return std::isfinite(xc) && std::isfinite(yc);
}

HoleResult locate_hole(const cv::Mat &gray8, const BoxSpec &b, const HoleParams &P, double dx_px) {
    HoleResult out;
    out.name = b.name;
    const int cx = int(std::nearbyint(b.cx_px + dx_px)), cy = b.cy_px, hw = b.hw_px, m = P.search_px;
    out.box_px = cv::Rect(cx - hw, cy - hw, 2 * hw, 2 * hw);
    const cv::Rect rr(cx - hw - m, cy - hw - m, 2 * (hw + m), 2 * (hw + m));
    if (rr.x < 0 || rr.y < 0 || rr.br().x > gray8.cols || rr.br().y > gray8.rows) { out.message = b.name + " 박스가 사진 밖"; return out; }
    cv::Mat roi, sm;
    gray8(rr).convertTo(roi, CV_32F);
    cv::GaussianBlur(roi, roi, cv::Size(0, 0), P.blur_sigma_px);
    if (P.locate_radius_px > 0) {
        // 대략 위치 = 반지름 locate_radius_px 원판 평균 밝기가 가장 어두운 곳 (홀 크기의 어두운 덩어리를 고름 — 작은 먼지 점은 원판 평균에 거의 안 걸림)
        const int R = P.locate_radius_px;
        cv::Mat disc = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * R + 1, 2 * R + 1));
        disc.convertTo(disc, CV_32F, 1.0 / cv::sum(disc)[0]);
        cv::filter2D(roi, sm, -1, disc, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
    } else {
        cv::GaussianBlur(roi, sm, cv::Size(0, 0), P.locate_sigma_px);   // Python 검토판과 같은 방식
    }
    double vmin;
    cv::Point pmin;
    cv::minMaxLoc(sm, &vmin, nullptr, &pmin, nullptr);
    // 가림·조명 검사 : 박스 위·아래 테두리 4줄 밝기 중앙값 − 홀 최소 밝기
    std::vector<float> border;
    for (int r : {0, 1, 2, 3, roi.rows - 4, roi.rows - 3, roi.rows - 2, roi.rows - 1})
        for (int c = 0; c < roi.cols; ++c) border.push_back(roi.at<float>(r, c));
    std::nth_element(border.begin(), border.begin() + border.size() / 2, border.end());
    out.contrast = border[border.size() / 2] - vmin;
    if (out.contrast < P.min_contrast) { out.message = b.name + " 홀이 보이지 않음 (명암 차 " + std::to_string(int(out.contrast)) + ")"; return out; }
    double xx = pmin.x, yy = pmin.y;
    const int h = P.window_half_px;
    for (int it = 0; it < 3; ++it) {
        const int a = int(std::nearbyint(xx)) - h, c = int(std::nearbyint(yy)) - h;
        const cv::Rect win(a, c, 2 * h + 1, 2 * h + 1);
        if (win.x < 0 || win.y < 0 || win.br().x > roi.cols || win.br().y > roi.rows) { out.message = b.name + " 홀이 박스 가장자리 (search_px 늘리기)"; return out; }
        double xc, yc;
        if (!radial_symmetry_center(roi(win), xc, yc, P.robust_iter, P.max_line_dist_px)) { out.message = b.name + " 방사 대칭 계산 실패"; return out; }
        xx = a + xc;
        yy = c + yc;
    }
    // 모양 검사 : 마지막 창에서 '충분히 어두운' 화소(테두리 − 명암 차의 절반보다 어두움)의 무게중심이 방사 대칭 중심과 가까워야 함
    //  둥근 홀(링)은 두 중심이 거의 같고, 가림 · 그림자 · 모서리처럼 둥글지 않은 어두운 영역은 크게 어긋남
    {
        const int a = int(std::nearbyint(xx)) - h, c = int(std::nearbyint(yy)) - h;
        const cv::Rect win(a, c, 2 * h + 1, 2 * h + 1);
        const float thr = float(border[border.size() / 2] - 0.5 * out.contrast);
        double sw = 0, sx = 0, sy = 0;
        if (win.x >= 0 && win.y >= 0 && win.br().x <= roi.cols && win.br().y <= roi.rows)
            for (int r = 0; r < win.height; ++r)
                for (int q = 0; q < win.width; ++q) {
                    const double w = thr - roi.at<float>(win.y + r, win.x + q);
                    if (w > 0) { sw += w; sx += w * (win.x + q); sy += w * (win.y + r); }
                }
        out.center_gap_px = sw > 0 ? std::hypot(sx / sw - xx, sy / sw - yy) : 1e9;
        if (out.center_gap_px > P.max_center_gap_px) {
            out.message = b.name + " 홀 모양이 둥글지 않음 (중심 차 " + std::to_string(out.center_gap_px).substr(0, 4) + " px, 가림·이물 확인)";
            return out;
        }
    }
    out.center_px = cv::Point2d(rr.x + xx, rr.y + yy);
    out.ok = true;
    return out;
}

} // namespace ksh
