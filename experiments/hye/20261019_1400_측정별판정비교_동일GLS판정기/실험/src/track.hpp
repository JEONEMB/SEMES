// 모서리 755점 추적. 원본: analysis_grid_corners_20261019/track_corners.py
//   center: refine.py 와 같은 맞춤(9x9 창 한 번)으로 ID별 기준 위치를 정한다.
//   사진마다: QR별 템플릿 매칭(±130px)으로 정수 이동 -> 시작점 = center 맞춤 위치 + 이동
//            9x9 창으로 맞춘 뒤 창을 맞춘 위치로 옮겨 다시 맞춤(최대 3번), 마지막 위치에서 11x11 창으로 한 번 더 맞춰 두 창 차이를 진단에 쓴다.
#pragma once
#include "corner_fit.hpp"

namespace tk {

struct Meas {
    double x = 0, y = 0, rmse = 0, stab = 0, contrast = 0;
    bool valid = false;
    std::string reason;
};

struct Tracker {
    cv::Mat center;
    std::vector<cf::QRInfo> qrs;
    std::vector<cf::Corner> corners;
    std::vector<double> cx, cy;  // center 맞춤 위치 (refine.py refined_corners.csv 의 x, y)

    void init(const cv::Mat& cen) {
        center = cen;
        qrs = cf::detect_qrs(cen);
        corners = cf::structural_corners(qrs);
        cx.assign(corners.size(), 0);
        cy.assign(corners.size(), 0);
        for (size_t i = 0; i < corners.size(); ++i) {
            const auto& c = corners[i];
            auto b = cf::kBoxes[c.qr - 1];
            std::vector<double> X, Y, I;
            cf::window(cen, c.x, c.y, 4, b[0], b[1], b[2], b[3], X, Y, I);
            auto F = cf::fit(X, Y, I, c.x, c.y, c.inv, c.patch);
            cx[i] = F.p[0];
            cy[i] = F.p[1];
        }
    }

    // QR별 정수 이동 (템플릿 = center 판 영역, ±130px, TM_CCOEFF_NORMED)
    std::array<cv::Point, 3> coarse(const cv::Mat& img) const {
        std::array<cv::Point, 3> out;
        for (int k = 0; k < 3; ++k) {
            auto b = cf::kBoxes[k];
            const int pad = 130;
            cv::Mat tpl = center(cv::Range(b[1], b[3]), cv::Range(b[0], b[2]));
            cv::Mat res;
            cv::matchTemplate(img(cv::Range(b[1] - pad, b[3] + pad), cv::Range(b[0] - pad, b[2] + pad)), tpl, res, cv::TM_CCOEFF_NORMED);
            cv::Point loc;
            cv::minMaxLoc(res, nullptr, nullptr, nullptr, &loc);
            out[k] = {loc.x - pad, loc.y - pad};
        }
        return out;
    }

    std::vector<Meas> track(const cv::Mat& img) const {
        auto sh = coarse(img);
        std::vector<Meas> out(corners.size());
        std::vector<double> X, Y, I;
        for (size_t i = 0; i < corners.size(); ++i) {
            const auto& c = corners[i];
            int k = c.qr;
            int dx = sh[k - 1].x, dy = sh[k - 1].y;
            double sx = cx[i] + dx, sy = cy[i] + dy;
            auto b = cf::kBoxes[k - 1];
            double a = b[0] + dx, cc = b[2] + dx, bb = b[1] + dy, d = b[3] + dy;
            double ex = sx, ey = sy;
            cf::FitResult f1;
            for (int it = 0; it < 3; ++it) {
                cf::window(img, ex, ey, 4, a, bb, cc, d, X, Y, I);
                f1 = cf::fit(X, Y, I, sx, sy, c.inv, c.patch);
                bool moved = std::nearbyint(f1.p[0]) != std::nearbyint(ex) || std::nearbyint(f1.p[1]) != std::nearbyint(ey);
                ex = f1.p[0];
                ey = f1.p[1];
                if (!moved) break;
            }
            cf::window(img, ex, ey, 5, a, bb, cc, d, X, Y, I);
            auto f2 = cf::fit(X, Y, I, sx, sy, c.inv, c.patch);
            Meas& m = out[i];
            m.x = f1.p[0];
            m.y = f1.p[1];
            m.rmse = f1.rmse;
            m.stab = std::hypot(f1.p[0] - f2.p[0], f1.p[1] - f2.p[1]);
            m.contrast = f1.p[5];
            m.reason = cf::reasons(f1.ok, f2.ok, f1.rmse, f1.p[5], m.stab, std::max(std::abs(f1.p[0] - sx), std::abs(f1.p[1] - sy)), std::max(f1.p[2], f1.p[3]));
            m.valid = m.reason.empty();
        }
        return out;
    }
};

}  // namespace tk
