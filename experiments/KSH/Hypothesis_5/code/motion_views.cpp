// =====================================================================================
// motion_views.cpp — 운동 모델 시각화 (화면 구성은 motion_views.hpp 참고)
// =====================================================================================
#include "motion_views.hpp"
#include "plot_utils.hpp"

#include <opencv2/imgproc.hpp>
#include <cmath>
#include <vector>

namespace mm {

namespace {

const cv::Size kCanvas(1600, 900);

cv::Mat background(const cv::Mat& refSmall, cv::Size sz) {
    cv::Mat bg;
    if (refSmall.empty()) bg = cv::Mat(sz, CV_8UC3, cv::Scalar(70, 70, 70));
    else cv::resize(refSmall, bg, sz, 0, 0, cv::INTER_AREA);
    bg = bg * 0.6;   // 화살표가 잘 보이도록 어둡게
    return bg;
}

// 화면 전체 격자 화살표 : 가장 긴 화살표가 패널에서 targetPx 가 되도록 자동 확대 (× gainMul)
//  반환값 = 실제 확대 배율 (영상 px 기준) — 화면에 표기
template <class F>
double drawArrows(cv::Mat& panel, const CameraModel& cam, F field, double gainMul, const cv::Scalar& c, int step = 160,
                  double targetPx = 45.0) {
    const double s = double(panel.cols) / cam.width;
    double dmax = 1e-12;
    for (int y = step / 2; y < cam.height; y += step)
        for (int x = step / 2; x < cam.width; x += step) dmax = std::max(dmax, cv::norm(field(cv::Point2d(x, y))));
    const double gain = targetPx / (dmax * s) * gainMul;
    if (dmax < 1e-6) {  // 움직임 없음 → 점만
        for (int y = step / 2; y < cam.height; y += step)
            for (int x = step / 2; x < cam.width; x += step) cv::circle(panel, cv::Point(int(x * s), int(y * s)), 1, c, cv::FILLED);
        return 0.0;
    }
    for (int y = step / 2; y < cam.height; y += step)
        for (int x = step / 2; x < cam.width; x += step) {
            const cv::Point2d p(x, y), d = field(p);
            const cv::Point a(int(p.x * s), int(p.y * s));
            const cv::Point b(int((p.x + d.x * gain) * s), int((p.y + d.y * gain) * s));
            if (a == b) { cv::circle(panel, a, 1, c, cv::FILLED); continue; }
            cv::arrowedLine(panel, a, b, c, 1, cv::LINE_AA, 0, 0.25);
        }
    return gain;
}

void drawQuads(cv::Mat& panel, const CameraModel& cam, const Pose& pose, const cv::Scalar& c) {
    const double s = double(panel.cols) / cam.width;
    for (const auto& q : cam.qrCorners(pose)) {
        std::vector<cv::Point> p;
        for (const auto& v : q) p.emplace_back(int(v.x * s), int(v.y * s));
        cv::polylines(panel, p, true, c, 1, cv::LINE_AA);
    }
}

// 격자 위 값 지도 (step px 간격) → 패널 크기로 확대
template <class F>
cv::Mat valueMap(const CameraModel& cam, F value, int step, double& absMax) {
    cv::Mat m(cam.height / step, cam.width / step, CV_64F);
    absMax = 0;
    for (int j = 0; j < m.rows; ++j)
        for (int i = 0; i < m.cols; ++i) {
            const double v = value(cv::Point2d(i * step + step / 2.0, j * step + step / 2.0));
            m.at<double>(j, i) = v;
            absMax = std::max(absMax, std::abs(v));
        }
    return m;
}

void blit(cv::Mat& canvas, const cv::Mat& img, cv::Point at) { img.copyTo(canvas(cv::Rect(at, img.size()))); }

cv::Mat heat(const cv::Mat& m, double lim, cv::Size sz) {
    cv::Mat c = signedColorMap(m, std::max(lim, 1e-6)), out;
    cv::resize(c, out, sz, 0, 0, cv::INTER_LINEAR);
    return out;
}

} // namespace

// -------------------------------------------------------------------------------------
cv::Mat renderFieldView(const CameraModel& cam, const Pose& pose, double gain, const cv::Mat& refSmall) {
    cv::Mat cv_(kCanvas, CV_8UC3, col::white);
    text(cv_, fmt("[1] Field  |  Y = %+.3f mm,  Theta = %+.4f deg   (plate depth %.0f mm)", pose.y_mm, pose.t_deg, cam.z_plate),
         cv::Point(20, 26), 0.6, col::black, 1);

    // 위 : 변위 화살표 + QR 위치
    cv::Mat top = background(refSmall, cv::Size(960, 540));
    drawQuads(top, cam, Pose{}, col::gray);
    drawQuads(top, cam, pose, col::green);
    const double g1 = drawArrows(top, cam, [&](const cv::Point2d& p) { return cam.displacement(p, cam.z_plate, pose); }, gain, col::cyan);
    blit(cv_, top, cv::Point(20, 40));
    text(cv_, fmt("arrows x%.1f", g1), cv::Point(28, 570), 0.5, col::yellow);

    // 아래 : 평균(플레이트 중심) 이동을 뺀 나머지 = 원근 성분
    const cv::Point2d d0 = cam.displacement(cam.plate_center, cam.z_plate, pose);
    double mu = 0, mv = 0;
    const cv::Mat ru = valueMap(cam, [&](const cv::Point2d& p) { return cam.displacement(p, cam.z_plate, pose).x - d0.x; }, 40, mu);
    const cv::Mat rv = valueMap(cam, [&](const cv::Point2d& p) { return cam.displacement(p, cam.z_plate, pose).y - d0.y; }, 40, mv);
    const double lim = std::max(mu, mv);
    text(cv_, fmt("du - du(plate)   [range +-%.3f px]", lim), cv::Point(20, 606), 0.5);
    text(cv_, fmt("dv - dv(plate)   [range +-%.3f px]", lim), cv::Point(510, 606), 0.5);
    blit(cv_, heat(ru, lim, cv::Size(470, 264)), cv::Point(20, 614));
    blit(cv_, heat(rv, lim, cv::Size(470, 264)), cv::Point(510, 614));

    // 오른쪽 : 수치
    const QrPrediction q0 = predictQr(cam, Pose{}), q = predictQr(cam, pose);
    const cv::Point2d bg = cam.displacement(cam.plate_center, cam.z_bg, pose);
    const cv::Point2d lin = cam.displacementLinear(cam.plate_center, cam.z_plate, pose);
    int y = 70; const int x = 1010;
    auto line = [&](const std::string& s, const cv::Scalar& c = col::black) { text(cv_, s, cv::Point(x, y), 0.5, c); y += 24; };
    line("Plate (QR 3) at depth " + fmt("%.0f mm", cam.z_plate), col::blue);
    line(fmt("  center shift   du = %+.3f px, dv = %+.3f px", d0.x, d0.y));
    line(fmt("  small-angle    du = %+.3f px  (err %.4f)", lin.x, lin.x - d0.x));
    line(fmt("  QR width L/C/R  %.3f  %.3f  %.3f", q.w[0], q.w[1], q.w[2]));
    line(fmt("  width diff R-L  %.4f px  (change %+.4f)", q.w_diff, q.w_diff - q0.w_diff));
    line(fmt("  height diff R-L %.4f px  (change %+.4f)", q.h_diff, q.h_diff - q0.h_diff));
    y += 10;
    line("Background at depth " + fmt("%.0f mm", cam.z_bg), col::orange);
    line(fmt("  shift at plate position du = %+.3f px", bg.x));
    line(fmt("  parallax (plate - bg)   = %+.3f px", d0.x - bg.x));
    y += 10;
    line("How to read", col::purple);
    line("  cyan arrows : motion of each image point");
    line("  green boxes : QR now,  gray : QR at reference");
    line("  heat maps   : motion left after removing");
    line("                the plate-center shift");
    line("  Y  -> heat map flat (lens distortion only)");
    line("  Theta -> U-shape in du, saddle in dv");
    line("  => only Theta changes the QR shapes");
    y += 10;
    line("Keys: 1-5 view, s save, n/p image, q quit", col::gray);
    return cv_;
}

// -------------------------------------------------------------------------------------
cv::Mat renderSeparationView(const CameraModel& cam, double yEq, double gain, const cv::Mat& refSmall) {
    const Sensitivity s = sensitivities(cam);
    const Pose pY{yEq, 0.0}, pT{0.0, yEq * s.equal_shift_deg_per_mm};
    cv::Mat cv_(kCanvas, CV_8UC3, col::white);
    text(cv_, fmt("[2] Separation  |  Y = %.2f mm  vs  Theta = %.4f deg  give the SAME plate shift (%.2f px)", pY.y_mm, pT.t_deg,
                  s.plate_per_mm * yEq), cv::Point(20, 26), 0.6);

    const char* titles[3] = {"Y only : motion - plate shift", "Theta only : motion - plate shift", "Difference : Theta - Y"};
    auto fY = [&](const cv::Point2d& p) { return cam.displacement(p, cam.z_plate, pY); };
    auto fT = [&](const cv::Point2d& p) { return cam.displacement(p, cam.z_plate, pT); };
    const cv::Point2d cY = fY(cam.plate_center), cT = fT(cam.plate_center);

    double mY = 0, mT = 0, mD = 0;
    const cv::Mat hY = valueMap(cam, [&](const cv::Point2d& p) { return fY(p).x - cY.x; }, 40, mY);
    const cv::Mat hT = valueMap(cam, [&](const cv::Point2d& p) { return fT(p).x - cT.x; }, 40, mT);
    const cv::Mat hD = valueMap(cam, [&](const cv::Point2d& p) { return fT(p).x - fY(p).x; }, 40, mD);
    const double lim = std::max({mY, mT, mD, 1e-6});

    const cv::Size psz(500, 281);
    for (int k = 0; k < 3; ++k) {
        const int x0 = 20 + k * 530;
        text(cv_, titles[k], cv::Point(x0, 58), 0.52, col::black);
        cv::Mat pan = background(refSmall, psz);
        double g = 0;
        if (k == 0) g = drawArrows(pan, cam, [&](const cv::Point2d& p) { return fY(p) - cY; }, gain, col::cyan, 200, 30);
        if (k == 1) g = drawArrows(pan, cam, [&](const cv::Point2d& p) { return fT(p) - cT; }, gain, col::yellow, 200, 30);
        if (k == 2) g = drawArrows(pan, cam, [&](const cv::Point2d& p) { return fT(p) - fY(p); }, gain, col::red, 200, 30);
        blit(cv_, pan, cv::Point(x0, 66));
        text(cv_, g > 0 ? fmt("arrows x%.0f", g) : std::string("no motion left (flat)"), cv::Point(x0 + 6, 340), 0.45, col::yellow);
        const cv::Mat& h = k == 0 ? hY : (k == 1 ? hT : hD);
        blit(cv_, heat(h, lim, psz), cv::Point(x0, 360));
        const double hm = k == 0 ? mY : (k == 1 ? mT : mD);
        if (hm < 1e-3 * lim) text(cv_, "flat : the whole image moves by the same amount", cv::Point(x0 + 40, 505), 0.5, col::gray);
    }
    text(cv_, "heat map = du (px), common scale :", cv::Point(20, 662), 0.5, col::gray);
    colorBar(cv_, cv::Rect(330, 650, 300, 14), lim, "px");

    // 아래 : QR 별 수치
    int y = 712;
    auto row = [&](const std::string& t, const cv::Scalar& c = col::black) { text(cv_, t, cv::Point(20, y), 0.52, c); y += 24; };
    row("                         QR L        QR C        QR R      width diff R-L   height diff R-L", col::blue);
    const QrPrediction q0 = predictQr(cam, Pose{}), qY = predictQr(cam, pY), qT = predictQr(cam, pT);
    auto du = [&](const Pose& p, int k) { return cam.displacement(cam.qrCenterRef(k), cam.z_plate, p).x; };
    row(fmt("Y  du (px)         %+9.4f", du(pY, 0)) + fmt("   %+9.4f   %+9.4f", du(pY, 1), du(pY, 2)) +
        fmt("      %+8.4f        %+8.4f", qY.w_diff - q0.w_diff, qY.h_diff - q0.h_diff));
    row(fmt("Th du (px)         %+9.4f", du(pT, 0)) + fmt("   %+9.4f   %+9.4f", du(pT, 1), du(pT, 2)) +
        fmt("      %+8.4f        %+8.4f", qT.w_diff - q0.w_diff, qT.h_diff - q0.h_diff));
    row(fmt("Th - Y (px)        %+9.4f", du(pT, 0) - du(pY, 0)) + fmt("   %+9.4f   %+9.4f", du(pT, 1) - du(pY, 1), du(pT, 2) - du(pY, 2)) +
        fmt("      %+8.4f        %+8.4f", qT.w_diff - qY.w_diff, qT.h_diff - qY.h_diff), col::red);
    row(fmt("=> per 0.01 deg the Y/Theta difference is only about %.4f px in width diff R-L", s.wdiff_per_deg * 0.01), col::purple);
    return cv_;
}

// -------------------------------------------------------------------------------------
cv::Mat renderProfileView(const CameraModel& cam, double yEq) {
    const Sensitivity s = sensitivities(cam);
    const Pose pY{yEq, 0.0}, pT{0.0, yEq * s.equal_shift_deg_per_mm};
    cv::Mat cv_(kCanvas, CV_8UC3, col::white);
    text(cv_, fmt("[3] Profile along the QR row (v = %.0f px)  |  Y = %.2f mm vs Theta = %.4f deg", cam.plate_center.y, pY.y_mm,
                  pT.t_deg), cv::Point(20, 26), 0.6);

    const double vRow = cam.plate_center.y;
    std::vector<cv::Point2d> uY, uT, uTl, rY, rT, vTa, vTb;
    const double cY = cam.displacement(cam.plate_center, cam.z_plate, pY).x;
    const double cT = cam.displacement(cam.plate_center, cam.z_plate, pT).x;
    double lo1 = 1e9, hi1 = -1e9, lo2 = 1e9, hi2 = -1e9, lo3 = 1e9, hi3 = -1e9;
    for (int x = 0; x <= cam.width; x += 32) {
        const cv::Point2d p(x, vRow);
        const double a = cam.displacement(p, cam.z_plate, pY).x, b = cam.displacement(p, cam.z_plate, pT).x;
        const double bl = cam.displacementLinear(p, cam.z_plate, pT).x;
        uY.emplace_back(x, a); uT.emplace_back(x, b); uTl.emplace_back(x, bl);
        rY.emplace_back(x, a - cY); rT.emplace_back(x, b - cT);
        const double va = cam.displacement(cv::Point2d(x, vRow - 300), cam.z_plate, pT).y;
        const double vb = cam.displacement(cv::Point2d(x, vRow + 300), cam.z_plate, pT).y;
        vTa.emplace_back(x, va); vTb.emplace_back(x, vb);
        lo1 = std::min({lo1, a, b}); hi1 = std::max({hi1, a, b});
        lo2 = std::min({lo2, a - cY, b - cT}); hi2 = std::max({hi2, a - cY, b - cT});
        lo3 = std::min({lo3, va, vb}); hi3 = std::max({hi3, va, vb});
    }
    auto pad = [](double& lo, double& hi) { const double m = std::max(1e-3, 0.1 * (hi - lo)); lo -= m; hi += m; };
    pad(lo1, hi1); pad(lo2, hi2); pad(lo3, hi3);

    Plot p1(cv_, cv::Rect(90, 70, 1460, 200), 0, cam.width, lo1, hi1);
    p1.axes("du : horizontal motion (px)  |  nearly the same for Y and Theta", "image x (px)", "px");
    p1.line(uY, col::blue); p1.line(uT, col::orange); p1.line(uTl, col::gray, 1);
    p1.legend({{"Y", col::blue}, {"Theta (exact)", col::orange}, {"Theta (small-angle)", col::gray}});

    Plot p2(cv_, cv::Rect(90, 360, 1460, 200), 0, cam.width, lo2, hi2);
    p2.axes("du - du(plate center) (px)  |  Theta: U-shape (perspective),  Y: flat", "image x (px)", "px");
    for (int k = 0; k < 3; ++k) p2.vline(cam.qrCenterRef(k).x, col::green, k == 0 ? "QR L" : (k == 1 ? "QR C" : "QR R"));
    p2.line(rY, col::blue); p2.line(rT, col::orange);
    p2.legend({{"Y", col::blue}, {"Theta", col::orange}});

    Plot p3(cv_, cv::Rect(90, 650, 1460, 200), 0, cam.width, lo3, hi3);
    p3.axes("dv : vertical motion for Theta (px)  |  u*v term, opposite sign above / below the QR row", "image x (px)", "px");
    p3.vline(cam.cx, col::gray, "cx");
    p3.line(vTa, col::purple); p3.line(vTb, col::green);
    p3.legend({{"Theta, 300 px above", col::purple}, {"Theta, 300 px below", col::green}, {"Y : always 0", col::blue}});
    return cv_;
}

// -------------------------------------------------------------------------------------
cv::Mat renderDepthView(const CameraModel& cam, const Pose& pose) {
    const Sensitivity s = sensitivities(cam);
    const double tEq = s.equal_shift_deg_per_mm;   // Y 1 mm 와 같은 플레이트 이동의 Θ
    cv::Mat cv_(kCanvas, CV_8UC3, col::white);
    text(cv_, "[4] Depth  |  Y motion falls as 1/Z, Theta motion hardly depends on Z  =>  plate vs background parallax",
         cv::Point(20, 26), 0.6);

    std::vector<cv::Point2d> cY, cT, cP;
    double hi = 0, lo = 0;
    for (double z = 250; z <= 1000; z += 5) {
        const double a = cam.displacement(cam.plate_center, z, Pose{1.0, 0}).x;
        const double b = cam.displacement(cam.plate_center, z, Pose{0, tEq}).x;
        const double c = cam.displacement(cam.plate_center, z, pose).x;
        cY.emplace_back(z, a); cT.emplace_back(z, b); cP.emplace_back(z, c);
        hi = std::max({hi, a, b, c}); lo = std::min({lo, a, b, c});
    }
    Plot p(cv_, cv::Rect(90, 70, 900, 560), 250, 1000, lo - 1, hi + 1);
    p.axes("shift at the plate-center pixel vs object depth Z (px)", "depth Z (mm)", "px", 5, 6);
    p.vline(cam.z_plate, col::green, "plate");
    p.vline(cam.z_bg, col::orange, "background");
    p.line(cY, col::blue); p.line(cT, col::red); p.line(cP, col::purple, 1);
    p.legend({{"Y = 1 mm", col::blue}, {fmt("Theta = %.3f deg", tEq), col::red}, {"current pose", col::purple}});

    // 2 × 2 시차 분해 : [s_plate; s_bg] = A [Y; Θ]
    const cv::Matx22d A(s.plate_per_mm, s.plate_per_deg, s.bg_per_mm, s.bg_per_deg);
    const double sp = cam.displacement(cam.plate_center, cam.z_plate, pose).x;
    const double sb = cam.displacement(cam.plate_center, cam.z_bg, pose).x;
    const cv::Vec2d est = A.inv() * cv::Vec2d(sp, sb);
    cv::Mat w; cv::SVD::compute(cv::Mat(A), w);
    const double cond = w.at<double>(0) / w.at<double>(1);
    const cv::Matx22d Ci = (A.t() * A).inv();
    const double sig = 0.011;   // 이동량 측정 잡음 (px, EDA 정지 잡음)

    int y = 90; const int x = 1040;
    auto line = [&](const std::string& t, const cv::Scalar& c = col::black) { text(cv_, t, cv::Point(x, y), 0.5, c); y += 24; };
    line("Sensitivity matrix A  (px per mm, px per deg)", col::blue);
    line(fmt("  plate      : %7.3f   %8.3f", s.plate_per_mm, s.plate_per_deg));
    line(fmt("  background : %7.3f   %8.3f", s.bg_per_mm, s.bg_per_deg));
    line(fmt("  condition number = %.1f", cond));
    y += 8;
    line("Current pose -> measured shifts", col::blue);
    line(fmt("  Y = %+.3f mm, Theta = %+.4f deg", pose.y_mm, pose.t_deg));
    line(fmt("  s_plate = %+.3f px, s_bg = %+.3f px", sp, sb));
    line(fmt("  solve A [Y,Th] = s -> Y %+.3f, Th %+.4f", est[0], est[1]), col::purple);
    y += 8;
    line(fmt("If each shift has noise %.3f px :", sig), col::blue);
    line(fmt("  sigma Y     = %.1f um", 1000.0 * sig * std::sqrt(Ci(0, 0))));
    line(fmt("  sigma Theta = %.4f deg", sig * std::sqrt(Ci(1, 1))));
    line("  (2 regions only; more regions -> smaller)", col::gray);
    y += 8;
    line("Plate alone (perspective cue only) :", col::blue);
    line(fmt("  width diff R-L : %.3f px/deg", s.wdiff_per_deg));
    line(fmt("  0.01 deg -> %.4f px", s.wdiff_per_deg * 0.01));
    // 플레이트 이동을 그대로 둔 채 Θ 0.01° 만 바꿀 때(ΔY = −0.01/tEq) 배경이 움직이는 양 = 시차 신호
    const double dT = 0.01, dY = -dT / tEq;
    line("Plate + background (parallax cue) :", col::blue);
    line(fmt("  0.01 deg at fixed plate shift -> bg moves %.4f px", std::abs(s.bg_per_mm * dY + s.bg_per_deg * dT)));
    return cv_;
}

} // namespace mm
