// =====================================================================================
// m1_phase_ecc.cpp — M1 구현 + 검증
// =====================================================================================
#include "m1_phase_ecc.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <cmath>
#include <iostream>

namespace yt {
namespace m1 {

namespace {
bool inside(const cv::Rect& r, const cv::Size& s) { return r.x >= 0 && r.y >= 0 && r.x + r.width <= s.width && r.y + r.height <= s.height; }
cv::Point iround(const cv::Point2d& p) { return cv::Point(int(std::lround(p.x)), int(std::lround(p.y))); }
} // namespace

cv::Point2d phaseShift(const cv::Mat& a, const cv::Mat& b) {
    cv::Mat win;
    cv::createHanningWindow(win, a.size(), CV_32F);
    cv::Mat a2 = a.clone(), b2 = b.clone();   // 제자리 곱셈 방지
    return cv::phaseCorrelate(a2, b2, win);
}

namespace {
// OpenCV 디버그 라이브러리(opencv_world3415d)는 예외를 던지기 전에 오류 문구를 콘솔에 찍음 (OPENCV_DUMP_ERRORS 기본값).
// ECC 미수렴은 '측정 실패' 로 처리하는 정상 경로이므로, ECC 호출 동안만 문구 출력을 막고 건수를 셈.
int quietError(int, const char*, const char*, const char*, int, void*) { return 0; }
int gEccFail = 0;
} // namespace

int eccFailCount() { return gEccFail; }
void resetEccFailCount() { gEccFail = 0; }

Shift eccTranslation(const cv::Mat& tmpl, const cv::Mat& img, const cv::Point2d& init) {
    Shift s;
    cv::Mat w = cv::Mat::eye(2, 3, CV_32F);
    w.at<float>(0, 2) = float(init.x);
    w.at<float>(1, 2) = float(init.y);
    void* prevData = nullptr;
    const cv::ErrorCallback prev = cv::redirectError(quietError, nullptr, &prevData);
    try {
        s.cc = cv::findTransformECC(tmpl, img, w, cv::MOTION_TRANSLATION,
                                    cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 1e-7), cv::noArray());
    } catch (const cv::Exception&) {
        cv::redirectError(prev, prevData);
        ++gEccFail;
        return s;   // ok = false → 호출 쪽에서 NaN(측정 실패)으로 기록
    }
    cv::redirectError(prev, prevData);
    s.d = cv::Point2d(w.at<float>(0, 2), w.at<float>(1, 2));
    s.ok = true;
    return s;
}

cv::Point2d coarseShift(const cv::Mat& ref, const cv::Mat& cur, const cv::Point& c, const cv::Size& win) {
    const cv::Rect R(c.x - win.width / 2, c.y - win.height / 2, win.width, win.height);
    if (!inside(R, ref.size()) || !inside(R, cur.size())) return cv::Point2d(0, 0);
    return phaseShift(ref(R), cur(R));
}

Shift measureRegion(const cv::Mat& ref, const cv::Mat& cur, const Region& r, const cv::Point2d& guess, bool wide) {
    Shift s;
    cv::Point2d g = guess;
    if (wide) {
        const int W = 2 * r.hw;
        const cv::Rect A(r.cx - W, r.cy - W, 2 * W, 2 * W);
        const cv::Rect B = A + iround(guess);
        if (!inside(A, ref.size()) || !inside(B, cur.size())) return s;
        g = cv::Point2d(iround(guess)) + phaseShift(ref(A), cur(B));
    }
    const cv::Point gi = iround(g);
    const cv::Rect T(r.cx - r.hw, r.cy - r.hw, 2 * r.hw, 2 * r.hw);
    const cv::Rect C = T + gi;
    if (!inside(T, ref.size()) || !inside(C, cur.size())) return s;
    s = eccTranslation(ref(T), cur(C), g - cv::Point2d(gi));
    if (s.ok) s.d += cv::Point2d(gi);
    // QR 은 플레이트와 같이 움직여야 함 : 거친 이동에서 15 px 이상 벗어나면 이웃 QR 에 붙은 것
    if (s.ok && r.group == 0 && cv::norm(s.d - guess) > 15.0) s.ok = false;
    return s;
}

Frame measureFrame(const cv::Mat& ref, const cv::Mat& cur) {
    Timer t;
    Frame f;
    f.coarse = coarseShift(ref, cur, cv::Point(int(kPlateCenter.x), int(kPlateCenter.y)), cv::Size(900, 400));
    for (const auto& r : regions()) f.r.push_back(measureRegion(ref, cur, r, f.coarse, r.group != 0));
    f.ms = t.ms();
    return f;
}

// =====================================================================================
// 검증
// =====================================================================================
bool verify(const std::vector<Sample>& ds, const std::string& outDir) {
    Report rep(outDir, "M1 위상 상관 + ECC — 검증 결과");
    const Sample* refS = findStem(ds, "center");
    if (!refS) { rep.check("기준 영상", false, "center 없음"); return false; }
    const cv::Mat ref = loadGray32(refS->path);

    // ---------------- V1-1 합성 : 부분 화소 이동 (QR 3개, x 0~1 px, y 0.25 px 고정)
    rep.h("V1-1 합성 부분 화소 이동 (정답을 아는 이동)");
    rep.line("기준 영상의 QR 영역을 푸리에 위상 회전으로 정확히 (dx, 0.25) px 옮긴 뒤, 같은 절차(ECC)로 측정해 정답과 비교.");
    {
        Csv csv(outDir + "/m1_synthetic_subpixel.csv");
        csv << "region" << "dx_true" << "dy_true" << "dx_meas" << "dy_meas" << "err_x" << "err_y" << "cc";
        csv.endl();
        std::vector<double> ex, ey;
        for (int k = 0; k < 3; ++k) {
            const Region& r = regions()[k];
            const int P = 64;
            const cv::Rect B(r.cx - r.hw - P, r.cy - r.hw - P, 2 * (r.hw + P), 2 * (r.hw + P));
            const cv::Mat big = ref(B).clone();
            const Region loc{r.name, 0, r.hw + P, r.hw + P, r.hw};
            for (int i = 0; i <= 20; ++i) {
                const double dx = 0.05 * i, dy = 0.25;
                const cv::Mat sh = fourierShift(big, dx, dy);
                const Shift m = measureRegion(big, sh, loc, phaseShift(big, sh), false);
                csv << r.name << dx << dy << m.d.x << m.d.y << m.d.x - dx << m.d.y - dy << m.cc;
                csv.endl();
                ex.push_back(m.d.x - dx); ey.push_back(m.d.y - dy);
            }
        }
        const Stat sx = stat(ex), sy = stat(ey);
        rep.table({"축", "RMS 오차 (px)", "최대 오차 (px)", "평균 (px)"},
                  {{"x", fmt("%.4f", sx.rms), fmt("%.4f", sx.maxAbs), fmt("%+.4f", sx.mean)},
                   {"y", fmt("%.4f", sy.rms), fmt("%.4f", sy.maxAbs), fmt("%+.4f", sy.mean)}});
        rep.check("부분 화소 정확도", sx.rms <= 0.03 && sy.rms <= 0.03 && sx.maxAbs <= 0.05 && sy.maxAbs <= 0.05,
                  fmt("x RMS %.4f, y RMS %.4f px (기준 RMS ≤ 0.03, 최대 ≤ 0.05)", sx.rms, sy.rms));
        rep.line("남는 오차는 ECC 내부 선형 보간의 1 px 주기 치우침 (M2 에서 따로 다룸).");
    }

    // ---------------- V1-2 합성 : 큰 이동 (Θ 1° 수준)
    rep.h("V1-2 합성 큰 이동 (Θ 1~2° 수준)");
    {
        const cv::Rect B(int(kPlateCenter.x) - 700, int(kPlateCenter.y) - 300, 1400, 600);
        const cv::Mat big = ref(B).clone();
        std::vector<std::vector<std::string>> rows;
        bool ok = true;
        for (const cv::Point2d tr : {cv::Point2d(80.37, 0.21), cv::Point2d(-160.62, 0.48), cv::Point2d(14.18, -0.07)}) {
            const cv::Mat sh = fourierShift(big, tr.x, tr.y);
            const cv::Point2d c = coarseShift(big, sh, cv::Point(700, 300), cv::Size(900, 400));
            const Region loc{"QR_C", 0, 700 + (1907 - int(kPlateCenter.x)), 300 + (1065 - int(kPlateCenter.y)), 110};
            const Shift m = measureRegion(big, sh, loc, c, false);
            const double e = cv::norm(m.d - tr);
            ok = ok && m.ok && e <= 0.05;
            rows.push_back({fmt("(%+.2f, %+.2f)", tr.x, tr.y), fmt("(%+.2f, %+.2f)", c.x, c.y), fmt("(%+.4f, %+.4f)", m.d.x, m.d.y),
                            fmt("%.4f", e)});
        }
        rep.table({"넣은 이동 (px)", "거친 이동", "최종 측정", "오차 (px)"}, rows);
        rep.check("큰 이동 포착", ok, "세 경우 모두 오차 ≤ 0.05 px");
    }

    // ---------------- V1-3 실제 사진 전체
    rep.h("V1-3 실제 사진 전체 (기준 : 1차 center, 2차 center1)");
    std::vector<double> tT, sT, tY, sY, times;
    double minQR = 1, minWall = 1;
    int fails = 0, wallFails = 0;   // QR / 벽 측정 실패
    {
        Csv csv(outDir + "/m1_shifts.csv");
        csv << "stem" << "set" << "shot" << "y_mm" << "t_deg" << "coarse_dx" << "coarse_dy";
        for (const auto& r : regions()) csv << r.name + "_dx" << r.name + "_dy" << r.name + "_cc";
        csv << "ms";
        csv.endl();
        cv::Mat ref2;
        int done = 0;
        for (const auto& s : ds) {
            if (s.stem == "center" || s.stem == "center1") continue;
            const cv::Mat& R = s.set_id == 1 ? ref : (ref2.empty() ? (ref2 = loadGray32(findStem(ds, "center1")->path)) : ref2);
            const cv::Mat cur = loadGray32(s.path);
            const Frame f = measureFrame(R, cur);
            csv << s.stem << s.set_id << s.shot << s.y_mm << s.t_deg << f.coarse.x << f.coarse.y;
            double plate = 0; int np = 0;
            for (size_t k = 0; k < f.r.size(); ++k) {
                const auto& m = f.r[k];
                if (m.ok) csv << m.d.x << m.d.y << m.cc;
                else { csv << std::string("") << std::string("") << m.cc; if (regions()[k].group == 0) ++fails; else ++wallFails; }
                if (regions()[k].group == 0) { minQR = std::min(minQR, m.cc); if (m.ok) { plate += m.d.x; ++np; } }
                else minWall = std::min(minWall, m.cc);
            }
            csv << f.ms;
            csv.endl();
            times.push_back(f.ms);
            if (s.set_id == 1 && !s.is_ref && np == 3) {
                if (s.y_mm == 0) { tT.push_back(s.t_deg); sT.push_back(plate / 3); }
                if (s.t_deg == 0) { tY.push_back(s.y_mm); sY.push_back(plate / 3); }
            }
            if (++done % 50 == 0) std::cout << "    " << done << " 장\n";
        }
    }
    // Θ 스윕 : s = c + b·Θ + q·Θ² ,  Y 스윕 : s = c + a·Y
    std::vector<std::vector<double>> A; std::vector<double> yhat;
    for (double t : tT) A.push_back({1.0, t, t * t});
    const auto bT = lstsq(A, sT);
    for (double t : tT) yhat.push_back(bT[0] + bT[1] * t + bT[2] * t * t);
    const double r2T = r2(sT, yhat);
    A.clear(); yhat.clear();
    for (double y : tY) A.push_back({1.0, y});
    const auto bY = lstsq(A, sY);
    for (double y : tY) yhat.push_back(bY[0] + bY[1] * y);
    const double r2Y = r2(sY, yhat);
    const Stat tm = stat(times);
    rep.table({"항목", "값"},
              {{"측정 사진 수", fmt("%d", int(times.size()))},
               {"Θ 스윕 플레이트 감도", fmt("%.3f px/° (Θ² %.3f), R² %.6f", bT[1], bT[2], r2T)},
               {"Y 스윕 플레이트 감도", fmt("%.3f px/mm, R² %.6f", bY[1], r2Y)},
               {"ECC 상관 최소 (QR / 벽)", fmt("%.4f / %.4f", minQR, minWall)},
               {"측정 실패 영역 수 (QR / 벽)", fmt("%d / %d", fails, wallFails)},
               {"처리 시간 / 장 (영상 읽기 제외)", fmt("평균 %.1f ms, 최대 %.1f ms", tm.mean, tm.maxAbs)}});
    rep.check("Θ 감도 (기존 실측 80.0 px/° ±2 %)", std::abs(bT[1] - 80.0) <= 1.6 && r2T >= 0.999, fmt("%.3f px/°, R² %.6f", bT[1], r2T));
    rep.check("Y 감도 (기존 실측 14.19 px/mm ±2 %)", std::abs(bY[1] - 14.19) <= 0.28 && r2Y >= 0.999, fmt("%.3f px/mm, R² %.6f", bY[1], r2Y));
    rep.check("QR 측정 신뢰도", minQR >= 0.95 && fails == 0, fmt("QR ECC 최소 %.4f (≥ 0.95), 실패 %d", minQR, fails));
    return rep.allOk();
}

} // namespace m1
} // namespace yt
