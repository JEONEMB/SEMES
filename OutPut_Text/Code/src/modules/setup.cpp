// setup.cpp — 초기 세팅 구현
#include "ksh/setup.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace ksh {

SetupResult run_setup(const std::vector<cv::Mat> &centers, const FeatureParams &params, const PoseModel &model, double max_s_spread_px,
                      double max_d_spread_px) {
    SetupResult out;
    out.params = params;
    if (centers.empty()) { out.message = "센터 사진이 없음"; return out; }
    // ① 지금 박스로 첫 센터 사진 측정
    const FeatureMeasurement f0 = measure_features(centers[0], params);
    if (!f0.ok) { out.message = "센터 사진 측정 실패 : " + f0.message; return out; }
    // ② 박스 갱신 (찾은 중심 반올림)
    for (size_t i = 0; i < out.params.qr.boxes.size(); ++i) {
        out.params.qr.boxes[i].cx_px = int(std::lround(f0.qr.centers_px[i].x));
        out.params.qr.boxes[i].cy_px = int(std::lround(f0.qr.centers_px[i].y));
    }
    for (size_t i = 0; i < out.params.holes.boxes.size(); ++i) {
        out.params.holes.boxes[i].cx_px = int(std::lround(f0.holes[i].center_px.x));
        out.params.holes.boxes[i].cy_px = int(std::lround(f0.holes[i].center_px.y));
    }
    if (f0.qr.centers_px.size() >= 3) {
        std::vector<double> xs;
        for (const auto &c : f0.qr.centers_px) xs.push_back(c.x);
        std::sort(xs.begin(), xs.end());
        for (size_t i = 2; i < xs.size(); ++i) {
            const double diff = std::abs((xs[i] - xs[i - 1]) - (xs[i - 1] - xs[i - 2]));
            if (diff > 2.0) {
                char b[160];
                std::snprintf(b, sizeof(b), "QR 간격이 고르지 않음 (차이 %.2f px) — 카메라 기울어짐·QR 판 확인", diff);
                out.warnings.push_back(b);
            }
        }
    }
    // ③ 새 박스로 다시 재서 영점 = 평균
    double sy = 0, st = 0, smin = 1e30, smax = -1e30, dmin = 1e30, dmax = -1e30;
    for (size_t k = 0; k < centers.size(); ++k) {
        FeatureMeasurement f = measure_features(centers[k], out.params);
        if (!f.ok) { out.message = "센터 사진 " + std::to_string(k + 1) + " 측정 실패 : " + f.message; return out; }
        double y, t;
        if (!invert_pose_model(model, f.s_px, f.d_px, y, t)) { out.message = "보정식이 Y·Θ 를 구별하지 못함 (model_file 확인)"; return out; }
        sy += y; st += t;
        smin = std::min(smin, f.s_px); smax = std::max(smax, f.s_px);
        dmin = std::min(dmin, f.d_px); dmax = std::max(dmax, f.d_px);
        out.measured.push_back(f);
    }
    out.zero.y_mm = sy / double(centers.size());
    out.zero.t_deg = st / double(centers.size());
    out.s_spread_px = smax - smin;
    out.d_spread_px = dmax - dmin;
    char b[200];
    if (out.s_spread_px > max_s_spread_px) {
        std::snprintf(b, sizeof(b), "센터 사진끼리 s 차이 %.3f px (> %.2f) — 로봇이 같은 자리에 섰는지 확인", out.s_spread_px, max_s_spread_px);
        out.warnings.push_back(b);
    }
    if (out.d_spread_px > max_d_spread_px) {
        std::snprintf(b, sizeof(b), "센터 사진끼리 d 차이 %.3f px (> %.2f) — 로봇이 같은 자리에 섰는지 확인", out.d_spread_px, max_d_spread_px);
        out.warnings.push_back(b);
    }
    out.ok = true;
    return out;
}

} // namespace ksh
