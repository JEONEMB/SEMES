// features.cpp — ③ 특징 측정 구현
#include "ksh/features.hpp"

#include <algorithm>
#include <string>

namespace ksh {

FeatureMeasurement measure_features(const cv::Mat &gray8, const FeatureParams &params) {
    FeatureMeasurement out;
    // ① QR 3개 (대략 위치 dx → 박스마다 사각형 중심)
    out.qr = locate_qr(gray8, params.qr);
    if (!out.qr.ok) { out.message = "QR : " + out.qr.message; return out; }
    if (params.holes.boxes.empty()) { out.message = "설정에 background_rois 가 없음"; return out; }
    // ② 홀 (박스를 QR 가로 이동 dx 만큼 옮긴 뒤 측정)
    double hx = 0.0;
    for (const auto &b : params.holes.boxes) {
        HoleResult r = locate_hole(gray8, b, params.holes, out.qr.coarse_dx_px);
        out.holes.push_back(r);
        if (!r.ok) { out.message = "배경 : " + r.message; return out; }
        hx += r.center_px.x;
    }
    hx /= double(params.holes.boxes.size());
    // 홀끼리 일관성 : 홀은 모두 같은 깊이(뒷벽)라 (측정 − 박스) 가로 차이가 거의 같아야 함
    double lo = 1e30, hi = -1e30;
    for (size_t k = 0; k < out.holes.size(); ++k) {
        const double r = out.holes[k].center_px.x - (params.holes.boxes[k].cx_px + out.qr.coarse_dx_px);
        lo = std::min(lo, r);
        hi = std::max(hi, r);
    }
    if (hi - lo > params.holes.max_spread_px) {
        out.message = "배경 : 홀 위치가 서로 맞지 않음 (차이 폭 " + std::to_string(hi - lo).substr(0, 5) + " px, 가림·이물 확인)";
        return out;
    }
    // ③ s, d
    out.s_px = out.qr.center_px.x;
    out.d_px = out.s_px - hx;
    out.ok = true;
    return out;
}

} // namespace ksh
