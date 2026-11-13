// features.cpp — ③ 특징 측정 구현
#include "ksh/features.hpp"

namespace ksh {

FeatureMeasurement measure_features(const cv::Mat &gray8, const FeatureParams &params) {
    FeatureMeasurement out;
    // ① QR 3개
    out.qr = locate_qr(gray8, params.qr);
    if (!out.qr.ok) { out.message = "QR : " + out.qr.message; return out; }
    if (params.screws.screws.empty()) { out.message = "설정에 나사 ROI 가 없음"; return out; }
    // ② 나사 (QR 위치로 ROI 를 옮긴 뒤 측정)
    double sx = 0.0;
    for (const auto &spec : params.screws.screws) {
        ScrewResult r = locate_screw(gray8, spec, params.screws, out.qr.center_px);
        out.screws.push_back(r);
        if (!r.ok) { out.message = r.message; return out; }
        sx += r.center_px.x;
    }
    sx /= double(params.screws.screws.size());
    // ③ s, d
    out.s_px = out.qr.center_px.x;
    out.d_px = out.s_px - sx;
    out.ok = true;
    return out;
}

} // namespace ksh
