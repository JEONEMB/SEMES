// features.cpp — ③ 시차 특징 측정 구현
#include "ksh/features.hpp"

namespace ksh {

RoiShift measure_roi(const cv::Mat &ref32, const cv::Mat &cur32, const RoiSpec &roi, const cv::Point2d &guess_px, bool wide_search,
                     int wide_window_scale, const FineParams &fine) {
    RoiShift m;
    m.name = roi.name;
    m.ref_rect_px = roi_rect(roi);
    cv::Point2d init_px = guess_px;
    if (wide_search) {
        const int half_px = wide_window_scale * roi.half_width_px;
        const cv::Rect A = rect_around(roi.center_px, half_px, half_px);
        const cv::Rect B = A + round_point(guess_px);
        if (!rect_inside(A, ref32.size()) || !rect_inside(B, cur32.size())) return m;
        init_px = cv::Point2d(round_point(guess_px)) + phase_shift(ref32(A), cur32(B));
    }
    FineResult r;
    m.ok = fine_alignment(ref32, cur32, m.ref_rect_px, init_px, fine, r);
    if (m.ok) { m.shift_px = r.shift_px; m.zncc = r.zncc; }
    return m;
}

bool compute_features(const std::vector<RoiShift> &rois, size_t n_qr, double &s_px, double &d_px) {
    if (n_qr == 0 || rois.size() <= n_qr) return false;
    double qr_sum_px = 0.0, bg_sum_px = 0.0;
    for (size_t i = 0; i < rois.size(); ++i) (i < n_qr ? qr_sum_px : bg_sum_px) += rois[i].shift_px.x;
    s_px = qr_sum_px / double(n_qr);
    d_px = s_px - bg_sum_px / double(rois.size() - n_qr);
    return true;
}

FeatureMeasurement measure_features(const cv::Mat &ref32, const cv::Mat &cur32, const FeatureParams &params) {
    FeatureMeasurement out;
    if (ref32.empty() || cur32.empty() || ref32.size() != cur32.size()) { out.message = "기준·현재 영상 크기가 다름"; return out; }
    // ① Coarse Alignment
    if (!coarse_alignment(ref32, cur32, params.coarse, out.coarse_px)) { out.message = "Coarse Alignment 창이 사진 밖"; return out; }
    // ② Fine Alignment : QR 은 Coarse 결과에서 바로, 배경은 넓은 창 위상 상관 후
    bool all_ok = true;
    for (const auto &r : params.qr_rois) {
        RoiShift m = measure_roi(ref32, cur32, r, out.coarse_px, false, params.wide_window_scale, params.fine);
        if (m.ok && cv::norm(m.shift_px - out.coarse_px) > params.qr_max_dev_px) m.ok = false;
        all_ok = all_ok && m.ok;
        out.rois.push_back(m);
    }
    for (const auto &r : params.background_rois) {
        const RoiShift m = measure_roi(ref32, cur32, r, out.coarse_px, true, params.wide_window_scale, params.fine);
        all_ok = all_ok && m.ok;
        out.rois.push_back(m);
    }
    if (!all_ok) { out.message = "ROI 측정 실패 (Fine Alignment 미수렴 또는 사진 밖)"; return out; }
    // ③ 특징
    if (!compute_features(out.rois, params.qr_rois.size(), out.s_px, out.d_px)) { out.message = "QR 또는 배경 ROI 가 설정에 없음"; return out; }
    out.ok = true;
    return out;
}

} // namespace ksh
