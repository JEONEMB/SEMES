// features.cpp — ③ 시차 특징 측정 구현
#include "ksh/features.hpp"

#include <cmath>
#include <cstdio>
#include <string>

namespace ksh {

// ROI 하나의 이동 측정
//  QR(wide_search = false) : Coarse 결과를 출발점으로 바로 Fine (QR 3개가 닮아 넓게 찾으면 이웃 QR 에 붙을 수 있음)
//  배경(wide_search = true) : 깊이가 달라 플레이트와 이동량이 다르므로 ROI 주변 넓은 창 위상 상관으로 출발점을 다시 잡은 뒤 Fine
RoiShift measure_roi(const cv::Mat &ref32, const cv::Mat &cur32, const RoiSpec &roi, const cv::Point2d &guess_px, bool wide_search,
                     int wide_window_scale, const FineParams &fine, double wide_min_response) {
    RoiShift m;
    m.name = roi.name;
    m.ref_rect_px = roi_rect(roi);
    cv::Point2d init_px = guess_px;
    if (wide_search) {
        const int half_x_px = wide_window_scale * roi.half_width_px;
        const int half_y_px = wide_window_scale * (roi.half_height_px > 0 ? roi.half_height_px : roi.half_width_px);
        const cv::Rect A = rect_around(roi.center_px, half_x_px, half_y_px);
        const cv::Rect B = A + round_point(guess_px);
        if (!rect_inside(A, ref32.size()) || !rect_inside(B, cur32.size())) { m.message = "탐색 창이 사진 밖"; return m; }
        init_px = cv::Point2d(round_point(guess_px)) + phase_shift(ref32(A), cur32(B), &m.response);
        if (m.response < wide_min_response) {
            char buf[96];
            std::snprintf(buf, sizeof(buf), "위상 상관 봉우리 약함 (%.3f < %.3f)", m.response, wide_min_response);
            m.message = buf;
            return m;
        }
    }
    FineResult r;
    m.ok = fine_alignment(ref32, cur32, m.ref_rect_px, init_px, fine, r);
    m.zncc = r.zncc;
    m.converged = r.converged;
    if (m.ok) { m.shift_px = r.shift_px; return m; }
    // 실패 이유 (fine_alignment 는 계산을 끝냈으면 out 을 채워 둠)
    char buf[96];
    if (r.iterations > 0 && !r.converged) std::snprintf(buf, sizeof(buf), "정합 미수렴 (%d회)", r.iterations);
    else if (r.converged && r.zncc < fine.min_zncc) std::snprintf(buf, sizeof(buf), "정합 상관 낮음 (ZNCC %.3f < %.3f)", r.zncc, fine.min_zncc);
    else std::snprintf(buf, sizeof(buf), "정합 실패 (사진 밖 · 민무늬 · 탐색 범위 초과)");
    m.message = buf;
    return m;
}

// ③ 특징 (가로 이동만 사용 — Y·Θ 는 모두 가로 이동으로 나타남)
//  s = QR 평균 이동,  d = s − 배경 평균 이동 (시차)
bool compute_features(const std::vector<RoiShift> &rois, size_t n_qr, double &s_px, double &d_px) {
    if (n_qr == 0 || rois.size() <= n_qr) return false;
    double qr_sum_px = 0.0, bg_sum_px = 0.0;
    for (size_t i = 0; i < rois.size(); ++i) (i < n_qr ? qr_sum_px : bg_sum_px) += rois[i].shift_px.x;
    s_px = qr_sum_px / double(n_qr);
    d_px = s_px - bg_sum_px / double(rois.size() - n_qr);
    return true;
}

// ①②③ 전체 : Coarse → ROI 5곳 Fine → s, d. ROI 하나라도 실패하면 전체 실패 (실패한 ROI 이름과 이유를 message 에)
FeatureMeasurement measure_features(const cv::Mat &ref32, const cv::Mat &cur32, const FeatureParams &params) {
    FeatureMeasurement out;
    if (ref32.empty() || cur32.empty() || ref32.size() != cur32.size()) { out.message = "기준·현재 영상 크기가 다름"; return out; }
    // ① Coarse Alignment
    if (!coarse_alignment(ref32, cur32, params.coarse, out.coarse_px, &out.coarse_response)) {
        char buf[128];
        if (std::isnan(out.coarse_response)) std::snprintf(buf, sizeof(buf), "Coarse Alignment 창이 사진 밖");
        else std::snprintf(buf, sizeof(buf), "Coarse Alignment 봉우리 약함 (%.3f < %.3f)", out.coarse_response, params.coarse.min_response);
        out.message = buf;
        return out;
    }
    // ② Fine Alignment : QR 은 Coarse 결과에서 바로, 배경은 넓은 창 위상 상관 후
    bool all_ok = true;
    for (const auto &r : params.qr_rois) {
        RoiShift m = measure_roi(ref32, cur32, r, out.coarse_px, false, params.wide_window_scale, params.fine);
        if (m.ok && cv::norm(m.shift_px - out.coarse_px) > params.qr_max_dev_px) { m.ok = false; m.message = "Coarse 결과에서 너무 벗어남 (이웃 QR)"; }
        all_ok = all_ok && m.ok;
        out.rois.push_back(m);
    }
    for (const auto &r : params.background_rois) {
        const RoiShift m = measure_roi(ref32, cur32, r, out.coarse_px, true, params.wide_window_scale, params.fine, params.wide_min_response);
        all_ok = all_ok && m.ok;
        out.rois.push_back(m);
    }
    if (!all_ok) {
        out.message = "ROI 측정 실패 :";
        for (const auto &m : out.rois) if (!m.ok) out.message += " " + m.name + " " + m.message + ";";
        return out;
    }
    // ③ 특징
    if (!compute_features(out.rois, params.qr_rois.size(), out.s_px, out.d_px)) { out.message = "QR 또는 배경 ROI 가 설정에 없음"; return out; }
    out.ok = true;
    return out;
}

} // namespace ksh
