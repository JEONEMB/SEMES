// relocate.cpp — 기준 사진(center) 교체 시 ROI 재배치 구현
#include "ksh/relocate.hpp"

#include "ksh/coarse_align.hpp"
#include "ksh/image_util.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace ksh {

namespace {
// printf 형식 문자열 → std::string (실패 메시지용)
template <typename... A> std::string fmt(const char *f, A... a) {
    char buf[200];
    std::snprintf(buf, sizeof(buf), f, a...);
    return buf;
}
} // namespace

bool relocate_rois(const cv::Mat &template32, const cv::Mat &new32, const FeatureParams &base, const RelocationParams &rp,
                   FeatureParams &out, RelocationResult &res) {
    res = RelocationResult();
    if (template32.empty() || new32.empty() || template32.size() != new32.size()) { res.message = "템플릿과 새 기준 사진 크기가 다름"; return false; }
    const cv::Size size = template32.size();

    // ---------- ① 전체 화면 저해상도 위상 상관 : 큰 이동을 먼저 찾음 (창이 크므로 수백 px 이동도 봉우리가 잡힘)
    const int k = std::max(1, rp.downscale);
    cv::Mat a, b;
    cv::resize(template32, a, cv::Size(size.width / k, size.height / k), 0, 0, cv::INTER_AREA);   // 축소 (평균 → 잡음 감소)
    cv::resize(new32, b, a.size(), 0, 0, cv::INTER_AREA);
    res.global_px = phase_shift(a, b, &res.global_response) * double(k);   // 축소 좌표 → 원본 좌표
    if (res.global_response < rp.min_response) {
        res.message = fmt("전체 화면 위상 상관 봉우리 약함 (%.3f < %.3f) — 다른 장면이거나 심하게 가려짐", res.global_response, rp.min_response);
        return false;
    }

    // ---------- ② Coarse 창 위상 상관 : 템플릿의 창 W 와, 새 사진에서 ① 만큼 옮긴 창 B 를 비교 → 원본 해상도로 다듬음
    const cv::Rect W(base.coarse.center_px.x - base.coarse.window_px.width / 2, base.coarse.center_px.y - base.coarse.window_px.height / 2,
                     base.coarse.window_px.width, base.coarse.window_px.height);
    const cv::Point g = round_point(res.global_px);
    const cv::Rect B = W + g;
    if (!rect_inside(W, size) || !rect_inside(B, size)) { res.message = fmt("Coarse 창이 새 사진 밖으로 나감 (이동 %d, %d px)", g.x, g.y); return false; }
    res.coarse_px = cv::Point2d(g) + phase_shift(template32(W), new32(B), &res.coarse_response);
    if (res.coarse_response < base.coarse.min_response) {
        res.message = fmt("Coarse 창 위상 상관 봉우리 약함 (%.3f < %.3f)", res.coarse_response, base.coarse.min_response);
        return false;
    }
    if (std::hypot(res.coarse_px.x, res.coarse_px.y) > rp.max_shift_px) {
        res.message = fmt("이동이 너무 큼 (%.1f, %.1f px > %.0f px) — 카메라 장착을 확인", res.coarse_px.x, res.coarse_px.y, rp.max_shift_px);
        return false;
    }

    // ---------- ③ ROI 마다 정밀 정합 : 측정과 같은 방식 (QR 은 Coarse 에서 바로, 배경은 넓은 창 탐색 후)
    for (const auto &r : base.qr_rois) {
        RoiShift m = measure_roi(template32, new32, r, res.coarse_px, false, base.wide_window_scale, base.fine);
        if (m.ok && cv::norm(m.shift_px - res.coarse_px) > base.qr_max_dev_px) { m.ok = false; m.message = "Coarse 결과에서 너무 벗어남 (이웃 QR)"; }
        res.rois.push_back(m);
    }
    for (const auto &r : base.background_rois)
        res.rois.push_back(measure_roi(template32, new32, r, res.coarse_px, true, base.wide_window_scale, base.fine, base.wide_min_response));
    for (const auto &m : res.rois)
        if (!m.ok) { res.message = "ROI 재배치 실패 : " + m.name + " " + m.message; return false; }

    // ---------- ④ QR 3개 이동이 서로 같아야 함 (평행이동만 있었다는 확인). 다르면 회전·배율이 바뀐 것 → 보정식이 맞지 않음
    double x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9;
    for (size_t i = 0; i < base.qr_rois.size(); ++i) {
        const cv::Point2d s = res.rois[i].shift_px;
        x0 = std::min(x0, s.x); x1 = std::max(x1, s.x); y0 = std::min(y0, s.y); y1 = std::max(y1, s.y);
    }
    res.qr_spread_px = std::max(x1 - x0, y1 - y0);
    if (res.qr_spread_px > rp.max_spread_px) {
        res.message = fmt("QR 3개 이동 차이 %.2f px > %.2f px — 카메라 회전·거리 변화 의심 (보정식 재학습 필요)", res.qr_spread_px, rp.max_spread_px);
        return false;
    }

    // ---------- 새 좌표 : Coarse 창 · ROI 중심을 측정한 이동만큼 옮김 (정수 화소, 소수 부분은 측정 때 정합이 흡수)
    FeatureParams moved = base;
    moved.coarse.center_px += round_point(res.coarse_px);
    for (size_t i = 0; i < moved.qr_rois.size(); ++i) moved.qr_rois[i].center_px += round_point(res.rois[i].shift_px);
    for (size_t i = 0; i < moved.background_rois.size(); ++i)
        moved.background_rois[i].center_px += round_point(res.rois[base.qr_rois.size() + i].shift_px);
    // 옮긴 영역이 사진 안에 있는지 (가장자리 ROI 는 옮기면 밖으로 나갈 수 있음)
    const cv::Rect Wn(moved.coarse.center_px.x - moved.coarse.window_px.width / 2, moved.coarse.center_px.y - moved.coarse.window_px.height / 2,
                      moved.coarse.window_px.width, moved.coarse.window_px.height);
    if (!rect_inside(Wn, size)) { res.message = "옮긴 Coarse 창이 사진 밖"; return false; }
    for (const auto *list : {&moved.qr_rois, &moved.background_rois})
        for (const auto &r : *list)
            if (!rect_inside(roi_rect(r), size)) { res.message = "옮긴 ROI 가 사진 밖 : " + r.name; return false; }

    out = moved;
    res.ok = true;
    return true;
}

} // namespace ksh
