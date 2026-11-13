// =====================================================================================
// alignment.cpp — 제출 형식 진입점 : 설정·영점을 들고 ksh::estimate_pose 를 부름
//  알고리즘 본문은 ksh 모듈에 있다 (qr_locate, screw_locate, features, pose_model, estimator)
// =====================================================================================
#include "alignment.hpp"

#include "ksh/align_config.hpp"
#include "ksh/estimator.hpp"

#include <algorithm>
#include <iostream>
#include <limits>

namespace {

// 진입점이 들고 있는 상태 (제출 형식 alignment(src) 는 인자가 사진 하나뿐이라 설정·영점을 여기 보관)
//  여러 스레드에서 동시에 alignment() 를 부르지 말 것
ksh::AlignConfig g_cfg;     // 읽은 설정 (QR·나사 ROI, 보정식, 영점)
bool g_loaded = false;      // 설정을 읽었는지
ksh::ZeroOffset g_zero;     // 지금 쓰는 영점 (설정 zero: 또는 alignment_set_reference)
AlignmentInfo g_info;       // 마지막 alignment() 호출의 측정 상세
ReferenceInfo g_ref_info;   // 마지막 기준 사진 등록 결과

bool ensure_ready() { return g_loaded || alignment_init(); }

// 측정 결과 → 공개 ROI 목록 (QR 3개 → 나사)
std::vector<AlignmentRoi> to_rois(const ksh::FeatureMeasurement &f) {
    std::vector<AlignmentRoi> v;
    static const char *qr_names[3] = {"QR_L", "QR_C", "QR_R"};
    if (f.qr.ok)
        for (int i = 0; i < 3; ++i) {
            AlignmentRoi a;
            a.name = qr_names[i];
            a.ref_rect_px = f.qr.squares_px[i];
            a.center_px = f.qr.centers_px[i];
            a.zncc = std::max(0.0, 1.0 - f.qr.edge_fit_rms_px);
            a.ok = true;
            v.push_back(a);
        }
    for (const auto &s : f.screws) {
        AlignmentRoi a;
        a.name = s.name;
        a.ref_rect_px = s.roi_px;
        a.center_px = s.center_px;
        a.zncc = s.radius_px > 0 ? std::max(0.0, 1.0 - s.fit_rms_px / s.radius_px) : 0.0;
        a.ok = s.ok;
        v.push_back(a);
    }
    return v;
}

// 내부 추정 결과 → 공개 진단 구조체
AlignmentInfo to_info(const ksh::PoseEstimate &e) {
    AlignmentInfo info;
    info.ok = e.ok;
    info.message = e.message;
    info.coarse_px = e.features.qr.center_px;
    info.rois = to_rois(e.features);
    info.s_px = e.features.s_px;
    info.d_px = e.features.d_px;
    info.elapsed_ms = e.elapsed_ms;
    return info;
}

} // namespace

// 설정 · 보정식 · 영점 읽기
bool alignment_init(const std::string &config_path) {
    g_loaded = false;
    std::string error;
    if (!ksh::load_align_config(config_path, g_cfg, error)) {
        std::cerr << "[alignment] " << error << "\n";
        return false;
    }
    g_zero = g_cfg.zero;
    g_loaded = true;
    return true;
}

// 기준(센터) 사진 → 영점. 사진 자체는 보관하지 않음
bool alignment_set_reference(const cv::Mat &reference, bool save) {
    ReferenceInfo info;
    auto fail = [&](const std::string &msg) {
        info.message = msg;
        g_ref_info = info;
        std::cerr << "[alignment] 기준 사진 등록 실패 : " << msg << "\n";
        return false;   // 이전 영점은 그대로 둠
    };
    if (!ensure_ready()) return fail("설정을 읽지 못함");
    if (reference.empty()) return fail("빈 사진");
    if (g_cfg.image_size_px.area() > 0 && reference.size() != g_cfg.image_size_px)
        return fail("크기 " + std::to_string(reference.cols) + "×" + std::to_string(reference.rows) + " ≠ 설정 " +
                    std::to_string(g_cfg.image_size_px.width) + "×" + std::to_string(g_cfg.image_size_px.height));
    // 영점 없이 잰 값 = 새 영점
    const ksh::PoseEstimate e = ksh::estimate_pose(reference, g_cfg.features, g_cfg.model, ksh::ZeroOffset());
    info.rois = to_rois(e.features);
    if (!e.ok) return fail(e.message);
    g_zero.y_mm = e.y_mm;
    g_zero.t_deg = e.t_deg;
    info.ok = true;
    info.zero_y_mm = e.y_mm;
    info.zero_t_deg = e.t_deg;
    if (save) {
        std::string error;
        if (!ksh::save_zero_offset(g_cfg.path, g_zero, error)) { info.message = error; std::cerr << "[alignment] " << error << "\n"; }
    }
    g_ref_info = info;
    return true;
}

void alignment_clear_reference() {
    if (g_loaded) g_zero = g_cfg.zero;
    g_ref_info = ReferenceInfo();
}

const ReferenceInfo &alignment_reference_info() { return g_ref_info; }

// 마지막 호출의 측정 상세 (참조 반환 — 다음 alignment() 호출 때 바뀜)
const AlignmentInfo &alignment_last_info() { return g_info; }

// 사진 한 장 → 센터 대비 어긋남 (y mm, t °). 실패하면 NaN
Alignment_t alignment(cv::Mat &src)
{
    Alignment_t result;
    result.y = std::numeric_limits<double>::quiet_NaN();
    result.t = std::numeric_limits<double>::quiet_NaN();
    if (!ensure_ready()) {
        g_info = AlignmentInfo();
        g_info.message = "설정을 읽지 못함";
        return result;
    }
    const ksh::PoseEstimate e = ksh::estimate_pose(src, g_cfg.features, g_cfg.model, g_zero);
    g_info = to_info(e);
    if (e.ok) {
        result.y = e.y_mm;
        result.t = e.t_deg;
    }
    return result;
}
