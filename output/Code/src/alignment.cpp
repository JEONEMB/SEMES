// =====================================================================================
// alignment.cpp — 제출 형식 진입점 : 설정·기준 사진을 들고 ksh::estimate_pose 를 부름
//  알고리즘 본문은 ksh 모듈에 있다 (coarse_align, fine_align, features, pose_model, estimator)
// =====================================================================================
#include "alignment.hpp"

#include "ksh/align_config.hpp"
#include "ksh/estimator.hpp"
#include "ksh/image_util.hpp"
#include "ksh/relocate.hpp"

#include <iostream>
#include <limits>

namespace {

// 진입점이 들고 있는 상태 (제출 형식 alignment(src) 는 인자가 사진 하나뿐이라 설정·기준 사진을 여기 보관)
//  여러 스레드에서 동시에 alignment() 를 부르지 말 것
ksh::AlignConfig g_cfg;     // 읽은 설정 (ROI, 정합 설정, 보정식)
bool g_loaded = false;      // 설정을 읽었는지
cv::Mat g_ref32;            // 기준 사진 (CV_32F 회색)
AlignmentInfo g_info;       // 마지막 alignment() 호출의 측정 상세
ksh::FeatureParams g_base_features;   // 설정 파일에 적힌 ROI·창 (템플릿 좌표). 재배치는 항상 여기서 출발 (누적되지 않게)
cv::Mat g_template32;       // ROI 템플릿 사진 (CV_32F 회색, relocation.template_path)
ReferenceInfo g_ref_info;   // 마지막 기준 사진 설정 결과

// 설정을 아직 안 읽었으면 기본 경로로 읽고, 기준 사진이 있는지 확인
bool ensure_ready() {
    if (!g_loaded && !alignment_init()) return false;
    return !g_ref32.empty();
}

// 내부 추정 결과 → 공개 진단 구조체 (alignment.hpp 의 AlignmentInfo)
AlignmentInfo to_info(const ksh::PoseEstimate &e) {
    AlignmentInfo info;
    info.ok = e.ok;
    info.message = e.message;
    info.coarse_px = e.features.coarse_px;
    for (const auto &r : e.features.rois) {
        AlignmentRoi a;
        a.name = r.name;
        a.ref_rect_px = r.ref_rect_px;
        a.shift_px = r.shift_px;
        a.zncc = r.zncc;
        a.ok = r.ok;
        info.rois.push_back(a);
    }
    info.s_px = e.features.s_px;
    info.d_px = e.features.d_px;
    info.elapsed_ms = e.elapsed_ms;
    return info;
}

} // namespace

// 설정 · 보정식 · 기준 사진(reference_path) 읽기. 기준 사진을 못 읽어도 설정을 읽었으면 true
bool alignment_init(const std::string &config_path) {
    g_loaded = false;
    g_ref32.release();
    std::string error;
    if (!ksh::load_align_config(config_path, g_cfg, error)) {
        std::cerr << "[alignment] " << error << "\n";
        return false;
    }
    g_loaded = true;
    g_base_features = g_cfg.features;
    g_template32.release();
    if (g_cfg.relocation.enable) {
        // ROI 템플릿 : ROI 좌표를 정한 사진. 없으면 재배치를 할 수 없으므로 설정 실패로 알림
        const cv::Mat tpl = ksh::read_image(g_cfg.relocation.template_path, cv::IMREAD_GRAYSCALE);
        if (tpl.empty()) {
            std::cerr << "[alignment] ROI 템플릿 사진을 읽을 수 없음 : " << g_cfg.relocation.template_path << "\n";
            g_loaded = false;
            return false;
        }
        g_template32 = ksh::to_gray32(tpl);
    }
    const cv::Mat ref = ksh::read_image(ksh::resolve_path(config_path, g_cfg.reference_path), cv::IMREAD_GRAYSCALE);
    if (ref.empty()) {
        std::cerr << "[alignment] 기준 사진을 읽을 수 없음 : " << g_cfg.reference_path << "\n";
        return true;   // 설정은 읽음. alignment_set_reference 로 기준을 따로 줄 수 있음
    }
    return alignment_set_reference(ref);
}

// 기준 사진 교체 : 크기가 설정(image_width_px × image_height_px)과 같아야 함 (ROI 좌표가 이 크기 기준)
bool alignment_set_reference(const cv::Mat &reference) {
    ReferenceInfo info;
    auto fail = [&](const std::string &msg) {
        info.message = msg;
        g_ref_info = info;
        std::cerr << "[alignment] 기준 사진 설정 실패 : " << msg << "\n";
        return false;   // 이전 기준 사진·ROI 는 그대로 둠
    };
    if (!g_loaded && !alignment_init()) return fail("설정을 읽지 못함");
    if (reference.empty()) return fail("빈 사진");
    if (g_cfg.image_size_px.area() > 0 && reference.size() != g_cfg.image_size_px)
        return fail("크기 " + std::to_string(reference.cols) + "×" + std::to_string(reference.rows) + " ≠ 설정 " +
                    std::to_string(g_cfg.image_size_px.width) + "×" + std::to_string(g_cfg.image_size_px.height));
    const cv::Mat ref32 = ksh::to_gray32(reference);

    ksh::FeatureParams features = g_base_features;   // 재배치하지 않으면 설정 좌표 그대로
    if (g_cfg.relocation.enable) {
        // 템플릿(교보재 center) → 새 center 위상 상관으로 ROI·Coarse 창 옮기기
        ksh::RelocationResult r;
        const bool ok = ksh::relocate_rois(g_template32, ref32, g_base_features, g_cfg.relocation, features, r);
        info.relocated = true;
        info.shift_px = r.coarse_px;
        info.qr_spread_px = r.qr_spread_px;
        const size_t nq = g_base_features.qr_rois.size();
        for (size_t i = 0; i < r.rois.size(); ++i) {
            AlignmentRoi a;
            a.name = r.rois[i].name;
            const ksh::RoiSpec &spec = ok ? (i < nq ? features.qr_rois[i] : features.background_rois[i - nq])
                                          : (i < nq ? g_base_features.qr_rois[i] : g_base_features.background_rois[i - nq]);
            a.ref_rect_px = ksh::roi_rect(spec);
            a.shift_px = r.rois[i].shift_px;
            a.zncc = r.rois[i].zncc;
            a.ok = r.rois[i].ok;
            info.rois.push_back(a);
        }
        if (!ok) return fail(r.message);
    }
    g_cfg.features = features;
    g_ref32 = ref32;
    info.ok = true;
    g_ref_info = info;
    return true;
}

// 마지막 기준 사진 설정 결과
const ReferenceInfo &alignment_reference_info() { return g_ref_info; }

// 마지막 호출의 측정 상세 (참조 반환 — 다음 alignment() 호출 때 바뀜)
const AlignmentInfo &alignment_last_info() { return g_info; }

// 사진 한 장 → 기준 대비 어긋남 (y mm, t °). 실패하면 NaN
Alignment_t alignment(cv::Mat &src)
{
    Alignment_t result;
    result.y = std::numeric_limits<double>::quiet_NaN();
    result.t = std::numeric_limits<double>::quiet_NaN();
    if (!ensure_ready()) {
        g_info = AlignmentInfo();
        g_info.message = "설정 또는 기준 사진 없음";
        return result;
    }
    const ksh::PoseEstimate e = ksh::estimate_pose(g_ref32, src, g_cfg.features, g_cfg.model);
    g_info = to_info(e);
    if (e.ok) {
        result.y = e.y_mm;
        result.t = e.t_deg;
    }
    return result;
}
