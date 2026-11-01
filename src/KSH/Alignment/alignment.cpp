// =====================================================================================
// alignment.cpp — 제출 형식 진입점 : 설정·기준 사진을 들고 ksh::estimate_pose 를 부름
//  알고리즘 본문은 ksh 모듈에 있다 (coarse_align, fine_align, features, pose_model, estimator)
// =====================================================================================
#include "alignment.hpp"

#include "ksh/align_config.hpp"
#include "ksh/estimator.hpp"
#include "ksh/image_util.hpp"

#include <iostream>
#include <limits>

namespace {

ksh::AlignConfig g_cfg;
bool g_loaded = false;
cv::Mat g_ref32;            // 기준 사진 (CV_32F 회색)
AlignmentInfo g_info;

bool ensure_ready() {
    if (!g_loaded && !alignment_init()) return false;
    return !g_ref32.empty();
}

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
        a.ecc_cc = r.zncc;
        a.ok = r.ok;
        info.rois.push_back(a);
    }
    info.s_px = e.features.s_px;
    info.d_px = e.features.d_px;
    info.elapsed_ms = e.elapsed_ms;
    return info;
}

} // namespace

bool alignment_init(const std::string &config_path) {
    g_loaded = false;
    g_ref32.release();
    std::string error;
    if (!ksh::load_align_config(config_path, g_cfg, error)) {
        std::cerr << "[alignment] " << error << "\n";
        return false;
    }
    g_loaded = true;
    const cv::Mat ref = cv::imread(ksh::resolve_path(config_path, g_cfg.reference_path), cv::IMREAD_GRAYSCALE);
    if (ref.empty()) {
        std::cerr << "[alignment] 기준 사진을 읽을 수 없음 : " << g_cfg.reference_path << "\n";
        return true;   // 설정은 읽음. alignment_set_reference 로 기준을 따로 줄 수 있음
    }
    return alignment_set_reference(ref);
}

bool alignment_set_reference(const cv::Mat &reference) {
    if (reference.empty()) return false;
    if (g_cfg.image_size_px.area() > 0 && reference.size() != g_cfg.image_size_px) {
        std::cerr << "[alignment] 기준 사진 크기 " << reference.cols << "×" << reference.rows << " ≠ 설정 "
                  << g_cfg.image_size_px.width << "×" << g_cfg.image_size_px.height << "\n";
        return false;
    }
    g_ref32 = ksh::to_gray32(reference);
    return true;
}

const AlignmentInfo &alignment_last_info() { return g_info; }

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
