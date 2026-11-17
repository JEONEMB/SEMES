// =====================================================================================
// alignment.cpp — 제출 형식 진입점 : 설정·영점을 들고 ksh::estimate_pose 를 부름 (텍스트 설정판)
//  알고리즘 본문은 ksh 모듈에 있다 (qr_locate, hole_locate, features, pose_model, estimator, setup)
// =====================================================================================
#include "alignment.hpp"

#include "ksh/align_config.hpp"
#include "ksh/estimator.hpp"
#include "ksh/image_util.hpp"
#include "ksh/setup.hpp"

#include <algorithm>
#include <iostream>
#include <limits>

namespace {

// 진입점이 들고 있는 상태 (제출 형식 alignment(src) 는 인자가 사진 하나뿐이라 설정·영점을 여기 보관)
//  여러 스레드에서 동시에 alignment() 를 부르지 말 것
ksh::AlignConfig g_cfg;
bool g_loaded = false;
ksh::ZeroOffset g_zero;
AlignmentInfo g_info;
ReferenceInfo g_ref_info;
SetupInfo g_setup_info;

bool ensure_ready() { return g_loaded || alignment_init(); }

// 측정 결과 → 공개 ROI 목록 (QR → 홀)
std::vector<AlignmentRoi> to_rois(const ksh::FeatureMeasurement &f, const ksh::FeatureParams &p) {
    std::vector<AlignmentRoi> v;
    for (size_t i = 0; i < f.qr.centers_px.size() && i < p.qr.boxes.size(); ++i) {
        AlignmentRoi a;
        a.name = p.qr.boxes[i].name;
        a.ref_rect_px = f.qr.boxes_px[i];
        a.center_px = f.qr.centers_px[i];
        a.zncc = 1.0;
        a.ok = true;
        v.push_back(a);
    }
    for (const auto &h : f.holes) {
        AlignmentRoi a;
        a.name = h.name;
        a.ref_rect_px = h.box_px;
        a.center_px = h.center_px;
        a.zncc = std::min(1.0, h.contrast / 255.0);
        a.ok = h.ok;
        v.push_back(a);
    }
    return v;
}

AlignmentInfo to_info(const ksh::PoseEstimate &e, const ksh::FeatureParams &p) {
    AlignmentInfo info;
    info.ok = e.ok;
    info.message = e.message;
    info.coarse_px = cv::Point2d(e.features.qr.coarse_dx_px, 0.0);
    info.rois = to_rois(e.features, p);
    info.s_px = e.features.s_px;
    info.d_px = e.features.d_px;
    info.elapsed_ms = e.elapsed_ms;
    return info;
}

} // namespace

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

bool alignment_set_reference(const cv::Mat &reference, bool save) {
    ReferenceInfo info;
    auto fail = [&](const std::string &msg) {
        info.message = msg;
        g_ref_info = info;
        std::cerr << "[alignment] 영점 등록 실패 : " << msg << "\n";
        return false;   // 이전 영점은 그대로 둠
    };
    if (!ensure_ready()) return fail("설정을 읽지 못함");
    if (reference.empty()) return fail("빈 사진");
    if (g_cfg.image_size_px.area() > 0 && reference.size() != g_cfg.image_size_px)
        return fail("크기 " + std::to_string(reference.cols) + "×" + std::to_string(reference.rows) + " ≠ 설정 " +
                    std::to_string(g_cfg.image_size_px.width) + "×" + std::to_string(g_cfg.image_size_px.height));
    const ksh::PoseEstimate e = ksh::estimate_pose(reference, g_cfg.features, g_cfg.model, ksh::ZeroOffset());   // 영점 없이 잰 값
    info.rois = to_rois(e.features, g_cfg.features);
    if (!e.ok) return fail(e.message);
    g_zero.y_mm = e.y_mm + g_cfg.center_to_encoder.y_mm;    // 엔코더 기준이면 센터 사진 자세와 엔코더 영점의 차이를 더함
    g_zero.t_deg = e.t_deg + g_cfg.center_to_encoder.t_deg;
    info.ok = true;
    info.zero_y_mm = g_zero.y_mm;
    info.zero_t_deg = g_zero.t_deg;
    if (save) {
        std::string error;
        if (!ksh::save_zero_offset(g_cfg.path, g_zero, error)) { info.message = error; std::cerr << "[alignment] " << error << "\n"; }
    }
    g_ref_info = info;
    return true;
}

bool alignment_setup(const std::vector<cv::Mat> &centers, bool save) {
    SetupInfo info;
    auto fail = [&](const std::string &msg) {
        info.message = msg;
        g_setup_info = info;
        std::cerr << "[alignment] 초기 세팅 실패 : " << msg << "\n";
        return false;   // 설정은 그대로 둠
    };
    if (!ensure_ready()) return fail("설정을 읽지 못함");
    std::vector<cv::Mat> gray;
    for (const auto &c : centers) {
        if (c.empty()) return fail("빈 사진");
        if (g_cfg.image_size_px.area() > 0 && c.size() != g_cfg.image_size_px) return fail("사진 크기가 설정과 다름");
        gray.push_back(ksh::to_gray8(c));
    }
    ksh::SetupResult r = ksh::run_setup(gray, g_cfg.features, g_cfg.model);
    r.zero.y_mm += g_cfg.center_to_encoder.y_mm;    // 엔코더 기준 (센터 사진 자세 − 엔코더 영점)
    r.zero.t_deg += g_cfg.center_to_encoder.t_deg;
    info.n_centers = int(centers.size());
    if (!r.ok) return fail(r.message);
    info.warnings = r.warnings;
    info.rois = to_rois(r.measured.front(), r.params);
    info.zero_y_mm = r.zero.y_mm;
    info.zero_t_deg = r.zero.t_deg;
    info.s_spread_px = r.s_spread_px;
    info.d_spread_px = r.d_spread_px;
    // 메모리 상태 갱신 (바로 alignment() 에 반영)
    g_cfg.features = r.params;
    g_cfg.zero = r.zero;
    g_zero = r.zero;
    if (save) {
        std::vector<ksh::BoxSpec> boxes = r.params.qr.boxes;
        boxes.insert(boxes.end(), r.params.holes.boxes.begin(), r.params.holes.boxes.end());
        std::string error;
        if (!ksh::save_roi_centers(g_cfg.path, boxes, error, true) || !ksh::save_zero_offset(g_cfg.path, r.zero, error, false))
            return fail("설정 파일 쓰기 실패 : " + error);
    }
    info.ok = true;
    g_setup_info = info;
    g_ref_info = ReferenceInfo();
    g_ref_info.ok = true;
    g_ref_info.relocated = true;
    g_ref_info.zero_y_mm = r.zero.y_mm;
    g_ref_info.zero_t_deg = r.zero.t_deg;
    g_ref_info.rois = info.rois;
    return true;
}

void alignment_clear_reference() {
    if (g_loaded) g_zero = g_cfg.zero;
    g_ref_info = ReferenceInfo();
}

const ReferenceInfo &alignment_reference_info() { return g_ref_info; }
const SetupInfo &alignment_setup_info() { return g_setup_info; }
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
    g_info = to_info(e, g_cfg.features);
    if (e.ok) {
        result.y = e.y_mm;
        result.t = e.t_deg;
    }
    return result;
}
