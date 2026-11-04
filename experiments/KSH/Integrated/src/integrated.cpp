// integrated.cpp — 통합 정렬기 구현 : ksh 추정 → shin 판·QR 검증 → 판정
#include "integrated/integrated.hpp"

#include "pipeline_internal.hpp"   // shin : detectPlate, createQrRois
#include "ksh/image_util.hpp"
#include "ksh/pose_model.hpp"

#include <chrono>
#include <cmath>
#include <iostream>
#include <sstream>

namespace integrated {

namespace {

double ms_since(const std::chrono::steady_clock::time_point &t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// shin 검출 코드의 진행 출력(std::cout)을 검출하는 동안만 끔 — 원본 코드는 고치지 않음
struct MuteCout {
    std::ostringstream sink;
    std::streambuf *old;
    MuteCout() : old(std::cout.rdbuf(sink.rdbuf())) {}
    ~MuteCout() { std::cout.rdbuf(old); }
};

cv::Mat to_bgr(const cv::Mat &src) {
    cv::Mat bgr;
    if (src.channels() == 1) cv::cvtColor(src, bgr, cv::COLOR_GRAY2BGR);
    else if (src.channels() == 4) cv::cvtColor(src, bgr, cv::COLOR_BGRA2BGR);
    else bgr = src;
    return bgr;
}

double real(const cv::FileNode &n, double fallback) { return n.empty() ? fallback : double(n); }
int integer(const cv::FileNode &n, int fallback) { return n.empty() ? fallback : int(n); }
bool flag(const cv::FileNode &n, bool fallback) { return n.empty() ? fallback : int(n) != 0; }

} // namespace

bool load_settings(const std::string &path, Settings &s, std::string &error) {
    s = Settings();
    s.config_path = path;
    if (!ksh::load_align_config(path, s.ksh, error)) return false;   // ROI·정렬·모델·허용 범위 (ksh 형식)
    s.reference_path = s.ksh.reference_path;
    cv::FileStorage fs;
    try {
        if (!fs.open(path, cv::FileStorage::READ)) { error = "설정 파일을 열 수 없음 : " + path; return false; }
    } catch (const cv::Exception &e) { error = e.what(); return false; }
    const cv::FileNode n = fs["integrated"];
    if (!n.empty()) {
        s.verify_plate = flag(n["verify_plate"], s.verify_plate);
        s.reject_on_guard = flag(n["reject_on_guard"], s.reject_on_guard);
        s.reject_unverified = flag(n["reject_unverified"], s.reject_unverified);
        s.plate_guard_px = real(n["plate_guard_px"], s.plate_guard_px);
        s.corner_check_px = real(n["corner_check_px"], s.corner_check_px);
        s.min_corner_points = integer(n["min_corner_points"], s.min_corner_points);
        s.center_gate_y_um = real(n["center_gate_y_um"], s.center_gate_y_um);
        s.center_gate_t_deg = real(n["center_gate_t_deg"], s.center_gate_t_deg);
    }
    return true;
}

bool Aligner::load(const std::string &config_path, std::string &error, int verify_override) {
    ref32_.release();
    if (!load_settings(config_path, s_, error)) return false;
    if (verify_override >= 0) s_.verify_plate = verify_override != 0;
    const std::string ref_path = ksh::resolve_path(config_path, s_.reference_path);
    const cv::Mat ref = cv::imread(ref_path, cv::IMREAD_COLOR);
    if (ref.empty()) { error = "기준 사진을 읽을 수 없음 : " + ref_path; return false; }
    return set_reference(ref, error);
}

bool Aligner::detect(const cv::Mat &image, PlateDetectionResult &det) const {
    // shin alignment() 의 검출 단계와 같은 순서 (시각화·파일 저장 없이)
    using namespace alignment_detail;
    det = PlateDetectionResult{};
    if (image.empty() || image.depth() != CV_8U) return false;
    const cv::Mat bgr = to_bgr(image);
    AlignmentOptions opt;
    opt.visualize = false;
    MuteCout mute;
    try {
        const PlateGeometry g = detectPlate(bgr, opt);
        det.found = g.found;
        det.score = g.score;
        det.searchRoi = g.searchRoi;
        det.corners = g.corners;
        if (!det.found) return false;
        createQrRois(bgr, opt, det);
        for (int i = 0; i < 3; ++i)
            if (det.qrRois[i].valid) analyzeQrFeatures(bgr, opt, det.qrRois[i], i);
    } catch (const cv::Exception &) {
        det.found = false;
        return false;
    }
    return det.found;
}

bool Aligner::set_reference(const cv::Mat &reference, std::string &error) {
    if (reference.empty()) { error = "기준 사진이 비어 있음"; return false; }
    ref32_ = ksh::to_gray32(reference);
    ref_corners_.clear();
    ref_plate_ok_ = false;
    if (s_.verify_plate) {
        ref_plate_ok_ = detect(reference, ref_det_);
        if (ref_plate_ok_)
            for (const auto &qr : ref_det_.qrRois)
                for (const auto &f : qr.features)
                    if (f.accepted) ref_corners_[{qr.payload, f.id}] = f.refined;
        if (!ref_plate_ok_) { error = "기준 사진에서 판을 찾지 못함 (검증을 쓰려면 기준 사진을 다시 찍을 것)"; return false; }
    }
    return true;
}

void Aligner::verify(const cv::Mat &image, const ksh::PoseEstimate &e, Result &r) const {
    const auto t0 = std::chrono::steady_clock::now();
    r.plate_checked = true;
    PlateDetectionResult det;
    r.plate_found = detect(image, det);
    std::string contradiction, unverified;
    if (!r.plate_found) unverified = "판 검출 실패";
    if (r.plate_found) {
        // ① 같은 판인가 : 세 QR payload 가 기준 사진과 같고 모듈 재생성이 일치
        r.payload_match = true;
        bool all_decoded = true;
        for (int i = 0; i < 3; ++i) {
            const auto &q = det.qrRois[i];
            r.payloads[i] = q.payload;
            all_decoded = all_decoded && q.decoded && q.modelMatched;
            r.payload_match = r.payload_match && q.payload == ref_det_.qrRois[i].payload;
        }
        if (!all_decoded) unverified = "QR 해독 실패";
        else if (!r.payload_match) contradiction = "QR payload 가 기준 사진과 다름 (다른 판·주소)";
        // ② 판 중심 이동 ≈ ksh Coarse 이동 (서로 다른 방법 : 허프 직선 vs 위상 상관)
        r.plate_shift_diff_px = (det.center.x - ref_det_.center.x) - e.features.coarse_px.x;
        if (std::abs(r.plate_shift_diff_px) > s_.plate_guard_px && contradiction.empty()) contradiction = "판 중심 이동과 Coarse 이동이 다름";
        // ③ Fourier 코너 QR 평균 가로 이동 ≈ IC-GN QR 평균 가로 이동
        double sum_qr = 0.0; int n_qr = 0;
        for (const auto &qr : det.qrRois) {
            double sum = 0.0; int n = 0;
            for (const auto &f : qr.features) {
                if (!f.accepted) continue;
                const auto it = ref_corners_.find({qr.payload, f.id});
                if (it == ref_corners_.end()) continue;
                sum += f.refined.x - it->second.x; ++n;
            }
            r.corner_points += n;
            if (n >= s_.min_corner_points) { sum_qr += sum / n; ++n_qr; }
        }
        if (n_qr == 3) {
            r.corner_s_px = sum_qr / 3.0;
            if (e.ok) {
                r.corner_s_diff_px = r.corner_s_px - e.features.s_px;
                if (std::abs(r.corner_s_diff_px) > s_.corner_check_px && contradiction.empty()) contradiction = "코너 QR 이동과 IC-GN QR 이동이 다름";
            }
        } else if (unverified.empty()) {
            unverified = "공통 코너 부족";
        }
    }
    r.guard_ok = contradiction.empty();
    r.verified = unverified.empty() && r.guard_ok;
    if (!contradiction.empty()) r.message = "검증 모순 : " + contradiction;
    else if (!unverified.empty()) r.message = "검증 못 함 : " + unverified;
    r.ms_verify = ms_since(t0);
}

Result Aligner::estimate(const cv::Mat &image) const {
    Result r;
    if (ref32_.empty()) { r.message = "기준 사진 없음"; return r; }
    const auto t0 = std::chrono::steady_clock::now();
    const ksh::PoseEstimate e = ksh::estimate_pose(ref32_, image, s_.ksh.features, s_.ksh.model);
    r.ms_estimate = ms_since(t0);
    r.estimate_ok = e.ok;
    r.coarse_px = e.features.coarse_px;
    if (e.ok) { r.y_mm = e.y_mm; r.t_deg = e.t_deg; r.s_px = e.features.s_px; r.d_px = e.features.d_px; }
    else r.message = "추정 : " + e.message;
    if (s_.verify_plate && ref_plate_ok_) verify(image, e, r);
    const bool unverified = r.plate_checked && r.guard_ok && !r.verified;
    r.ok = r.estimate_ok && (r.guard_ok || !s_.reject_on_guard) && !(unverified && s_.reject_unverified);
    if (!r.ok) { r.y_mm = kNaN; r.t_deg = kNaN; }
    else if (!r.message.empty()) r.message += " (값은 유지 — 설정 reject_on_guard / reject_unverified)";
    return r;
}

bool Aligner::center_gate(const cv::Mat &new_center, double &dy_um, double &dt_deg, std::string &message) const {
    dy_um = dt_deg = kNaN;
    if (ref32_.empty()) { message = "현재 기준 사진 없음"; return false; }
    const ksh::PoseEstimate e = ksh::estimate_pose(ref32_, new_center, s_.ksh.features, s_.ksh.model);
    if (!e.ok) { message = "새 center 측정 실패 : " + e.message; return false; }
    // 두 center 의 차이만 본다 → 영점 상수 c 를 빼고 역변환
    ksh::PoseModel m0 = s_.ksh.model;
    m0.s.c_px = 0.0; m0.d.c_px = 0.0;
    double y = 0.0, t = 0.0;
    if (!ksh::invert_pose_model(m0, e.features.s_px, e.features.d_px, y, t)) { message = "역변환 실패"; return false; }
    dy_um = y * 1000.0; dt_deg = t;
    const bool ok = std::abs(dy_um) <= s_.center_gate_y_um && std::abs(dt_deg) <= s_.center_gate_t_deg;
    message = ok ? "채택 (반복 정밀도 범위 안)" : "다시 찍기 권장 (반복 정밀도 범위 밖 — 접근 방향·고정 상태 확인)";
    return ok;
}

} // namespace integrated
