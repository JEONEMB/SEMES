// =====================================================================================
// test_real_photos.cpp — 실사진 점검 (사진이 있을 때만 실행)
//
//  사용 : align_photo_check <센터 사진> [사진 ...]     (작업 폴더 = Code, config/alignment.yml 사용)
//  R1 센터 사진 → (0, 0) (설정 영점이 이 사진으로 잡혀 있을 때)
//  R2 같은 사진을 두 번 → 결과가 비트 단위로 같음 (상태 없는 계산)
//  R3 밝기 ×0.85 + 15 → 결과 변화 Y < 5 µm, Θ < 0.0005°        (조명 변화)
//  R4 가우시안 잡음 σ 2 (8bit) → 결과 변화 Y < 10 µm, Θ < 0.001° (센서 잡음)
//  R5 사진 전체 +100 px 가로 이동 → s +100 px, d 변화 < 0.05 px  (대략 위치 · 박스 이동)
//  R6 홀 W2 가림 / QR_C 가림 → NaN 과 실패 이유
//  R7 추가 사진 : 추정값과 파일 이름 라벨의 오차, 처리 시간
// =====================================================================================
#include "alignment.hpp"
#include "ksh/align_config.hpp"
#include "ksh/dataset.hpp"
#include "ksh/image_util.hpp"

#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>   // 콘솔 한글 출력 (UTF-8)
#endif

namespace {
int g_fail = 0, g_pass = 0;
void check(bool ok, const std::string &what) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    (ok ? g_pass : g_fail)++;
}
Alignment_t run(const cv::Mat &m) {
    cv::Mat c = m.clone();
    return alignment(c);
}
} // namespace

int main(int argc, char **argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);   // 소스·문자열이 UTF-8 → 콘솔도 UTF-8 로 (안 하면 한글이 CP949 로 깨짐)
#endif
    if (argc < 2) { std::printf("사용 : align_photo_check <센터 사진> [사진 ...]\n"); return 2; }
    if (!alignment_init("config/alignment.yml")) return 1;
    const cv::Mat center = ksh::read_image(argv[1], cv::IMREAD_GRAYSCALE);
    if (center.empty()) { std::printf("센터 사진을 읽지 못함\n"); return 1; }
    char b[200];

    std::printf("== R1 센터 사진 (센터 = −center_to_encoder : 센터 사진 자세가 엔코더 영점에서 떨어진 만큼)\n");
    ksh::AlignConfig cfg0;
    std::string e0;
    ksh::load_align_config("config/alignment.yml", cfg0, e0);
    const Alignment_t r0 = run(center);
    const AlignmentInfo i0 = alignment_last_info();
    std::snprintf(b, sizeof(b), "센터 → (%.5f mm, %.5f°) ≈ (%.5f, %.5f), s %.3f, d %.3f px", r0.y, r0.t, -cfg0.center_to_encoder.y_mm,
                  -cfg0.center_to_encoder.t_deg, i0.s_px, i0.d_px);
    check(std::abs(r0.y + cfg0.center_to_encoder.y_mm) < 1e-4 && std::abs(r0.t + cfg0.center_to_encoder.t_deg) < 1e-5, b);

    std::printf("== R2 반복\n");
    const Alignment_t r0b = run(center);
    check(r0b.y == r0.y && r0b.t == r0.t, "같은 사진 두 번 → 같은 결과");

    std::printf("== R3 밝기 변화\n");
    cv::Mat br;
    center.convertTo(br, CV_8U, 0.85, 15);
    const Alignment_t r3 = run(br);
    std::snprintf(b, sizeof(b), "×0.85 + 15 → 변화 Y %.2f µm, Θ %.5f°", (r3.y - r0.y) * 1000, r3.t - r0.t);
    check(std::abs(r3.y - r0.y) < 0.005 && std::abs(r3.t - r0.t) < 0.0005, b);

    std::printf("== R4 잡음\n");
    cv::Mat noise(center.size(), CV_16S), nz;
    cv::randn(noise, 0, 2);
    cv::Mat c16;
    center.convertTo(c16, CV_16S);
    cv::add(c16, noise, c16);
    c16.convertTo(nz, CV_8U);
    const Alignment_t r4 = run(nz);
    std::snprintf(b, sizeof(b), "σ 2 잡음 → 변화 Y %.2f µm, Θ %.5f°", (r4.y - r0.y) * 1000, r4.t - r0.t);
    check(std::abs(r4.y - r0.y) < 0.010 && std::abs(r4.t - r0.t) < 0.001, b);

    std::printf("== R5 사진 전체 가로 이동\n");
    cv::Mat sh;
    const cv::Mat A = (cv::Mat_<double>(2, 3) << 1, 0, 100, 0, 1, 0);
    cv::warpAffine(center, sh, A, center.size(), cv::INTER_NEAREST, cv::BORDER_REPLICATE);
    const Alignment_t r5 = run(sh);
    const AlignmentInfo i5 = alignment_last_info();
    std::snprintf(b, sizeof(b), "+100 px → Δs %.4f px, Δd %.4f px (대략 이동 %.1f px), 추정 (%.3f mm, %.4f°)", i5.s_px - i0.s_px, i5.d_px - i0.d_px,
                  i5.coarse_px.x, r5.y, r5.t);
    check(i5.ok && std::abs(i5.s_px - i0.s_px - 100.0) < 0.01 && std::abs(i5.d_px - i0.d_px) < 0.05, b);

    std::printf("== R6 가림\n");
    {
        ksh::AlignConfig cfg;
        std::string e;
        ksh::load_align_config("config/alignment.yml", cfg, e);
        cv::Mat o1 = center.clone(), o2 = center.clone();
        const auto &w2 = cfg.features.holes.boxes[1];
        cv::rectangle(o1, cv::Rect(w2.cx_px - 40, w2.cy_px - 40, 80, 80), cv::Scalar(140), cv::FILLED);
        const Alignment_t a = run(o1);
        check(std::isnan(a.y) && std::isnan(a.t), "W2 홀 가림 → NaN : " + alignment_last_info().message);
        const auto &qc = cfg.features.qr.boxes[1];
        cv::rectangle(o2, cv::Rect(qc.cx_px - 60, qc.cy_px - 130, 120, 260), cv::Scalar(40), cv::FILLED);
        const Alignment_t c = run(o2);
        check(std::isnan(c.y) && std::isnan(c.t), "QR_C 가림 → NaN : " + alignment_last_info().message);
    }

    if (argc > 2) std::printf("== R7 추가 사진 (라벨 = 파일 이름, 영점 = 설정 zero)\n");
    for (int k = 2; k < argc; ++k) {
        const cv::Mat m = ksh::read_image(argv[k], cv::IMREAD_GRAYSCALE);
        const auto t0 = std::chrono::steady_clock::now();
        const Alignment_t r = run(m);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        ksh::LabeledPhoto L;
        const std::string stem = std::filesystem::u8path(argv[k]).stem().u8string();
        if (ksh::parse_label(stem, L))
            std::printf("  %-22s 추정 (%+.4f mm, %+.4f°)  라벨 (%+.2f, %+.2f)  오차 %+.1f µm, %+.4f°  %.0f ms\n", stem.c_str(), r.y, r.t, L.y_mm, L.t_deg,
                        (r.y - L.y_mm) * 1000, r.t - L.t_deg, ms);
        else std::printf("  %-22s 추정 (%+.4f mm, %+.4f°)  %.0f ms\n", stem.c_str(), r.y, r.t, ms);
    }
    std::printf("\n결과 : 통과 %d, 실패 %d\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
