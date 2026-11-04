// integrated_main.cpp — 통합 정렬기 예제
//  integrated_main <사진> [--config 파일] [--ref 기준사진] [--no-verify]
//  integrated_main --center-check <새 center 사진> [--config 파일]   : 새 기준 사진 채택 검사
#include "integrated/integrated.hpp"

#include <iomanip>
#include <iostream>

int main(int argc, char **argv) {
    std::string cfg = "config/integrated.yml", image, ref, center;
    bool no_verify = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--config" && i + 1 < argc) cfg = argv[++i];
        else if (a == "--ref" && i + 1 < argc) ref = argv[++i];
        else if (a == "--center-check" && i + 1 < argc) center = argv[++i];
        else if (a == "--no-verify") no_verify = true;
        else if (image.empty() && a.rfind("--", 0) != 0) image = a;
        else { std::cerr << "알 수 없는 인자 : " << a << "\n"; return 1; }
    }
    if (image.empty() && center.empty()) {
        std::cerr << "사용법 : integrated_main <사진> [--config 파일] [--ref 기준사진] [--no-verify]\n"
                     "        integrated_main --center-check <새 center 사진> [--config 파일]\n";
        return 1;
    }
    integrated::Aligner al;
    std::string err;
    if (!al.load(cfg, err, no_verify ? 0 : -1)) { std::cerr << err << "\n"; return 1; }
    if (!ref.empty() && !al.set_reference(cv::imread(ref, cv::IMREAD_COLOR), err)) { std::cerr << err << "\n"; return 1; }

    std::cout << std::fixed;
    if (!center.empty()) {
        double dy = 0, dt = 0; std::string msg;
        const bool ok = al.center_gate(cv::imread(center, cv::IMREAD_COLOR), dy, dt, msg);
        std::cout << std::setprecision(1) << "새 center − 현재 기준 : Y " << dy << " µm, Θ " << std::setprecision(4) << dt << "°  (한계 ±"
                  << std::setprecision(1) << al.settings().center_gate_y_um << " µm · ±" << std::setprecision(4) << al.settings().center_gate_t_deg << "°)\n"
                  << msg << "\n";
        return ok ? 0 : 5;
    }
    const cv::Mat img = cv::imread(image, cv::IMREAD_COLOR);
    if (img.empty()) { std::cerr << "사진을 읽을 수 없음 : " << image << "\n"; return 1; }
    const integrated::Result r = al.estimate(img);
    std::cout << std::setprecision(4)
              << "추정 Y " << r.y_mm << " mm, Θ " << r.t_deg << "°  → 보정 이동 (" << -r.y_mm << " mm, " << -r.t_deg << "°)\n"
              << "특징 s " << r.s_px << " px, d " << r.d_px << " px, Coarse (" << r.coarse_px.x << ", " << r.coarse_px.y << ") px\n";
    if (r.plate_checked)
        std::cout << "검증 : 판 " << (r.plate_found ? "찾음" : "못 찾음") << ", payload " << r.payloads[0] << "/" << r.payloads[1] << "/" << r.payloads[2]
                  << (r.payload_match ? " (기준과 같음)" : " (불일치)") << ", 판 중심 − Coarse " << r.plate_shift_diff_px << " px, 코너 − IC-GN QR 이동 "
                  << r.corner_s_diff_px << " px (" << r.corner_points << "점)\n";
    std::cout << std::setprecision(1) << "시간 : 추정 " << r.ms_estimate << " ms, 검증 " << r.ms_verify << " ms\n"
              << (r.ok ? "판정 : 사용 가능" : "판정 : 사용 불가") << (r.message.empty() ? "" : " — " + r.message) << "\n";
    return r.ok ? 0 : 2;
}
