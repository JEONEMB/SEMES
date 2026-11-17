// =====================================================================================
// main.cpp — alignment() 사용 예 + 초기 세팅 (제출 형식, 교보재 텍스트 설정판)
//
//  실행 (작업 폴더 = Code 폴더, config/alignment.yml 을 상대 경로로 찾음)
//   1) 인자 없음                         : demo.image 한 장 추정 + 결과 창 (설치 점검용)
//   2) <사진> [센터 사진]                 : 사진 한 장 추정. 센터 사진을 주면 그 측정값을 영점으로 (파일에는 안 씀)
//   3) --setup <센터1> [센터2 센터3 ...]  : 초기 세팅 — ROI 박스 7개 좌표와 영점을 센터 사진으로 다시 맞춰 설정 파일에 기록 (원본 .bak)
//   4) --zero <센터 사진>                 : 영점만 다시 기록 (박스는 그대로)
//   옵션 --no-window                     : 결과 창 없이 (콘솔만)
//        --overlay <파일.png>             : 박스·찾은 중심을 그린 확인 그림 저장 (--setup · 사진 추정 모두)
//  실행에 사진 파일(기준·템플릿)은 필요 없다. 센터 사진은 초기 세팅·영점 기록 때만 쓴다
// =====================================================================================
#include <iostream>
#include <opencv2/opencv.hpp>

#include "alignment.hpp"
#include "ksh/align_config.hpp"
#include "ksh/dataset.hpp"
#include "ksh/image_util.hpp"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iomanip>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace {

const char *kConfigPath = "config/alignment.yml";

// 박스 · 중심 그리기 (성공 초록 / 실패 빨강)
cv::Mat draw_rois(const cv::Mat &img, const std::vector<AlignmentRoi> &rois, const std::string &text, int width_px) {
    cv::Mat view;
    if (img.channels() == 1) cv::cvtColor(img, view, cv::COLOR_GRAY2BGR);
    else view = img.clone();
    for (const auto &roi : rois) {
        const cv::Scalar col = roi.ok ? cv::Scalar(0, 200, 0) : cv::Scalar(0, 0, 255);
        cv::rectangle(view, roi.ref_rect_px, col, 4);
        if (roi.ok) cv::drawMarker(view, cv::Point(int(std::lround(roi.center_px.x)), int(std::lround(roi.center_px.y))), cv::Scalar(0, 0, 255),
                                   cv::MARKER_CROSS, 50, 3);
        cv::putText(view, roi.name, roi.ref_rect_px.tl() + cv::Point(0, -12), cv::FONT_HERSHEY_SIMPLEX, 1.4, col, 3);
    }
    cv::putText(view, text, cv::Point(60, 120), cv::FONT_HERSHEY_SIMPLEX, 2.6, cv::Scalar(0, 0, 0), 12);
    cv::putText(view, text, cv::Point(60, 120), cv::FONT_HERSHEY_SIMPLEX, 2.6, cv::Scalar(255, 255, 255), 4);
    if (width_px > 0 && view.cols > width_px) cv::resize(view, view, cv::Size(width_px, view.rows * width_px / view.cols), 0, 0, cv::INTER_AREA);
    return view;
}

} // namespace

int main(int argc, char **argv)
{
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;

    bool show_window = true, zero_mode = false, setup_mode = false;
    std::string overlay_path;
    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--no-window") show_window = false;
        else if (a == "--zero") zero_mode = true;
        else if (a == "--setup") setup_mode = true;
        else if (a == "--overlay" && i + 1 < argc) overlay_path = argv[++i];
        else args.push_back(a);
    }

    ksh::AlignConfig cfg;
    std::string error;
    if (!ksh::load_align_config(kConfigPath, cfg, error) || !alignment_init(kConfigPath)) {
        std::cerr << error << "\n설정 파일 " << kConfigPath << " 을 읽지 못함 (작업 폴더를 Code 폴더로 설정)\n";
        return 1;
    }

    // ========== 초기 세팅 : --setup <센터 사진들> ==========
    if (setup_mode) {
        if (args.empty()) { std::cerr << "사용 : alignment --setup <센터1> [센터2 센터3 ...]\n"; return 1; }
        std::vector<cv::Mat> centers;
        for (const auto &p : args) {
            cv::Mat m = ksh::read_image(p, cv::IMREAD_GRAYSCALE);
            if (m.empty()) { std::cerr << "사진을 읽지 못함 : " << p << "\n"; return 1; }
            centers.push_back(m);
        }
        if (!alignment_setup(centers, true)) { std::cerr << "초기 세팅 실패 : " << alignment_setup_info().message << "\n"; return 1; }
        const SetupInfo &si = alignment_setup_info();
        std::cout << "초기 세팅 완료 (센터 " << si.n_centers << "장) → " << kConfigPath << " (원본 .bak)\n";
        for (const auto &r : si.rois)
            std::cout << "  " << std::left << std::setw(5) << r.name << std::right << " 박스 중심 (" << r.ref_rect_px.x + r.ref_rect_px.width / 2 << ", "
                      << r.ref_rect_px.y + r.ref_rect_px.height / 2 << ")  찾은 중심 (" << std::fixed << std::setprecision(2) << r.center_px.x << ", "
                      << r.center_px.y << ")" << std::defaultfloat << "\n";
        std::cout << std::fixed << std::setprecision(5) << "  영점 : y = " << si.zero_y_mm << " mm, t = " << si.zero_t_deg << " deg\n"
                  << std::setprecision(3) << "  센터 사진끼리 차이 : s " << si.s_spread_px << " px, d " << si.d_spread_px << " px\n";
        for (const auto &w : si.warnings) std::cout << "  경고 : " << w << "\n";
        if (!overlay_path.empty()) {
            const cv::Mat v = draw_rois(ksh::read_image(args[0], cv::IMREAD_COLOR), si.rois, "SETUP", 1920);
            cv::imwrite(overlay_path, v);
            std::cout << "  확인 그림 : " << overlay_path << "\n";
        }
        return si.warnings.empty() ? 0 : 2;
    }

    // ========== 영점만 기록 : --zero <센터 사진> ==========
    if (zero_mode) {
        if (args.empty()) { std::cerr << "사용 : alignment --zero <센터 사진>\n"; return 1; }
        if (!alignment_set_reference(ksh::read_image(args[0], cv::IMREAD_GRAYSCALE), true)) {
            std::cerr << "영점 설정 실패 : " << alignment_reference_info().message << "\n";
            return 1;
        }
        const ReferenceInfo &ri = alignment_reference_info();
        std::cout << std::fixed << std::setprecision(5) << "Zero saved      : y = " << ri.zero_y_mm << " mm, t = " << ri.zero_t_deg << " deg → "
                  << kConfigPath << " (이전 파일 .bak)" << std::endl;
        return 0;
    }

    // ========== 1. 추정할 사진과 (선택) 센터 사진 ==========
    const std::string image_path = !args.empty() ? args[0] : cfg.demo.image;
    if (args.size() >= 2) {
        if (!alignment_set_reference(ksh::read_image(args[1], cv::IMREAD_GRAYSCALE))) {
            std::cerr << "센터 사진 설정 실패 : " << args[1] << " — " << alignment_reference_info().message << "\n";
            return 1;
        }
        const ReferenceInfo &ri = alignment_reference_info();
        std::cout << std::fixed << std::setprecision(5) << "Reference zero  : y = " << ri.zero_y_mm << " mm, t = " << ri.zero_t_deg << " deg"
                  << std::defaultfloat << std::endl;
    }
    cv::Mat image = ksh::read_image(image_path, (show_window || !overlay_path.empty()) ? cv::IMREAD_COLOR : cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "사진을 읽지 못함 : " << image_path << "\n";
        return 1;
    }

    // ========== 2. 추정 : 사진 한 장 → Y [mm], Θ [°] ==========
    Alignment_t alignment_result = alignment(image);

    // ========== 3. 결과 출력 ==========
    const AlignmentInfo &info = alignment_last_info();
    std::cout << "Image: " << image_path << std::endl;
    if (!info.ok) {
        std::cout << "Alignment failed: " << info.message << std::endl;
    } else {
        std::cout << std::fixed << std::setprecision(4) << "Alignment result: y = " << alignment_result.y << " mm"
                  << ", t = " << alignment_result.t << " deg" << std::endl
                  << "Correction move : y = " << -alignment_result.y << " mm, t = " << -alignment_result.t << " deg" << std::endl
                  << std::setprecision(3) << "Features        : s = " << info.s_px << " px, d = " << info.d_px << " px (coarse dx "
                  << info.coarse_px.x << " px)" << std::endl
                  << std::setprecision(1) << "Elapsed         : " << info.elapsed_ms << " ms" << std::endl;
        ksh::LabeledPhoto L;
        if (ksh::parse_label(std::filesystem::u8path(image_path).stem().u8string(), L) && !L.is_reference)
            std::cout << std::setprecision(4) << "Label (truth)   : y = " << L.y_mm << " mm, t = " << L.t_deg << " deg  → error " << std::setprecision(1)
                      << (alignment_result.y - L.y_mm) * 1000.0 << " um, " << std::setprecision(4) << alignment_result.t - L.t_deg << " deg"
                      << std::endl;
    }

    // ========== 4. 결과 그림 ==========
    char buf[160];
    if (info.ok) std::snprintf(buf, sizeof(buf), "Y = %+.4f mm   Theta = %+.4f deg", alignment_result.y, alignment_result.t);
    else std::snprintf(buf, sizeof(buf), "FAILED");
    if (!overlay_path.empty()) cv::imwrite(overlay_path, draw_rois(image, info.rois, buf, 1920));
    if (show_window) {
        cv::imshow("Alignment", draw_rois(image, info.rois, buf, cfg.demo.display_width_px));
        cv::waitKey(0);
    }
    return info.ok ? 0 : 1;
}
