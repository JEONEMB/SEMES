// =====================================================================================
// main.cpp — alignment() 사용 예 (제출 형식, A507 장비용)
//
//  실행 방법 (작업 폴더 = Code 폴더, config/alignment.yml 을 상대 경로로 찾음)
//   1) 인자 없음                  : alignment.yml 의 demo.image 한 장을 추정하고 결과 창 표시 (설치 점검용)
//   2) <사진> [센터 사진]          : 지정한 사진 한 장 추정. 센터 사진을 주면 그 사진을 영점으로 삼아 비교 (교보재와 같은 사용법)
//   3) --zero <센터 사진>          : 센터 사진으로 영점을 잡아 config/alignment.yml 의 zero: 에 기록 (원본은 .bak)
//   옵션 --no-window              : 결과 창을 띄우지 않음 (라즈베리파이처럼 화면이 없을 때)
//  기준 사진이 없어도 동작한다 (설정 파일 두 개만 있으면 됨). 센터 사진은 영점을 다시 잡을 때만 쓴다
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
#include <windows.h>   // 콘솔 한글 출력 (UTF-8)
#endif

namespace {

const char *kConfigPath = "config/alignment.yml";

// =====================================================================================
// show_result — 결과 창 : 찾은 QR 사각형과 나사 ROI 를 그리고 추정값을 글자로 표시
// =====================================================================================
void show_result(const cv::Mat &img, const Alignment_t &r, int display_width_px) {
    cv::Mat view;
    if (img.channels() == 1) cv::cvtColor(img, view, cv::COLOR_GRAY2BGR);
    else view = img.clone();

    const AlignmentInfo &info = alignment_last_info();
    for (const auto &roi : info.rois) {
        // 측정 성공 = 초록, 실패 = 빨강
        const cv::Scalar col = roi.ok ? cv::Scalar(0, 200, 0) : cv::Scalar(0, 0, 255);
        cv::rectangle(view, roi.ref_rect_px, col, 4);
        if (roi.ok) cv::drawMarker(view, cv::Point(int(std::lround(roi.center_px.x)), int(std::lround(roi.center_px.y))), col,
                                   cv::MARKER_CROSS, 40, 3);
        cv::putText(view, roi.name, roi.ref_rect_px.tl() + cv::Point(0, -12), cv::FONT_HERSHEY_SIMPLEX, 1.4, col, 3);
    }
    char buf[160];
    if (info.ok) std::snprintf(buf, sizeof(buf), "Y = %+.4f mm   Theta = %+.4f deg   (%.0f ms)", r.y, r.t, info.elapsed_ms);
    else std::snprintf(buf, sizeof(buf), "FAILED");
    cv::putText(view, buf, cv::Point(60, 120), cv::FONT_HERSHEY_SIMPLEX, 3.0, cv::Scalar(0, 0, 0), 12);
    cv::putText(view, buf, cv::Point(60, 120), cv::FONT_HERSHEY_SIMPLEX, 3.0, cv::Scalar(255, 255, 255), 4);
    if (display_width_px > 0 && view.cols > display_width_px)
        cv::resize(view, view, cv::Size(display_width_px, view.rows * display_width_px / view.cols), 0, 0, cv::INTER_AREA);
    cv::imshow("Alignment", view);
    cv::waitKey(0);
}

} // namespace

int main(int argc, char **argv)
{
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;

    // 인자 정리 : 옵션(--no-window, --zero)과 위치 인자(사진, 센터 사진)
    bool show_window = true, zero_mode = false;
    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--no-window") show_window = false;
        else if (a == "--zero") zero_mode = true;
        else args.push_back(a);
    }

    ksh::AlignConfig cfg;
    std::string error;
    if (!ksh::load_align_config(kConfigPath, cfg, error) || !alignment_init(kConfigPath)) {
        std::cerr << error << "\n설정 파일 " << kConfigPath << " 을 읽지 못함 (작업 폴더를 Code 폴더로 설정)\n";
        return 1;
    }

    // ========== 0. 영점 기록 모드 : --zero <센터 사진> ==========
    if (zero_mode) {
        if (args.empty()) { std::cerr << "사용 : alignment --zero <센터 사진>\n"; return 1; }
        const cv::Mat ref = ksh::read_image(args[0], cv::IMREAD_GRAYSCALE);
        if (!alignment_set_reference(ref, true)) {
            std::cerr << "영점 설정 실패 : " << alignment_reference_info().message << "\n";
            return 1;
        }
        const ReferenceInfo &ri = alignment_reference_info();
        std::cout << std::fixed << std::setprecision(5) << "Zero saved      : y = " << ri.zero_y_mm << " mm, t = " << ri.zero_t_deg
                  << " deg  → " << kConfigPath << " (이전 파일 .bak)" << std::endl;
        return 0;
    }

    // ========== 1. 추정할 사진과 (선택) 센터 사진 ==========
    const std::string image_path = !args.empty() ? args[0] : cfg.demo.image;
    const std::string ref_path = args.size() >= 2 ? args[1] : std::string();
    if (!ref_path.empty()) {
        // 센터 사진을 주면 그 측정값을 영점으로 (파일에는 쓰지 않음)
        if (!alignment_set_reference(ksh::read_image(ref_path, cv::IMREAD_GRAYSCALE))) {
            std::cerr << "센터 사진 설정 실패 : " << ref_path << " — " << alignment_reference_info().message << "\n";
            return 1;
        }
        const ReferenceInfo &ri = alignment_reference_info();
        std::cout << std::fixed << std::setprecision(5) << "Reference zero  : y = " << ri.zero_y_mm << " mm, t = " << ri.zero_t_deg << " deg"
                  << std::defaultfloat << std::endl;
    }

    cv::Mat image = ksh::read_image(image_path, show_window ? cv::IMREAD_COLOR : cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "사진을 읽지 못함 : " << image_path << "\n";
        return 1;
    }

    // ========== 2. 추정 : 사진 한 장 → Y [mm], Θ [°] ==========
    //  ① QR 3개 → ② 나사 ROI 추정·원 맞춤 → ③ 특징 s, d → ④ 보정식 역산
    Alignment_t alignment_result = alignment(image);

    // ========== 3. 결과 출력 ==========
    const AlignmentInfo &info = alignment_last_info();
    std::cout << "Image: " << image_path << std::endl;
    if (!info.ok) {
        std::cout << "Alignment failed: " << info.message << std::endl;
    } else {
        std::cout << std::fixed << std::setprecision(4)
                  << "Alignment result: y = " << alignment_result.y << " mm"
                  << ", t = " << alignment_result.t << " deg" << std::endl
                  << "Correction move : y = " << -alignment_result.y << " mm"
                  << ", t = " << -alignment_result.t << " deg" << std::endl
                  // 중간값 : s = QR 3개 중심 가로 위치, d = s − 나사 가로 위치 평균 (시차)
                  << std::setprecision(3) << "Features        : s = " << info.s_px << " px, d = " << info.d_px << " px" << std::endl
                  << std::setprecision(1) << "Elapsed         : " << info.elapsed_ms << " ms" << std::endl;

        // 파일 이름이 라벨 형식이면(예 "Y+0.50 T+0.10 (1)") 정답과 오차도 출력 — 확인용, 추정에는 쓰지 않음
        ksh::LabeledPhoto L;
        if (ksh::parse_label(std::filesystem::u8path(image_path).stem().u8string(), L) && !L.is_reference)
            std::cout << std::setprecision(4) << "Label (truth)   : y = " << L.y_mm << " mm, t = " << L.t_deg << " deg  → error "
                      << std::setprecision(1) << (alignment_result.y - L.y_mm) * 1000.0 << " um, " << std::setprecision(4)
                      << alignment_result.t - L.t_deg << " deg" << std::endl;
    }

    // ========== 4. 결과 창 ==========
    if (show_window) show_result(image, alignment_result, cfg.demo.display_width_px);
    return 0;
}
