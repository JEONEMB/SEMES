// =====================================================================================
// main.cpp — 운동 모델 데모 : 로봇 자세 (Y, Θ) 가 바뀔 때 영상이 어떻게 움직이는가
//
//  실행 위치 : 프로젝트 폴더 (img/, config/ 가 보이는 곳 — Visual Studio 기본 작업 폴더)
//  인수      : [img 폴더] [--batch] [--no-gui] [--config 파일] [--out 폴더]
//    verify ... Y·Θ 분리 방법 검증 실행 (src/yt/yt_verify.hpp 참고) — 예 : QR_Image.exe verify all
//    (없음)    대화형 화면. 슬라이더로 Y·Θ 를 바꾸며 5개 화면을 봄
//    --batch   화면 PNG 저장 + 모든 사진으로 모델 vs 실측 비교 (results/motion_model/)
//    --no-gui  창을 띄우지 않음 (--batch 와 함께)
//  화면 : 1 Field, 2 Separation, 3 Profile, 4 Depth, 5 Real photo vs model
//  키   : 1~5 화면 전환, n/p 다음·이전 사진(5번), s 현재 화면 저장, h 도움말, q/ESC 종료
// =====================================================================================
#include "camera_model.hpp"
#include "motion_views.hpp"
#include "real_compare.hpp"
#include "yt/yt_verify.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace fs = std::filesystem;
using namespace mm;

namespace {

const char* kWin = "Motion Model (Y, Theta)";

void printModel(const CameraModel& cam) {
    const Sensitivity s = sensitivities(cam);
    std::printf("\n[운동 모델] fx %.1f  fy %.1f  (cx, cy) (%.0f, %.0f)  플레이트 Z %.1f mm  배경 Z %.1f mm  회전축 a %.1f mm\n",
                cam.fx, cam.fy, cam.cx, cam.cy, cam.z_plate, cam.z_bg, cam.axis_a);
    std::printf("  항목                        모델        실측(EDA·Center_check)\n");
    std::printf("  플레이트 이동 / Y (px/mm)  %8.3f      14.25\n", s.plate_per_mm);
    std::printf("  플레이트 이동 / Θ (px/°)   %8.3f      80.06\n", s.plate_per_deg);
    std::printf("  폭 차 R−L / Θ (px/°)       %8.3f      0.659 (1차), 0.698 (2차)\n", s.wdiff_per_deg);
    std::printf("  폭 차 R−L / Y (px/mm)      %8.4f     −0.026\n", s.wdiff_per_mm);
    std::printf("  아래 벽 이동 / Y (px/mm)   %8.3f      11.07\n", s.bg_per_mm);
    std::printf("  아래 벽 이동 / Θ (px/°)    %8.3f      81.3\n", s.bg_per_deg);
    std::printf("  같은 플레이트 이동 : Y 1 mm ↔ Θ %.4f°\n\n", s.equal_shift_deg_per_mm);
}

void printHelp() {
    std::printf("키 : 1 Field | 2 Separation | 3 Profile | 4 Depth | 5 실제 사진 비교\n"
                "     n / p 다음·이전 사진 (5번 화면) | s 현재 화면 PNG 저장 | h 도움말 | q, ESC 종료\n"
                "슬라이더 : Y (0.01 mm, 가운데 = 0), Theta (0.001°, 가운데 = 0), Arrow gain (화살표 길이 배율, 100 = 자동)\n\n");
}

void save(const cv::Mat& img, const std::string& dir, const std::string& name) {
    fs::create_directories(dir);
    const std::string p = dir + "/" + name;
    cv::imwrite(p, img);
    std::cout << "  저장 : " << p << "\n";
}

} // namespace

int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    // 첫 인수가 verify 이면 Y·Θ 분리 검증 실행기로 넘김
    if (argc >= 2 && std::string(argv[1]) == "verify") return yt::verifyMain(argc - 1, argv + 1);

    std::string imgDir = "img", cfg = "config/camera_model.yml", outDir = "results/motion_model";
    bool batch = false, gui = true;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--batch") batch = true;
        else if (a == "--no-gui") gui = false;
        else if (a == "--config" && i + 1 < argc) cfg = argv[++i];
        else if (a == "--out" && i + 1 < argc) outDir = argv[++i];
        else imgDir = a;
    }

    CameraModel cam;
    if (cam.load(cfg)) std::cout << "모델 설정 : " << cfg << "\n";
    else {
        fs::create_directories(fs::path(cfg).parent_path());
        cam.save(cfg);
        std::cout << "모델 설정 파일이 없어 기본값으로 만들었습니다 : " << cfg << "\n";
    }
    printModel(cam);

    // 배경으로 쓸 기준 영상 (1/4 축소)
    cv::Mat refSmall;
    const auto entries = scanImages(imgDir);
    for (const auto& e : entries)
        if (e.stem == "center") { refSmall = cv::imread(e.path, cv::IMREAD_REDUCED_COLOR_4); break; }
    std::cout << "사진 " << entries.size() << " 장 (" << imgDir << ")" << (refSmall.empty() ? " — center.bmp 없음, 배경 없이 표시" : "") << "\n";

    std::vector<ImageEntry> photos;   // 비교용 (기준 영상 제외)
    for (const auto& e : entries) if (!e.is_ref) photos.push_back(e);

    if (batch) {
        std::cout << "\n[batch] 화면 저장\n";
        const Pose demo{0.5, 0.2};
        save(renderFieldView(cam, demo, 1.0, refSmall), outDir, "view1_field.png");
        save(renderSeparationView(cam, 1.0, 1.0, refSmall), outDir, "view2_separation.png");
        save(renderProfileView(cam, 1.0), outDir, "view3_profile.png");
        save(renderDepthView(cam, demo), outDir, "view4_depth.png");
        RealComparer cmp(cam, imgDir);
        for (const char* name : {"Y+1.00", "T+0.18", "T+1.00", "Y+0.50 T+1.00 (1)", "Y+1.00 T+2.00 (1)"})
            for (const auto& e : photos)
                if (e.stem == name) {
                    CompareResult r; cv::Mat panel;
                    if (cmp.compare(e, r, &panel)) save(panel, outDir, std::string("view5_real_") + name + ".png");
                    break;
                }
        std::cout << "\n[batch] 모든 사진으로 모델 vs 실측 비교\n";
        runBatch(cam, entries, imgDir, outDir);
    }
    if (!gui) return 0;

    printHelp();
    cv::namedWindow(kWin, cv::WINDOW_NORMAL);
    cv::resizeWindow(kWin, 1600, 900);
    cv::createTrackbar("Y (0.01 mm)", kWin, nullptr, 400);
    cv::createTrackbar("Theta (0.001 deg)", kWin, nullptr, 4000);
    cv::createTrackbar("Arrow gain", kWin, nullptr, 500);
    cv::setTrackbarPos("Y (0.01 mm)", kWin, 250);
    cv::setTrackbarPos("Theta (0.001 deg)", kWin, 2200);
    cv::setTrackbarPos("Arrow gain", kWin, 100);

    RealComparer cmp(cam, imgDir);
    int view = 0, photo = 0, lastKey[5] = {-1, -1, -1, -1, -1};
    for (size_t i = 0; i < photos.size(); ++i) if (photos[i].stem == "T+0.18") photo = int(i);
    cv::Mat frame, comparePanel;
    int lastPhoto = -1;
    for (;;) {
        const int ty = cv::getTrackbarPos("Y (0.01 mm)", kWin), tt = cv::getTrackbarPos("Theta (0.001 deg)", kWin);
        const int tg = std::max(1, cv::getTrackbarPos("Arrow gain", kWin));
        const Pose pose{(ty - 200) / 100.0, (tt - 2000) / 1000.0};
        const int state[5] = {view, ty, tt, tg, photo};
        bool changed = frame.empty();
        for (int i = 0; i < 5; ++i) if (state[i] != lastKey[i]) changed = true;
        if (changed) {
            for (int i = 0; i < 5; ++i) lastKey[i] = state[i];
            switch (view) {
                case 0: frame = renderFieldView(cam, pose, tg / 100.0, refSmall); break;
                case 1: frame = renderSeparationView(cam, std::abs(pose.y_mm) > 1e-6 ? std::abs(pose.y_mm) : 1.0, tg / 100.0, refSmall); break;
                case 2: frame = renderProfileView(cam, std::abs(pose.y_mm) > 1e-6 ? std::abs(pose.y_mm) : 1.0); break;
                case 3: frame = renderDepthView(cam, pose); break;
                default:
                    if (photos.empty()) { frame = cv::Mat(900, 1600, CV_8UC3, cv::Scalar(255, 255, 255)); break; }
                    if (photo != lastPhoto) {
                        CompareResult r;
                        std::cout << "  비교 : " << photos[photo].stem << "\n";
                        if (!cmp.compare(photos[photo], r, &comparePanel)) comparePanel = cv::Mat(900, 1600, CV_8UC3, cv::Scalar(255, 255, 255));
                        lastPhoto = photo;
                    }
                    frame = comparePanel;
            }
            cv::imshow(kWin, frame);
        }
        const int k = cv::waitKey(30);
        if (k == 'q' || k == 27) break;
        if (k >= '1' && k <= '5') view = k - '1';
        if (k == 'h') printHelp();
        if (!photos.empty() && k == 'n') { photo = (photo + 1) % int(photos.size()); view = 4; }
        if (!photos.empty() && k == 'p') { photo = (photo + int(photos.size()) - 1) % int(photos.size()); view = 4; }
        if (k == 's') {
            char name[128];
            std::snprintf(name, sizeof(name), "view%d_Y%+.2f_T%+.3f.png", view + 1, pose.y_mm, pose.t_deg);
            save(frame, outDir, view == 4 && !photos.empty() ? "view5_real_" + photos[photo].stem + ".png" : name);
        }
        if (cv::getWindowProperty(kWin, cv::WND_PROP_VISIBLE) < 1) break;
    }
    cv::destroyAllWindows();
    return 0;
}
