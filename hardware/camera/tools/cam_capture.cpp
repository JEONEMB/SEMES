// cam_capture — stage_camera 시험 도구
//
//  카메라로 찍기
//    cam_capture --calib camera_calib.yml -o shot.png [--raw raw.png] [--settle] [--average 4]
//                [--count 5 --interval 1000] [--exposure 8333] [--gain 2.0] [--zoom 1.0]
//    --count 를 주면 shot_001.png, shot_002.png ... 로 저장
//
//  카메라 없이 저장된 원본 사진 변환 (4056x3040 → 3840x2160)
//    cam_capture --calib camera_calib.yml --rectify in.png -o out.png
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <thread>

#include "stage_camera/camera.hpp"
#include "stage_camera/rectifier.hpp"

using namespace stage_camera;
using Clock = std::chrono::steady_clock;

namespace {

void usage()
{
    std::fprintf(stderr,
                 "사용법:\n"
                 "  cam_capture --calib YML -o OUT.png [--raw RAW.png] [--settle] [--average N]\n"
                 "              [--count N --interval MS] [--exposure US] [--gain G] [--zoom Z]\n"
                 "              [--settle-shift PX] [--settle-frames N] [--settle-timeout MS]\n"
                 "  cam_capture --calib YML --rectify IN.png -o OUT.png [--zoom Z]\n");
}

double ms(Clock::time_point a, Clock::time_point b)
{
    return std::chrono::duration<double, std::milli>(b - a).count();
}

// out.png + 3 → out_003.png
std::string numbered(const std::string &path, int i)
{
    const size_t dot = path.find_last_of('.');
    char num[16];
    std::snprintf(num, sizeof(num), "_%03d", i);
    if (dot == std::string::npos)
        return path + num;
    return path.substr(0, dot) + num + path.substr(dot);
}

void save(const std::string &path, const cv::Mat &img)
{
    if (!cv::imwrite(path, img))
        throw std::runtime_error("저장 실패: " + path);
}

}  // namespace

int main(int argc, char **argv)
{
    CameraOptions opt;
    std::string out, raw_out, rectify_in;
    bool settle = false;
    int average = 1, count = 1, interval_ms = 0;
    bool have_calib = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s 뒤에 값이 필요함\n", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--calib") {
            opt.calib_path = val();
            have_calib = true;
        } else if (a == "-o") {
            out = val();
        } else if (a == "--raw") {
            raw_out = val();
        } else if (a == "--rectify") {
            rectify_in = val();
        } else if (a == "--settle") {
            settle = true;
        } else if (a == "--average") {
            average = std::atoi(val());
        } else if (a == "--count") {
            count = std::atoi(val());
        } else if (a == "--interval") {
            interval_ms = std::atoi(val());
        } else if (a == "--exposure") {
            opt.exposure_us = std::atoi(val());
        } else if (a == "--gain") {
            opt.analogue_gain = static_cast<float>(std::atof(val()));
        } else if (a == "--zoom") {
            opt.zoom = std::atof(val());
        } else if (a == "--settle-shift") {
            opt.settle.max_shift_px = std::atof(val());
        } else if (a == "--settle-frames") {
            opt.settle.stable_frames = std::atoi(val());
        } else if (a == "--settle-timeout") {
            opt.settle.timeout_ms = std::atoi(val());
        } else {
            usage();
            return 2;
        }
    }
    if (!have_calib || out.empty() || average < 1 || count < 1) {
        usage();
        return 2;
    }
    if (opt.exposure_us % 8333 > 50 && 8333 - opt.exposure_us % 8333 > 50)
        std::fprintf(stderr, "주의: 노출 %dus 가 8333us 의 배수가 아니라 조명 줄무늬가 생길 수 있음\n",
                     opt.exposure_us);

    try {
        if (!rectify_in.empty()) {
            const auto t0 = Clock::now();
            Rectifier rect(opt.calib_path, opt.zoom);
            const auto t1 = Clock::now();
            const cv::Mat in = cv::imread(rectify_in, cv::IMREAD_GRAYSCALE);
            if (in.empty())
                throw std::runtime_error("읽기 실패: " + rectify_in);
            const auto t2 = Clock::now();
            const cv::Mat res = rect.apply(in);
            const auto t3 = Clock::now();
            save(out, res);
            std::printf("변환표 생성 %.0f ms, 변환 %.1f ms, 원본 밖 픽셀 %.3f%%\n", ms(t0, t1), ms(t2, t3),
                        rect.outsideFraction() * 100);
            std::printf("저장: %s (%dx%d)\n", out.c_str(), res.cols, res.rows);
            return 0;
        }

        const auto t0 = Clock::now();
        Camera cam(opt);
        std::printf("카메라 준비 %.0f ms (노출 %dus, 게인 %.2f, 원본 밖 픽셀 %.3f%%)\n", ms(t0, Clock::now()),
                    opt.exposure_us, opt.analogue_gain, cam.rectifier().outsideFraction() * 100);

        for (int i = 1; i <= count; ++i) {
            const auto s0 = Clock::now();
            const Shot s = settle ? cam.captureSettled(average) : cam.capture(average);
            const auto s1 = Clock::now();

            const std::string path = count > 1 ? numbered(out, i) : out;
            save(path, s.image);
            if (!raw_out.empty())
                save(count > 1 ? numbered(raw_out, i) : raw_out, s.raw);

            std::printf("[%d] %s  seq %lld  노출 %dus 게인 %.2f  밝기 평균 %.1f  %.0f ms",
                        i, path.c_str(), static_cast<long long>(s.info.sequence), s.info.exposure_us,
                        s.info.analogue_gain, cv::mean(s.image)[0], ms(s0, s1));
            if (settle)
                std::printf("  안정 %s (%d프레임, %dms, 마지막 %.2fpx, 최대 %.2fpx)",
                            s.settle.settled ? "OK" : "시간초과", s.settle.frames, s.settle.waited_ms,
                            s.settle.last_shift_px, s.settle.max_shift_px);
            std::printf("\n");
            std::fflush(stdout);

            if (i < count && interval_ms > 0)
                std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
        }
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "오류: %s\n", e.what());
        return 1;
    }
}
