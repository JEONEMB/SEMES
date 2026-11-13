// 확인 앱용 알고리즘 실행기: alignment(cv::Mat&) 를 별도 프로세스로 돌린다.
// (알고리즘이 죽거나 멈춰도 웹 서버는 살아 있고, 다음 요청 때 다시 띄운다.)
//
// 통신 (stdin/stdout, 바이너리 입력 + 한 줄 텍스트 출력)
//   시작:  "READY[ 기능...]\n"   기능: ref = 영점 등록(ALGR) 가능, roi = QR·나사 위치를 같이 돌려줌
//   추론:  "ALGN" + uint32 width + uint32 height (리틀엔디언) + width*height 바이트 (8bit 흑백)
//          → "OK <y> <t> <ms>[ | 이름 x y w h cx cy; ...]\n"  또는  "ERR <메시지>\n" (측정 실패 = NaN 도 ERR)
//          ROI 는 ALGO_HAS_REFERENCE 일 때만 (QR_L/QR_C/QR_R 사각형, 나사 S_LB/S_RT)
//   영점:  "ALGR" + 같은 형식의 센터 사진 → alignment_set_reference
//          → "OK <영점 y> <영점 t>[ | ROI...]\n"  또는  "ERR <메시지>\n"
//
// 옵션: --gray         흑백 1채널 그대로 넘김 (기본은 SEMES BMP 를 imread 한 것과 같은 BGR 3채널)
//       --config PATH  알고리즘 설정 파일 (alignment_init) 또는 자료 폴더 (initialize_joint)
//
// ALGO_HAS_REFERENCE: 김성현 A507 버전처럼 alignment_init / alignment_set_reference /
//   alignment_last_info / alignment_reference_info 가 있는 알고리즘 (CMake -DALGO_DIR=... 이 켬)
// ALGO_HAS_JOINT: 이혜원 값 7개 모듈처럼 initialize_joint(자료 폴더) 를 먼저 부르고 alignment() 를 쓰는 알고리즘
//   (CMake -DHYE_DIR=... -DHYE_MODULE=... 이 켬). 영점은 자료에 고정이라 ALGR 은 지원하지 않음
// ALGO_HAS_QR_POSE: 신윤석 QR 전용 단일 프레임 추정 qr_pose::estimate (CMake -DQR_POSE_DIR=... 이 켬).
//   --config 는 보정 파일(yml, 환경변수 QR_POSE_MODEL 로 넘김). 영점은 그 파일의 센터 템플릿에 고정
//
// 알고리즘 코드가 printf/std::cout 로 찍는 내용이 통신을 깨지 않도록
// 표준 출력(fd 1)은 표준 에러로 돌리고, 통신은 따로 복제한 fd 로 한다.
#include "alignment.hpp"
#ifdef ALGO_HAS_QR_POSE
#include "qr_pose.hpp"
#include <cstdlib>
#include <fstream>
#endif

#include <opencv2/imgproc/imgproc.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

static bool read_all(void *buf, size_t n)
{
    return std::fread(buf, 1, n, stdin) == n;
}

static std::string one_line(std::string s)
{
    for (char &c : s)
        if (c == '\n' || c == '\r')
            c = ' ';
    return s;
}

#ifdef ALGO_HAS_REFERENCE
// 측정한 QR·나사 위치 (알고리즘 좌표): " | 이름 x y w h cx cy; ..." — 화면에 무엇을 보고 계산했는지 그리기 위해
static std::string rois_text(const std::vector<AlignmentRoi> &rois)
{
    std::string out;
    char buf[160];
    for (const AlignmentRoi &r : rois) {
        if (!r.ok)
            continue;
        std::snprintf(buf, sizeof buf, "%s%s %d %d %d %d %.2f %.2f", out.empty() ? " | " : "; ", r.name.c_str(),
                      r.ref_rect_px.x, r.ref_rect_px.y, r.ref_rect_px.width, r.ref_rect_px.height,
                      r.center_px.x, r.center_px.y);
        out += buf;
    }
    return out;
}
#endif

// 사진 한 장 → (y mm, t °). 알고리즘마다 진입 함수가 다르다
static Alignment_t run_algorithm(cv::Mat &src)
{
#ifdef ALGO_HAS_QR_POSE
    const qr_pose::Pose p = qr_pose::estimate(src);
    Alignment_t r;
    r.y = p.y_mm;
    r.t = p.theta_deg;
    return r;
#else
    return alignment(src);
#endif
}

static std::string last_message()
{
#ifdef ALGO_HAS_REFERENCE
    return one_line(alignment_last_info().message);
#else
    return "";
#endif
}

int main(int argc, char **argv)
{
    bool color = true;
    std::string config;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--gray") == 0)
            color = false;
        else if (std::strcmp(argv[i], "--config") == 0 && i + 1 < argc)
            config = argv[++i];
    }

    std::fflush(stdout);
    const int proto_fd = dup(STDOUT_FILENO);
    dup2(STDERR_FILENO, STDOUT_FILENO);
    FILE *proto = fdopen(proto_fd, "w");
    if (!proto)
        return 3;

#ifdef ALGO_HAS_REFERENCE
    if (!(config.empty() ? alignment_init() : alignment_init(config))) {
        std::fprintf(proto, "ERR 설정 파일을 읽지 못함: %s\n", config.empty() ? "(기본)" : config.c_str());
        std::fflush(proto);
        return 4;
    }
    std::fprintf(proto, "READY ref roi\n");
#elif defined(ALGO_HAS_JOINT)
    try {
        if (config.empty())
            throw std::runtime_error("--config 로 자료(assets) 폴더를 주세요");
        initialize_joint(config);
    } catch (const std::exception &e) {
        std::fprintf(proto, "ERR 초기화 실패: %s\n", one_line(e.what()).c_str());
        std::fflush(proto);
        return 4;
    }
    std::fprintf(proto, "READY\n");
#elif defined(ALGO_HAS_QR_POSE)
    if (!config.empty()) {
        if (!std::ifstream(config)) {
            std::fprintf(proto, "ERR 보정 파일을 열 수 없음: %s\n", config.c_str());
            std::fflush(proto);
            return 4;
        }
        setenv("QR_POSE_MODEL", config.c_str(), 1);   // 첫 estimate() 때 이 파일을 읽는다
    }
    std::fprintf(proto, "READY\n");
#else
    if (!config.empty())
        std::fprintf(stderr, "--config 무시: 이 알고리즘은 설정 파일 API 가 없음\n");
    std::fprintf(proto, "READY\n");
#endif
    std::fflush(proto);

    for (;;) {
        char magic[4];
        uint32_t w = 0, h = 0;
        if (!read_all(magic, 4))
            return 0;  // 입력 끝 = 정상 종료
        const bool is_ref = std::memcmp(magic, "ALGR", 4) == 0;
        if ((!is_ref && std::memcmp(magic, "ALGN", 4) != 0) || !read_all(&w, 4) || !read_all(&h, 4)) {
            std::fprintf(proto, "ERR bad header\n");
            std::fflush(proto);
            return 2;
        }
        if (w == 0 || h == 0 || w > 10000 || h > 10000) {
            std::fprintf(proto, "ERR bad size %ux%u\n", w, h);
            std::fflush(proto);
            return 2;
        }
        cv::Mat gray(static_cast<int>(h), static_cast<int>(w), CV_8UC1);
        if (!read_all(gray.data, static_cast<size_t>(w) * h))
            return 0;

        cv::Mat src;
        if (color)
            cv::cvtColor(gray, src, cv::COLOR_GRAY2BGR);
        else
            src = gray;

        try {
            if (is_ref) {
#ifdef ALGO_HAS_REFERENCE
                if (alignment_set_reference(src, false)) {
                    const ReferenceInfo &ri = alignment_reference_info();
                    std::fprintf(proto, "OK %.9g %.9g%s\n", ri.zero_y_mm, ri.zero_t_deg, rois_text(ri.rois).c_str());
                } else {
                    std::fprintf(proto, "ERR %s\n", one_line(alignment_reference_info().message).c_str());
                }
#else
                std::fprintf(proto, "ERR 이 알고리즘은 영점 등록을 지원하지 않음\n");
#endif
            } else {
                const auto t0 = std::chrono::steady_clock::now();
                const Alignment_t r = run_algorithm(src);
                const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                if (std::isnan(static_cast<double>(r.y)) || std::isnan(static_cast<double>(r.t)))
                    std::fprintf(proto, "ERR 측정 실패 %s\n", last_message().c_str());
                else
#ifdef ALGO_HAS_REFERENCE
                    std::fprintf(proto, "OK %.9g %.9g %.3f%s\n", static_cast<double>(r.y), static_cast<double>(r.t), ms,
                                 rois_text(alignment_last_info().rois).c_str());
#else
                    std::fprintf(proto, "OK %.9g %.9g %.3f\n", static_cast<double>(r.y), static_cast<double>(r.t), ms);
#endif
            }
        } catch (const std::exception &e) {
            std::fprintf(proto, "ERR %s\n", one_line(e.what()).c_str());
        } catch (...) {
            std::fprintf(proto, "ERR unknown exception\n");
        }
        std::fflush(proto);
    }
}
