// Unity 시뮬레이터용 알고리즘 실행기 (임시, 작동 시험용)
// 원본: GitLab dev verify_web/native/align_worker.cpp (이영은) — 같은 통신 규약을 Windows 에서 쓰도록 고친 것
// 알고리즘: GitLab dev legacy/KSH/Alignment (김성현, 2026-11-01 버전) — verify_web 이 쓰는 A507 버전이 아님
//   11-01 버전에는 alignment_reference_info() 가 없어, 영점 등록(ALGR) 성공 시 영점 측정값 대신 "OK 0 0" 을 돌려준다
// 상태: OpenCV 3.4.15 미설치로 아직 빌드·실행 시험 안 함
//
// 통신 (stdin/stdout, 바이너리 입력 + 한 줄 텍스트 출력) — verify_web 과 같음
//   시작:  "READY\n"
//   추론:  "ALGN" + uint32 width + uint32 height (리틀엔디언) + width*height 바이트 (8bit 흑백)
//          → "OK <y> <t> <ms>\n"  또는  "ERR <메시지>\n" (측정 실패 = NaN 도 ERR)
//   영점:  "ALGR" + 같은 형식의 센터 사진 → alignment_set_reference → "OK 0 0\n" 또는 "ERR ...\n"
// 옵션: --gray (흑백 1채널 그대로), --config PATH (alignment_init)
#include "alignment.hpp"

#include <opencv2/imgproc/imgproc.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <string>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#define DUP _dup
#define DUP2 _dup2
#define FILENO _fileno
#define FDOPEN _fdopen
#else
#include <unistd.h>
#define DUP dup
#define DUP2 dup2
#define FILENO fileno
#define FDOPEN fdopen
#endif

static bool read_all(void *buf, size_t n) { return std::fread(buf, 1, n, stdin) == n; }

static std::string one_line(std::string s)
{
    for (char &c : s)
        if (c == '\n' || c == '\r')
            c = ' ';
    return s;
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
#ifdef _WIN32
    _setmode(_fileno(stdin), _O_BINARY);
#endif
    // 알고리즘이 printf/std::cout 로 찍어도 통신이 깨지지 않게: 표준 출력은 표준 에러로, 통신은 복제한 fd 로
    std::fflush(stdout);
    const int proto_fd = DUP(FILENO(stdout));
    DUP2(FILENO(stderr), FILENO(stdout));
    FILE *proto = FDOPEN(proto_fd, "w");
    if (!proto)
        return 3;

    if (!(config.empty() ? alignment_init() : alignment_init(config))) {
        std::fprintf(proto, "ERR 설정 파일을 읽지 못함: %s\n", config.empty() ? "(기본)" : config.c_str());
        std::fflush(proto);
        return 4;
    }
    std::fprintf(proto, "READY\n");
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
                if (alignment_set_reference(src))
                    std::fprintf(proto, "OK 0 0\n");
                else
                    std::fprintf(proto, "ERR 영점 등록 실패 %s\n", one_line(alignment_last_info().message).c_str());
            } else {
                const auto t0 = std::chrono::steady_clock::now();
                const Alignment_t r = alignment(src);
                const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                if (std::isnan(static_cast<double>(r.y)) || std::isnan(static_cast<double>(r.t)))
                    std::fprintf(proto, "ERR 측정 실패 %s\n", one_line(alignment_last_info().message).c_str());
                else
                    std::fprintf(proto, "OK %.9g %.9g %.3f\n", static_cast<double>(r.y), static_cast<double>(r.t), ms);
            }
        } catch (const std::exception &e) {
            std::fprintf(proto, "ERR %s\n", one_line(e.what()).c_str());
        } catch (...) {
            std::fprintf(proto, "ERR unknown exception\n");
        }
        std::fflush(proto);
    }
}
