// 확인 앱용 알고리즘 실행기: 저장소의 alignment(cv::Mat&) 를 별도 프로세스로 돌린다.
// (알고리즘이 죽거나 멈춰도 웹 서버는 살아 있고, 다음 요청 때 다시 띄운다.)
//
// 통신 (stdin/stdout, 바이너리 입력 + 한 줄 텍스트 출력)
//   시작:  "READY\n"
//   입력:  "ALGN" + uint32 width + uint32 height (리틀엔디언) + width*height 바이트 (8bit 흑백)
//   출력:  "OK <y> <t> <ms>\n"  또는  "ERR <메시지>\n"
//
// 옵션: --gray  흑백 1채널 그대로 넘김 (기본은 SEMES BMP 를 imread 한 것과 같은 BGR 3채널)
//
// 알고리즘 코드가 printf/std::cout 로 찍는 내용이 통신을 깨지 않도록
// 표준 출력(fd 1)은 표준 에러로 돌리고, 통신은 따로 복제한 fd 로 한다.
#include "alignment.hpp"

#include <opencv2/imgproc/imgproc.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <string>
#include <unistd.h>

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

int main(int argc, char **argv)
{
    bool color = true;
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], "--gray") == 0)
            color = false;

    std::fflush(stdout);
    const int proto_fd = dup(STDOUT_FILENO);
    dup2(STDERR_FILENO, STDOUT_FILENO);
    FILE *proto = fdopen(proto_fd, "w");
    if (!proto)
        return 3;

    std::fprintf(proto, "READY\n");
    std::fflush(proto);

    for (;;) {
        char magic[4];
        uint32_t w = 0, h = 0;
        if (!read_all(magic, 4))
            return 0;  // 입력 끝 = 정상 종료
        if (std::memcmp(magic, "ALGN", 4) != 0 || !read_all(&w, 4) || !read_all(&h, 4)) {
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

        const auto t0 = std::chrono::steady_clock::now();
        try {
            const Alignment_t r = alignment(src);
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            std::fprintf(proto, "OK %.9g %.9g %.3f\n", static_cast<double>(r.y), static_cast<double>(r.t), ms);
        } catch (const std::exception &e) {
            std::fprintf(proto, "ERR %s\n", one_line(e.what()).c_str());
        } catch (...) {
            std::fprintf(proto, "ERR unknown exception\n");
        }
        std::fflush(proto);
    }
}
