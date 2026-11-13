// stagectl — stage::Stage 시험 도구 (mm, ° 단위)
//
//  sudo ./build/stagectl ping | status | config
//  sudo ./build/stagectl home [y|t|both]
//  sudo ./build/stagectl move  <y_mm> [t_deg]       절대 이동
//  sudo ./build/stagectl rel   <dy_mm> [dt_deg]     상대 이동
//  sudo ./build/stagectl cycle <a_mm> <b_mm> <횟수>  a ↔ b 왕복, 이동 시간·도착 위치 기록
//  sudo ./build/stagectl stop
//  sudo ./build/stagectl speed <y_mm_s> [t_deg_s]
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "stage/stage.hpp"

using namespace stage;

namespace {

void usage()
{
    std::fprintf(stderr,
                 "사용법: stagectl <명령> [인자]\n"
                 "  ping | status | config\n"
                 "  home [y|t|both]               원점 복귀 (기본 y)\n"
                 "  move <y_mm> [t_deg]           절대 이동\n"
                 "  rel <dy_mm> [dt_deg]          상대 이동\n"
                 "  cycle <a_mm> <b_mm> <횟수>    a ↔ b 왕복 반복 (원점 복귀가 먼저 되어 있어야 함)\n"
                 "  stop\n"
                 "  speed <y_mm_s> [t_deg_s]      0 = 현재 값 유지\n");
}

void printStatus(const Status &s)
{
    std::printf("상태 %s  Y %.3f mm  Θ %.3f°  (스텝 %d, %d)\n", toString(s.state), s.pos.yMm(), s.pos.tDeg(),
                s.y_steps, s.t_steps);
    std::printf("원점 Y %s Θ %s  리밋 Y- %s Y+ %s  Θ원점SW %s\n", s.yHomed() ? "O" : "X", s.tHomed() ? "O" : "X",
                s.yLimitNeg() ? "눌림" : "-", s.yLimitPos() ? "눌림" : "-", s.tHomeSwitch() ? "눌림" : "-");
}

void printMove(const char *what, const MoveResult &r)
{
    std::printf("%s: %s  위치 Y %.3f mm Θ %.3f°  %.0f ms\n", what, toString(r.result), r.pos.yMm(), r.pos.tDeg(),
                r.elapsed_ms);
}

double arg(int argc, char **argv, int i, double def)
{
    return i < argc ? std::atof(argv[i]) : def;
}

}  // namespace

int main(int argc, char **argv)
{
    if (argc < 2) {
        usage();
        return 2;
    }
    const std::string cmd = argv[1];
    try {
        Stage st;
        if (cmd == "ping") {
            const Version v = st.ping();
            std::printf("PONG 프로토콜 %d, 펌웨어 %d.%d.%d\n", v.proto, v.fw_major, v.fw_minor, v.fw_patch);
        } else if (cmd == "status") {
            printStatus(st.status());
        } else if (cmd == "config") {
            const Limits l = st.limits();
            std::printf("소프트 리밋 Y %.3f ~ %.3f mm, Θ %.3f ~ %.3f°\n", l.y_min_um / 1000.0, l.y_max_um / 1000.0,
                        l.t_min_mdeg / 1000.0, l.t_max_mdeg / 1000.0);
        } else if (cmd == "home") {
            const std::string a = argc > 2 ? argv[2] : "y";
            const Axis ax = a == "t" ? Axis::T : a == "both" ? Axis::Both : Axis::Y;
            printMove("HOME", st.home(ax));
        } else if (cmd == "move" && argc >= 3) {
            const Position p = Position::fromMmDeg(arg(argc, argv, 2, 0), arg(argc, argv, 3, 0));
            const Limits l = st.limits();
            if (!l.contains(p))
                std::fprintf(stderr, "주의: 목표가 소프트 리밋 밖 → STM32 가 거부할 것\n");
            printMove("MOVE_ABS", st.moveTo(p));
        } else if (cmd == "rel" && argc >= 3) {
            printMove("MOVE_REL", st.moveBy(Position::fromMmDeg(arg(argc, argv, 2, 0), arg(argc, argv, 3, 0))));
        } else if (cmd == "cycle" && argc >= 5) {
            const Position a = Position::fromMmDeg(std::atof(argv[2]), 0);
            const Position b = Position::fromMmDeg(std::atof(argv[3]), 0);
            const int n = std::atoi(argv[4]);
            if (!st.homeIntact()) {
                std::fprintf(stderr, "원점이 없음. 먼저 stagectl home\n");
                return 1;
            }
            std::vector<double> err_a, ms;
            for (int i = 1; i <= n; ++i) {
                const MoveResult ra = st.moveTo(a);
                const MoveResult rb = st.moveTo(b);
                std::printf("[%3d] → a %s %.3f mm (%.0f ms)  → b %s %.3f mm (%.0f ms)\n", i, toString(ra.result),
                            ra.pos.yMm(), ra.elapsed_ms, toString(rb.result), rb.pos.yMm(), rb.elapsed_ms);
                if (!ra.ok() || !rb.ok()) {
                    std::fprintf(stderr, "이동 실패로 중단\n");
                    return 1;
                }
                err_a.push_back(ra.pos.y_um - a.y_um);
                ms.push_back(ra.elapsed_ms);
                ms.push_back(rb.elapsed_ms);
            }
            double mx = 0, sum = 0;
            for (double e : err_a) mx = std::max(mx, std::fabs(e));
            for (double m : ms) sum += m;
            std::printf("보고 위치 오차 최대 %.0f µm (STM32 계산값, 실제 위치는 카메라로 확인), 평균 이동 %.0f ms\n", mx,
                        sum / ms.size());
        } else if (cmd == "stop") {
            st.stop();
            std::printf("STOP 전송\n");
        } else if (cmd == "speed" && argc >= 3) {
            st.setSpeed(static_cast<uint32_t>(std::lround(arg(argc, argv, 2, 0) * 1000)),
                        static_cast<uint32_t>(std::lround(arg(argc, argv, 3, 0) * 1000)));
            std::printf("속도 설정 완료\n");
        } else {
            usage();
            return 2;
        }
        return 0;
    } catch (const StageError &e) {
        std::fprintf(stderr, "오류 [%s]: %s\n", toString(e.kind()), e.what());
        return 1;
    }
}
