// stage.hpp — 보정 앱이 쓰는 스테이지 API (/dev/robot_stage ioctl 래퍼)
//
//  stage::Stage st;                               // /dev/robot_stage 열기
//  st.home();                                     // 원점 복귀, DONE 까지 기다림
//  auto r = st.moveTo({1500, 0});                 // Y 1.5mm 로 이동, DONE 까지 기다림
//  if (r.result != stage::DoneResult::Ok) { ... } // LIMIT / FAULT / STOPPED
//
//  단위: Y = µm, Θ = 0.001° (프로토콜 그대로, STM32 가 펄스로 변환)
//
//  오류 처리
//   - 이동이 "끝났지만 결과가 나쁨"(LIMIT, FAULT, STOPPED)은 MoveResult 로 돌려준다 → 앱이 판단
//   - 명령 자체가 안 됨(NAK, 응답 없음, 장치 없음, 원점 잃음)은 StageError 예외
//   - STM32 가 재부팅하면 원점이 사라진다. 이후 이동은 NAK(NOT_HOMED) 또는 DONE 시간 초과가 되며,
//     둘 다 StageError::Kind::HomeLost 로 알려 준다 → 앱은 home() 부터 다시
#ifndef STAGE_STAGE_HPP
#define STAGE_STAGE_HPP

#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace stage {

using Millis = std::chrono::milliseconds;

struct Position {
    int32_t y_um = 0;    // 1 = 1 µm
    int32_t t_mdeg = 0;  // 1 = 0.001°

    // mm, ° → 프로토콜 단위 (가장 가까운 정수로 반올림)
    static Position fromMmDeg(double y_mm, double t_deg);
    double yMm() const { return y_um / 1000.0; }
    double tDeg() const { return t_mdeg / 1000.0; }
};

enum class Axis : uint8_t { Y = 1, T = 2, Both = 3 };

enum class State { Idle, Moving, Homing, Fault, Unknown };

enum class DoneResult { Ok, Stopped, Limit, Fault, Unknown };

enum class Nak { None, UnknownType, BadLen, NotHomed, OutOfRange, Busy, Fault, BadParam, Unknown };

struct Version {
    int proto = 0, fw_major = 0, fw_minor = 0, fw_patch = 0;
};

struct Status {
    State state = State::Unknown;
    uint8_t flags = 0;  // RS_FLAG_* 원본
    Position pos;
    int32_t y_steps = 0, t_steps = 0;

    bool yHomed() const;
    bool tHomed() const;
    bool yLimitNeg() const;  // Y- 끝 스위치 눌림
    bool yLimitPos() const;  // Y+ 끝 스위치 눌림
    bool tHomeSwitch() const;
};

// 소프트 리밋. 유일한 출처는 STM32 (GET_CONFIG)
struct Limits {
    int32_t y_min_um = 0, y_max_um = 0, t_min_mdeg = 0, t_max_mdeg = 0;
    bool contains(const Position &p) const;
    Position clamp(const Position &p) const;
};

struct MoveResult {
    DoneResult result = DoneResult::Unknown;
    Position pos;          // 완료 시점 위치 (STM32 보고)
    double elapsed_ms = 0; // 명령 전송 ~ DONE
    bool ok() const { return result == DoneResult::Ok; }
};

class StageError : public std::runtime_error {
public:
    enum class Kind {
        NoDevice,   // /dev/robot_stage 없음, 또는 ttyACM0 에 line discipline 이 안 붙음
        Nak,        // STM32 가 거부 (nak() 에 사유)
        Timeout,    // 응답 없음 / DONE 시간 초과 (이동은 아직 진행 중일 수 있음)
        HomeLost,   // 원점을 잃음 (STM32 재부팅 등) → home() 다시
        Interrupted,
        Io,         // 그 밖의 시스템 오류
    };
    StageError(Kind kind, const std::string &msg, Nak nak = Nak::None, int err = 0)
        : std::runtime_error(msg), kind_(kind), nak_(nak), errno_(err)
    {
    }
    Kind kind() const { return kind_; }
    Nak nak() const { return nak_; }
    int sysErrno() const { return errno_; }

private:
    Kind kind_;
    Nak nak_;
    int errno_;
};

const char *toString(State s);
const char *toString(DoneResult r);
const char *toString(Nak n);
const char *toString(StageError::Kind k);

class Stage {
public:
    static constexpr Millis kDefaultHomeTimeout{60000};
    static constexpr Millis kDefaultMoveTimeout{20000};

    explicit Stage(const std::string &path = "/dev/robot_stage");  // 실패 시 StageError(NoDevice)
    ~Stage();
    Stage(const Stage &) = delete;
    Stage &operator=(const Stage &) = delete;

    // ---- 조회
    Version ping();
    Status status();
    Limits limits();  // STM32 에 매번 묻는다 (값이 바뀔 수 있음)

    // ---- 이동 (DONE 까지 기다림)
    MoveResult home(Axis axes = Axis::Y, Millis timeout = kDefaultHomeTimeout);
    MoveResult moveTo(const Position &target, Millis timeout = kDefaultMoveTimeout);
    MoveResult moveBy(const Position &delta, Millis timeout = kDefaultMoveTimeout);

    // ---- 나눠서 쓰기 (보내고 다른 일을 하다가 waitDone)
    void startHome(Axis axes = Axis::Y);
    void startMoveTo(const Position &target);
    void startMoveBy(const Position &delta);
    MoveResult waitDone(Millis timeout);

    void stop();
    void setSpeed(uint32_t y_um_per_s, uint32_t t_mdeg_per_s);  // 0 = 현재 값 유지

    // 원점이 유지되고 있는지 STATUS 로 확인한다.
    // 이 객체로 home() 에 성공한 적이 있는데 지금 Y 원점 플래그가 없으면 false (STM32 재부팅 등)
    bool homeIntact();

private:
    void ioctlOrThrow(unsigned long req, void *arg, const char *what, uint8_t *nak = nullptr);
    void startMove(unsigned long req, const Position &p, const char *what);

    int fd_ = -1;
    std::string path_;
    bool homed_y_ = false;  // 이 객체로 Y 원점 복귀에 성공했는지
    std::chrono::steady_clock::time_point started_{};
};

}  // namespace stage

#endif  // STAGE_STAGE_HPP
