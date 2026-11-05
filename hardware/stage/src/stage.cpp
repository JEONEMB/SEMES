#include "stage/stage.hpp"

#include <errno.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>

#include "robot_stage_ioctl.h"
#include "robot_stage_protocol.h"

namespace stage {

namespace {

using Clock = std::chrono::steady_clock;

Nak nakFrom(uint8_t r)
{
    switch (r) {
    case 0: return Nak::None;
    case RS_NAK_UNKNOWN_TYPE: return Nak::UnknownType;
    case RS_NAK_BAD_LEN: return Nak::BadLen;
    case RS_NAK_NOT_HOMED: return Nak::NotHomed;
    case RS_NAK_OUT_OF_RANGE: return Nak::OutOfRange;
    case RS_NAK_BUSY: return Nak::Busy;
    case RS_NAK_FAULT: return Nak::Fault;
    case RS_NAK_BAD_PARAM: return Nak::BadParam;
    }
    return Nak::Unknown;
}

State stateFrom(uint8_t s)
{
    switch (s) {
    case RS_STATE_IDLE: return State::Idle;
    case RS_STATE_MOVING: return State::Moving;
    case RS_STATE_HOMING: return State::Homing;
    case RS_STATE_FAULT: return State::Fault;
    }
    return State::Unknown;
}

DoneResult resultFrom(uint8_t r)
{
    switch (r) {
    case RS_DONE_OK: return DoneResult::Ok;
    case RS_DONE_STOPPED: return DoneResult::Stopped;
    case RS_DONE_LIMIT: return DoneResult::Limit;
    case RS_DONE_FAULT: return DoneResult::Fault;
    }
    return DoneResult::Unknown;
}

int32_t roundToInt32(double v)
{
    return static_cast<int32_t>(std::lround(v));
}

}  // namespace

// ------------------------------------------------------------------ 값 타입

Position Position::fromMmDeg(double y_mm, double t_deg)
{
    return {roundToInt32(y_mm * 1000.0), roundToInt32(t_deg * 1000.0)};
}

bool Status::yHomed() const { return flags & RS_FLAG_Y_HOMED; }
bool Status::tHomed() const { return flags & RS_FLAG_T_HOMED; }
bool Status::yLimitNeg() const { return flags & RS_FLAG_Y_LIMIT_NEG; }
bool Status::yLimitPos() const { return flags & RS_FLAG_Y_LIMIT_POS; }
bool Status::tHomeSwitch() const { return flags & RS_FLAG_T_HOME_SW; }

bool Limits::contains(const Position &p) const
{
    return p.y_um >= y_min_um && p.y_um <= y_max_um && p.t_mdeg >= t_min_mdeg && p.t_mdeg <= t_max_mdeg;
}

Position Limits::clamp(const Position &p) const
{
    return {std::clamp(p.y_um, y_min_um, y_max_um), std::clamp(p.t_mdeg, t_min_mdeg, t_max_mdeg)};
}

const char *toString(State s)
{
    switch (s) {
    case State::Idle: return "IDLE";
    case State::Moving: return "MOVING";
    case State::Homing: return "HOMING";
    case State::Fault: return "FAULT";
    case State::Unknown: break;
    }
    return "?";
}

const char *toString(DoneResult r)
{
    switch (r) {
    case DoneResult::Ok: return "OK";
    case DoneResult::Stopped: return "STOPPED";
    case DoneResult::Limit: return "LIMIT";
    case DoneResult::Fault: return "FAULT";
    case DoneResult::Unknown: break;
    }
    return "?";
}

const char *toString(Nak n)
{
    switch (n) {
    case Nak::None: return "-";
    case Nak::UnknownType: return "UNKNOWN_TYPE";
    case Nak::BadLen: return "BAD_LEN";
    case Nak::NotHomed: return "NOT_HOMED";
    case Nak::OutOfRange: return "OUT_OF_RANGE";
    case Nak::Busy: return "BUSY";
    case Nak::Fault: return "FAULT";
    case Nak::BadParam: return "BAD_PARAM";
    case Nak::Unknown: break;
    }
    return "?";
}

const char *toString(StageError::Kind k)
{
    switch (k) {
    case StageError::Kind::NoDevice: return "장치 없음";
    case StageError::Kind::Nak: return "STM32 거부";
    case StageError::Kind::Timeout: return "시간 초과";
    case StageError::Kind::HomeLost: return "원점 잃음";
    case StageError::Kind::Interrupted: return "중단됨";
    case StageError::Kind::Io: return "입출력 오류";
    }
    return "?";
}

// ------------------------------------------------------------------ Stage

Stage::Stage(const std::string &path) : path_(path)
{
    fd_ = ::open(path.c_str(), O_RDWR | O_CLOEXEC);
    if (fd_ < 0) {
        const int e = errno;
        std::string why = std::strerror(e);
        if (e == ENOENT)
            why = "없음 (robot_stage.ko 를 insmod 했는지 확인)";
        else if (e == EACCES)
            why = "권한 없음 (sudo 또는 udev 규칙 필요)";
        throw StageError(StageError::Kind::NoDevice, path + " 열기 실패: " + why, Nak::None, e);
    }
}

Stage::~Stage()
{
    if (fd_ >= 0)
        ::close(fd_);
}

void Stage::ioctlOrThrow(unsigned long req, void *arg, const char *what, uint8_t *nak)
{
    if (::ioctl(fd_, req, arg) == 0)
        return;
    const int e = errno;
    const std::string w = what;
    switch (e) {
    case EIO: {
        const Nak n = nakFrom(nak ? *nak : 0);
        // 원점이 없어서 거부된 것은 앱이 따로 처리해야 하므로 종류를 나눈다
        const auto kind = n == Nak::NotHomed ? StageError::Kind::HomeLost : StageError::Kind::Nak;
        throw StageError(kind, w + ": STM32 가 거부함 (" + toString(n) + ")", n, e);
    }
    case ETIMEDOUT:
        throw StageError(StageError::Kind::Timeout, w + ": STM32 응답 없음 (재전송 후에도)", Nak::None, e);
    case ENODEV:
        throw StageError(StageError::Kind::NoDevice,
                         w + ": ttyACM0 에 line discipline 이 붙어 있지 않음 (ldattach 29, USB 연결 확인)",
                         Nak::None, e);
    case EINTR:
        throw StageError(StageError::Kind::Interrupted, w + ": 신호로 중단됨", Nak::None, e);
    default:
        throw StageError(StageError::Kind::Io, w + ": " + std::strerror(e), Nak::None, e);
    }
}

Version Stage::ping()
{
    rs_ioc_version v{};
    ioctlOrThrow(RS_IOC_PING, &v, "PING");
    return {v.proto_ver, v.fw_major, v.fw_minor, v.fw_patch};
}

Status Stage::status()
{
    rs_ioc_status s{};
    ioctlOrThrow(RS_IOC_GET_STATUS, &s, "GET_STATUS");
    Status r;
    r.state = stateFrom(s.state);
    r.flags = s.flags;
    r.pos = {s.y, s.t};
    r.y_steps = s.y_steps;
    r.t_steps = s.t_steps;
    return r;
}

Limits Stage::limits()
{
    rs_ioc_config c{};
    ioctlOrThrow(RS_IOC_GET_CONFIG, &c, "GET_CONFIG");
    return {c.y_min, c.y_max, c.t_min, c.t_max};
}

void Stage::setSpeed(uint32_t y_um_per_s, uint32_t t_mdeg_per_s)
{
    rs_ioc_speed s{};
    s.y = y_um_per_s;
    s.t = t_mdeg_per_s;
    ioctlOrThrow(RS_IOC_SET_SPEED, &s, "SET_SPEED", &s.nak_reason);
}

void Stage::stop()
{
    rs_ioc_stop s{};
    ioctlOrThrow(RS_IOC_STOP, &s, "STOP", &s.nak_reason);
}

void Stage::startHome(Axis axes)
{
    rs_ioc_home h{};
    h.axis_mask = static_cast<uint8_t>(axes);
    started_ = Clock::now();
    ioctlOrThrow(RS_IOC_HOME, &h, "HOME", &h.nak_reason);
}

void Stage::startMove(unsigned long req, const Position &p, const char *what)
{
    rs_ioc_move m{};
    m.y = p.y_um;
    m.t = p.t_mdeg;
    started_ = Clock::now();
    ioctlOrThrow(req, &m, what, &m.nak_reason);
}

void Stage::startMoveTo(const Position &target)
{
    startMove(RS_IOC_MOVE_ABS, target, "MOVE_ABS");
}

void Stage::startMoveBy(const Position &delta)
{
    startMove(RS_IOC_MOVE_REL, delta, "MOVE_REL");
}

MoveResult Stage::waitDone(Millis timeout)
{
    const auto deadline = Clock::now() + timeout;
    rs_ioc_done w{};
    for (;;) {
        const auto left = std::chrono::duration_cast<Millis>(deadline - Clock::now()).count();
        // 0 은 드라이버에서 "무한 대기"이므로 최소 1ms
        w.timeout_ms = static_cast<uint32_t>(std::max<long long>(left, 1));
        if (::ioctl(fd_, RS_IOC_WAIT_DONE, &w) == 0)
            break;
        const int e = errno;
        if (e == EINTR && Clock::now() < deadline)
            continue;  // 신호로 깨어남 → 남은 시간만큼 다시 대기
        if (e == ETIMEDOUT) {
            // DONE 이 안 옴. STM32 가 재부팅했으면 원점 플래그가 지워져 있다
            bool lost = false;
            try {
                const Status s = status();
                lost = homed_y_ && !s.yHomed();
            } catch (const StageError &) {
                // 상태 조회도 안 되면 아래의 시간 초과로 보고
            }
            if (lost) {
                homed_y_ = false;
                throw StageError(StageError::Kind::HomeLost,
                                 "WAIT_DONE: DONE 이 오지 않았고 원점이 사라짐 (STM32 재부팅 의심)", Nak::None, e);
            }
            throw StageError(StageError::Kind::Timeout,
                             "WAIT_DONE: " + std::to_string(timeout.count()) + " ms 안에 DONE 이 오지 않음 (이동 중일 수 있음)",
                             Nak::None, e);
        }
        if (e == ENOENT)
            throw StageError(StageError::Kind::Io, "WAIT_DONE: 기다릴 이동이 없음", Nak::None, e);
        if (e == ENODEV)
            throw StageError(StageError::Kind::NoDevice, "WAIT_DONE: 장치 연결이 끊김", Nak::None, e);
        throw StageError(StageError::Kind::Io, std::string("WAIT_DONE: ") + std::strerror(e), Nak::None, e);
    }

    MoveResult r;
    r.result = resultFrom(w.result);
    r.pos = {w.y, w.t};
    r.elapsed_ms = std::chrono::duration<double, std::milli>(Clock::now() - started_).count();
    if (w.cmd_type == RS_CMD_HOME && r.ok())
        homed_y_ = true;  // 축 구분은 STATUS 로 다시 확인하지 않고, HOME 성공 = Y 원점 있음으로 본다
    return r;
}

MoveResult Stage::home(Axis axes, Millis timeout)
{
    homed_y_ = false;
    startHome(axes);
    MoveResult r = waitDone(timeout);
    if (r.ok() && !(static_cast<uint8_t>(axes) & RS_AXIS_Y))
        homed_y_ = false;  // Θ 만 원점 복귀한 경우
    return r;
}

MoveResult Stage::moveTo(const Position &target, Millis timeout)
{
    startMoveTo(target);
    return waitDone(timeout);
}

MoveResult Stage::moveBy(const Position &delta, Millis timeout)
{
    startMoveBy(delta);
    return waitDone(timeout);
}

bool Stage::homeIntact()
{
    const Status s = status();
    if (homed_y_ && !s.yHomed()) {
        homed_y_ = false;
        return false;
    }
    return s.yHomed();
}

}  // namespace stage
