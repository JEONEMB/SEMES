#include "y_motor.h"
#include "y_motor_port.h"

static volatile YMotorStatus motor;
static uint32_t period_us;
static volatile uint8_t stop_requested;
static uint32_t stop_hz;
static uint8_t homing_mode, homing_stop_high;

static int Guard(uint8_t dir_high)
{
    uint8_t bits = homing_mode ? YMotorPort_RawLimits() : YMotorPort_Limits();
    if (bits == 3U) return Y_MOTOR_LIMIT;
    if (homing_mode) {
        /* Seeking left may depart the right end; release goes right. */
        if (dir_high && (bits & 2U)) return Y_MOTOR_LIMIT;
        if ((bits & 1U) == homing_stop_high) return Y_MOTOR_SWITCH;
        return 0;
    }
    return bits & (dir_high ? 2U : 1U) ? Y_MOTOR_LIMIT : 0;
}

static uint32_t SqrtFloor(uint32_t value)
{
    uint32_t lo = 0U, hi = Y_MOTOR_MAX_HZ;
    while (lo < hi) {
        uint32_t mid = (lo + hi + 1U) / 2U;
        if (mid * mid <= value) lo = mid;
        else hi = mid - 1U;
    }
    return lo;
}

static void Enable(uint8_t enabled)
{
    YMotorPort_Enable(enabled);
    motor.enabled = enabled;
}

static void Finish(YMotorState state)
{
    YMotorPort_Abort();
    Enable(0U);
    motor.state = state;
    stop_requested = 0U;
}

static void Abort(YMotorState state)
{
    if (motor.state == Y_MOTOR_RUNNING) {
        if (YMotorPort_Completed()) { ++motor.completed_pulses; }
        else { motor.count_exact = 0U; }
        Finish(state);
    } else {
        YMotorPort_Abort();
        Enable(0U);
    }
}

void YMotor_Init(void)
{
    uint32_t key = YMotorPort_Lock();
    /* Reinitializing active hardware is forbidden: preserve its pulse count. */
    if (motor.state != Y_MOTOR_RUNNING) {
        motor = (YMotorStatus){0};
        motor.count_exact = 1U;
        motor.ready = (uint8_t)YMotorPort_Init();
        Enable(0U);
        stop_requested = 0U;
    }
    YMotorPort_Unlock(key);
}

YMotorResult YMotor_SetEnabled(uint8_t enabled)
{
    uint32_t key = YMotorPort_Lock();
    YMotorResult result = Y_MOTOR_OK;
    if (enabled > 1U) { result = Y_MOTOR_BAD_ARGUMENT; }
    else if (motor.state == Y_MOTOR_RUNNING) { result = Y_MOTOR_BUSY; }
    else if (enabled && !motor.ready) { result = Y_MOTOR_NOT_READY; }
    else if (enabled && !Y_MOTOR_OUTPUT_ALLOWED) { result = Y_MOTOR_LOCKED; }
    else if (enabled && YMotorPort_Limits()) { result = Y_MOTOR_LIMIT_ACTIVE; }
    else { Enable(enabled); }
    YMotorPort_Unlock(key);
    return result;
}

static YMotorResult Start(uint8_t dir_high, uint32_t pulses, uint32_t frequency_hz,
                          uint8_t is_home, uint8_t stop_high)
{
    uint32_t key = YMotorPort_Lock();
    YMotorResult result = Y_MOTOR_OK;
    if (!motor.ready) { result = Y_MOTOR_NOT_READY; }
    else if (motor.state == Y_MOTOR_RUNNING) { result = Y_MOTOR_BUSY; }
    else if (dir_high > 1U || pulses == 0U ||
             frequency_hz < Y_MOTOR_MIN_HZ || frequency_hz > Y_MOTOR_MAX_HZ) {
        result = Y_MOTOR_BAD_ARGUMENT;
    } else if (!Y_MOTOR_OUTPUT_ALLOWED) { result = Y_MOTOR_LOCKED; }
    else {
        homing_mode = is_home; homing_stop_high = stop_high;
        int guard = Guard(dir_high);
        if (guard == Y_MOTOR_LIMIT) {
            YMotorPort_Unlock(key); return Y_MOTOR_LIMIT_ACTIVE;
        }
        period_us = (1000000U + frequency_hz - 1U) / frequency_hz;
        motor.requested_pulses = pulses;
        motor.completed_pulses = 0U;
        motor.frequency_hz = frequency_hz;
        motor.dir_high = dir_high;
        motor.count_exact = 1U;
        motor.state = Y_MOTOR_RUNNING;
        stop_requested = 0U;
        if (guard == Y_MOTOR_SWITCH) {
            Finish(Y_MOTOR_SWITCH);
            YMotorPort_Unlock(key); return Y_MOTOR_OK;
        }
        YMotorPort_Direction(dir_high);
        Enable(1U);
        /* First rising edge follows >=2ms of DIR/ENN settling. */
        uint32_t first = period_us;
        if (first < Y_MOTOR_ENABLE_SETUP_US + Y_MOTOR_PULSE_US) {
            first = Y_MOTOR_ENABLE_SETUP_US + Y_MOTOR_PULSE_US;
        }
        YMotorPort_Arm(first);
    }
    YMotorPort_Unlock(key);
    return result;
}

YMotorResult YMotor_Start(uint8_t dir_high, uint32_t pulses, uint32_t hz)
{ return Start(dir_high,pulses,hz,0U,0U); }
YMotorResult YMotor_StartHoming(uint8_t dir_high, uint32_t pulses, uint32_t hz, uint8_t stop_high)
{
    if (stop_high > 1U) return Y_MOTOR_BAD_ARGUMENT;
    return Start(dir_high,pulses,hz,1U,stop_high);
}

void YMotor_Stop(void)
{
    uint32_t key = YMotorPort_Lock();
    if (motor.state == Y_MOTOR_RUNNING) { stop_requested = 1U; }
    else { Enable(0U); }
    YMotorPort_Unlock(key);
}

void YMotor_Abort(void)
{
    uint32_t key = YMotorPort_Lock();
    Abort(Y_MOTOR_ABORTED);
    YMotorPort_Unlock(key);
}

void YMotor_Decelerate(void)
{
    uint32_t key = YMotorPort_Lock();
    /* A retransmitted STOP must not reset the deceleration to full speed. */
    if (motor.state == Y_MOTOR_RUNNING && stop_requested == 0U) {
        stop_hz = motor.frequency_hz;
        stop_requested = 2U;
    } else if (motor.state != Y_MOTOR_RUNNING) Enable(0U);
    YMotorPort_Unlock(key);
}

void YMotor_Poll(void)
{
    uint32_t key = YMotorPort_Lock();
    if (motor.state == Y_MOTOR_RUNNING) {
        int guard = Guard(motor.dir_high);
        if (guard) Abort((YMotorState)guard);
    }
    YMotorPort_Unlock(key);
}

void YMotor_GetStatus(YMotorStatus *status)
{
    if (status == 0) { return; }
    uint32_t key = YMotorPort_Lock();
    *status = motor;
    YMotorPort_Unlock(key);
}

void YMotor_IRQHandler(void)
{
    if (!YMotorPort_Completed()) { return; }
    if (motor.state != Y_MOTOR_RUNNING) { return; }
    ++motor.completed_pulses;
    int guard = Guard(motor.dir_high);
    if (guard) { Finish((YMotorState)guard); }
    else if (stop_requested == 1U) { Finish(Y_MOTOR_STOPPED); }
    else if (stop_requested == 2U) {
        uint32_t squared = stop_hz * stop_hz;
        if (motor.completed_pulses == motor.requested_pulses ||
            squared <= 2U * Y_MOTOR_STOP_ACCEL_PPS2 + Y_MOTOR_MIN_HZ * Y_MOTOR_MIN_HZ) {
            Finish(Y_MOTOR_STOPPED);
        } else {
            /* v_next^2 = v^2 - 2*a*one_step. Floor rounding never increases
             * the stopping distance; hardware OPM still owns each pulse. */
            stop_hz = SqrtFloor(squared - 2U * Y_MOTOR_STOP_ACCEL_PPS2);
            YMotorPort_Arm((1000000U + stop_hz - 1U) / stop_hz);
        }
    }
    else if (motor.completed_pulses == motor.requested_pulses) { Finish(Y_MOTOR_DONE); }
    else { YMotorPort_Arm(period_us); }
}
