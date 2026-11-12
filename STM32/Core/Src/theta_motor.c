#include "theta_motor.h"
#include "theta_motor_port.h"

static volatile ThetaMotorStatus motor;
static uint32_t period_us;
static volatile uint8_t stop_requested;
static uint32_t stop_hz;
static uint8_t homing_mode, homing_stop_high;

static int Guard(uint8_t dir_high)
{
    (void)dir_high;
    /* PA4 is an index/home switch, not an end-of-travel switch. Normal moves
     * may cross it; angular soft limits are enforced by the service. */
    if (homing_mode && ThetaMotorPort_RawLimits() == homing_stop_high)
        return THETA_MOTOR_SWITCH;
    return 0;
}

static uint32_t SqrtFloor(uint32_t value)
{
    uint32_t lo = 0U, hi = THETA_MOTOR_MAX_HZ;
    while (lo < hi) {
        uint32_t mid = (lo + hi + 1U) / 2U;
        if (mid * mid <= value) lo = mid;
        else hi = mid - 1U;
    }
    return lo;
}

static void Enable(uint8_t enabled)
{
    ThetaMotorPort_Enable(enabled);
    motor.enabled = enabled;
}

static void Finish(ThetaMotorState state)
{
    ThetaMotorPort_Abort();
    /* Keep torque only between HOME seek and release. Never hold on a limit,
     * STOP, error, ordinary MOVE/jog completion, or the final release edge. */
    Enable(state == THETA_MOTOR_SWITCH && homing_mode && homing_stop_high && !stop_requested);
    motor.state = state;
    stop_requested = 0U;
}

static void Abort(ThetaMotorState state)
{
    if (motor.state == THETA_MOTOR_RUNNING) {
        if (ThetaMotorPort_Completed()) { ++motor.completed_pulses; }
        else { motor.count_exact = 0U; }
        Finish(state);
    } else {
        ThetaMotorPort_Abort();
        Enable(0U);
    }
}

void ThetaMotor_Init(void)
{
    uint32_t key = ThetaMotorPort_Lock();
    /* Reinitializing active hardware is forbidden: preserve its pulse count. */
    if (motor.state != THETA_MOTOR_RUNNING) {
        motor = (ThetaMotorStatus){0};
        motor.count_exact = 1U;
        motor.ready = (uint8_t)ThetaMotorPort_Init();
        Enable(0U);
        stop_requested = 0U;
    }
    ThetaMotorPort_Unlock(key);
}

ThetaMotorResult ThetaMotor_SetEnabled(uint8_t enabled)
{
    uint32_t key = ThetaMotorPort_Lock();
    ThetaMotorResult result = THETA_MOTOR_OK;
    if (enabled > 1U) { result = THETA_MOTOR_BAD_ARGUMENT; }
    else if (motor.state == THETA_MOTOR_RUNNING) { result = THETA_MOTOR_BUSY; }
    else if (enabled && !motor.ready) { result = THETA_MOTOR_NOT_READY; }
    else if (enabled && !THETA_MOTOR_OUTPUT_ALLOWED) { result = THETA_MOTOR_LOCKED; }
    else { Enable(enabled); }
    ThetaMotorPort_Unlock(key);
    return result;
}

static ThetaMotorResult Start(uint8_t dir_high, uint32_t pulses, uint32_t frequency_hz,
                          uint8_t is_home, uint8_t stop_high)
{
    uint32_t key = ThetaMotorPort_Lock();
    ThetaMotorResult result = THETA_MOTOR_OK;
    if (!motor.ready) { result = THETA_MOTOR_NOT_READY; }
    else if (motor.state == THETA_MOTOR_RUNNING) { result = THETA_MOTOR_BUSY; }
    else if (dir_high > 1U || pulses == 0U ||
             frequency_hz < THETA_MOTOR_MIN_HZ || frequency_hz > THETA_MOTOR_MAX_HZ) {
        result = THETA_MOTOR_BAD_ARGUMENT;
    } else if (!THETA_MOTOR_OUTPUT_ALLOWED) { result = THETA_MOTOR_LOCKED; }
    else {
        homing_mode = is_home; homing_stop_high = stop_high;
        int guard = Guard(dir_high);
        if (guard == THETA_MOTOR_LIMIT) {
            ThetaMotorPort_Unlock(key); return THETA_MOTOR_LIMIT_ACTIVE;
        }
        period_us = (1000000U + frequency_hz - 1U) / frequency_hz;
        motor.requested_pulses = pulses;
        motor.completed_pulses = 0U;
        motor.frequency_hz = frequency_hz;
        motor.dir_high = dir_high;
        motor.count_exact = 1U;
        motor.state = THETA_MOTOR_RUNNING;
        stop_requested = 0U;
        if (guard == THETA_MOTOR_SWITCH) {
            Finish(THETA_MOTOR_SWITCH);
            ThetaMotorPort_Unlock(key); return THETA_MOTOR_OK;
        }
        ThetaMotorPort_Direction(dir_high);
        Enable(1U);
        /* First rising edge follows >=2ms of DIR/ENN settling. */
        uint32_t first = period_us;
        if (first < THETA_MOTOR_ENABLE_SETUP_US + THETA_MOTOR_PULSE_US) {
            first = THETA_MOTOR_ENABLE_SETUP_US + THETA_MOTOR_PULSE_US;
        }
        ThetaMotorPort_Arm(first);
    }
    ThetaMotorPort_Unlock(key);
    return result;
}

ThetaMotorResult ThetaMotor_Start(uint8_t dir_high, uint32_t pulses, uint32_t hz)
{ return Start(dir_high,pulses,hz,0U,0U); }
ThetaMotorResult ThetaMotor_StartHoming(uint8_t dir_high, uint32_t pulses, uint32_t hz, uint8_t stop_high)
{
    if (stop_high > 1U) return THETA_MOTOR_BAD_ARGUMENT;
    return Start(dir_high,pulses,hz,1U,stop_high);
}

ThetaMotorResult ThetaMotor_HoldHomeContact(void)
{
    uint32_t key = ThetaMotorPort_Lock();
    ThetaMotorResult result = THETA_MOTOR_OK;
    if (motor.state == THETA_MOTOR_RUNNING) result = THETA_MOTOR_BUSY;
    else if (!motor.ready) result = THETA_MOTOR_NOT_READY;
    else if (!THETA_MOTOR_OUTPUT_ALLOWED) result = THETA_MOTOR_LOCKED;
    else {
        ThetaMotorPort_Abort();
        homing_mode = homing_stop_high = 1U;
        stop_requested = 0U;
        motor.requested_pulses = motor.completed_pulses = motor.frequency_hz = 0U;
        motor.state = THETA_MOTOR_SWITCH;
        motor.count_exact = 1U;
        Enable(1U);
    }
    ThetaMotorPort_Unlock(key);
    return result;
}

void ThetaMotor_Stop(void)
{
    uint32_t key = ThetaMotorPort_Lock();
    if (motor.state == THETA_MOTOR_RUNNING) { stop_requested = 1U; }
    else { Enable(0U); }
    ThetaMotorPort_Unlock(key);
}

void ThetaMotor_Abort(void)
{
    uint32_t key = ThetaMotorPort_Lock();
    Abort(THETA_MOTOR_ABORTED);
    ThetaMotorPort_Unlock(key);
}

void ThetaMotor_Decelerate(void)
{
    uint32_t key = ThetaMotorPort_Lock();
    /* A retransmitted STOP must not reset the deceleration to full speed. */
    if (motor.state == THETA_MOTOR_RUNNING && stop_requested == 0U) {
        stop_hz = motor.frequency_hz;
        stop_requested = 2U;
    } else if (motor.state != THETA_MOTOR_RUNNING) Enable(0U);
    ThetaMotorPort_Unlock(key);
}

void ThetaMotor_Poll(void)
{
    uint32_t key = ThetaMotorPort_Lock();
    if (motor.state == THETA_MOTOR_RUNNING) {
        int guard = Guard(motor.dir_high);
        if (guard) Abort((ThetaMotorState)guard);
    }
    ThetaMotorPort_Unlock(key);
}

void ThetaMotor_GetStatus(ThetaMotorStatus *status)
{
    if (status == 0) { return; }
    uint32_t key = ThetaMotorPort_Lock();
    *status = motor;
    ThetaMotorPort_Unlock(key);
}

void ThetaMotor_IRQHandler(void)
{
    if (!ThetaMotorPort_Completed()) { return; }
    if (motor.state != THETA_MOTOR_RUNNING) { return; }
    ++motor.completed_pulses;
    int guard = Guard(motor.dir_high);
    if (guard) { Finish((ThetaMotorState)guard); }
    else if (stop_requested == 1U) { Finish(THETA_MOTOR_STOPPED); }
    else if (stop_requested == 2U) {
        uint32_t squared = stop_hz * stop_hz;
        if (motor.completed_pulses == motor.requested_pulses ||
            squared <= 2U * THETA_MOTOR_STOP_ACCEL_PPS2 + THETA_MOTOR_MIN_HZ * THETA_MOTOR_MIN_HZ) {
            Finish(THETA_MOTOR_STOPPED);
        } else {
            /* v_next^2 = v^2 - 2*a*one_step. Floor rounding never increases
             * the stopping distance; hardware OPM still owns each pulse. */
            stop_hz = SqrtFloor(squared - 2U * THETA_MOTOR_STOP_ACCEL_PPS2);
            ThetaMotorPort_Arm((1000000U + stop_hz - 1U) / stop_hz);
        }
    }
    else if (motor.completed_pulses == motor.requested_pulses) { Finish(THETA_MOTOR_DONE); }
    else { ThetaMotorPort_Arm(period_us); }
}
