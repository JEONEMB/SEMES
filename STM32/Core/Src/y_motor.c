#include "y_motor.h"
#include "y_motor_port.h"

static volatile YMotorStatus motor;
static uint32_t period_us;
static volatile uint8_t stop_requested;

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

YMotorResult YMotor_Start(uint8_t dir_high, uint32_t pulses, uint32_t frequency_hz)
{
    uint32_t key = YMotorPort_Lock();
    YMotorResult result = Y_MOTOR_OK;
    if (!motor.ready) { result = Y_MOTOR_NOT_READY; }
    else if (motor.state == Y_MOTOR_RUNNING) { result = Y_MOTOR_BUSY; }
    else if (dir_high > 1U || pulses == 0U ||
             frequency_hz < Y_MOTOR_MIN_HZ || frequency_hz > Y_MOTOR_MAX_HZ) {
        result = Y_MOTOR_BAD_ARGUMENT;
    } else if (!Y_MOTOR_OUTPUT_ALLOWED) { result = Y_MOTOR_LOCKED; }
    else if (YMotorPort_Limits()) { result = Y_MOTOR_LIMIT_ACTIVE; }
    else {
        period_us = (1000000U + frequency_hz - 1U) / frequency_hz;
        motor.requested_pulses = pulses;
        motor.completed_pulses = 0U;
        motor.frequency_hz = frequency_hz;
        motor.dir_high = dir_high;
        motor.count_exact = 1U;
        motor.state = Y_MOTOR_RUNNING;
        stop_requested = 0U;
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

void YMotor_Poll(void)
{
    uint32_t key = YMotorPort_Lock();
    if (motor.state == Y_MOTOR_RUNNING && YMotorPort_Limits()) { Abort(Y_MOTOR_LIMIT); }
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
    if (YMotorPort_Limits()) { Finish(Y_MOTOR_LIMIT); }
    else if (stop_requested) { Finish(Y_MOTOR_STOPPED); }
    else if (motor.completed_pulses == motor.requested_pulses) { Finish(Y_MOTOR_DONE); }
    else { YMotorPort_Arm(period_us); }
}
