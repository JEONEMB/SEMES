#include "main.h"
#include "app_config.h"
#include "y_home.h"
#include "y_motor.h"
#include "y_motor_port.h"

static YHomeState state;
static YHomePhase phase;
static uint8_t stop_requested, stable_value;
static uint32_t started_ms, settled_ms, stable_ms;
volatile YHomeDebug y_home_debug;

static void Snapshot(void)
{
    YMotorStatus motor;
    YMotor_GetStatus(&motor);
    y_home_debug.state = state;
    y_home_debug.phase = phase;
    y_home_debug.elapsed_ms = (uint32_t)(HAL_GetTick() - started_ms);
    y_home_debug.switch_bits = YMotorPort_RawLimits();
    y_home_debug.completed_pulses = y_home_debug.requested_pulses ? motor.completed_pulses : 0U;
    y_home_debug.motor_state = (uint8_t)motor.state;
    y_home_debug.enabled = motor.enabled;
}

static void Finish(YHomeState result, YHomeReason reason)
{
    YMotor_Abort();
    state = result;
    y_home_debug.reason = reason;
    Snapshot();
}

static int StartPhase(YHomePhase next)
{
    phase = next;
    uint8_t release = next == Y_HOME_RELEASE;
    uint32_t pulses = release ? APP_Y_HOME_RELEASE_PULSES : APP_Y_HOME_SEEK_PULSES;
    uint32_t hz = release ? APP_Y_HOME_RELEASE_HZ : APP_Y_HOME_SEEK_HZ;
    y_home_debug.dir_high = release ? APP_Y_POSITIVE_DIR_LEVEL : !APP_Y_POSITIVE_DIR_LEVEL;
    y_home_debug.requested_pulses = pulses;
    y_home_debug.frequency_hz = hz;
    YMotorResult result = YMotor_StartHoming(y_home_debug.dir_high, pulses, hz, !release);
    if (result != Y_MOTOR_OK) {
        Finish(Y_HOME_FAULT, Y_HOME_REASON_START_FAILED);
        return 0;
    }
    Snapshot();
    return 1;
}

static void BeginStable(YHomePhase next, uint32_t now, uint8_t bits)
{
    phase = next;
    settled_ms = stable_ms = now;
    stable_value = bits & 1U;
}

void YHome_Init(void)
{
    state = Y_HOME_IDLE;
    phase = Y_HOME_PHASE_IDLE;
    stop_requested = 0;
    y_home_debug = (YHomeDebug){0};
}

YHomeState YHome_State(void) { return state; }

int YHome_Start(void)
{
    YMotorStatus motor;
    YMotor_GetStatus(&motor);
    if (state == Y_HOME_RUNNING || motor.state == Y_MOTOR_RUNNING) return 0;
    started_ms = HAL_GetTick();
    phase = Y_HOME_PHASE_IDLE;
    y_home_debug = (YHomeDebug){0};
    if (!APP_Y_SWITCH_HOME || !Y_MOTOR_USE_LIMITS || !motor.ready || !Y_MOTOR_OUTPUT_ALLOWED) {
        Finish(Y_HOME_FAULT, Y_HOME_REASON_NOT_READY);
        return 0;
    }
    uint8_t bits = YMotorPort_RawLimits();
    if (bits == 3U) {
        Finish(Y_HOME_LIMIT, Y_HOME_REASON_BOTH_LIMITS);
        return 0;
    }
    stop_requested = 0;
    state = Y_HOME_RUNNING;
    if (bits & 1U) {
        /* Already pressed: qualify contact without emitting a seek pulse.
         * A previous LIMIT/ABORTED motor state must not block HOME recovery. */
        YMotor_Abort();
        BeginStable(Y_HOME_PRESS_STABLE, started_ms, bits);
        Snapshot();
        return 1;
    }
    return StartPhase(Y_HOME_SEEK);
}

void YHome_Stop(void)
{
    if (state == Y_HOME_RUNNING) {
        stop_requested = 1;
        YMotor_Decelerate();
    }
}

void YHome_Poll(void)
{
    if (state != Y_HOME_RUNNING) return;
    YMotor_Poll();
    uint32_t now = HAL_GetTick();
    uint8_t bits = YMotorPort_RawLimits();
    YMotorStatus motor;
    YMotor_GetStatus(&motor);
    Snapshot();
    if (bits == 3U) {
        Finish(Y_HOME_LIMIT, Y_HOME_REASON_BOTH_LIMITS);
        return;
    }
    if ((phase == Y_HOME_RELEASE || phase == Y_HOME_RELEASE_STABLE) &&
        ((bits & 2U) || motor.state == Y_MOTOR_LIMIT)) {
        Finish(Y_HOME_LIMIT, Y_HOME_REASON_RIGHT_LIMIT);
        return;
    }
    if (phase == Y_HOME_SEEK && motor.state == Y_MOTOR_LIMIT) {
        Finish(Y_HOME_LIMIT, Y_HOME_REASON_BOTH_LIMITS);
        return;
    }
    /* Hard overall deadline, independent of pulse budgets. No 10s phase timeout. */
    if ((uint32_t)(now - started_ms) >= APP_Y_HOME_TIMEOUT_MS) {
        Finish(Y_HOME_FAULT, Y_HOME_REASON_TIMEOUT);
        return;
    }
    if (stop_requested) {
        if (motor.state != Y_MOTOR_RUNNING)
            Finish(Y_HOME_STOPPED, Y_HOME_REASON_STOP);
        return;
    }
    if (phase == Y_HOME_SEEK || phase == Y_HOME_RELEASE) {
        if (motor.state == Y_MOTOR_RUNNING) return;
        /* The pulse ISR checks the switch before the last-pulse budget check:
         * a valid edge on the final pulse is SWITCH, not travel exhaustion. */
        if (motor.state != Y_MOTOR_SWITCH) {
            Finish(Y_HOME_FAULT, motor.state == Y_MOTOR_DONE ?
                   Y_HOME_REASON_TRAVEL_EXHAUSTED : Y_HOME_REASON_MOTOR_STATE);
            return;
        }
        BeginStable(phase == Y_HOME_SEEK ? Y_HOME_PRESS_STABLE : Y_HOME_RELEASE_STABLE,
                    now, bits);
    }
    /* Both stability phases require stopped pulses and disabled driver. */
    if (motor.state == Y_MOTOR_RUNNING || motor.enabled) {
        Finish(Y_HOME_FAULT, Y_HOME_REASON_MOTOR_STATE);
        return;
    }
    uint8_t value = bits & 1U;
    uint8_t expected = phase == Y_HOME_PRESS_STABLE;
    if (value != stable_value) {
        stable_value = value;
        stable_ms = now;
    }
    if ((uint32_t)(now - settled_ms) > APP_Y_HOME_SETTLE_TIMEOUT_MS) {
        Finish(Y_HOME_FAULT, Y_HOME_REASON_UNSTABLE);
        return;
    }
    if (stable_value != expected || (uint32_t)(now - stable_ms) < APP_Y_HOME_STABLE_MS) return;
    if (phase == Y_HOME_RELEASE_STABLE) {
        /* Final released, stopped position is the origin; no negative relatch. */
        Finish(Y_HOME_OK, Y_HOME_REASON_NONE);
        return;
    }
    (void)StartPhase(Y_HOME_RELEASE);
}
