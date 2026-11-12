#include "main.h"
#include "theta_config.h"
#include "theta_units.h"
#include "theta_home.h"
#include "theta_motor.h"
#include "theta_motor_port.h"

static ThetaHomeState state;
static ThetaHomePhase phase;
static uint8_t stop_requested, stable_value;
static uint32_t started_ms, settled_ms, stable_ms;
volatile ThetaHomeDebug theta_home_debug;

static void Snapshot(void)
{
    ThetaMotorStatus motor;
    ThetaMotor_GetStatus(&motor);
    theta_home_debug.state = state;
    theta_home_debug.phase = phase;
    theta_home_debug.elapsed_ms = (uint32_t)(HAL_GetTick() - started_ms);
    theta_home_debug.switch_bits = ThetaMotorPort_RawLimits();
    theta_home_debug.completed_pulses = theta_home_debug.requested_pulses ? motor.completed_pulses : 0U;
    theta_home_debug.motor_state = (uint8_t)motor.state;
    theta_home_debug.enabled = motor.enabled;
}

static void Finish(ThetaHomeState result, ThetaHomeReason reason)
{
    ThetaMotor_Abort();
    state = result;
    theta_home_debug.reason = reason;
    Snapshot();
}

static int StartPhase(ThetaHomePhase next)
{
    phase = next;
    uint8_t release = next == THETA_HOME_RELEASE;
    uint32_t pulses = release ? ThetaUnits_ToSteps(APP_THETA_HOME_RELEASE_MDEG) : ThetaUnits_ToSteps(APP_THETA_HOME_SEEK_MDEG);
    uint32_t hz = release ? ThetaUnits_ToHz(APP_THETA_HOME_RELEASE_MDEG_S) : ThetaUnits_ToHz(APP_THETA_HOME_SEEK_MDEG_S);
    theta_home_debug.dir_high = release ? !APP_THETA_CCW_DIR_HIGH : APP_THETA_CCW_DIR_HIGH;
    theta_home_debug.requested_pulses = pulses;
    theta_home_debug.frequency_hz = hz;
    ThetaMotorResult result = ThetaMotor_StartHoming(theta_home_debug.dir_high, pulses, hz, !release);
    if (result != THETA_MOTOR_OK) {
        Finish(THETA_HOME_FAULT, THETA_HOME_REASON_START_FAILED);
        return 0;
    }
    Snapshot();
    return 1;
}

static void BeginStable(ThetaHomePhase next, uint32_t now, uint8_t bits)
{
    phase = next;
    settled_ms = stable_ms = now;
    stable_value = bits & 1U;
}

void ThetaHome_Init(void)
{
    state = THETA_HOME_IDLE;
    phase = THETA_HOME_PHASE_IDLE;
    stop_requested = 0;
    theta_home_debug = (ThetaHomeDebug){0};
}

ThetaHomeState ThetaHome_State(void) { return state; }

int ThetaHome_Start(void)
{
    ThetaMotorStatus motor;
    ThetaMotor_GetStatus(&motor);
    if (state == THETA_HOME_RUNNING || motor.state == THETA_MOTOR_RUNNING) return 0;
    started_ms = HAL_GetTick();
    phase = THETA_HOME_PHASE_IDLE;
    theta_home_debug = (ThetaHomeDebug){0};
    if (!APP_THETA_ENABLED || !motor.ready || !THETA_MOTOR_OUTPUT_ALLOWED) {
        Finish(THETA_HOME_FAULT, THETA_HOME_REASON_NOT_READY);
        return 0;
    }
    if (APP_THETA_MANUAL_HOME) {
        /* Explicit HOME sets the stopped current position as the origin.
         * Service commits coordinates/HOMED only on successful HOME completion.
         * No STEP or switch qualification; raw input remains reportable. */
        stop_requested = 0;
        Finish(THETA_HOME_OK, THETA_HOME_REASON_NONE);
        return 1;
    }
    uint8_t bits = ThetaMotorPort_RawLimits();
    stop_requested = 0;
    state = THETA_HOME_RUNNING;
    if (bits & 1U) {
        /* Already pressed: qualify contact without emitting a seek pulse.
         * A previous LIMIT/ABORTED motor state must not block HOME recovery. */
        if (ThetaMotor_HoldHomeContact() != THETA_MOTOR_OK) {
            Finish(THETA_HOME_FAULT, THETA_HOME_REASON_START_FAILED);
            return 0;
        }
        BeginStable(THETA_HOME_PRESS_STABLE, started_ms, bits);
        Snapshot();
        return 1;
    }
    return StartPhase(THETA_HOME_SEEK);
}

void ThetaHome_Stop(void)
{
    if (state == THETA_HOME_RUNNING) {
        stop_requested = 1;
        ThetaMotor_Decelerate();
    }
}

void ThetaHome_Poll(void)
{
    if (state != THETA_HOME_RUNNING) return;
    ThetaMotor_Poll();
    uint32_t now = HAL_GetTick();
    uint8_t bits = ThetaMotorPort_RawLimits();
    ThetaMotorStatus motor;
    ThetaMotor_GetStatus(&motor);
    Snapshot();
    /* Hard overall deadline, independent of pulse budgets. No 10s phase timeout. */
    if ((uint32_t)(now - started_ms) >= APP_THETA_HOME_TIMEOUT_MS) {
        Finish(THETA_HOME_FAULT, THETA_HOME_REASON_TIMEOUT);
        return;
    }
    if (stop_requested) {
        if (motor.state != THETA_MOTOR_RUNNING)
            Finish(THETA_HOME_STOPPED, THETA_HOME_REASON_STOP);
        return;
    }
    if (phase == THETA_HOME_SEEK || phase == THETA_HOME_RELEASE) {
        if (motor.state == THETA_MOTOR_RUNNING) return;
        /* The pulse ISR checks the switch before the last-pulse budget check:
         * a valid edge on the final pulse is SWITCH, not travel exhaustion. */
        if (motor.state != THETA_MOTOR_SWITCH) {
            Finish(THETA_HOME_FAULT, motor.state == THETA_MOTOR_DONE ?
                   THETA_HOME_REASON_TRAVEL_EXHAUSTED : THETA_HOME_REASON_MOTOR_STATE);
            return;
        }
        BeginStable(phase == THETA_HOME_SEEK ? THETA_HOME_PRESS_STABLE : THETA_HOME_RELEASE_STABLE,
                    now, bits);
    }
    /* Contact qualification keeps torque without STEP pulses. Final release
     * qualifies with EN off, so success cannot conceal spring-back after disable. */
    if (motor.state == THETA_MOTOR_RUNNING ||
        (phase == THETA_HOME_RELEASE_STABLE && motor.enabled)) {
        Finish(THETA_HOME_FAULT, THETA_HOME_REASON_MOTOR_STATE);
        return;
    }
    uint8_t value = bits & 1U;
    uint8_t expected = phase == THETA_HOME_PRESS_STABLE;
    if (value != stable_value) {
        stable_value = value;
        stable_ms = now;
    }
    if ((uint32_t)(now - settled_ms) > APP_THETA_HOME_SETTLE_TIMEOUT_MS) {
        Finish(THETA_HOME_FAULT, THETA_HOME_REASON_UNSTABLE);
        return;
    }
    if (stable_value != expected || (uint32_t)(now - stable_ms) < APP_THETA_HOME_STABLE_MS) return;
    if (phase == THETA_HOME_RELEASE_STABLE) {
        /* Final released, stopped position is the origin; no second approach. */
        Finish(THETA_HOME_OK, THETA_HOME_REASON_NONE);
        return;
    }
    (void)StartPhase(THETA_HOME_RELEASE);
}
