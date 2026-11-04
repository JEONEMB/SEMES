#include "main.h"
#include "app_config.h"
#include "y_home.h"
#include "y_motor.h"
#include "y_motor_port.h"

enum { RELEASE_INITIAL, SEEK, RELEASE, LATCH };
static YHomeState state;
static uint8_t phase, settling, stop_requested, stable_value;
static uint32_t started_ms, settled_ms, stable_ms;

static int StartPhase(uint8_t next)
{
    phase=next; settling=0;
    uint8_t release=(next==RELEASE || next==RELEASE_INITIAL);
    uint32_t pulses=next==SEEK ? APP_Y_HOME_SEEK_PULSES : APP_Y_HOME_RELEASE_PULSES;
    uint32_t hz=release ? APP_Y_HOME_RELEASE_HZ :
        next==SEEK ? APP_Y_HOME_SEEK_HZ : APP_Y_HOME_LATCH_HZ;
    return YMotor_StartHoming(release ? 1U : 0U,pulses,hz,!release)==Y_MOTOR_OK;
}
void YHome_Init(void) { state=Y_HOME_IDLE; settling=stop_requested=0; }
YHomeState YHome_State(void) { return state; }
int YHome_Start(void)
{
    YMotorStatus motor; YMotor_GetStatus(&motor);
    if (state==Y_HOME_RUNNING || motor.state==Y_MOTOR_RUNNING ||
        !APP_Y_SWITCH_HOME || !Y_MOTOR_USE_LIMITS || !motor.ready || !Y_MOTOR_OUTPUT_ALLOWED ||
        YMotorPort_RawLimits()==3U) return 0;
    stop_requested=0; started_ms=HAL_GetTick();
    state=Y_HOME_RUNNING;
    if (!StartPhase((YMotorPort_RawLimits() & 1U) ? RELEASE_INITIAL : SEEK)) {
        state=Y_HOME_FAULT; return 0;
    }
    return 1;
}
void YHome_Stop(void)
{
    if (state==Y_HOME_RUNNING) { stop_requested=1; YMotor_Decelerate(); }
}
static void Fail(YHomeState reason) { YMotor_Abort(); state=reason; }
void YHome_Poll(void)
{
    if (state!=Y_HOME_RUNNING) return;
    YMotor_Poll();
    uint32_t now=HAL_GetTick();
    YMotorStatus motor; YMotor_GetStatus(&motor);
    if (motor.state==Y_MOTOR_LIMIT || YMotorPort_RawLimits()==3U) { Fail(Y_HOME_LIMIT); return; }
    if ((uint32_t)(now-started_ms)>=APP_Y_HOME_TIMEOUT_MS) { Fail(Y_HOME_FAULT); return; }
    if (stop_requested) {
        if (motor.state!=Y_MOTOR_RUNNING) state=Y_HOME_STOPPED;
        return;
    }
    if (motor.state==Y_MOTOR_RUNNING) return;
    /* End of travel budget without a switch edge is a homing failure. */
    if (motor.state!=Y_MOTOR_SWITCH) { Fail(Y_HOME_FAULT); return; }
    uint8_t value=YMotorPort_RawLimits() & 1U;
    if (!settling) { settling=1; settled_ms=stable_ms=now; stable_value=value; }
    uint8_t expected=(phase==SEEK || phase==LATCH);
    if (value!=stable_value) { stable_value=value; stable_ms=now; }
    if ((uint32_t)(now-settled_ms)>200U) { Fail(Y_HOME_FAULT); return; }
    if (stable_value!=expected || (uint32_t)(now-stable_ms)<APP_Y_HOME_STABLE_MS) return;
    if (phase==LATCH) { state=Y_HOME_OK; return; }
    uint8_t next=phase==RELEASE_INITIAL ? SEEK : phase==SEEK ? RELEASE : LATCH;
    if (!StartPhase(next)) Fail(Y_HOME_FAULT);
}
