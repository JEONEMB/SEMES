#ifndef THETA_HOME_H
#define THETA_HOME_H
#include <stdint.h>
typedef enum { THETA_HOME_IDLE, THETA_HOME_RUNNING, THETA_HOME_OK, THETA_HOME_STOPPED,
               THETA_HOME_FAULT, THETA_HOME_LIMIT } ThetaHomeState;
typedef enum {
    THETA_HOME_PHASE_IDLE, THETA_HOME_SEEK, THETA_HOME_PRESS_STABLE,
    THETA_HOME_RELEASE, THETA_HOME_RELEASE_STABLE
} ThetaHomePhase;
typedef enum {
    THETA_HOME_REASON_NONE,
    THETA_HOME_REASON_TRAVEL_EXHAUSTED, THETA_HOME_REASON_UNSTABLE,
    THETA_HOME_REASON_TIMEOUT, THETA_HOME_REASON_START_FAILED,
    THETA_HOME_REASON_MOTOR_STATE, THETA_HOME_REASON_STOP, THETA_HOME_REASON_NOT_READY
} ThetaHomeReason;
/* ST-LINK watch: snapshot retained at completion/failure until the next HOME.
 * No serial debug text or new protocol messages. Read while halted for consistency. */
typedef struct {
    ThetaHomeState state;
    ThetaHomePhase phase;
    ThetaHomeReason reason;
    uint32_t elapsed_ms, requested_pulses, completed_pulses, frequency_hz;
    uint8_t switch_bits, dir_high, motor_state, enabled;
} ThetaHomeDebug;
extern volatile ThetaHomeDebug theta_home_debug;
void ThetaHome_Init(void);
int ThetaHome_Start(void);
void ThetaHome_Poll(void);
void ThetaHome_Stop(void);
ThetaHomeState ThetaHome_State(void);
#endif
