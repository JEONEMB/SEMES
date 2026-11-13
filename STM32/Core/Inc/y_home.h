#ifndef Y_HOME_H
#define Y_HOME_H
#include <stdint.h>
typedef enum { Y_HOME_IDLE, Y_HOME_RUNNING, Y_HOME_OK, Y_HOME_STOPPED,
               Y_HOME_FAULT, Y_HOME_LIMIT } YHomeState;
typedef enum {
    Y_HOME_PHASE_IDLE, Y_HOME_SEEK, Y_HOME_PRESS_STABLE,
    Y_HOME_RELEASE, Y_HOME_RELEASE_STABLE
} YHomePhase;
typedef enum {
    Y_HOME_REASON_NONE, Y_HOME_REASON_BOTH_LIMITS, Y_HOME_REASON_RIGHT_LIMIT,
    Y_HOME_REASON_TRAVEL_EXHAUSTED, Y_HOME_REASON_UNSTABLE,
    Y_HOME_REASON_TIMEOUT, Y_HOME_REASON_START_FAILED,
    Y_HOME_REASON_MOTOR_STATE, Y_HOME_REASON_STOP, Y_HOME_REASON_NOT_READY
} YHomeReason;
/* ST-LINK watch: snapshot retained at completion/failure until the next HOME.
 * No serial debug text or new protocol messages. Read while halted for consistency. */
typedef struct {
    YHomeState state;
    YHomePhase phase;
    YHomeReason reason;
    uint32_t elapsed_ms, requested_pulses, completed_pulses, frequency_hz;
    uint8_t switch_bits, dir_high, motor_state, enabled;
} YHomeDebug;
extern volatile YHomeDebug y_home_debug;
void YHome_Init(void);
int YHome_Start(void);
void YHome_Poll(void);
void YHome_Stop(void);
YHomeState YHome_State(void);
#endif
