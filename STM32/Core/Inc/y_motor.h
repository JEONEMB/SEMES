#ifndef Y_MOTOR_H
#define Y_MOTOR_H
#include <stdint.h>

/* Pulse output enabled for PC jog and standard Y moves after origin setup.
 * No automatic start. Override to 0 to lock all pulse output. */
#ifndef Y_MOTOR_OUTPUT_ALLOWED
#define Y_MOTOR_OUTPUT_ALLOWED 1
#endif
/* NC end switches required in the physical HOME build. */
#ifndef Y_MOTOR_USE_LIMITS
#define Y_MOTOR_USE_LIMITS 1
#endif
#if Y_MOTOR_USE_LIMITS != 0 && Y_MOTOR_USE_LIMITS != 1
#error "Y_MOTOR_USE_LIMITS must be 0 or 1"
#endif
#if Y_MOTOR_OUTPUT_ALLOWED != 0 && Y_MOTOR_OUTPUT_ALLOWED != 1
#error "Y_MOTOR_OUTPUT_ALLOWED must be 0 or 1"
#endif
#define Y_MOTOR_MIN_HZ 16U
#define Y_MOTOR_MAX_HZ 3200U
#define Y_MOTOR_MAX_PULSES UINT32_MAX /* wire/count representation limit only */
#define Y_MOTOR_PULSE_US 5U
#define Y_MOTOR_ENABLE_SETUP_US 2000U
#define Y_MOTOR_STOP_ACCEL_PPS2 8000U /* 5mm/s^2 at 1600 pulses/mm */

typedef enum {
    Y_MOTOR_OK, Y_MOTOR_LOCKED, Y_MOTOR_NOT_READY, Y_MOTOR_BUSY,
    Y_MOTOR_BAD_ARGUMENT, Y_MOTOR_LIMIT_ACTIVE
} YMotorResult;
typedef enum {
    Y_MOTOR_IDLE, Y_MOTOR_RUNNING, Y_MOTOR_DONE, Y_MOTOR_STOPPED,
    Y_MOTOR_ABORTED, Y_MOTOR_LIMIT, Y_MOTOR_SWITCH
} YMotorState;
typedef struct {
    uint32_t requested_pulses;
    uint32_t completed_pulses; /* complete high+low pulses in the latest burst */
    uint32_t frequency_hz;    /* requested upper bound; IRQ latency adds delay */
    YMotorState state;
    uint8_t dir_high;         /* user confirmed: 1=right, 0=left */
    uint8_t enabled;
    uint8_t count_exact;      /* 0: an aborted partial pulse may have stepped */
    uint8_t ready;
} YMotorStatus;

/* Call after MX_TIM3_Init / MX_GPIO_Init. Never starts motion. */
void YMotor_Init(void);
YMotorResult YMotor_SetEnabled(uint8_t enabled);
YMotorResult YMotor_Start(uint8_t dir_high, uint32_t pulses, uint32_t frequency_hz);
YMotorResult YMotor_StartHoming(uint8_t dir_high, uint32_t pulses, uint32_t hz, uint8_t stop_high);
void YMotor_Stop(void);  /* finish the in-flight pulse, then disable EN */
void YMotor_Decelerate(void); /* standard STOP: ramp down within remaining pulses */
void YMotor_Abort(void); /* immediate disable; possibly incomplete final pulse */
void YMotor_Poll(void);  /* raw switch guard; call independently of UART TX */
void YMotor_GetStatus(YMotorStatus *status);
void YMotor_IRQHandler(void); /* TIM3 update only; no HAL IRQ dispatch */
#endif
