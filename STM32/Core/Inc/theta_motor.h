#ifndef THETA_MOTOR_H
#define THETA_MOTOR_H
#include <stdint.h>

#include "theta_config.h"
#define THETA_MOTOR_OUTPUT_ALLOWED APP_THETA_ENABLED

typedef enum {
    THETA_MOTOR_OK, THETA_MOTOR_LOCKED, THETA_MOTOR_NOT_READY, THETA_MOTOR_BUSY,
    THETA_MOTOR_BAD_ARGUMENT, THETA_MOTOR_LIMIT_ACTIVE
} ThetaMotorResult;
typedef enum {
    THETA_MOTOR_IDLE, THETA_MOTOR_RUNNING, THETA_MOTOR_DONE, THETA_MOTOR_STOPPED,
    THETA_MOTOR_ABORTED, THETA_MOTOR_LIMIT, THETA_MOTOR_SWITCH
} ThetaMotorState;
typedef struct {
    uint32_t requested_pulses;
    uint32_t completed_pulses; /* complete high+low pulses in the latest burst */
    uint32_t frequency_hz;    /* requested upper bound; IRQ latency adds delay */
    ThetaMotorState state;
    uint8_t dir_high;         /* electrical DIR level; CCW mapping in theta_config.h */
    uint8_t enabled;
    uint8_t count_exact;      /* 0: an aborted partial pulse may have stepped */
    uint8_t ready;
} ThetaMotorStatus;

/* Call after MX_TIM1_Init / MX_GPIO_Init. Never starts motion. */
void ThetaMotor_Init(void);
ThetaMotorResult ThetaMotor_SetEnabled(uint8_t enabled);
ThetaMotorResult ThetaMotor_Start(uint8_t dir_high, uint32_t pulses, uint32_t frequency_hz);
ThetaMotorResult ThetaMotor_StartHoming(uint8_t dir_high, uint32_t pulses, uint32_t hz, uint8_t stop_high);
/* HOME-only stationary hold while qualifying the home contact. No STEP output. */
ThetaMotorResult ThetaMotor_HoldHomeContact(void);
void ThetaMotor_Stop(void);  /* finish the in-flight pulse, then disable EN */
void ThetaMotor_Decelerate(void); /* standard STOP: ramp down within remaining pulses */
void ThetaMotor_Abort(void); /* immediate disable; possibly incomplete final pulse */
void ThetaMotor_Poll(void);  /* raw switch guard; call independently of UART TX */
void ThetaMotor_GetStatus(ThetaMotorStatus *status);
void ThetaMotor_IRQHandler(void); /* TIM1 update only; no HAL IRQ dispatch */
#endif
