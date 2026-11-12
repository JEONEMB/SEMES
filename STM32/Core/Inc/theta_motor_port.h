#ifndef THETA_MOTOR_PORT_H
#define THETA_MOTOR_PORT_H
#include <stdint.h>
/* TIM1 adapter. Mocked in tests; all calls made with IRQs masked or in TIM1 ISR. */
uint32_t ThetaMotorPort_Lock(void);
void ThetaMotorPort_Unlock(uint32_t key);
int ThetaMotorPort_Init(void);
void ThetaMotorPort_Enable(uint8_t enabled);
void ThetaMotorPort_Direction(uint8_t high);
void ThetaMotorPort_Arm(uint32_t period_us);
int ThetaMotorPort_Completed(void); /* consume one completed hardware one-shot */
void ThetaMotorPort_Abort(void);
uint8_t ThetaMotorPort_Limits(void); /* no end limits on rotary axis */
uint8_t ThetaMotorPort_RawLimits(void);
#endif
