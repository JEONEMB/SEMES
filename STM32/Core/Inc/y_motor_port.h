#ifndef Y_MOTOR_PORT_H
#define Y_MOTOR_PORT_H
#include <stdint.h>
/* TIM3 adapter. Mocked in tests; all calls made with IRQs masked or in TIM3 ISR. */
uint32_t YMotorPort_Lock(void);
void YMotorPort_Unlock(uint32_t key);
int YMotorPort_Init(void);
void YMotorPort_Enable(uint8_t enabled);
void YMotorPort_Direction(uint8_t high);
void YMotorPort_Arm(uint32_t period_us);
int YMotorPort_Completed(void); /* consume one completed hardware one-shot */
void YMotorPort_Abort(void);
uint8_t YMotorPort_Limits(void); /* bit0 PA0, bit1 PA1; NC: high = open */
uint8_t YMotorPort_RawLimits(void);
#endif
