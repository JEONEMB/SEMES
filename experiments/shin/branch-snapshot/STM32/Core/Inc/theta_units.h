#ifndef THETA_UNITS_H
#define THETA_UNITS_H
#include <stdint.h>
#include "theta_config.h"
/* Positive angle = CCW viewed from above. Keep target angle separate from
 * rounded pulse position to avoid cumulative rounding on relative moves. */
static inline int32_t ThetaUnits_ToSteps(int64_t mdeg) {
    int64_t n=mdeg * APP_THETA_MOTOR_PULSES_PER_REV * APP_THETA_REDUCTION;
    return (int32_t)(n>=0 ? (n+180000)/360000 : -((-n+180000)/360000));
}
static inline int32_t ThetaUnits_FromSteps(int32_t steps) {
#if APP_THETA_MOTOR_PULSES_PER_REV > 0
    int64_t n=(int64_t)steps*360000;
    int32_t d=APP_THETA_MOTOR_PULSES_PER_REV*APP_THETA_REDUCTION;
    return (int32_t)(n>=0 ? (n+d/2)/d : -((-n+d/2)/d));
#else
    (void)steps; return 0;
#endif
}
static inline uint32_t ThetaUnits_ToHz(uint32_t mdeg_s) {
    return (uint32_t)(((uint64_t)mdeg_s*APP_THETA_MOTOR_PULSES_PER_REV*APP_THETA_REDUCTION+180000)/360000);
}
static inline int ThetaUnits_ValidSpeed(uint32_t speed) {
    uint64_t hz=((uint64_t)speed*APP_THETA_MOTOR_PULSES_PER_REV*APP_THETA_REDUCTION+180000)/360000;
    return hz>=THETA_MOTOR_MIN_HZ && hz<=THETA_MOTOR_MAX_HZ;
}
#endif
