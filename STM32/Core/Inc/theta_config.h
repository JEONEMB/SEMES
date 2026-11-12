#ifndef THETA_CONFIG_H
#define THETA_CONFIG_H
/* Commissioning profile: SW1..6 OFF OFF ON ON OFF OFF, 24V, reduction 100:1.
 * 3200 pulses/motor revolution per SM-NK17 pulse manual. User requested EN High.
 * CCW=DIR High is a commissioning assumption: verify direction BEFORE HOME.
 * Positive=CCW from above; +/-10deg relative to the manually selected origin. */
#ifndef APP_THETA_MANUAL_HOME
#define APP_THETA_MANUAL_HOME 1
#endif
#ifndef APP_THETA_ENABLED
#define APP_THETA_ENABLED 1
#endif
#ifndef APP_THETA_MOTOR_PULSES_PER_REV
#define APP_THETA_MOTOR_PULSES_PER_REV 3200
#endif
#define APP_THETA_REDUCTION 100U
#ifndef APP_THETA_EN_ACTIVE_HIGH
#define APP_THETA_EN_ACTIVE_HIGH 1
#endif
#ifndef APP_THETA_CCW_DIR_HIGH
#define APP_THETA_CCW_DIR_HIGH 1
#endif
#ifndef APP_THETA_MIN_MDEG
#define APP_THETA_MIN_MDEG (-10000)
#endif
#ifndef APP_THETA_MAX_MDEG
#define APP_THETA_MAX_MDEG 10000
#endif
#ifndef APP_THETA_HOME_SEEK_MDEG
#define APP_THETA_HOME_SEEK_MDEG 12000
#endif
#ifndef APP_THETA_HOME_RELEASE_MDEG
#define APP_THETA_HOME_RELEASE_MDEG 2000
#endif
/* NC COM -> GND, NC -> PA4: released=0, pressed/open=1. */
#define APP_THETA_DEFAULT_SPEED_MDEG_S 1000U
#define APP_THETA_HOME_SEEK_MDEG_S 1000U
#define APP_THETA_HOME_RELEASE_MDEG_S 500U
#define APP_THETA_HOME_TIMEOUT_MS 120000U
#define APP_THETA_HOME_STABLE_MS 10U
#define APP_THETA_HOME_SETTLE_TIMEOUT_MS 200U
#define THETA_MOTOR_MIN_HZ 16U
#define THETA_MOTOR_MAX_HZ 3200U
#define THETA_MOTOR_PULSE_US 5U
#define THETA_MOTOR_ENABLE_SETUP_US 2000U
#define THETA_MOTOR_STOP_ACCEL_PPS2 8000U
#if APP_THETA_ENABLED
#if APP_THETA_MOTOR_PULSES_PER_REV < 400 || APP_THETA_MOTOR_PULSES_PER_REV > 51200
#error "Confirm motor-side input pulses/revolution (not encoder CPR)"
#endif
#if (APP_THETA_EN_ACTIVE_HIGH != 0 && APP_THETA_EN_ACTIVE_HIGH != 1) || (APP_THETA_CCW_DIR_HIGH != 0 && APP_THETA_CCW_DIR_HIGH != 1)
#error "Confirm Theta EN and CCW DIR polarity"
#endif
#if APP_THETA_MIN_MDEG > 0 || APP_THETA_MAX_MDEG < 0 || APP_THETA_MIN_MDEG >= APP_THETA_MAX_MDEG || APP_THETA_MIN_MDEG < -360000 || APP_THETA_MAX_MDEG > 360000
#error "Confirm cable-safe Theta range including zero, within +/-360 degrees"
#endif
#if APP_THETA_HOME_SEEK_MDEG <= 0 || APP_THETA_HOME_SEEK_MDEG > 360000 || APP_THETA_HOME_RELEASE_MDEG <= 0 || APP_THETA_HOME_RELEASE_MDEG > 10000
#error "Confirm bounded Theta HOME search and release travel"
#endif
#endif
#endif
