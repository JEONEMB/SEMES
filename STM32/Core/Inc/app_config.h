#ifndef APP_CONFIG_H
#define APP_CONFIG_H

/* NC switches: PA0 left, PA1 right. DIR High=right, Low=left.
 * HOME is command-triggered only. Software range is user-requested 500mm;
 * the physical actuator is 50mm and end switches take priority. */
#ifndef APP_Y_SWITCH_HOME
#define APP_Y_SWITCH_HOME 1
#endif
#ifndef APP_Y_MANUAL_HOME
#define APP_Y_MANUAL_HOME 0
#endif
#ifndef APP_Y_MIN_UM
#define APP_Y_MIN_UM 0
#endif
#ifndef APP_Y_MAX_UM
#define APP_Y_MAX_UM 500000
#endif
#define APP_Y_DEFAULT_SPEED_UM_S 100U
#define APP_Y_HOME_SEEK_PULSES 83200U /* 52mm bounded search, not 500mm */
#define APP_Y_HOME_RELEASE_PULSES 3200U /* 2mm maximum to release a switch */
#define APP_Y_HOME_SEEK_HZ 800U
#define APP_Y_HOME_RELEASE_HZ 160U /* 3200 pulses require about 20 seconds */
#define APP_Y_HOME_TIMEOUT_MS 120000U
#define APP_Y_HOME_STABLE_MS 10U
#define APP_Y_HOME_SETTLE_TIMEOUT_MS 200U
#if APP_Y_SWITCH_HOME && APP_Y_MANUAL_HOME
#error "Choose physical HOME or manual-origin test, not both"
#endif

#define APP_RX_CAPACITY 512U
#define APP_TX_QUEUE_CAPACITY 16U
#define APP_RX_POLL_BUDGET 128U
#define APP_FW_MAJOR 0U
#define APP_FW_MINOR 2U
#define APP_FW_PATCH 0U
#define APP_SWITCH_STABLE_SAMPLES 10U

/* Unknown is explicit; never use these sentinels for pulse conversion.
 * Complete electrical/mechanical validation before replacing them. */
#define APP_SPEC_UNKNOWN (-1)
#define APP_Y_FULL_STEPS_PER_REV APP_SPEC_UNKNOWN
#define APP_Y_INPUT_MICROSTEPS APP_SPEC_UNKNOWN
#define APP_Y_LEAD_MM_PER_REV APP_SPEC_UNKNOWN
#define APP_Y_CURRENT_MA APP_SPEC_UNKNOWN
#define APP_Y_POSITIVE_DIR_LEVEL 1U /* User confirmed: +Y = right = DIR High */
#define APP_THETA_INPUT_PULSES_PER_REV APP_SPEC_UNKNOWN
#define APP_THETA_PULSE_REFERENCE APP_SPEC_UNKNOWN /* motor or output shaft */
#define APP_THETA_GEAR_RATIO APP_SPEC_UNKNOWN
#define APP_THETA_DISABLED_LEVEL APP_SPEC_UNKNOWN
#define APP_THETA_POSITIVE_DIR_LEVEL APP_SPEC_UNKNOWN
#define APP_MIN_PULSE_US APP_SPEC_UNKNOWN
#define APP_DIR_SETUP_US APP_SPEC_UNKNOWN
#define APP_DIR_HOLD_US APP_SPEC_UNKNOWN
#define APP_Y_SOFT_MIN_MM APP_SPEC_UNKNOWN
#define APP_Y_SOFT_MAX_MM APP_SPEC_UNKNOWN
#define APP_THETA_SOFT_MIN_DEG APP_SPEC_UNKNOWN
#define APP_THETA_SOFT_MAX_DEG APP_SPEC_UNKNOWN
#define APP_MAX_SPEED APP_SPEC_UNKNOWN
#define APP_ACCELERATION APP_SPEC_UNKNOWN
#define APP_HEARTBEAT_TIMEOUT_MS APP_SPEC_UNKNOWN

#endif
