#ifndef APP_CONFIG_H
#define APP_CONFIG_H

/* Communication/input bring-up ONLY. Changing this flag cannot enable motion:
 * this firmware deliberately contains no motion-start implementation. */
#define APP_MOTION_LOCKED 1
#if APP_MOTION_LOCKED != 1
#error "Motion requires a separate reviewed implementation and hardware validation."
#endif

#define APP_RX_CAPACITY 512U
#define APP_TX_QUEUE_CAPACITY 16U
#define APP_RX_POLL_BUDGET 128U
#define APP_FW_MAJOR 0U
#define APP_FW_MINOR 1U
#define APP_FW_PATCH 0U
#define APP_SWITCH_STABLE_SAMPLES 10U

/* Unknown is explicit; never use these sentinels for pulse conversion.
 * Complete electrical/mechanical validation before replacing them. */
#define APP_SPEC_UNKNOWN (-1)
#define APP_Y_FULL_STEPS_PER_REV APP_SPEC_UNKNOWN
#define APP_Y_INPUT_MICROSTEPS APP_SPEC_UNKNOWN
#define APP_Y_LEAD_MM_PER_REV APP_SPEC_UNKNOWN
#define APP_Y_CURRENT_MA APP_SPEC_UNKNOWN
#define APP_Y_POSITIVE_DIR_LEVEL APP_SPEC_UNKNOWN
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
