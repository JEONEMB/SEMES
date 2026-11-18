#ifndef ROBOT_STAGE_PROTOCOL_H
#define ROBOT_STAGE_PROTOCOL_H

/* Wire definitions transcribed from robot_stage_protocol.md, version 0x01.
 * This header was not supplied by the Pi implementation; compare before sharing.
 * Encode/decode bytes explicitly: never cast an unaligned UART buffer to a struct.
 */
#ifdef __KERNEL__
#include <linux/types.h>
typedef u8 rs_u8;
typedef u16 rs_u16;
typedef u32 rs_u32;
typedef s32 rs_i32;
#else
#include <stdint.h>
typedef uint8_t rs_u8;
typedef uint16_t rs_u16;
typedef uint32_t rs_u32;
typedef int32_t rs_i32;
#endif

#define RS_VERSION 0x01U
#define RS_MAX_PAYLOAD 32U
#define RS_MAX_FRAME 40U
#define RS_RX_TIMEOUT_MS 20U

enum rs_type {
    RS_PING = 0x01, RS_HOME, RS_MOVE_REL, RS_MOVE_ABS, RS_STOP,
    RS_GET_STATUS, RS_SET_SPEED, RS_GET_CONFIG,
    RS_ACK = 0x81, RS_NAK, RS_DONE, RS_STATUS, RS_PONG, RS_CONFIG,
    RS_EVENT = 0x90
};
enum rs_reason {
    RS_UNKNOWN_TYPE = 1, RS_BAD_LEN, RS_NOT_HOMED, RS_OUT_OF_RANGE,
    RS_BUSY, RS_FAULT, RS_BAD_PARAM
};
enum rs_state { RS_STATE_IDLE, RS_STATE_MOVING, RS_STATE_HOMING, RS_STATE_FAULT };
enum rs_result { RS_OK, RS_STOPPED, RS_LIMIT, RS_RESULT_FAULT };
enum rs_event { RS_LIMIT_HIT = 1, RS_EVENT_FAULT, RS_BOOT };

/* Payload layouts only; CRC and all multibyte fields are little endian. */
#define RS_PACKED __attribute__((packed))
struct rs_home { rs_u8 axis_mask; } RS_PACKED;
struct rs_move { rs_i32 y_um, t_mdeg; } RS_PACKED;
struct rs_speed { rs_u32 y_um_per_s, t_mdeg_per_s; } RS_PACKED;
struct rs_ack { rs_u8 cmd_type; } RS_PACKED;
struct rs_nak { rs_u8 cmd_type, reason; } RS_PACKED;
struct rs_done { rs_u8 result, cmd_type; rs_u16 reserved; rs_i32 y_um, t_mdeg; } RS_PACKED;
struct rs_status {
    rs_u8 state, flags; rs_u16 reserved;
    rs_i32 y_um, t_mdeg, y_steps, t_steps;
} RS_PACKED;
struct rs_pong { rs_u8 proto_ver, fw_major, fw_minor, fw_patch; } RS_PACKED;
struct rs_config { rs_i32 y_min_um, y_max_um, t_min_mdeg, t_max_mdeg; } RS_PACKED;
struct rs_event_payload { rs_u8 event; rs_i32 y_um, t_mdeg; } RS_PACKED;
#undef RS_PACKED

static inline rs_u16 rs_crc16(const rs_u8 *data, unsigned length)
{
    rs_u16 crc = 0xFFFFU;
    for (unsigned i = 0; i < length; ++i) {
        crc ^= (rs_u16)((rs_u16)data[i] << 8);
        for (unsigned bit = 0; bit < 8U; ++bit) {
            crc = (rs_u16)((crc & 0x8000U) ? ((rs_u32)crc << 1) ^ 0x1021U
                                         : (rs_u32)crc << 1);
        }
    }
    return crc;
}

static inline rs_u16 rs_get_u16(const rs_u8 *p)
{
    return (rs_u16)((rs_u16)p[0] | ((rs_u16)p[1] << 8));
}
static inline void rs_put_u16(rs_u8 *p, rs_u16 value)
{
    p[0] = (rs_u8)value; p[1] = (rs_u8)(value >> 8);
}
static inline void rs_put_u32(rs_u8 *p, rs_u32 value)
{
    for (unsigned i = 0; i < 4U; ++i) { p[i] = (rs_u8)(value >> (i * 8U)); }
}
#endif
