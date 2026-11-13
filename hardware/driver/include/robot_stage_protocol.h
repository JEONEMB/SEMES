/*
 * robot_stage 통신 프로토콜 공용 정의
 *
 * Raspberry Pi(robot_stage 커널 드라이버)와 STM32(NUCLEO-F446RE 펌웨어)가
 * 함께 include하는 헤더입니다. 상세 규칙은 docs/protocol.md를 따릅니다.
 * 이 파일의 값을 바꾸면 양쪽을 모두 다시 빌드해야 합니다.
 */
#ifndef __ROBOT_STAGE_PROTOCOL_H__
#define __ROBOT_STAGE_PROTOCOL_H__

#ifdef __KERNEL__
#include <linux/types.h>
#include <linux/build_bug.h>
typedef u8  rs_u8;
typedef u16 rs_u16;
typedef s32 rs_i32;
typedef u32 rs_u32;
#define RS_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
#else
#include <stdint.h>
typedef uint8_t  rs_u8;
typedef uint16_t rs_u16;
typedef int32_t  rs_i32;
typedef uint32_t rs_u32;
#ifdef __cplusplus
#define RS_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
#else
#define RS_STATIC_ASSERT(cond, msg) _Static_assert(cond, msg)
#endif
#endif

/* ---------------------------------------------------------------------------
 * 프레임 구조
 *
 *  | SOF1 | SOF2 | VER | TYPE | SEQ | LEN | PAYLOAD(LEN) | CRC16(LE) |
 *  |  1   |  1   |  1  |  1   |  1  |  1  |    0~32      |     2     |
 *
 *  - 모든 다바이트 값은 리틀 엔디언
 *  - CRC 계산 범위: VER부터 PAYLOAD 끝까지 (SOF 제외)
 * ------------------------------------------------------------------------- */
#define RS_SOF1             0xAA
#define RS_SOF2             0x55
#define RS_PROTO_VER        0x01

#define RS_HEADER_LEN       6   /* SOF1 ~ LEN */
#define RS_CRC_LEN          2
#define RS_MAX_PAYLOAD      32
#define RS_FRAME_OVERHEAD   (RS_HEADER_LEN + RS_CRC_LEN)
#define RS_MAX_FRAME        (RS_FRAME_OVERHEAD + RS_MAX_PAYLOAD)

/* 프레임 내 오프셋 */
#define RS_OFF_SOF1         0
#define RS_OFF_SOF2         1
#define RS_OFF_VER          2
#define RS_OFF_TYPE         3
#define RS_OFF_SEQ          4
#define RS_OFF_LEN          5
#define RS_OFF_PAYLOAD      6

/* ---------------------------------------------------------------------------
 * 메시지 종류 (TYPE)
 *  0x01~0x7F: Pi -> STM32 명령
 *  0x80~0xFF: STM32 -> Pi 응답/이벤트
 * ------------------------------------------------------------------------- */
enum rs_msg_type {
	/* Pi -> STM32 */
	RS_CMD_PING       = 0x01,
	RS_CMD_HOME       = 0x02,
	RS_CMD_MOVE_REL   = 0x03,
	RS_CMD_MOVE_ABS   = 0x04,
	RS_CMD_STOP       = 0x05,
	RS_CMD_GET_STATUS = 0x06,
	RS_CMD_SET_SPEED  = 0x07,
	RS_CMD_GET_CONFIG = 0x08,

	/* STM32 -> Pi */
	RS_RSP_ACK        = 0x81,
	RS_RSP_NAK        = 0x82,
	RS_RSP_DONE       = 0x83,
	RS_RSP_STATUS     = 0x84,
	RS_RSP_PONG       = 0x85,
	RS_RSP_CONFIG     = 0x86,
	RS_EVT_EVENT      = 0x90,
};

/* NAK 사유 */
enum rs_nak_reason {
	RS_NAK_UNKNOWN_TYPE = 1,
	RS_NAK_BAD_LEN      = 2,
	RS_NAK_NOT_HOMED    = 3,
	RS_NAK_OUT_OF_RANGE = 4,
	RS_NAK_BUSY         = 5,
	RS_NAK_FAULT        = 6,
	RS_NAK_BAD_PARAM    = 7,
};

/* DONE 결과 */
enum rs_done_result {
	RS_DONE_OK      = 0,
	RS_DONE_STOPPED = 1,  /* STOP 명령으로 중단 */
	RS_DONE_LIMIT   = 2,  /* 이동 중 리밋 스위치 감지 */
	RS_DONE_FAULT   = 3,  /* 드라이버 알람 등 고장 */
};

/* 장치 상태 (STATUS.state) */
enum rs_state {
	RS_STATE_IDLE   = 0,
	RS_STATE_MOVING = 1,
	RS_STATE_HOMING = 2,
	RS_STATE_FAULT  = 3,
};

/* STATUS.flags 비트 */
#define RS_FLAG_Y_HOMED     (1u << 0)
#define RS_FLAG_T_HOMED     (1u << 1)
#define RS_FLAG_Y_LIMIT_NEG (1u << 2)  /* Y- 끝 스위치 눌림 */
#define RS_FLAG_Y_LIMIT_POS (1u << 3)  /* Y+ 끝 스위치 눌림 */
#define RS_FLAG_T_HOME_SW   (1u << 4)  /* Θ 원점 스위치 눌림 */

/* HOME.axis_mask 비트 */
#define RS_AXIS_Y           (1u << 0)
#define RS_AXIS_T           (1u << 1)

/* EVENT 종류 */
enum rs_event {
	RS_EVT_LIMIT_HIT = 1,  /* 대기 중 리밋 스위치 감지 */
	RS_EVT_FAULT     = 2,
	RS_EVT_BOOT      = 3,  /* STM32 리셋 후 최초 1회 */
};

/* ---------------------------------------------------------------------------
 * 페이로드 구조체
 *  - packed로 선언해 컴파일러가 padding을 넣지 않게 함
 *  - Pi(aarch64)와 STM32(Cortex-M4) 모두 리틀 엔디언이므로 그대로 복사 가능
 * ------------------------------------------------------------------------- */
#define RS_PACKED __attribute__((packed))

struct rs_home {            /* RS_CMD_HOME */
	rs_u8  axis_mask;
} RS_PACKED;

struct rs_move {            /* RS_CMD_MOVE_REL, RS_CMD_MOVE_ABS */
	rs_i32 y_um;            /* Y, 1 = 1µm */
	rs_i32 t_mdeg;          /* Θ, 1 = 0.001° */
} RS_PACKED;

struct rs_speed {           /* RS_CMD_SET_SPEED, 0 = 현재 값 유지 */
	rs_u32 y_um_per_s;
	rs_u32 t_mdeg_per_s;
} RS_PACKED;

struct rs_ack {             /* RS_RSP_ACK */
	rs_u8  cmd_type;
} RS_PACKED;

struct rs_nak {             /* RS_RSP_NAK */
	rs_u8  cmd_type;
	rs_u8  reason;          /* enum rs_nak_reason */
} RS_PACKED;

struct rs_done {            /* RS_RSP_DONE */
	rs_u8  result;          /* enum rs_done_result */
	rs_u8  cmd_type;        /* 완료된 명령 (HOME/MOVE_REL/MOVE_ABS) */
	rs_u16 reserved;
	rs_i32 y_um;            /* 완료 시점 위치 */
	rs_i32 t_mdeg;
} RS_PACKED;

struct rs_status {          /* RS_RSP_STATUS */
	rs_u8  state;           /* enum rs_state */
	rs_u8  flags;           /* RS_FLAG_* */
	rs_u16 reserved;
	rs_i32 y_um;
	rs_i32 t_mdeg;
	rs_i32 y_steps;         /* 디버깅용 실제 스텝 수 */
	rs_i32 t_steps;
} RS_PACKED;

struct rs_pong {            /* RS_RSP_PONG */
	rs_u8  proto_ver;
	rs_u8  fw_major;
	rs_u8  fw_minor;
	rs_u8  fw_patch;
} RS_PACKED;

struct rs_config {          /* RS_RSP_CONFIG, 리밋 값의 유일한 출처는 STM32 */
	rs_i32 y_min_um;
	rs_i32 y_max_um;
	rs_i32 t_min_mdeg;
	rs_i32 t_max_mdeg;
} RS_PACKED;

struct rs_event_msg {       /* RS_EVT_EVENT */
	rs_u8  event;           /* enum rs_event */
	rs_i32 y_um;
	rs_i32 t_mdeg;
} RS_PACKED;

RS_STATIC_ASSERT(sizeof(struct rs_home)      == 1,  "rs_home size");
RS_STATIC_ASSERT(sizeof(struct rs_move)      == 8,  "rs_move size");
RS_STATIC_ASSERT(sizeof(struct rs_speed)     == 8,  "rs_speed size");
RS_STATIC_ASSERT(sizeof(struct rs_ack)       == 1,  "rs_ack size");
RS_STATIC_ASSERT(sizeof(struct rs_nak)       == 2,  "rs_nak size");
RS_STATIC_ASSERT(sizeof(struct rs_done)      == 12, "rs_done size");
RS_STATIC_ASSERT(sizeof(struct rs_status)    == 20, "rs_status size");
RS_STATIC_ASSERT(sizeof(struct rs_pong)      == 4,  "rs_pong size");
RS_STATIC_ASSERT(sizeof(struct rs_config)    == 16, "rs_config size");
RS_STATIC_ASSERT(sizeof(struct rs_event_msg) == 9,  "rs_event_msg size");

/* ---------------------------------------------------------------------------
 * CRC-16/CCITT-FALSE
 *  poly 0x1021, init 0xFFFF, 반사 없음, xorout 0x0000
 *  검증값: "123456789" -> 0x29B1
 * ------------------------------------------------------------------------- */
#define RS_CRC_INIT 0xFFFF

static inline rs_u16 rs_crc16(const rs_u8 *data, rs_u32 len)
{
	rs_u16 crc = RS_CRC_INIT;
	rs_u32 i;
	int bit;

	for (i = 0; i < len; i++) {
		crc ^= (rs_u16)data[i] << 8;
		for (bit = 0; bit < 8; bit++)
			crc = (crc & 0x8000) ? (rs_u16)((crc << 1) ^ 0x1021)
					     : (rs_u16)(crc << 1);
	}
	return crc;
}

#endif /* __ROBOT_STAGE_PROTOCOL_H__ */
