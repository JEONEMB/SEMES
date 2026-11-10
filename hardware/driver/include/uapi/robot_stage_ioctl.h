/* SPDX-License-Identifier: GPL-2.0 WITH Linux-syscall-note */
/*
 * robot_stage 앱 인터페이스 (/dev/robot_stage ioctl)
 *
 * 비전 앱과 드라이버가 함께 include하는 헤더입니다.
 *  - 위치/속도 단위는 protocol.md 5장을 따릅니다.
 *  - NAK 사유, DONE 결과, 상태 값의 의미는 robot_stage_protocol.h의 enum을 참고합니다.
 *
 * 반환값 (ioctl)
 *   0           성공
 *   -EIO        STM32가 NAK으로 거부 (사유는 구조체의 nak_reason)
 *   -ETIMEDOUT  재전송까지 응답 없음 / WAIT_DONE 시간 초과
 *   -ENODEV     ttyACM0에 line discipline이 붙어 있지 않음
 *   -ENOENT     WAIT_DONE: 기다릴 이동이 없음
 */
#ifndef _UAPI_ROBOT_STAGE_IOCTL_H
#define _UAPI_ROBOT_STAGE_IOCTL_H

#include <linux/ioctl.h>
#include <linux/types.h>

#define RS_DEV_PATH "/dev/robot_stage"

/* PING 응답 */
struct rs_ioc_version {
	__u8  proto_ver;
	__u8  fw_major;
	__u8  fw_minor;
	__u8  fw_patch;
};

/* HOME. in: axis_mask (bit0 Y, bit1 Θ), out: nak_reason */
struct rs_ioc_home {
	__u8  axis_mask;
	__u8  nak_reason;
};

/* MOVE_REL / MOVE_ABS. in: y, t, out: nak_reason */
struct rs_ioc_move {
	__s32 y;
	__s32 t;
	__u8  nak_reason;
	__u8  pad[3];
};

/* SET_SPEED. in: y, t (0 = 현재 값 유지), out: nak_reason */
struct rs_ioc_speed {
	__u32 y;
	__u32 t;
	__u8  nak_reason;
	__u8  pad[3];
};

/* STOP. out: nak_reason */
struct rs_ioc_stop {
	__u8  nak_reason;
};

/* GET_STATUS 응답 */
struct rs_ioc_status {
	__u8  state;
	__u8  flags;
	__u16 pad;
	__s32 y;
	__s32 t;
	__s32 y_steps;
	__s32 t_steps;
};

/* GET_CONFIG 응답 */
struct rs_ioc_config {
	__s32 y_min;
	__s32 y_max;
	__s32 t_min;
	__s32 t_max;
};

/* WAIT_DONE. in: timeout_ms (0 = 무한 대기), out: 나머지 */
struct rs_ioc_done {
	__u32 timeout_ms;
	__u8  result;
	__u8  cmd_type;
	__u16 pad;
	__s32 y;
	__s32 t;
};

#define RS_IOC_MAGIC		'R'

#define RS_IOC_PING		_IOR(RS_IOC_MAGIC, 1, struct rs_ioc_version)
#define RS_IOC_HOME		_IOWR(RS_IOC_MAGIC, 2, struct rs_ioc_home)
#define RS_IOC_MOVE_REL		_IOWR(RS_IOC_MAGIC, 3, struct rs_ioc_move)
#define RS_IOC_MOVE_ABS		_IOWR(RS_IOC_MAGIC, 4, struct rs_ioc_move)
#define RS_IOC_STOP		_IOR(RS_IOC_MAGIC, 5, struct rs_ioc_stop)
#define RS_IOC_GET_STATUS	_IOR(RS_IOC_MAGIC, 6, struct rs_ioc_status)
#define RS_IOC_SET_SPEED	_IOWR(RS_IOC_MAGIC, 7, struct rs_ioc_speed)
#define RS_IOC_GET_CONFIG	_IOR(RS_IOC_MAGIC, 8, struct rs_ioc_config)
#define RS_IOC_WAIT_DONE	_IOWR(RS_IOC_MAGIC, 9, struct rs_ioc_done)

#endif /* _UAPI_ROBOT_STAGE_IOCTL_H */
