/* SPDX-License-Identifier: GPL-2.0 */
/*
 * robot_stage 드라이버 내부 공용 정의
 *
 *  rs_ldisc.c  ttyACM0 ↔ 드라이버 (바이트 송수신)
 *  rs_cmd.c    명령 송신, 응답 대기/재전송, DONE 처리
 *  rs_cdev.c   /dev/robot_stage ↔ 앱 (ioctl)
 *
 * STM32는 하나뿐이므로 상태도 전역 하나(rs_dev)만 둡니다.
 */
#ifndef __RS_DEV_H__
#define __RS_DEV_H__

#include <linux/completion.h>
#include <linux/ktime.h>
#include <linux/mutex.h>
#include <linux/spinlock.h>
#include <linux/tty.h>
#include <linux/wait.h>

#include "rs_frame.h"

/* 응답 대기 시간과 재전송 횟수 (protocol.md 6.3절) */
#define RS_RSP_TIMEOUT_MS	100
#define RS_MAX_RETRY		3

struct rs_dev {
	/* 송신 */
	struct mutex cmd_lock;          /* 명령 → 응답을 한 번에 한 건씩 */
	struct mutex tx_lock;           /* tty 포인터와 tty 쓰기 보호 */
	struct tty_struct *tty;         /* 붙어 있는 tty, 없으면 NULL */
	u8 next_seq;

	/* 수신 (receive_buf에서만 사용) */
	struct rs_parser parser;
	ktime_t last_rx;

	/* 아래는 lock으로 보호 (ioctl 경로와 수신 경로가 공유) */
	spinlock_t lock;
	bool attached;

	/* 응답 대기 중인 명령 */
	bool rsp_waiting;
	u8 rsp_seq;
	u8 rsp_cmd;
	int rsp_err;
	struct rs_frame rsp;
	struct completion rsp_done;

	/* 진행 중인 이동 (HOME/MOVE) 과 DONE 결과 */
	bool motion_active;
	u8 motion_seq;
	bool done_ready;
	struct rs_done done;
	wait_queue_head_t done_wq;
};

extern struct rs_dev rs_dev;
extern bool rs_debug;

/* rs_cmd.c */
void rs_dev_init(struct rs_dev *d);
int rs_dev_attach(struct rs_dev *d, struct tty_struct *tty);
void rs_dev_detach(struct rs_dev *d);
void rs_dev_rx_frame(struct rs_dev *d, const struct rs_frame *f);
int rs_cmd_query(struct rs_dev *d, u8 type, void *out, size_t size);
int rs_cmd_action(struct rs_dev *d, u8 type, const void *payload, u8 len,
		  u8 *nak_reason);
int rs_cmd_wait_done(struct rs_dev *d, unsigned int timeout_ms,
		     struct rs_done *out);

/* rs_ldisc.c */
int rs_ldisc_register(void);
void rs_ldisc_unregister(void);

/* rs_cdev.c */
int rs_cdev_register(void);
void rs_cdev_unregister(void);

#endif /* __RS_DEV_H__ */
