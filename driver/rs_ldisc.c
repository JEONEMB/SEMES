// SPDX-License-Identifier: GPL-2.0
/*
 * robot_stage line discipline (N_DEVELOPMENT = 29)
 *
 *  - open:        ldattach가 ttyACM0에 붙일 때 호출. 상태 할당 후 테스트용 PING 송신
 *  - receive_buf: STM32에서 온 바이트를 파서에 넣고, 완성된 프레임을 로그로 출력
 *  - close:       ldattach 종료나 USB 분리 시 호출. 상태 해제
 *
 * 지금 단계에서는 송수신 확인용으로 로그만 남깁니다.
 * 명령/응답 매칭과 재전송은 다음 단계(rs_cmd.c)에서 붙입니다.
 */
#define pr_fmt(fmt) KBUILD_MODNAME ": " fmt

#include <linux/ktime.h>
#include <linux/module.h>
#include <linux/printk.h>
#include <linux/slab.h>
#include <linux/tty.h>
#include <linux/tty_ldisc.h>
#include <linux/workqueue.h>

#include "rs_frame.h"
#include "rs_ldisc.h"

/* 프레임 수신 도중 이 시간 동안 바이트가 없으면 받던 프레임을 버림 (6.4절) */
#define RS_RX_GAP_MS	20

/* tty 하나에 붙는 상태 */
struct rs_port {
	struct tty_struct *tty;
	struct rs_parser parser;
	ktime_t last_rx;               /* 마지막 바이트 수신 시각 */
	struct work_struct ping_work;  /* 테스트용: 연결 직후 PING 송신 */
};

static int rs_send(struct rs_port *port, u8 type, u8 seq,
		   const void *payload, u8 len)
{
	u8 buf[RS_MAX_FRAME];
	ssize_t sent;
	int n;

	n = rs_frame_build(buf, type, seq, payload, len);
	if (n < 0)
		return -EINVAL;

	/* 최대 40바이트라 보통 한 번에 다 들어감. 다 못 보내면 오류로 처리 */
	sent = port->tty->ops->write(port->tty, buf, n);
	if (sent != n) {
		pr_warn("tx type=0x%02x seq=%u: wrote %zd of %d\n",
			type, seq, sent, n);
		return sent < 0 ? sent : -EIO;
	}

	pr_info("tx type=0x%02x seq=%u len=%u\n", type, seq, len);
	return 0;
}

static void rs_ping_work(struct work_struct *work)
{
	struct rs_port *port = container_of(work, struct rs_port, ping_work);

	rs_send(port, RS_CMD_PING, 1, NULL, 0);
}

static void rs_handle_frame(struct rs_port *port, const struct rs_frame *f)
{
	pr_info("rx type=0x%02x seq=%u len=%u\n", f->type, f->seq, f->len);
	if (f->len)
		print_hex_dump(KERN_INFO, KBUILD_MODNAME ":   ", DUMP_PREFIX_NONE,
			       16, 1, f->payload, f->len, false);
}

static int rs_ldisc_open(struct tty_struct *tty)
{
	struct rs_port *port;

	if (!tty->ops->write)
		return -EOPNOTSUPP;

	port = kzalloc(sizeof(*port), GFP_KERNEL);
	if (!port)
		return -ENOMEM;

	port->tty = tty;
	rs_parser_reset(&port->parser);
	port->last_rx = ktime_get();
	INIT_WORK(&port->ping_work, rs_ping_work);

	tty->disc_data = port;
	/* receive_buf에 한 번에 넘겨받을 수 있는 최대 바이트 수. 0이면 아무것도 못 받음 */
	tty->receive_room = 65536;

	pr_info("attached to %s\n", tty->name);

	/* open은 tty 잠금을 잡은 채 불리므로 송신은 workqueue로 미룸 */
	schedule_work(&port->ping_work);
	return 0;
}

static void rs_ldisc_close(struct tty_struct *tty)
{
	struct rs_port *port = tty->disc_data;

	if (!port)
		return;

	/* 송신 중인 work가 끝날 때까지 기다린 뒤 해제 */
	cancel_work_sync(&port->ping_work);
	tty->disc_data = NULL;

	pr_info("detached from %s (dropped: ver=%u len=%u crc=%u)\n",
		tty->name, port->parser.err_ver, port->parser.err_len,
		port->parser.err_crc);
	kfree(port);
}

static void rs_ldisc_receive_buf(struct tty_struct *tty, const u8 *cp,
				 const u8 *fp, size_t count)
{
	struct rs_port *port = tty->disc_data;
	struct rs_frame frame;
	ktime_t now = ktime_get();
	size_t i;

	if (!port)
		return;

	if (port->parser.state != RS_RX_SOF1 &&
	    ktime_ms_delta(now, port->last_rx) > RS_RX_GAP_MS) {
		pr_warn("rx gap > %d ms, partial frame dropped\n", RS_RX_GAP_MS);
		rs_parser_reset(&port->parser);
	}
	port->last_rx = now;

	for (i = 0; i < count; i++) {
		/* fp: 바이트별 수신 상태. 프레이밍/패리티 오류 등이면 받던 프레임 버림 */
		if (fp && fp[i] != TTY_NORMAL) {
			rs_parser_reset(&port->parser);
			continue;
		}
		if (rs_parser_feed(&port->parser, cp[i], &frame))
			rs_handle_frame(port, &frame);
	}
}

static struct tty_ldisc_ops rs_ldisc_ops = {
	.owner       = THIS_MODULE,     /* 붙어 있는 동안 rmmod 거부 */
	.num         = N_DEVELOPMENT,
	.name        = "robot_stage",
	.open        = rs_ldisc_open,
	.close       = rs_ldisc_close,
	.receive_buf = rs_ldisc_receive_buf,
};

int rs_ldisc_register(void)
{
	return tty_register_ldisc(&rs_ldisc_ops);
}

void rs_ldisc_unregister(void)
{
	tty_unregister_ldisc(&rs_ldisc_ops);
}
