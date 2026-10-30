// SPDX-License-Identifier: GPL-2.0
/*
 * robot_stage line discipline (N_DEVELOPMENT = 29)
 *
 *  붙이기: sudo ldattach -s 115200 -8 -n -1 29 /dev/ttyACM0
 *
 *  - open:        ldattach가 ttyACM0에 붙일 때 호출. rs_dev에 tty 연결
 *  - receive_buf: STM32에서 온 바이트를 파서에 넣고, 완성된 프레임을 rs_cmd로 넘김
 *  - close:       ldattach 종료나 USB 분리 시 호출. 기다리던 명령을 모두 깨움
 */
#define pr_fmt(fmt) KBUILD_MODNAME ": " fmt

#include <linux/ktime.h>
#include <linux/module.h>
#include <linux/tty.h>
#include <linux/tty_ldisc.h>

#include "rs_dev.h"

/* 프레임 수신 도중 이 시간 동안 바이트가 없으면 받던 프레임을 버림 (6.4절) */
#define RS_RX_GAP_MS	20

static int rs_ldisc_open(struct tty_struct *tty)
{
	int ret;

	if (!tty->ops->write)
		return -EOPNOTSUPP;

	/* STM32는 하나뿐이라 두 번째 tty는 거부 */
	ret = rs_dev_attach(&rs_dev, tty);
	if (ret)
		return ret;

	tty->disc_data = &rs_dev;
	/* receive_buf에 한 번에 넘겨받을 수 있는 최대 바이트 수. 0이면 아무것도 못 받음 */
	tty->receive_room = 65536;

	pr_info("attached to %s\n", tty->name);
	return 0;
}

static void rs_ldisc_close(struct tty_struct *tty)
{
	struct rs_dev *d = tty->disc_data;

	if (!d)
		return;

	rs_dev_detach(d);
	tty->disc_data = NULL;

	pr_info("detached from %s (dropped: ver=%u len=%u crc=%u)\n",
		tty->name, d->parser.err_ver, d->parser.err_len,
		d->parser.err_crc);
}

static void rs_ldisc_receive_buf(struct tty_struct *tty, const u8 *cp,
				 const u8 *fp, size_t count)
{
	struct rs_dev *d = tty->disc_data;
	struct rs_frame frame;
	ktime_t now = ktime_get();
	size_t i;

	if (!d)
		return;

	if (d->parser.state != RS_RX_SOF1 &&
	    ktime_ms_delta(now, d->last_rx) > RS_RX_GAP_MS) {
		pr_warn("rx gap > %d ms, partial frame dropped\n", RS_RX_GAP_MS);
		rs_parser_reset(&d->parser);
	}
	d->last_rx = now;

	for (i = 0; i < count; i++) {
		/* fp: 바이트별 수신 상태. 프레이밍/패리티 오류 등이면 받던 프레임 버림 */
		if (fp && fp[i] != TTY_NORMAL) {
			rs_parser_reset(&d->parser);
			continue;
		}
		if (rs_parser_feed(&d->parser, cp[i], &frame))
			rs_dev_rx_frame(d, &frame);
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
