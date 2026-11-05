// SPDX-License-Identifier: GPL-2.0
/*
 * robot_stage 명령 처리 (protocol.md 6장)
 *
 *  - 명령마다 새 SEQ, 100 ms 안에 응답 없으면 같은 SEQ로 최대 3회 재전송
 *  - 응답은 SEQ와 TYPE이 맞아야 인정
 *  - HOME/MOVE는 ACK 후 DONE이 따로 옴 → motion_* / done_* 로 추적
 */
#define pr_fmt(fmt) KBUILD_MODNAME ": " fmt

#include <linux/jiffies.h>
#include <linux/random.h>
#include <linux/string.h>

#include "rs_dev.h"

struct rs_dev rs_dev;

static bool rs_is_motion(u8 type)
{
	return type == RS_CMD_HOME || type == RS_CMD_MOVE_REL ||
	       type == RS_CMD_MOVE_ABS;
}

/* STM32 → Pi 메시지의 정해진 페이로드 길이, 모르는 TYPE이면 -1 */
static int rs_rsp_len(u8 type)
{
	switch (type) {
	case RS_RSP_ACK:	return sizeof(struct rs_ack);
	case RS_RSP_NAK:	return sizeof(struct rs_nak);
	case RS_RSP_DONE:	return sizeof(struct rs_done);
	case RS_RSP_STATUS:	return sizeof(struct rs_status);
	case RS_RSP_PONG:	return sizeof(struct rs_pong);
	case RS_RSP_CONFIG:	return sizeof(struct rs_config);
	case RS_EVT_EVENT:	return sizeof(struct rs_event_msg);
	}
	return -1;
}

/* f가 cmd에 대한 응답인지 (SEQ는 호출하는 쪽에서 확인) */
static bool rs_rsp_matches(u8 cmd, const struct rs_frame *f)
{
	switch (cmd) {
	case RS_CMD_PING:	return f->type == RS_RSP_PONG;
	case RS_CMD_GET_STATUS:	return f->type == RS_RSP_STATUS;
	case RS_CMD_GET_CONFIG:	return f->type == RS_RSP_CONFIG;
	}
	/* 동작 명령: ACK/NAK의 첫 바이트가 명령 TYPE */
	return (f->type == RS_RSP_ACK || f->type == RS_RSP_NAK) &&
	       f->payload[0] == cmd;
}

void rs_dev_init(struct rs_dev *d)
{
	mutex_init(&d->cmd_lock);
	mutex_init(&d->tx_lock);
	spin_lock_init(&d->lock);
	init_completion(&d->rsp_done);
	init_waitqueue_head(&d->done_wq);
}

int rs_dev_attach(struct rs_dev *d, struct tty_struct *tty)
{
	mutex_lock(&d->tx_lock);
	if (d->tty) {
		mutex_unlock(&d->tx_lock);
		return -EBUSY;
	}
	d->tty = tty;

	memset(&d->parser, 0, sizeof(d->parser));
	rs_parser_reset(&d->parser);
	d->last_rx = ktime_get();
	/*
	 * STM32는 직전 명령과 SEQ, TYPE이 같으면 재전송으로 보고 실행하지 않음.
	 * 모듈을 다시 올릴 때마다 0부터 시작하면 첫 명령이 무시될 수 있어 랜덤 시작.
	 */
	d->next_seq = get_random_u8();

	spin_lock(&d->lock);
	d->attached = true;
	d->rsp_waiting = false;
	d->motion_active = false;
	d->done_ready = false;
	spin_unlock(&d->lock);

	mutex_unlock(&d->tx_lock);
	return 0;
}

void rs_dev_detach(struct rs_dev *d)
{
	/* 송신 중이면 끝날 때까지 기다린 뒤 tty를 놓음 */
	mutex_lock(&d->tx_lock);
	d->tty = NULL;
	mutex_unlock(&d->tx_lock);

	/* 기다리던 쪽을 모두 -ENODEV로 깨움 */
	spin_lock(&d->lock);
	d->attached = false;
	if (d->rsp_waiting) {
		d->rsp_waiting = false;
		d->rsp_err = -ENODEV;
		complete(&d->rsp_done);
	}
	d->motion_active = false;
	spin_unlock(&d->lock);
	wake_up_all(&d->done_wq);
}

static int rs_send(struct rs_dev *d, u8 type, u8 seq,
		   const void *payload, u8 len)
{
	u8 buf[RS_MAX_FRAME];
	ssize_t sent;
	int n, ret = 0;

	n = rs_frame_build(buf, type, seq, payload, len);
	if (n < 0)
		return -EINVAL;

	mutex_lock(&d->tx_lock);
	if (!d->tty) {
		ret = -ENODEV;
		goto out;
	}
	/* 최대 40바이트라 보통 한 번에 다 들어감. 다 못 보내면 오류로 처리 */
	sent = d->tty->ops->write(d->tty, buf, n);
	if (sent != n) {
		pr_warn("tx type=0x%02x seq=%u: wrote %zd of %d\n",
			type, seq, sent, n);
		ret = sent < 0 ? sent : -EIO;
		goto out;
	}
	if (rs_debug)
		pr_info("tx type=0x%02x seq=%u len=%u\n", type, seq, len);
out:
	mutex_unlock(&d->tx_lock);
	return ret;
}

/* 명령 하나를 보내고 응답(ACK/NAK 또는 조회 응답)을 rsp에 받음 */
static int rs_xfer(struct rs_dev *d, u8 type, const void *payload, u8 len,
		   struct rs_frame *rsp)
{
	long left;
	int ret = -ENODEV;
	int try;
	u8 seq;

	if (mutex_lock_killable(&d->cmd_lock))
		return -EINTR;

	seq = d->next_seq++;

	for (try = 0; try <= RS_MAX_RETRY; try++) {
		spin_lock(&d->lock);
		if (!d->attached) {
			spin_unlock(&d->lock);
			ret = -ENODEV;
			break;
		}
		reinit_completion(&d->rsp_done);
		d->rsp_waiting = true;
		d->rsp_seq = seq;
		d->rsp_cmd = type;
		d->rsp_err = 0;
		spin_unlock(&d->lock);

		if (try)
			pr_warn("type=0x%02x seq=%u: no response, retry %d\n",
				type, seq, try);

		ret = rs_send(d, type, seq, payload, len);
		if (!ret) {
			left = wait_for_completion_killable_timeout(&d->rsp_done,
					msecs_to_jiffies(RS_RSP_TIMEOUT_MS));
			if (left > 0)
				ret = 0;
			else if (left == 0)
				ret = -ETIMEDOUT;
			else
				ret = -EINTR;
		}

		spin_lock(&d->lock);
		d->rsp_waiting = false;
		/* 타임아웃 직후 응답이 들어온 경우도 성공으로 처리 */
		if (ret == -ETIMEDOUT && try_wait_for_completion(&d->rsp_done))
			ret = 0;
		if (!ret) {
			ret = d->rsp_err;
			if (!ret)
				*rsp = d->rsp;
		}
		spin_unlock(&d->lock);

		if (ret != -ETIMEDOUT)
			break;
	}

	mutex_unlock(&d->cmd_lock);

	if (ret == -ETIMEDOUT)
		pr_warn("type=0x%02x seq=%u: no response after %d retries\n",
			type, seq, RS_MAX_RETRY);
	return ret;
}

int rs_cmd_query(struct rs_dev *d, u8 type, void *out, size_t size)
{
	struct rs_frame rsp;
	int ret;

	ret = rs_xfer(d, type, NULL, 0, &rsp);
	if (ret)
		return ret;
	if (rsp.len != size)
		return -EPROTO;
	memcpy(out, rsp.payload, size);
	return 0;
}

int rs_cmd_action(struct rs_dev *d, u8 type, const void *payload, u8 len,
		  u8 *nak_reason)
{
	struct rs_frame rsp;
	struct rs_nak nak;
	int ret;

	*nak_reason = 0;
	ret = rs_xfer(d, type, payload, len, &rsp);
	if (ret)
		return ret;
	if (rsp.type == RS_RSP_NAK) {
		memcpy(&nak, rsp.payload, sizeof(nak));
		*nak_reason = nak.reason;
		return -EIO;
	}
	return 0;
}

int rs_cmd_wait_done(struct rs_dev *d, unsigned int timeout_ms,
		     struct rs_done *out)
{
	long timeout = timeout_ms ? msecs_to_jiffies(timeout_ms)
				  : MAX_SCHEDULE_TIMEOUT;
	long left;
	int ret;

	spin_lock(&d->lock);
	ret = (d->done_ready || d->motion_active) ? 0 :
	      d->attached ? -ENOENT : -ENODEV;
	spin_unlock(&d->lock);
	if (ret)
		return ret;

	left = wait_event_interruptible_timeout(d->done_wq,
			READ_ONCE(d->done_ready) || !READ_ONCE(d->motion_active),
			timeout);
	if (left < 0)
		return -EINTR;

	spin_lock(&d->lock);
	if (d->done_ready) {
		*out = d->done;
		d->done_ready = false;
		ret = 0;
	} else if (!d->attached) {
		ret = -ENODEV;
	} else if (d->motion_active) {
		ret = -ETIMEDOUT;   /* 앱은 GET_STATUS로 상태 확인 (6.3절) */
	} else {
		ret = -ENOENT;      /* 다른 쪽에서 먼저 가져감 */
	}
	spin_unlock(&d->lock);
	return ret;
}

/* lock 잡은 상태에서 호출. DONE 처리 */
static void rs_rx_done(struct rs_dev *d, const struct rs_frame *f)
{
	if (d->motion_active && f->seq == d->motion_seq) {
		/* 정상: ACK 받은 이동의 DONE */
	} else if (d->rsp_waiting && f->seq == d->rsp_seq &&
		   rs_is_motion(d->rsp_cmd)) {
		/* ACK가 유실되고 DONE이 먼저 옴 → 명령은 수락된 것으로 처리 */
		d->rsp.type = RS_RSP_ACK;
		d->rsp.seq = f->seq;
		d->rsp.len = 1;
		d->rsp.payload[0] = d->rsp_cmd;
		d->rsp_waiting = false;
		complete(&d->rsp_done);
	} else {
		pr_warn("rx DONE seq=%u: no matching motion, ignored\n", f->seq);
		return;
	}

	memcpy(&d->done, f->payload, sizeof(d->done));
	d->done_ready = true;
	d->motion_active = false;
	wake_up_all(&d->done_wq);
}

/* 수신 경로(receive_buf)에서 완성된 프레임마다 호출 */
void rs_dev_rx_frame(struct rs_dev *d, const struct rs_frame *f)
{
	int want = rs_rsp_len(f->type);

	if (rs_debug)
		pr_info("rx type=0x%02x seq=%u len=%u\n", f->type, f->seq, f->len);

	if (want < 0 || f->len != want) {
		pr_warn("rx type=0x%02x len=%u: unexpected, ignored\n",
			f->type, f->len);
		return;
	}

	if (f->type == RS_EVT_EVENT) {
		/* TODO: read/poll로 앱에 전달 (다음 단계) */
		pr_info("event %u\n", f->payload[0]);
		return;
	}

	spin_lock(&d->lock);
	if (f->type == RS_RSP_DONE) {
		rs_rx_done(d, f);
	} else if (d->rsp_waiting && f->seq == d->rsp_seq &&
		   rs_rsp_matches(d->rsp_cmd, f)) {
		d->rsp = *f;
		d->rsp_waiting = false;
		/* 이동 명령이 수락됨 → 이후 같은 SEQ의 DONE을 기다림 */
		if (f->type == RS_RSP_ACK && rs_is_motion(d->rsp_cmd)) {
			d->motion_active = true;
			d->motion_seq = f->seq;
			d->done_ready = false;
		}
		complete(&d->rsp_done);
	} else if (rs_debug) {
		pr_info("rx type=0x%02x seq=%u: not waited, ignored\n",
			f->type, f->seq);
	}
	spin_unlock(&d->lock);
}
