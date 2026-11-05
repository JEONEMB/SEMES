// SPDX-License-Identifier: GPL-2.0
/*
 * robot_stage 프레임 조립/해석 (protocol.md 3장, 6.4절)
 *
 *  | 0xAA | 0x55 | VER | TYPE | SEQ | LEN | PAYLOAD(LEN) | CRC_L | CRC_H |
 *  CRC16 범위: VER ~ PAYLOAD
 */
#include "rs_frame.h"

int rs_frame_build(rs_u8 *buf, rs_u8 type, rs_u8 seq,
		   const void *payload, rs_u8 len)
{
	const rs_u8 *src = payload;
	rs_u16 crc;
	rs_u32 i;

	if (len > RS_MAX_PAYLOAD)
		return -1;

	buf[RS_OFF_SOF1] = RS_SOF1;
	buf[RS_OFF_SOF2] = RS_SOF2;
	buf[RS_OFF_VER]  = RS_PROTO_VER;
	buf[RS_OFF_TYPE] = type;
	buf[RS_OFF_SEQ]  = seq;
	buf[RS_OFF_LEN]  = len;
	for (i = 0; i < len; i++)
		buf[RS_OFF_PAYLOAD + i] = src[i];

	/* VER(4바이트 헤더) + PAYLOAD */
	crc = rs_crc16(&buf[RS_OFF_VER], 4 + len);
	buf[RS_OFF_PAYLOAD + len]     = crc & 0xFF;   /* 리틀 엔디언 */
	buf[RS_OFF_PAYLOAD + len + 1] = crc >> 8;

	return RS_FRAME_OVERHEAD + len;
}

void rs_parser_reset(struct rs_parser *p)
{
	p->state = RS_RX_SOF1;
	p->pos = 0;
	p->need = 0;
}

int rs_parser_feed(struct rs_parser *p, rs_u8 byte, struct rs_frame *out)
{
	rs_u16 crc_rx, crc_calc;
	rs_u8 len;
	rs_u32 i;

	switch (p->state) {
	case RS_RX_SOF1:
		if (byte == RS_SOF1) {
			p->buf[0] = byte;
			p->state = RS_RX_SOF2;
		}
		return 0;

	case RS_RX_SOF2:
		if (byte == RS_SOF2) {
			p->buf[1] = byte;
			p->pos = 2;
			p->state = RS_RX_BODY;
		} else if (byte != RS_SOF1) {
			/* 0xAA 0xAA 0x55 처럼 와도 두 번째 0xAA를 SOF1로 보고 계속 */
			p->state = RS_RX_SOF1;
		}
		return 0;

	case RS_RX_BODY:
		p->buf[p->pos++] = byte;

		if (p->pos == RS_OFF_VER + 1 && byte != RS_PROTO_VER) {
			p->err_ver++;
			rs_parser_reset(p);
			return 0;
		}
		if (p->pos == RS_HEADER_LEN) {
			if (byte > RS_MAX_PAYLOAD) {
				p->err_len++;
				rs_parser_reset(p);
				return 0;
			}
			p->need = RS_FRAME_OVERHEAD + byte;
		}
		if (p->pos < RS_HEADER_LEN || p->pos < p->need)
			return 0;

		/* 프레임 전체 수신 완료 → CRC 검사 */
		len = p->buf[RS_OFF_LEN];
		crc_calc = rs_crc16(&p->buf[RS_OFF_VER], 4 + len);
		crc_rx = p->buf[RS_OFF_PAYLOAD + len] |
			 (rs_u16)p->buf[RS_OFF_PAYLOAD + len + 1] << 8;
		if (crc_rx != crc_calc) {
			p->err_crc++;
			rs_parser_reset(p);
			return 0;
		}

		out->type = p->buf[RS_OFF_TYPE];
		out->seq  = p->buf[RS_OFF_SEQ];
		out->len  = len;
		for (i = 0; i < len; i++)
			out->payload[i] = p->buf[RS_OFF_PAYLOAD + i];

		rs_parser_reset(p);
		return 1;
	}

	return 0;
}
