/* SPDX-License-Identifier: GPL-2.0 */
/*
 * robot_stage 프레임 조립/해석
 *
 * 커널 API를 쓰지 않으므로 유저 공간에서도 그대로 컴파일됩니다.
 * (tools/rs_frame_test.c 에서 protocol.md 9장 예시로 검증)
 */
#ifndef __RS_FRAME_H__
#define __RS_FRAME_H__

#include "robot_stage_protocol.h"

/* 해석이 끝난 프레임 하나 */
struct rs_frame {
	rs_u8 type;
	rs_u8 seq;
	rs_u8 len;
	rs_u8 payload[RS_MAX_PAYLOAD];
};

/* 수신 상태 */
enum rs_rx_state {
	RS_RX_SOF1,     /* 0xAA 기다리는 중 */
	RS_RX_SOF2,     /* 0x55 기다리는 중 */
	RS_RX_BODY,     /* VER ~ CRC 받는 중 */
};

/* 수신 파서. 바이트를 하나씩 넣으면 프레임이 완성될 때 알려줌 */
struct rs_parser {
	enum rs_rx_state state;
	rs_u8  buf[RS_MAX_FRAME];   /* SOF부터 CRC까지 프레임 전체 */
	rs_u32 pos;                 /* buf에 채운 바이트 수 */
	rs_u32 need;                /* 프레임 전체 길이 (LEN을 받은 뒤 결정) */

	/* 버린 프레임 수 (디버깅용) */
	rs_u32 err_ver;
	rs_u32 err_len;
	rs_u32 err_crc;
};

/*
 * 프레임 조립
 *  buf는 RS_MAX_FRAME 바이트 이상이어야 함
 *  반환: 프레임 길이(8 + len), len이 32를 넘으면 -1
 */
int rs_frame_build(rs_u8 *buf, rs_u8 type, rs_u8 seq,
		   const void *payload, rs_u8 len);

/* 파서를 SOF 탐색 상태로 되돌림 (초기화, 20 ms 수신 중단 시 사용) */
void rs_parser_reset(struct rs_parser *p);

/*
 * 수신 바이트 하나 처리
 *  반환: 1 = 프레임 완성(out에 채움), 0 = 아직 진행 중이거나 버림
 */
int rs_parser_feed(struct rs_parser *p, rs_u8 byte, struct rs_frame *out);

#endif /* __RS_FRAME_H__ */
