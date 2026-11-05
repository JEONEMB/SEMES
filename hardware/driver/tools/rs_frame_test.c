// SPDX-License-Identifier: GPL-2.0
/*
 * rs_frame.c 유저 공간 테스트
 *
 *  빌드/실행: driver 폴더에서 make test
 *  기준 값: protocol.md 9장 예시 프레임
 */
#include <stdio.h>
#include <string.h>

#include "rs_frame.h"

static int fails;

#define CHECK(cond, ...)						\
	do {								\
		if (!(cond)) {						\
			printf("  FAIL %s:%d: ", __FILE__, __LINE__);	\
			printf(__VA_ARGS__);				\
			printf("\n");					\
			fails++;					\
		}							\
	} while (0)

/* protocol.md 9장 */
struct example {
	const char *name;
	rs_u8 type, seq, len;
	rs_u8 bytes[RS_MAX_FRAME];
	int size;
};

static const struct example examples[] = {
	{ "PING", RS_CMD_PING, 1, 0,
	  { 0xAA, 0x55, 0x01, 0x01, 0x01, 0x00, 0x75, 0xF6 }, 8 },
	{ "PONG", RS_RSP_PONG, 1, 4,
	  { 0xAA, 0x55, 0x01, 0x85, 0x01, 0x04, 0x01, 0x00, 0x01, 0x00,
	    0x90, 0x2D }, 12 },
	{ "HOME", RS_CMD_HOME, 2, 1,
	  { 0xAA, 0x55, 0x01, 0x02, 0x02, 0x01, 0x03, 0x07, 0x3B }, 9 },
	{ "MOVE_REL", RS_CMD_MOVE_REL, 12, 8,
	  { 0xAA, 0x55, 0x01, 0x03, 0x0C, 0x08, 0x1E, 0xFB, 0xFF, 0xFF,
	    0x5E, 0x01, 0x00, 0x00, 0xA4, 0x18 }, 16 },
	{ "ACK", RS_RSP_ACK, 12, 1,
	  { 0xAA, 0x55, 0x01, 0x81, 0x0C, 0x01, 0x03, 0xE2, 0x66 }, 9 },
	{ "NAK", RS_RSP_NAK, 12, 2,
	  { 0xAA, 0x55, 0x01, 0x82, 0x0C, 0x02, 0x03, 0x03, 0x81, 0x69 }, 10 },
	{ "DONE", RS_RSP_DONE, 12, 12,
	  { 0xAA, 0x55, 0x01, 0x83, 0x0C, 0x0C, 0x00, 0x03, 0x00, 0x00,
	    0xC6, 0x5C, 0x00, 0x00, 0x46, 0x05, 0x00, 0x00, 0x49, 0x84 }, 20 },
};

#define N_EXAMPLES (int)(sizeof(examples) / sizeof(examples[0]))

/* 바이트 배열을 파서에 넣고 완성된 프레임 수를 반환 */
static int feed_all(struct rs_parser *p, const rs_u8 *data, int n,
		    struct rs_frame *frames, int max)
{
	int i, got = 0;
	struct rs_frame f;

	for (i = 0; i < n; i++) {
		if (rs_parser_feed(p, data[i], &f) && got < max)
			frames[got++] = f;
	}
	return got;
}

static void test_crc(void)
{
	const char *s = "123456789";

	printf("[crc]\n");
	CHECK(rs_crc16((const rs_u8 *)s, 9) == 0x29B1,
	      "\"123456789\" -> 0x%04X", rs_crc16((const rs_u8 *)s, 9));
}

static void test_build(void)
{
	rs_u8 buf[RS_MAX_FRAME];
	int i, n;

	printf("[build]\n");
	for (i = 0; i < N_EXAMPLES; i++) {
		const struct example *e = &examples[i];

		n = rs_frame_build(buf, e->type, e->seq,
				   &e->bytes[RS_OFF_PAYLOAD], e->len);
		CHECK(n == e->size, "%s: size %d != %d", e->name, n, e->size);
		CHECK(memcmp(buf, e->bytes, e->size) == 0,
		      "%s: bytes differ", e->name);
	}

	n = rs_frame_build(buf, RS_CMD_PING, 0, NULL, RS_MAX_PAYLOAD + 1);
	CHECK(n == -1, "len 33 must fail, got %d", n);
}

static void test_parse_examples(void)
{
	struct rs_parser p = { 0 };
	struct rs_frame f[1];
	int i;

	printf("[parse examples]\n");
	rs_parser_reset(&p);
	for (i = 0; i < N_EXAMPLES; i++) {
		const struct example *e = &examples[i];

		CHECK(feed_all(&p, e->bytes, e->size, f, 1) == 1,
		      "%s: not parsed", e->name);
		CHECK(f[0].type == e->type && f[0].seq == e->seq &&
		      f[0].len == e->len, "%s: header mismatch", e->name);
		CHECK(memcmp(f[0].payload, &e->bytes[RS_OFF_PAYLOAD],
			     e->len) == 0, "%s: payload mismatch", e->name);
	}
}

static void test_parse_errors(void)
{
	const struct example *ping = &examples[0];
	const struct example *done = &examples[6];
	struct rs_parser p = { 0 };
	struct rs_frame f[4];
	rs_u8 buf[128];
	int n;

	printf("[parse errors / resync]\n");

	/* 앞에 쓰레기 바이트 + 0xAA 0xAA 0x55 */
	rs_parser_reset(&p);
	n = 0;
	buf[n++] = 0x00; buf[n++] = 0x55; buf[n++] = 0xAA;
	memcpy(&buf[n], ping->bytes, ping->size); n += ping->size;
	CHECK(feed_all(&p, buf, n, f, 4) == 1 && f[0].type == RS_CMD_PING,
	      "garbage prefix");

	/* 연속된 두 프레임 */
	rs_parser_reset(&p);
	memcpy(buf, done->bytes, done->size);
	memcpy(&buf[done->size], ping->bytes, ping->size);
	CHECK(feed_all(&p, buf, done->size + ping->size, f, 4) == 2 &&
	      f[0].type == RS_RSP_DONE && f[1].type == RS_CMD_PING,
	      "back-to-back frames");

	/* CRC 오류 → 버리고 다음 프레임은 정상 수신 */
	rs_parser_reset(&p);
	memcpy(buf, ping->bytes, ping->size);
	buf[ping->size - 1] ^= 0xFF;
	memcpy(&buf[ping->size], ping->bytes, ping->size);
	CHECK(feed_all(&p, buf, ping->size * 2, f, 4) == 1, "bad crc");
	CHECK(p.err_crc == 1, "err_crc %u", p.err_crc);

	/* VER 오류 */
	rs_parser_reset(&p);
	memcpy(buf, ping->bytes, ping->size);
	buf[RS_OFF_VER] = 0x02;
	CHECK(feed_all(&p, buf, ping->size, f, 4) == 0, "bad ver");
	CHECK(p.err_ver == 1, "err_ver %u", p.err_ver);

	/* LEN 33 */
	rs_parser_reset(&p);
	memcpy(buf, ping->bytes, ping->size);
	buf[RS_OFF_LEN] = RS_MAX_PAYLOAD + 1;
	CHECK(feed_all(&p, buf, ping->size, f, 4) == 0, "bad len");
	CHECK(p.err_len == 1, "err_len %u", p.err_len);

	/* 수신 도중 끊김(20 ms) → reset 후 새 프레임 정상 수신 */
	rs_parser_reset(&p);
	CHECK(feed_all(&p, done->bytes, 10, f, 4) == 0, "partial");
	rs_parser_reset(&p);
	CHECK(feed_all(&p, ping->bytes, ping->size, f, 4) == 1,
	      "after timeout reset");
}

int main(void)
{
	test_crc();
	test_build();
	test_parse_examples();
	test_parse_errors();

	if (fails) {
		printf("\n%d FAILED\n", fails);
		return 1;
	}
	printf("\nALL PASSED\n");
	return 0;
}
