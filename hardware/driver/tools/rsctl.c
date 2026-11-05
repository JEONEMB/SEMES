// SPDX-License-Identifier: GPL-2.0
/*
 * rsctl: /dev/robot_stage 테스트 도구 (비전 앱 대신 ioctl 호출)
 *
 *  빌드: driver 폴더에서 make tools
 *  사용: sudo ./tools/rsctl <명령> [인자]   (인자 없이 실행하면 도움말)
 */
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include "robot_stage_protocol.h"
#include "robot_stage_ioctl.h"

/* HOME/MOVE 후 DONE 기다리는 최대 시간 */
#define DEFAULT_WAIT_MS 60000

static const char *nak_name(int r)
{
	switch (r) {
	case RS_NAK_UNKNOWN_TYPE:	return "UNKNOWN_TYPE";
	case RS_NAK_BAD_LEN:		return "BAD_LEN";
	case RS_NAK_NOT_HOMED:		return "NOT_HOMED";
	case RS_NAK_OUT_OF_RANGE:	return "OUT_OF_RANGE";
	case RS_NAK_BUSY:		return "BUSY";
	case RS_NAK_FAULT:		return "FAULT";
	case RS_NAK_BAD_PARAM:		return "BAD_PARAM";
	}
	return "?";
}

static const char *result_name(int r)
{
	switch (r) {
	case RS_DONE_OK:	return "OK";
	case RS_DONE_STOPPED:	return "STOPPED";
	case RS_DONE_LIMIT:	return "LIMIT";
	case RS_DONE_FAULT:	return "FAULT";
	}
	return "?";
}

static const char *state_name(int s)
{
	switch (s) {
	case RS_STATE_IDLE:	return "IDLE";
	case RS_STATE_MOVING:	return "MOVING";
	case RS_STATE_HOMING:	return "HOMING";
	case RS_STATE_FAULT:	return "FAULT";
	}
	return "?";
}

static void usage(void)
{
	fprintf(stderr,
		"usage: rsctl <command> [args]\n"
		"  ping                  펌웨어 버전 확인\n"
		"  status                현재 상태/위치\n"
		"  config                허용 범위\n"
		"  home <mask>           원점 복귀 (1=Y, 2=Θ, 3=둘 다), DONE까지 대기\n"
		"  move-rel <y> <t>      상대 이동, DONE까지 대기\n"
		"  move-abs <y> <t>      절대 이동, DONE까지 대기\n"
		"  stop                  정지\n"
		"  speed <y> <t>         속도 설정 (0 = 유지)\n"
		"  wait [timeout_ms]     진행 중인 이동의 DONE 대기\n"
		"단위: Y 1 = 1 µm, Θ 1 = 0.001° (protocol.md 5장)\n");
	exit(2);
}

static long num(const char *s)
{
	char *end;
	long v;

	errno = 0;
	v = strtol(s, &end, 0);
	if (errno || *s == '\0' || *end != '\0') {
		fprintf(stderr, "invalid number: %s\n", s);
		exit(2);
	}
	return v;
}

/* ioctl 실패 메시지. NAK이면 사유까지 출력 */
static int fail(const char *what, int nak_reason)
{
	if (errno == EIO && nak_reason)
		fprintf(stderr, "%s: NAK %s (%d)\n", what,
			nak_name(nak_reason), nak_reason);
	else
		fprintf(stderr, "%s: %s\n", what, strerror(errno));
	return 1;
}

static int wait_done(int fd, unsigned int timeout_ms)
{
	struct rs_ioc_done w = { .timeout_ms = timeout_ms };

	if (ioctl(fd, RS_IOC_WAIT_DONE, &w) < 0)
		return fail("wait", 0);
	printf("DONE %s (cmd 0x%02x)  y=%d t=%d\n",
	       result_name(w.result), w.cmd_type, w.y, w.t);
	return w.result == RS_DONE_OK ? 0 : 1;
}

int main(int argc, char **argv)
{
	const char *cmd;
	int fd;

	if (argc < 2)
		usage();
	cmd = argv[1];

	fd = open(RS_DEV_PATH, O_RDWR);
	if (fd < 0) {
		perror(RS_DEV_PATH);
		return 1;
	}

	if (!strcmp(cmd, "ping") && argc == 2) {
		struct rs_ioc_version v;

		if (ioctl(fd, RS_IOC_PING, &v) < 0)
			return fail("ping", 0);
		printf("PONG proto %u, fw %u.%u.%u\n",
		       v.proto_ver, v.fw_major, v.fw_minor, v.fw_patch);

	} else if (!strcmp(cmd, "status") && argc == 2) {
		struct rs_ioc_status s;

		if (ioctl(fd, RS_IOC_GET_STATUS, &s) < 0)
			return fail("status", 0);
		printf("state  %s\n", state_name(s.state));
		printf("flags  0x%02x [%s%s%s%s%s ]\n", s.flags,
		       s.flags & RS_FLAG_Y_HOMED ? " Y_HOMED" : "",
		       s.flags & RS_FLAG_T_HOMED ? " T_HOMED" : "",
		       s.flags & RS_FLAG_Y_LIMIT_NEG ? " Y-" : "",
		       s.flags & RS_FLAG_Y_LIMIT_POS ? " Y+" : "",
		       s.flags & RS_FLAG_T_HOME_SW ? " T_SW" : "");
		printf("y      %d (steps %d)\n", s.y, s.y_steps);
		printf("t      %d (steps %d)\n", s.t, s.t_steps);

	} else if (!strcmp(cmd, "config") && argc == 2) {
		struct rs_ioc_config c;

		if (ioctl(fd, RS_IOC_GET_CONFIG, &c) < 0)
			return fail("config", 0);
		printf("y  %d ~ %d\n", c.y_min, c.y_max);
		printf("t  %d ~ %d\n", c.t_min, c.t_max);

	} else if (!strcmp(cmd, "home") && argc == 3) {
		struct rs_ioc_home h = { .axis_mask = num(argv[2]) };

		if (ioctl(fd, RS_IOC_HOME, &h) < 0)
			return fail("home", h.nak_reason);
		printf("ACK, waiting DONE...\n");
		return wait_done(fd, DEFAULT_WAIT_MS);

	} else if ((!strcmp(cmd, "move-rel") || !strcmp(cmd, "move-abs")) &&
		   argc == 4) {
		struct rs_ioc_move m = { .y = num(argv[2]), .t = num(argv[3]) };
		unsigned long req = !strcmp(cmd, "move-rel") ? RS_IOC_MOVE_REL
							      : RS_IOC_MOVE_ABS;

		if (ioctl(fd, req, &m) < 0)
			return fail(cmd, m.nak_reason);
		printf("ACK, waiting DONE...\n");
		return wait_done(fd, DEFAULT_WAIT_MS);

	} else if (!strcmp(cmd, "stop") && argc == 2) {
		struct rs_ioc_stop s = { 0 };

		if (ioctl(fd, RS_IOC_STOP, &s) < 0)
			return fail("stop", s.nak_reason);
		printf("ACK\n");

	} else if (!strcmp(cmd, "speed") && argc == 4) {
		struct rs_ioc_speed s = { .y = num(argv[2]), .t = num(argv[3]) };

		if (ioctl(fd, RS_IOC_SET_SPEED, &s) < 0)
			return fail("speed", s.nak_reason);
		printf("ACK\n");

	} else if (!strcmp(cmd, "wait") && (argc == 2 || argc == 3)) {
		return wait_done(fd, argc == 3 ? num(argv[2]) : DEFAULT_WAIT_MS);

	} else {
		usage();
	}

	close(fd);
	return 0;
}
