# robot_stage 커널 모듈 빌드
#
#   make            모듈 빌드 (robot_stage.ko)
#   make test       프레임 코드 유저 공간 테스트 (protocol.md 9장 예시)
#   make tools      테스트 도구 빌드 (tools/rsctl)
#   make clean      빌드 결과 삭제
#
# 커널 빌드 시스템(Kbuild)이 이 Makefile을 두 번 읽습니다.
#  1) 사용자가 make 실행 → KERNELRELEASE 없음 → 아래 else 부분이 커널 빌드 디렉터리로 넘김
#  2) 커널 빌드 시스템이 다시 읽음 → KERNELRELEASE 있음 → obj-m 목록만 사용

ifneq ($(KERNELRELEASE),)

obj-m := robot_stage.o
robot_stage-y := rs_main.o rs_frame.o rs_ldisc.o rs_cmd.o rs_cdev.o
ccflags-y := -I$(src)/include

else

KDIR ?= /lib/modules/$(shell uname -r)/build
PWD  := $(shell pwd)

all:
	$(MAKE) -C $(KDIR) M=$(PWD) modules

test:
	$(CC) -Wall -Wextra -I. -Iinclude -o tools/rs_frame_test \
		tools/rs_frame_test.c rs_frame.c
	./tools/rs_frame_test

tools:
	$(CC) -Wall -Wextra -Iinclude -Iinclude/uapi -o tools/rsctl tools/rsctl.c

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean
	rm -f tools/rs_frame_test tools/rsctl

.PHONY: all test tools clean

endif
