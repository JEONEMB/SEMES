// SPDX-License-Identifier: GPL-2.0
/*
 * robot_stage: Raspberry Pi ↔ STM32 스테이지 제어 드라이버
 *
 * 모듈 로드/언로드 시 line discipline을 등록/해제합니다.
 * 이후 단계에서 /dev/robot_stage도 여기서 등록합니다.
 */

/* pr_info() 등의 로그 앞에 "robot_stage: "를 자동으로 붙임 */
#define pr_fmt(fmt) KBUILD_MODNAME ": " fmt

#include <linux/init.h>
#include <linux/module.h>

#include "rs_ldisc.h"

static int __init rs_init(void)
{
	int ret;

	ret = rs_ldisc_register();
	if (ret) {
		pr_err("line discipline register failed: %d\n", ret);
		return ret;
	}

	pr_info("loaded\n");
	return 0;
}

static void __exit rs_exit(void)
{
	rs_ldisc_unregister();
	pr_info("unloaded\n");
}

module_init(rs_init);
module_exit(rs_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("lye");
MODULE_DESCRIPTION("Robot stage (Y/Theta) control over STM32 serial link");
