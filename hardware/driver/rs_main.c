// SPDX-License-Identifier: GPL-2.0
/*
 * robot_stage: Raspberry Pi ↔ STM32 스테이지 제어 드라이버
 *
 * 모듈 로드 시 /dev/robot_stage와 line discipline을 등록합니다.
 * 송수신 로그: sudo insmod robot_stage.ko debug=1
 *              또는 echo 1 | sudo tee /sys/module/robot_stage/parameters/debug
 */

/* pr_info() 등의 로그 앞에 "robot_stage: "를 자동으로 붙임 */
#define pr_fmt(fmt) KBUILD_MODNAME ": " fmt

#include <linux/init.h>
#include <linux/module.h>

#include "rs_dev.h"

bool rs_debug;
module_param_named(debug, rs_debug, bool, 0644);
MODULE_PARM_DESC(debug, "log every tx/rx frame");

static int __init rs_init(void)
{
	int ret;

	rs_dev_init(&rs_dev);

	ret = rs_cdev_register();
	if (ret) {
		pr_err("misc device register failed: %d\n", ret);
		return ret;
	}

	ret = rs_ldisc_register();
	if (ret) {
		pr_err("line discipline register failed: %d\n", ret);
		rs_cdev_unregister();
		return ret;
	}

	pr_info("loaded\n");
	return 0;
}

static void __exit rs_exit(void)
{
	rs_ldisc_unregister();
	rs_cdev_unregister();
	pr_info("unloaded\n");
}

module_init(rs_init);
module_exit(rs_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("lye");
MODULE_DESCRIPTION("Robot stage (Y/Theta) control over STM32 serial link");
