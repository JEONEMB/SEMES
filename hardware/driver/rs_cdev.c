// SPDX-License-Identifier: GPL-2.0
/*
 * /dev/robot_stage: 앱 ↔ 드라이버 통로 (ioctl)
 *
 *  앱이 넘긴 값(uapi 구조체)을 프로토콜 페이로드로 바꿔 rs_cmd로 보내고,
 *  응답을 다시 uapi 구조체로 바꿔 돌려줍니다.
 */
#define pr_fmt(fmt) KBUILD_MODNAME ": " fmt

#include <linux/fs.h>
#include <linux/miscdevice.h>
#include <linux/module.h>
#include <linux/uaccess.h>

#include "rs_dev.h"
#include "uapi/robot_stage_ioctl.h"

static long rs_ioc_ping(struct rs_dev *d, void __user *uarg)
{
	struct rs_ioc_version v = {};
	struct rs_pong p;
	int ret;

	ret = rs_cmd_query(d, RS_CMD_PING, &p, sizeof(p));
	if (ret)
		return ret;
	v.proto_ver = p.proto_ver;
	v.fw_major = p.fw_major;
	v.fw_minor = p.fw_minor;
	v.fw_patch = p.fw_patch;
	return copy_to_user(uarg, &v, sizeof(v)) ? -EFAULT : 0;
}

static long rs_ioc_status(struct rs_dev *d, void __user *uarg)
{
	struct rs_ioc_status s = {};
	struct rs_status p;
	int ret;

	ret = rs_cmd_query(d, RS_CMD_GET_STATUS, &p, sizeof(p));
	if (ret)
		return ret;
	s.state = p.state;
	s.flags = p.flags;
	s.y = p.y_um;
	s.t = p.t_mdeg;
	s.y_steps = p.y_steps;
	s.t_steps = p.t_steps;
	return copy_to_user(uarg, &s, sizeof(s)) ? -EFAULT : 0;
}

static long rs_ioc_config(struct rs_dev *d, void __user *uarg)
{
	struct rs_ioc_config c = {};
	struct rs_config p;
	int ret;

	ret = rs_cmd_query(d, RS_CMD_GET_CONFIG, &p, sizeof(p));
	if (ret)
		return ret;
	c.y_min = p.y_min_um;
	c.y_max = p.y_max_um;
	c.t_min = p.t_min_mdeg;
	c.t_max = p.t_max_mdeg;
	return copy_to_user(uarg, &c, sizeof(c)) ? -EFAULT : 0;
}

/* 동작 명령은 NAK이어도 nak_reason을 앱에 돌려줘야 하므로 항상 copy_to_user */
static long rs_ioc_home(struct rs_dev *d, void __user *uarg)
{
	struct rs_ioc_home h;
	struct rs_home p;
	int ret;

	if (copy_from_user(&h, uarg, sizeof(h)))
		return -EFAULT;
	p.axis_mask = h.axis_mask;
	ret = rs_cmd_action(d, RS_CMD_HOME, &p, sizeof(p), &h.nak_reason);
	if (copy_to_user(uarg, &h, sizeof(h)))
		return -EFAULT;
	return ret;
}

static long rs_ioc_move(struct rs_dev *d, u8 type, void __user *uarg)
{
	struct rs_ioc_move m;
	struct rs_move p;
	int ret;

	if (copy_from_user(&m, uarg, sizeof(m)))
		return -EFAULT;
	p.y_um = m.y;
	p.t_mdeg = m.t;
	ret = rs_cmd_action(d, type, &p, sizeof(p), &m.nak_reason);
	if (copy_to_user(uarg, &m, sizeof(m)))
		return -EFAULT;
	return ret;
}

static long rs_ioc_speed(struct rs_dev *d, void __user *uarg)
{
	struct rs_ioc_speed s;
	struct rs_speed p;
	int ret;

	if (copy_from_user(&s, uarg, sizeof(s)))
		return -EFAULT;
	p.y_um_per_s = s.y;
	p.t_mdeg_per_s = s.t;
	ret = rs_cmd_action(d, RS_CMD_SET_SPEED, &p, sizeof(p), &s.nak_reason);
	if (copy_to_user(uarg, &s, sizeof(s)))
		return -EFAULT;
	return ret;
}

static long rs_ioc_stop(struct rs_dev *d, void __user *uarg)
{
	struct rs_ioc_stop s = {};
	int ret;

	ret = rs_cmd_action(d, RS_CMD_STOP, NULL, 0, &s.nak_reason);
	if (copy_to_user(uarg, &s, sizeof(s)))
		return -EFAULT;
	return ret;
}

static long rs_ioc_wait_done(struct rs_dev *d, void __user *uarg)
{
	struct rs_ioc_done w;
	struct rs_done p;
	int ret;

	if (copy_from_user(&w, uarg, sizeof(w)))
		return -EFAULT;
	ret = rs_cmd_wait_done(d, w.timeout_ms, &p);
	if (ret)
		return ret;
	w.result = p.result;
	w.cmd_type = p.cmd_type;
	w.pad = 0;
	w.y = p.y_um;
	w.t = p.t_mdeg;
	return copy_to_user(uarg, &w, sizeof(w)) ? -EFAULT : 0;
}

static long rs_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
	void __user *uarg = (void __user *)arg;
	struct rs_dev *d = &rs_dev;

	switch (cmd) {
	case RS_IOC_PING:	return rs_ioc_ping(d, uarg);
	case RS_IOC_HOME:	return rs_ioc_home(d, uarg);
	case RS_IOC_MOVE_REL:	return rs_ioc_move(d, RS_CMD_MOVE_REL, uarg);
	case RS_IOC_MOVE_ABS:	return rs_ioc_move(d, RS_CMD_MOVE_ABS, uarg);
	case RS_IOC_STOP:	return rs_ioc_stop(d, uarg);
	case RS_IOC_GET_STATUS:	return rs_ioc_status(d, uarg);
	case RS_IOC_SET_SPEED:	return rs_ioc_speed(d, uarg);
	case RS_IOC_GET_CONFIG:	return rs_ioc_config(d, uarg);
	case RS_IOC_WAIT_DONE:	return rs_ioc_wait_done(d, uarg);
	}
	return -ENOTTY;
}

static const struct file_operations rs_fops = {
	.owner          = THIS_MODULE,
	.open           = nonseekable_open,
	.unlocked_ioctl = rs_ioctl,
};

static struct miscdevice rs_misc = {
	.minor = MISC_DYNAMIC_MINOR,
	.name  = "robot_stage",     /* /dev/robot_stage */
	.fops  = &rs_fops,
};

int rs_cdev_register(void)
{
	return misc_register(&rs_misc);
}

void rs_cdev_unregister(void)
{
	misc_deregister(&rs_misc);
}
