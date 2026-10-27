/* SPDX-License-Identifier: GPL-2.0 */
/*
 * robot_stage line discipline
 *
 *  ttyACM0에 붙어서 STM32와 프레임을 주고받습니다.
 *  붙이기: sudo ldattach -s 115200 -8 -n -1 29 /dev/ttyACM0
 */
#ifndef __RS_LDISC_H__
#define __RS_LDISC_H__

int rs_ldisc_register(void);
void rs_ldisc_unregister(void);

#endif /* __RS_LDISC_H__ */
