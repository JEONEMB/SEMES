#ifndef Y_TEST_H
#define Y_TEST_H
#include "robot_stage_transport.h"

/* Local commissioning extension; NOT part of the Pi v1 motion contract. */
#define Y_TEST_JOG 0x30U
#define Y_TEST_GET 0x31U
#define Y_TEST_REPORT 0xB0U
void YTest_Init(void);
int YTest_Busy(void);
int YTest_Handle(const RsFrame *request, RsFrame *reply);
int YTest_Poll(RsFrame *reply);
#endif
