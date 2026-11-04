#ifndef ROBOT_STAGE_SERVICE_H
#define ROBOT_STAGE_SERVICE_H
#include "robot_stage_transport.h"

typedef struct {
    rs_u8 cached, event_seq, motion_cached, pending, homed, fault;
    rs_u8 motor_started, stop_requested;
    rs_u16 motion_age;
    rs_i32 steps, commanded_um, start_steps, target_um;
    int direction;
    rs_u32 speed_um_s;
    RsFrame last_request, reply, motion_request, motion_reply;
} RsService;
void RsService_Init(RsService *service);
void RsService_Handle(RsService *service, const RsFrame *request,
                      rs_u8 switch_bits, RsFrame *reply);
void RsService_Event(RsService *service, rs_u8 event, RsFrame *reply);
int RsService_Poll(RsService *service, RsFrame *reply);
void RsService_InvalidatePosition(RsService *service);
#endif
