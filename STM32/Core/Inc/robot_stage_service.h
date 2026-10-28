#ifndef ROBOT_STAGE_SERVICE_H
#define ROBOT_STAGE_SERVICE_H
#include "robot_stage_transport.h"

/* Locked bring-up service: no command can start motion. */
typedef struct {
    rs_u8 cached, last_type, last_seq, event_seq;
    RsFrame reply;
} RsService;
void RsService_Init(RsService *service);
void RsService_Handle(RsService *service, const RsFrame *request,
                      rs_u8 switch_bits, RsFrame *reply);
void RsService_Event(RsService *service, rs_u8 event, RsFrame *reply);
#endif
