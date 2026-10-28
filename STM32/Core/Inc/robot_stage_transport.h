#ifndef ROBOT_STAGE_TRANSPORT_H
#define ROBOT_STAGE_TRANSPORT_H
#include "robot_stage_protocol.h"

typedef struct {
    rs_u8 type, seq, len, payload[RS_MAX_PAYLOAD];
} RsFrame;
typedef struct {
    rs_u8 bytes[RS_MAX_FRAME];
    unsigned count;
    rs_u32 last_byte_ms;
} RsParser;

void RsParser_Reset(RsParser *parser);
void RsParser_Expire(RsParser *parser, rs_u32 now);
/* Drain Next before pushing another byte. Timestamp is UART arrival time. */
void RsParser_Push(RsParser *parser, rs_u8 byte, rs_u32 arrival_ms);
int RsParser_Next(RsParser *parser, RsFrame *frame);
unsigned RsFrame_Encode(const RsFrame *frame, rs_u8 output[RS_MAX_FRAME]);
#endif
