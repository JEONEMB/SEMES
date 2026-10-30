#include "robot_stage_service.h"
#include "app_config.h"

void RsService_Init(RsService *service)
{
    service->cached = 0U;
    service->event_seq = 0U;
}

static void Nak(RsFrame *reply, rs_u8 type, rs_u8 reason)
{
    reply->type = RS_NAK; reply->len = 2U;
    reply->payload[0] = type; reply->payload[1] = reason;
}

void RsService_Handle(RsService *service, const RsFrame *request,
                      rs_u8 switch_bits, RsFrame *reply)
{
    if (service->cached && request->seq == service->last_seq &&
        request->type == service->last_type) {
        *reply = service->reply;
        return;
    }
    *reply = (RsFrame){0};
    reply->seq = request->seq;
    int expected = -1;
    switch (request->type) {
    case RS_PING: case RS_STOP: case RS_GET_STATUS: case RS_GET_CONFIG: expected = 0; break;
    case RS_HOME: expected = 1; break;
    case RS_MOVE_REL: case RS_MOVE_ABS: case RS_SET_SPEED: expected = 8; break;
    default: break;
    }
    if (expected < 0) {
        Nak(reply, request->type, RS_UNKNOWN_TYPE);
    } else if (request->len != (unsigned)expected) {
        Nak(reply, request->type, RS_BAD_LEN);
    } else {
        switch (request->type) {
        case RS_PING:
            reply->type = RS_PONG; reply->len = 4U;
            reply->payload[0] = RS_VERSION;
            reply->payload[1] = APP_FW_MAJOR;
            reply->payload[2] = APP_FW_MINOR;
            reply->payload[3] = APP_FW_PATCH;
            break;
        case RS_GET_STATUS:
            reply->type = RS_STATUS; reply->len = 20U;
            reply->payload[0] = RS_STATE_FAULT;
            /* No homed bits. Coordinates are zero placeholders, not measurements. */
            reply->payload[1] = (rs_u8)((switch_bits & 7U) << 2);
            break;
        case RS_STOP:
            /* App dispatch aborts the Y test before calling this handler.
             * Theta EN polarity remains unknown. */
            reply->type = RS_ACK; reply->len = 1U;
            reply->payload[0] = RS_STOP;
            break;
        case RS_HOME:
            Nak(reply, request->type,
                (request->payload[0] == 0U || (request->payload[0] & ~3U))
                ? RS_BAD_PARAM : RS_FAULT);
            break;
        default:
            /* GET_CONFIG has no truthful range until hardware is configured.
             * Explicit bring-up exception documented in the project context. */
            Nak(reply, request->type, RS_FAULT);
            break;
        }
    }
    service->cached = 1U;
    service->last_type = request->type; service->last_seq = request->seq;
    service->reply = *reply;
}

void RsService_Event(RsService *service, rs_u8 event, RsFrame *reply)
{
    *reply = (RsFrame){0};
    reply->type = RS_EVENT; reply->seq = service->event_seq++;
    reply->len = 9U; reply->payload[0] = event;
}
