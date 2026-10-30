#include "y_test.h"
#include "y_motor.h"

static uint8_t cached, pending;
static RsFrame last_request, last_reply;

static uint32_t Get32(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static void Report(uint8_t seq, RsFrame *reply)
{
    YMotorStatus s;
    YMotor_GetStatus(&s);
    *reply = (RsFrame){0};
    reply->type = Y_TEST_REPORT; reply->seq = seq; reply->len = 20;
    reply->payload[0] = (uint8_t)s.state;
    reply->payload[1] = s.dir_high;
    reply->payload[2] = s.count_exact;
    reply->payload[3] = s.enabled;
    rs_put_u32(reply->payload + 4, s.requested_pulses);
    rs_put_u32(reply->payload + 8, s.completed_pulses);
    rs_put_u32(reply->payload + 12, s.frequency_hz);
    reply->payload[16] = s.ready;
    reply->payload[17] = Y_MOTOR_OUTPUT_ALLOWED;
    reply->payload[18] = cached;
    reply->payload[19] = cached ? last_request.seq : 0;
}

static void Nak(const RsFrame *request, RsFrame *reply, uint8_t reason)
{
    *reply = (RsFrame){0};
    reply->type = RS_NAK; reply->seq = request->seq; reply->len = 2;
    reply->payload[0] = request->type; reply->payload[1] = reason;
}

void YTest_Init(void) { cached = pending = 0; }

int YTest_Handle(const RsFrame *request, RsFrame *reply)
{
    if (request->type != Y_TEST_JOG && request->type != Y_TEST_GET) return 0;
    if (request->len != (request->type == Y_TEST_JOG ? 9U : 0U)) {
        Nak(request, reply, RS_BAD_LEN); return 1;
    }
    if (request->type == Y_TEST_GET) { Report(request->seq, reply); return 1; }
    /* Keep the most recent accepted jog across PING/GET/STOP. Never replay
     * motion for that SEQ. Different payload with the same SEQ is rejected. */
    if (cached && request->seq == last_request.seq) {
        for (unsigned i = 0; i < 9; ++i) {
            if (request->payload[i] != last_request.payload[i]) {
                Nak(request, reply, RS_BAD_PARAM); return 1;
            }
        }
        *reply = last_reply; return 1;
    }
    if (pending) { Nak(request, reply, RS_BUSY); return 1; }
    YMotorResult result = YMotor_Start(request->payload[0],
        Get32(request->payload + 1), Get32(request->payload + 5));
    if (result != Y_MOTOR_OK) {
        Nak(request, reply, result == Y_MOTOR_BUSY ? RS_BUSY :
            result == Y_MOTOR_BAD_ARGUMENT ? RS_BAD_PARAM : RS_FAULT);
        return 1;
    }
    *reply = (RsFrame){0};
    reply->type = RS_ACK; reply->seq = request->seq; reply->len = 1;
    reply->payload[0] = Y_TEST_JOG;
    last_request = *request; last_reply = *reply;
    cached = pending = 1;
    return 1;
}

int YTest_Poll(RsFrame *reply)
{
    YMotorStatus s;
    YMotor_GetStatus(&s);
    if (!pending || s.state == Y_MOTOR_RUNNING) return 0;
    Report(last_request.seq, reply);
    last_reply = *reply;
    pending = 0;
    return 1;
}
