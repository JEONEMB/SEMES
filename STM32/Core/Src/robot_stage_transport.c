#include "robot_stage_transport.h"

static void Drop(RsParser *parser, unsigned count)
{
    parser->count -= count;
    for (unsigned i = 0; i < parser->count; ++i) {
        parser->bytes[i] = parser->bytes[i + count];
    }
}

void RsParser_Reset(RsParser *parser) { parser->count = 0U; }

void RsParser_Expire(RsParser *parser, rs_u32 now)
{
    if (parser->count && (rs_u32)(now - parser->last_byte_ms) >= RS_RX_TIMEOUT_MS) {
        RsParser_Reset(parser);
    }
}

void RsParser_Push(RsParser *parser, rs_u8 byte, rs_u32 arrival_ms)
{
    RsParser_Expire(parser, arrival_ms);
    parser->last_byte_ms = arrival_ms;
    if (parser->count == RS_MAX_FRAME) { Drop(parser, 1U); }
    parser->bytes[parser->count++] = byte;
}

int RsParser_Next(RsParser *parser, RsFrame *frame)
{
    while (parser->count) {
        rs_u8 *p = parser->bytes;
        if (p[0] != 0xAAU || (parser->count >= 2U && p[1] != 0x55U) ||
            (parser->count >= 3U && p[2] != RS_VERSION) ||
            (parser->count >= 6U && p[5] > RS_MAX_PAYLOAD)) {
            Drop(parser, 1U);
            continue;
        }
        if (parser->count < 6U) { return 0; }
        unsigned size = 8U + p[5];
        if (parser->count < size) { return 0; }
        if (rs_crc16(p + 2, 4U + p[5]) != rs_get_u16(p + size - 2U)) {
            Drop(parser, 1U);
            continue;
        }
        frame->type = p[3]; frame->seq = p[4]; frame->len = p[5];
        for (unsigned i = 0; i < frame->len; ++i) { frame->payload[i] = p[6U + i]; }
        Drop(parser, size);
        return 1;
    }
    return 0;
}

unsigned RsFrame_Encode(const RsFrame *frame, rs_u8 output[RS_MAX_FRAME])
{
    if (frame->len > RS_MAX_PAYLOAD) { return 0U; }
    output[0] = 0xAAU; output[1] = 0x55U; output[2] = RS_VERSION;
    output[3] = frame->type; output[4] = frame->seq; output[5] = frame->len;
    for (unsigned i = 0; i < frame->len; ++i) { output[6U + i] = frame->payload[i]; }
    rs_put_u16(output + 6U + frame->len, rs_crc16(output + 2, 4U + frame->len));
    return 8U + frame->len;
}
