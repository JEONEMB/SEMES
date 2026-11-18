#include "robot_stage_service.h"
#include "mock_hal.h"
#include <stddef.h>
#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (0)

_Static_assert(sizeof(struct rs_done) == 12, "DONE layout");
_Static_assert(sizeof(struct rs_status) == 20, "STATUS layout");
_Static_assert(sizeof(struct rs_config) == 16, "CONFIG layout");
_Static_assert(sizeof(struct rs_event_payload) == 9, "EVENT layout");

/* Bare-metal test executable supplies its tiny C runtime. */
void *memset(void *p, int value, size_t n)
{ unsigned char *b = p; while (n--) { *b++ = (unsigned char)value; } return p; }
void *memcpy(void *to, const void *from, size_t n)
{ unsigned char *d = to; const unsigned char *s = from; while (n--) { *d++ = *s++; } return to; }
static int Equal(const void *a, const void *b, unsigned n)
{ const uint8_t *x = a, *y = b; while (n--) { if (*x++ != *y++) return 0; } return 1; }

static const uint8_t vectors[][20] = {
    {0xAA,0x55,1,1,1,0,0x75,0xF6},
    {0xAA,0x55,1,0x85,1,4,1,0,1,0,0x90,0x2D},
    {0xAA,0x55,1,2,2,1,3,7,0x3B},
    {0xAA,0x55,1,3,12,8,0x1E,0xFB,0xFF,0xFF,0x5E,1,0,0,0xA4,0x18},
    {0xAA,0x55,1,0x81,12,1,3,0xE2,0x66},
    {0xAA,0x55,1,0x82,12,2,3,3,0x81,0x69},
    {0xAA,0x55,1,0x83,12,12,0,3,0,0,0xC6,0x5C,0,0,0x46,5,0,0,0x49,0x84}
};

int TestTransport(void)
{
    CHECK(rs_crc16((const uint8_t *)"123456789", 9) == 0x29B1);
    RsParser parser = {0}; RsFrame frame;
    uint8_t encoded[RS_MAX_FRAME];
    for (unsigned v = 0; v < 7; ++v) {
        unsigned n = 8U + vectors[v][5];
        for (unsigned split = 0; split <= n; ++split) {
            RsParser_Reset(&parser);
            for (unsigned i = 0; i < n; ++i) {
                RsParser_Push(&parser, vectors[v][i], i < split ? 0 : 19);
                CHECK(RsParser_Next(&parser, &frame) == (i == n - 1U));
            }
            CHECK(RsFrame_Encode(&frame, encoded) == n);
            CHECK(Equal(encoded, vectors[v], n));
        }
    }
    /* Noise, overlapping SOF, back-to-back frames, sequence wrap. */
    RsParser_Reset(&parser);
    RsParser_Push(&parser, 0xAA, 0); CHECK(!RsParser_Next(&parser, &frame));
    for (unsigned k = 0; k < 256; ++k) {
        RsFrame source = {.type = RS_PING, .seq = (uint8_t)k, .len = 0};
        unsigned n = RsFrame_Encode(&source, encoded);
        for (unsigned i = 0; i < n; ++i) {
            RsParser_Push(&parser, encoded[i], 0);
            CHECK(RsParser_Next(&parser, &frame) == (i == n - 1U));
        }
        CHECK(frame.seq == k);
    }
    /* CRC, version, excessive length: silent discard, then valid recovery. */
    for (unsigned mutation = 0; mutation < 3; ++mutation) {
        RsParser_Reset(&parser);
        memcpy(encoded, vectors[0], 8);
        encoded[mutation == 0 ? 7 : mutation == 1 ? 2 : 5] ^= 0x80;
        for (unsigned i = 0; i < 8; ++i) {
            RsParser_Push(&parser, encoded[i], 0);
            CHECK(!RsParser_Next(&parser, &frame));
        }
        for (unsigned i = 0; i < 8; ++i) {
            RsParser_Push(&parser, vectors[0][i], 0);
            CHECK(RsParser_Next(&parser, &frame) == (i == 7));
        }
    }
    /* Bad outer CRC must preserve a valid embedded frame. */
    uint8_t nested[16] = {0xAA,0x55,1,0x7F,0,8};
    memcpy(nested + 6, vectors[0], 8);
    RsParser_Reset(&parser);
    for (unsigned i = 0; i < 16; ++i) {
        RsParser_Push(&parser, nested[i], 0);
        CHECK(RsParser_Next(&parser, &frame) == (i == 15));
    }
    CHECK(frame.type == RS_PING);
    /* 20ms gap, including HAL tick wraparound, discards a partial frame. */
    for (unsigned wrap = 0; wrap < 2; ++wrap) {
        uint32_t start = wrap ? 0xFFFFFFF5U : 0;
        RsParser_Reset(&parser);
        for (unsigned i = 0; i < 4; ++i) {
            RsParser_Push(&parser, vectors[0][i], start);
            CHECK(!RsParser_Next(&parser, &frame));
        }
        for (unsigned i = 4; i < 8; ++i) {
            RsParser_Push(&parser, vectors[0][i], start + 20U);
            CHECK(!RsParser_Next(&parser, &frame));
        }
    }
    RsFrame max = {.type = 0x7F, .len = 32};
    for (unsigned i = 0; i < 32; ++i) max.payload[i] = (uint8_t)i;
    CHECK(RsFrame_Encode(&max, encoded) == 40);
    RsParser_Reset(&parser);
    for (unsigned i = 0; i < 40; ++i) {
        RsParser_Push(&parser, encoded[i], 0);
        CHECK(RsParser_Next(&parser, &frame) == (i == 39));
    }
    CHECK(Equal(frame.payload, max.payload, 32));
    max.len = 33; CHECK(RsFrame_Encode(&max, encoded) == 0);
    return 0;
}

int TestService(void)
{
    RsService service; RsService_Init(&service);
    RsFrame request = {.type = RS_PING, .seq = 1}, reply, old;
    uint8_t encoded[40];
    RsService_Handle(&service, &request, 0, &reply);
    CHECK(RsFrame_Encode(&reply, encoded) == 12);
    CHECK(Equal(encoded, vectors[1], 12));
    request.type = RS_GET_STATUS;
    RsService_Handle(&service, &request, 5, &reply);
    CHECK(reply.type == RS_STATUS && reply.len == 20);
    CHECK(reply.payload[0] == RS_STATE_FAULT && reply.payload[1] == 20);
    for (unsigned i = 2; i < 20; ++i) CHECK(reply.payload[i] == 0);
    old = reply;
    RsService_Handle(&service, &request, 2, &reply);
    CHECK(Equal(&old, &reply, sizeof(reply)));
    ++request.seq;
    RsService_Handle(&service, &request, 2, &reply);
    CHECK(reply.payload[1] == 8);
    const uint8_t lengths[] = {0,1,8,8,0,0,8,0};
    for (unsigned type = 1; type <= 8; ++type) {
        request.type = (uint8_t)type; request.seq++;
        request.len = lengths[type - 1] + 1;
        RsService_Handle(&service, &request, 0, &reply);
        CHECK(reply.type == RS_NAK && reply.payload[1] == RS_BAD_LEN);
        request.seq++; request.len = lengths[type - 1]; request.payload[0] = 3;
        RsService_Handle(&service, &request, 0, &reply);
        if (type == RS_HOME || type == RS_MOVE_REL || type == RS_MOVE_ABS ||
            type == RS_SET_SPEED || type == RS_GET_CONFIG) {
            CHECK(reply.type == RS_NAK && reply.payload[1] == RS_FAULT);
        } else if (type == RS_STOP) {
            CHECK(reply.type == RS_ACK && reply.payload[0] == RS_STOP);
        }
    }
    request.type = RS_HOME; request.len = 1;
    for (unsigned mask = 0; mask < 256; ++mask) {
        request.seq++; request.payload[0] = (uint8_t)mask;
        RsService_Handle(&service, &request, 0, &reply);
        CHECK(reply.payload[1] == (mask >= 1 && mask <= 3 ? RS_FAULT : RS_BAD_PARAM));
    }
    request.type = 0x7F;
    RsService_Handle(&service, &request, 0, &reply);
    CHECK(reply.payload[1] == RS_UNKNOWN_TYPE);
    for (unsigned i = 0; i < 257; ++i) {
        RsService_Event(&service, RS_BOOT, &reply);
        CHECK(reply.type == RS_EVENT && reply.len == 9 && reply.seq == (uint8_t)i);
    }
    return 0;
}

TestTimer timer1, timer3;
TestGpio gpio;
static uint32_t tick, primask;
static uint8_t *rx_target, *tx_source;
static uint16_t tx_length;
static UART_HandleTypeDef uart;
static uint8_t wire[4096];
static unsigned wire_length, aborts, gpio_level;
static HAL_StatusTypeDef next_tx_result;
uint32_t __get_PRIMASK(void) { return primask; }
void __disable_irq(void) { primask = 1; }
void __set_PRIMASK(uint32_t state) { primask = state; }
uint32_t HAL_GetTick(void) { return tick; }
void HAL_GPIO_WritePin(TestGpio *port, uint32_t pin, uint32_t level)
{ (void)port; (void)pin; gpio_level = level; }
void BSP_LED_Toggle(unsigned led) { (void)led; }
HAL_StatusTypeDef HAL_UART_Receive_IT(UART_HandleTypeDef *u, uint8_t *p, uint16_t n)
{ (void)u; (void)n; rx_target = p; return HAL_OK; }
HAL_StatusTypeDef HAL_UART_AbortReceive(UART_HandleTypeDef *u)
{ (void)u; ++aborts; return HAL_OK; }
HAL_StatusTypeDef HAL_UART_Transmit_IT(UART_HandleTypeDef *u, uint8_t *p, uint16_t n)
{
    if (next_tx_result != HAL_OK) {
        HAL_StatusTypeDef result = next_tx_result; next_tx_result = HAL_OK; return result;
    }
    tx_source = p; tx_length = n; u->gState = 1; return HAL_OK;
}
static void FinishTx(void)
{
    if (uart.gState) {
        memcpy(wire + wire_length, tx_source, tx_length);
        wire_length += tx_length; uart.gState = HAL_UART_STATE_READY;
    }
}
static void Drain(void) { for (unsigned i = 0; i < 100; ++i) { App_Poll(); FinishTx(); } }
static void Receive(const uint8_t *bytes, unsigned n)
{ for (unsigned i = 0; i < n; ++i) { *rx_target = bytes[i]; App_UartRxComplete(&uart); } }
static void ResetApp(void)
{
    uart = (UART_HandleTypeDef){0}; tick = wire_length = aborts = 0;
    gpio.IDR = 0; next_tx_result = HAL_OK;
    timer1 = timer3 = (TestTimer){.CCER = 5, .BDTR = 0x8000, .CR1 = 1};
    App_Init(&uart); Drain(); wire_length = 0;
}
int TestApp(void)
{
    ResetApp();
    CHECK(timer1.CCER == 0 && timer1.BDTR == 0 && timer1.CR1 == 0);
    CHECK(timer3.CCER == 4 && timer3.CR1 == 0 && gpio_level == 1);
    /* RX while TX is active cannot overwrite an in-flight packet. */
    Receive(vectors[0], 8); App_Poll();
    CHECK(uart.gState != HAL_UART_STATE_READY);
    for (unsigned i = 0; i < 20; ++i) Receive(vectors[0], 8);
    App_Poll(); Drain();
    CHECK(wire_length == 21U * 12U);
    for (unsigned i = 0; i < 21; ++i) CHECK(Equal(wire + i * 12, vectors[1], 12));
    /* Arrival timestamps, not drain time, decide timeout. */
    ResetApp(); Receive(vectors[0], 4); tick = 20; Receive(vectors[0] + 4, 4);
    Drain(); CHECK(wire_length == 0);
    Receive(vectors[0], 8); tick = 100; Drain();
    CHECK(wire_length == 12 && Equal(wire, vectors[1], 12));
    /* UART loss discards partial frame, preserves TX, accepts next full frame. */
    ResetApp(); Receive(vectors[0], 8); App_Poll();
    uart.ErrorCode = 1; *rx_target = 0xAA; App_UartRxComplete(&uart);
    App_UartError(&uart); App_Poll();
    CHECK(aborts == 1); Drain(); CHECK(wire_length == 12);
    Receive(vectors[0], 8); Drain(); CHECK(wire_length == 24);
    /* RX queue overrun cannot dispatch buffered/truncated requests. */
    ResetApp();
    for (unsigned i = 0; i < 70; ++i) Receive(vectors[0], 8);
    Drain(); CHECK(wire_length == 0);
    Receive(vectors[0], 8); Drain(); CHECK(wire_length == 12);
    /* Transient TX start failures retain the response. */
    ResetApp(); next_tx_result = HAL_ERROR; Receive(vectors[0], 8);
    Drain(); CHECK(wire_length == 12 && Equal(wire, vectors[1], 12));
    /* Boot once; stable Y input edges emit one independent event. */
    ResetApp();
    for (tick = 1; tick <= 10; ++tick) App_Poll();
    gpio.IDR = 1;
    for (; tick <= 21; ++tick) { App_Poll(); FinishTx(); }
    Drain(); CHECK(wire_length == 17 && wire[3] == RS_EVENT && wire[6] == RS_LIMIT_HIT);
    CHECK(wire[4] == 1);
    wire_length = 0;
    for (; tick <= 40; ++tick) { App_Poll(); FinishTx(); }
    CHECK(wire_length == 0);
    uart.gState = HAL_UART_STATE_READY;
    App_Init(&uart); Drain();
    CHECK(wire_length == 17 && wire[3] == RS_EVENT && wire[4] == 0 && wire[6] == RS_BOOT);
    return 0;
}
