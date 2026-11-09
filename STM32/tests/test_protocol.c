#include "robot_stage_service.h"
#include "mock_hal.h"
#include "y_motor.h"
#include "y_test.h"
#include "app_config.h"
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

static const uint8_t firmware_pong[] = {0xAA,0x55,1,0x85,1,4,1,0,2,1,0xE2,0x68};

int TestService(void)
{
    gpio.IDR = 0; test_rcc.CFGR = test_rcc.DCKCFGR = 0;
    YMotor_Init();
    RsService service; RsService_Init(&service);
    RsFrame request = {.type = RS_PING, .seq = 1}, reply, old;
    uint8_t encoded[40];
    RsService_Handle(&service, &request, 0, &reply);
    CHECK(RsFrame_Encode(&reply, encoded) == 12);
    CHECK(Equal(encoded, firmware_pong, 12));
    request.type = RS_GET_STATUS;
    RsService_Handle(&service, &request, 5, &reply);
    CHECK(reply.type == RS_STATUS && reply.len == 20);
    CHECK(reply.payload[0] == (Y_MOTOR_OUTPUT_ALLOWED ? RS_STATE_IDLE : RS_STATE_FAULT));
    CHECK(reply.payload[1] == (Y_MOTOR_USE_LIMITS ? 4 : 0));
    for (unsigned i = 2; i < 20; ++i) CHECK(reply.payload[i] == 0);
    old = reply;
    RsService_Handle(&service, &request, 2, &reply);
    CHECK(Equal(&old, &reply, sizeof(reply)));
    ++request.seq;
    RsService_Handle(&service, &request, 2, &reply);
    CHECK(reply.payload[1] == (Y_MOTOR_USE_LIMITS ? 8 : 0));
    request.type = RS_GET_CONFIG; request.seq++;
    RsService_Handle(&service, &request, 0, &reply);
    CHECK(reply.type == RS_CONFIG && reply.len == 16);
    uint8_t expected_max[4]; rs_put_u32(expected_max, APP_Y_MAX_UM);
    CHECK(Equal(reply.payload+4, expected_max, 4));
    for (unsigned i = 8; i < 16; ++i) CHECK(reply.payload[i] == 0);
    const uint8_t lengths[] = {0,1,8,8,0,0,8,0};
    for (unsigned type = 1; type <= 8; ++type) {
        request.type = (uint8_t)type; request.seq++;
        request.len = lengths[type - 1] + 1;
        RsService_Handle(&service, &request, 0, &reply);
        CHECK(reply.type == RS_NAK && reply.payload[1] == RS_BAD_LEN);
    }
    request.type = RS_HOME; request.len = 1;
    for (unsigned mask = 0; mask < 256; ++mask) {
        if (mask == 1) continue;
        request.seq++; request.payload[0] = (uint8_t)mask;
        RsService_Handle(&service, &request, 0, &reply);
        CHECK(reply.type == RS_NAK && reply.payload[1] == RS_BAD_PARAM);
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
TestRcc test_rcc;
uint32_t HAL_RCC_GetPCLK1Freq(void) { return 16000000U; }
void HAL_NVIC_DisableIRQ(int irq) { (void)irq; }
void HAL_NVIC_ClearPendingIRQ(int irq) { (void)irq; }
void HAL_NVIC_SetPriority(int irq, unsigned p, unsigned s) { (void)irq; (void)p; (void)s; }
void HAL_NVIC_EnableIRQ(int irq) { (void)irq; }
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
void TestSetTime(uint32_t now) { tick=now; }
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
int TestUartDuringPulseBurst(void)
{
    ResetApp();
    CHECK(YMotor_Start(0, 3, 100) == Y_MOTOR_OK);
    Receive(vectors[0], 8); Drain();
    CHECK(wire_length == 12 && Equal(wire, firmware_pong, 12));
    YMotorStatus status;
    YMotor_GetStatus(&status);
    CHECK(status.state == Y_MOTOR_RUNNING && status.completed_pulses == 0);
    YMotor_Abort();
    return 0;
}
int TestApp(void)
{
    ResetApp();
    CHECK(timer1.CCER == 0 && timer1.BDTR == 0 && timer1.CR1 == 0);
    CHECK((timer3.CR1 & TIM_CR1_CEN) == 0 && gpio_level == 1);
    /* RX while TX is active cannot overwrite an in-flight packet. */
    Receive(vectors[0], 8); App_Poll();
    CHECK(uart.gState != HAL_UART_STATE_READY);
    for (unsigned i = 0; i < 20; ++i) Receive(vectors[0], 8);
    App_Poll(); Drain();
    CHECK(wire_length == 21U * 12U);
    for (unsigned i = 0; i < 21; ++i) CHECK(Equal(wire + i * 12, firmware_pong, 12));
    /* Arrival timestamps, not drain time, decide timeout. */
    ResetApp(); Receive(vectors[0], 4); tick = 20; Receive(vectors[0] + 4, 4);
    Drain(); CHECK(wire_length == 0);
    Receive(vectors[0], 8); tick = 100; Drain();
    CHECK(wire_length == 12 && Equal(wire, firmware_pong, 12));
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
    Drain(); CHECK(wire_length == 12 && Equal(wire, firmware_pong, 12));
    /* Boot once; stable Y input edges emit one independent event. */
    ResetApp();
    for (tick = 1; tick <= 10; ++tick) App_Poll();
    gpio.IDR = 1;
    for (; tick <= 21; ++tick) { App_Poll(); FinishTx(); }
    Drain();
#if Y_MOTOR_USE_LIMITS
    CHECK(wire_length == 17 && wire[3] == RS_EVENT && wire[6] == RS_LIMIT_HIT);
    CHECK(wire[4] == 1);
#else
    CHECK(wire_length == 0);
#endif
    wire_length = 0;
    for (; tick <= 40; ++tick) { App_Poll(); FinishTx(); }
    CHECK(wire_length == 0);
    uart.gState = HAL_UART_STATE_READY;
    App_Init(&uart); Drain();
    CHECK(wire_length == 17 && wire[3] == RS_EVENT && wire[4] == 0 && wire[6] == RS_BOOT);
    return 0;
}

int TestPCJog(void)
{
#if Y_MOTOR_OUTPUT_ALLOWED
    ResetApp();
    RsFrame request = {.type = Y_TEST_JOG, .seq = 23, .len = 9};
    rs_put_u32(request.payload + 1, 160);
    rs_put_u32(request.payload + 5, 160);
    uint8_t bytes[RS_MAX_FRAME];
    unsigned n = RsFrame_Encode(&request, bytes);
    Receive(bytes, n); Drain();
    CHECK(wire_length == 9 && wire[3] == RS_ACK && wire[6] == Y_TEST_JOG);
    YMotorStatus s; YMotor_GetStatus(&s);
    CHECK(s.state == Y_MOTOR_RUNNING && s.requested_pulses == 160);
    wire_length = 0;
    Receive(vectors[0], 8); Drain();
    CHECK(wire_length == 12 && Equal(wire, firmware_pong, 12));
    wire_length = 0;
    request.type = RS_STOP; request.seq = 24; request.len = 0;
    n = RsFrame_Encode(&request, bytes); Receive(bytes, n); Drain();
    CHECK(wire_length == 9 + 28 && wire[3] == RS_ACK && wire[6] == RS_STOP);
    CHECK(wire[12] == Y_TEST_REPORT && wire[13] == 23 && wire[15] == Y_MOTOR_ABORTED);
    YMotor_GetStatus(&s); CHECK(!s.enabled && s.state == Y_MOTOR_ABORTED);
#endif
    return 0;
}

int TestPhysicalApp(void)
{
#if APP_Y_SWITCH_HOME && Y_MOTOR_OUTPUT_ALLOWED
    ResetApp();
    for (tick=1;tick<=10;++tick) Drain();
    RsFrame q={.type=RS_HOME,.seq=90,.len=1,.payload={1}};
    uint8_t bytes[RS_MAX_FRAME];
    unsigned n=RsFrame_Encode(&q,bytes); Receive(bytes,n); Drain();
    CHECK(wire_length==9 && wire[3]==RS_ACK);
    gpio.IDR=1;
    for (;tick<=22;++tick) Drain(); /* left contact -> release */
    gpio.IDR=0;
    for (;tick<=60;++tick) Drain(); /* stable release -> zero, no further movement */
    CHECK(wire_length==29 && wire[12]==RS_DONE && wire[13]==90 && wire[15]==RS_OK);
    wire_length=0;
    q=(RsFrame){.type=RS_GET_STATUS,.seq=91};
    n=RsFrame_Encode(&q,bytes); Receive(bytes,n); Drain();
    CHECK(wire_length==28 && wire[3]==RS_STATUS && wire[6]==RS_STATE_IDLE && wire[7]==1);
    for (unsigned i=10;i<26;++i) CHECK(wire[i]==0); /* y/theta coordinates and steps */
    wire_length=0;
    q=(RsFrame){.type=RS_MOVE_REL,.seq=92,.len=8}; rs_put_u32(q.payload,1000);
    n=RsFrame_Encode(&q,bytes); Receive(bytes,n); Drain();
    CHECK(wire_length==9 && wire[3]==RS_ACK);
    gpio.IDR=0;
    for (unsigned i=0;i<1600;++i) {
        TIM3->CR1 &= ~TIM_CR1_CEN; TIM3->SR |= TIM_SR_UIF; YMotor_IRQHandler();
    }
    Drain();
    CHECK(wire_length==29 && wire[12]==RS_DONE && wire[15]==RS_OK);
    CHECK(wire[19]==0xE8 && wire[20]==3);
#endif
    return 0;
}

int TestStandardApp(void)
{
#if Y_MOTOR_OUTPUT_ALLOWED
    ResetApp();
    RsFrame q = {.type=RS_HOME, .seq=70, .len=1, .payload={1}};
    uint8_t bytes[RS_MAX_FRAME];
    unsigned n=RsFrame_Encode(&q,bytes); Receive(bytes,n); Drain();
    CHECK(wire_length==29 && wire[3]==RS_ACK && wire[12]==RS_DONE && wire[13]==70);
    wire_length=0;
    q=(RsFrame){.type=RS_MOVE_REL, .seq=71, .len=8};
    rs_put_u32(q.payload,1000);
    n=RsFrame_Encode(&q,bytes); Receive(bytes,n); Drain();
    CHECK(wire_length==9 && wire[3]==RS_ACK);
    wire_length=0;
    q=(RsFrame){.type=Y_TEST_JOG, .seq=72, .len=9};
    rs_put_u32(q.payload+1,160); rs_put_u32(q.payload+5,160);
    n=RsFrame_Encode(&q,bytes); Receive(bytes,n); Drain();
    CHECK(wire_length==10 && wire[3]==RS_NAK && wire[7]==RS_BUSY);
    wire_length=0;
    q=(RsFrame){.type=RS_STOP, .seq=73};
    n=RsFrame_Encode(&q,bytes); Receive(bytes,n); Drain();
    CHECK(wire_length==9 && wire[3]==RS_ACK); /* ACK precedes ramp completion */
    for (unsigned i=0;i<700 && (TIM3->CR1 & TIM_CR1_CEN);++i) {
        TIM3->CR1 &= ~TIM_CR1_CEN; TIM3->SR |= TIM_SR_UIF; YMotor_IRQHandler();
    }
    Drain();
    CHECK(wire_length==29 && wire[3]==RS_ACK && wire[6]==RS_STOP);
    CHECK(wire[12]==RS_DONE && wire[13]==71 && wire[15]==RS_STOPPED && wire[16]==RS_MOVE_REL);
    wire_length=0;
    q=(RsFrame){.type=RS_HOME, .seq=74, .len=1, .payload={1}};
    n=RsFrame_Encode(&q,bytes); Receive(bytes,n); Drain();
    CHECK(wire_length==29 && wire[12]==RS_DONE);
    wire_length=0;
    q=(RsFrame){.type=RS_MOVE_ABS, .seq=75, .len=8};
    rs_put_u32(q.payload,1000);
    n=RsFrame_Encode(&q,bytes); Receive(bytes,n); Drain();
    CHECK(wire_length==9 && wire[3]==RS_ACK);
    for (unsigned i=0;i<1600;++i) {
        TIM3->CR1 &= ~TIM_CR1_CEN; TIM3->SR |= TIM_SR_UIF; YMotor_IRQHandler();
    }
    Drain();
    CHECK(wire_length==29 && wire[12]==RS_DONE && wire[15]==RS_OK);
    CHECK(wire[19]==0xE8 && wire[20]==3);
    wire_length=0;
    q=(RsFrame){.type=Y_TEST_JOG, .seq=76, .len=9};
    rs_put_u32(q.payload+1,160); rs_put_u32(q.payload+5,160);
    n=RsFrame_Encode(&q,bytes); Receive(bytes,n); Drain();
    CHECK(wire_length==9 && wire[3]==RS_ACK);
    wire_length=0;
    q=(RsFrame){.type=RS_HOME, .seq=77, .len=1, .payload={1}};
    n=RsFrame_Encode(&q,bytes); Receive(bytes,n); Drain();
    CHECK(wire[3]==RS_NAK && wire[7]==RS_BUSY);
    wire_length=0;
    q=(RsFrame){.type=RS_GET_STATUS, .seq=78};
    n=RsFrame_Encode(&q,bytes); Receive(bytes,n); Drain();
    CHECK(wire_length==28 && wire[3]==RS_STATUS && !(wire[7]&1));
    YMotor_Abort(); Drain();
#endif
    return 0;
}
