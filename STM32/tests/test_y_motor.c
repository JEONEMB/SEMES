#include "y_motor.h"
#include "y_test.h"
#include "mock_hal.h"
#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (0)

/* Explicit PWM2/OPM peripheral model: production register adapter is compiled.
 * This models documented timer behavior, not MCU electrical waveforms. */
#if Y_MOTOR_OUTPUT_ALLOWED
static unsigned rises, falls, high_ticks;
static unsigned level;
static void Tick(void)
{
    if (!(TIM3->CR1 & TIM_CR1_CEN)) { return; }
    if (++TIM3->CNT > TIM3->ARR) {
        TIM3->CNT = 0;
        TIM3->SR |= TIM_SR_UIF;
        if (TIM3->CR1 & TIM_CR1_OPM) TIM3->CR1 &= ~TIM_CR1_CEN;
    }
    unsigned next = (TIM3->CCER & TIM_CCER_CC1E) &&
        (TIM3->CCMR1 & TIM_CCMR1_OC1M) == TIM_OCMODE_PWM2 &&
        TIM3->CNT >= TIM3->CCR1;
    if (next && !level) ++rises;
    if (!next && level) ++falls;
    high_ticks += next;
    level = next;
}
static void Pulse(void)
{
    for (unsigned i = 0; i < 65536 && (TIM3->CR1 & TIM_CR1_CEN); ++i) Tick();
}
#endif

int TestYMotor(void)
{
    YMotorStatus status;
    gpio.IDR = 0; test_rcc.CFGR = test_rcc.DCKCFGR = 0;
    YMotor_Init();
    YMotor_GetStatus(&status);
    CHECK(status.ready && !status.enabled && status.state == Y_MOTOR_IDLE);
    CHECK(TIM3->PSC == 15 && !(TIM3->CR1 & TIM_CR1_CEN));
    CHECK(YMotor_Start(0,0,100) == Y_MOTOR_BAD_ARGUMENT);
    CHECK(YMotor_Start(2,1,100) == Y_MOTOR_BAD_ARGUMENT);
    CHECK(YMotor_Start(0,1601,100) == Y_MOTOR_BAD_ARGUMENT);
    CHECK(YMotor_Start(0,1,15) == Y_MOTOR_BAD_ARGUMENT);
    CHECK(YMotor_Start(0,1,3201) == Y_MOTOR_BAD_ARGUMENT);
#if !Y_MOTOR_OUTPUT_ALLOWED
    CHECK(YMotor_Start(0,1,100) == Y_MOTOR_LOCKED);
    CHECK(YMotor_SetEnabled(1) == Y_MOTOR_LOCKED);
    CHECK(!(TIM3->CR1 & TIM_CR1_CEN));
#else
    gpio.IDR = 1;
    CHECK(YMotor_Start(0,1,100) == Y_MOTOR_LIMIT_ACTIVE);
    CHECK(YMotor_SetEnabled(1) == Y_MOTOR_LIMIT_ACTIVE);
    gpio.IDR = 0;
    CHECK(YMotor_Start(1,3,3200) == Y_MOTOR_OK);
    CHECK(TIM3->CCR1 >= 2000); /* first edge DIR/enable settling */
    CHECK(YMotor_Start(0,1,100) == Y_MOTOR_BUSY);
    CHECK(YMotor_SetEnabled(0) == Y_MOTOR_BUSY);
    YMotor_Init(); /* no reset during active motion */
    for (unsigned p = 0; p < 3; ++p) {
        Pulse();
        CHECK(rises == p+1 && falls == p+1 && high_ticks == (p+1)*5);
        CHECK(!(TIM3->CR1 & TIM_CR1_CEN));
        for (unsigned i = 0; i < 1000; ++i) Tick(); /* delayed IRQ cannot overrun */
        CHECK(rises == p+1);
        YMotor_IRQHandler();
        YMotor_GetStatus(&status);
        CHECK(status.completed_pulses == p+1);
    }
    CHECK(status.state == Y_MOTOR_DONE && status.count_exact && !status.enabled);
    YMotor_IRQHandler();
    YMotor_GetStatus(&status); CHECK(status.completed_pulses == 3);
    CHECK((TIM3->CCMR1 & TIM_CCMR1_OC1M) == TIM_OCMODE_FORCED_INACTIVE);
    const unsigned counts[] = {1, Y_MOTOR_MAX_PULSES};
    for (unsigned c = 0; c < 2; ++c) {
        unsigned before = rises;
        CHECK(YMotor_Start(0, counts[c], c ? 3200 : 16) == Y_MOTOR_OK);
        for (unsigned p = 0; p < counts[c]; ++p) { Pulse(); YMotor_IRQHandler(); }
        YMotor_GetStatus(&status);
        CHECK(status.state == Y_MOTOR_DONE && status.completed_pulses == counts[c]);
        CHECK(rises - before == counts[c] && status.count_exact);
    }
    CHECK(YMotor_Start(0,10,100) == Y_MOTOR_OK);
    Pulse(); YMotor_IRQHandler();
    YMotor_Stop(); Pulse(); YMotor_IRQHandler();
    YMotor_GetStatus(&status);
    CHECK(status.state == Y_MOTOR_STOPPED && status.completed_pulses == 2 && status.count_exact);
    CHECK(YMotor_Start(0,10,100) == Y_MOTOR_OK);
    Pulse(); /* completion pending when immediate abort arrives */
    YMotor_Abort(); YMotor_IRQHandler(); YMotor_GetStatus(&status);
    CHECK(status.state == Y_MOTOR_ABORTED && status.completed_pulses == 1 && status.count_exact);
    CHECK(YMotor_Start(0,10,100) == Y_MOTOR_OK);
    YMotor_Abort(); YMotor_GetStatus(&status);
    CHECK(status.state == Y_MOTOR_ABORTED && !status.count_exact && !status.enabled);
    CHECK(YMotor_Start(0,10,100) == Y_MOTOR_OK);
    gpio.IDR = 2; YMotor_Poll(); YMotor_GetStatus(&status);
    CHECK(status.state == Y_MOTOR_LIMIT && !status.enabled && !status.count_exact);
    gpio.IDR = 0;
    CHECK(YMotor_Start(0,10,100) == Y_MOTOR_OK);
    Pulse(); gpio.IDR = 1; YMotor_IRQHandler(); YMotor_GetStatus(&status);
    CHECK(status.state == Y_MOTOR_LIMIT && status.completed_pulses == 1 && status.count_exact);
    CHECK(!(TIM3->CR1 & TIM_CR1_CEN));
    gpio.IDR = 0;
    /* UART parsing progresses between pulse IRQs, with production app linked. */
    extern int TestUartDuringPulseBurst(void);
    CHECK(TestUartDuringPulseBurst() == 0);
#endif
    test_rcc.CFGR = RCC_CFGR_PPRE1;
    YMotor_Init(); YMotor_GetStatus(&status);
    CHECK(!status.ready && YMotor_Start(0,1,100) == Y_MOTOR_NOT_READY);
    test_rcc.CFGR = 0;
    return 0;
}

int TestYTest(void)
{
    gpio.IDR = 0; test_rcc.CFGR = test_rcc.DCKCFGR = 0;
    YMotor_Init(); YTest_Init();
    RsFrame request = {.type = Y_TEST_JOG, .seq = 7, .len = 9};
    RsFrame reply;
    request.payload[0] = 1;
    rs_put_u32(request.payload + 1, 3);
    rs_put_u32(request.payload + 5, 160);
    request.len = 8;
    CHECK(YTest_Handle(&request, &reply) && reply.payload[1] == RS_BAD_LEN);
    request.len = 9;
    CHECK(YTest_Handle(&request, &reply));
#if !Y_MOTOR_OUTPUT_ALLOWED
    CHECK(reply.type == RS_NAK && reply.payload[1] == RS_FAULT);
    CHECK(!YTest_Poll(&reply));
#else
    CHECK(reply.type == RS_ACK && reply.seq == 7 && reply.payload[0] == Y_TEST_JOG);
    Pulse(); YMotor_IRQHandler();
    CHECK(YTest_Handle(&request, &reply) && reply.type == RS_ACK);
    YMotorStatus s; YMotor_GetStatus(&s);
    CHECK(s.completed_pulses == 1); /* duplicate did not restart */
    request.payload[0] = 0;
    CHECK(YTest_Handle(&request, &reply) && reply.payload[1] == RS_BAD_PARAM);
    request.payload[0] = 1; request.seq = 8;
    CHECK(YTest_Handle(&request, &reply) && reply.payload[1] == RS_BUSY);
    CHECK(!YTest_Poll(&reply));
    Pulse(); YMotor_IRQHandler(); Pulse(); YMotor_IRQHandler();
    CHECK(YTest_Poll(&reply));
    CHECK(reply.type == Y_TEST_REPORT && reply.seq == 7 && reply.len == 20);
    CHECK(reply.payload[0] == Y_MOTOR_DONE && reply.payload[8] == 3);
    CHECK(reply.payload[2] && !reply.payload[3] && reply.payload[18] && reply.payload[19] == 7);
    CHECK(!YTest_Poll(&reply));
    request.seq = 7;
    CHECK(YTest_Handle(&request, &reply) && reply.type == Y_TEST_REPORT);
    YMotor_GetStatus(&s); CHECK(s.state == Y_MOTOR_DONE && s.completed_pulses == 3);
    request.seq = 8;
    CHECK(YTest_Handle(&request, &reply) && reply.type == RS_ACK);
    YMotor_Abort();
    CHECK(YTest_Poll(&reply) && reply.seq == 8 && reply.payload[0] == Y_MOTOR_ABORTED);
    CHECK(!reply.payload[2]);
    request.seq = 9; gpio.IDR = 1;
    CHECK(YTest_Handle(&request, &reply) && reply.type == RS_NAK && reply.payload[1] == RS_FAULT);
    gpio.IDR = 0;
    CHECK(YTest_Handle(&request, &reply) && reply.type == RS_ACK);
    gpio.IDR = 2; YMotor_Poll();
    CHECK(YTest_Poll(&reply) && reply.payload[0] == Y_MOTOR_LIMIT);
#endif
    request.type = Y_TEST_GET; request.len = 0; request.seq = 100;
    CHECK(YTest_Handle(&request, &reply) && reply.type == Y_TEST_REPORT && reply.seq == 100);
    request.type = RS_MOVE_REL;
    CHECK(!YTest_Handle(&request, &reply));
    return 0;
}
