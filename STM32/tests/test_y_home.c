#include "robot_stage_service.h"
#include "y_home.h"
#include "y_motor.h"
#include "app_config.h"
#include "mock_hal.h"
#define CHECK(x) do { if (!(x)) return __LINE__; } while (0)
extern void TestSetTime(uint32_t now);

#if APP_Y_SWITCH_HOME && Y_MOTOR_OUTPUT_ALLOWED
static void CompletePulse(void)
{
    TIM3->CR1 &= ~TIM_CR1_CEN; TIM3->SR |= TIM_SR_UIF; YMotor_IRQHandler();
}
static void At(uint32_t time, uint32_t inputs)
{ TestSetTime(time); gpio.IDR=inputs; YHome_Poll(); }
static RsFrame Send(RsService *s, uint8_t type, uint8_t seq, uint8_t len, uint32_t value)
{
    RsFrame q={.type=type,.seq=seq,.len=len}, r;
    rs_put_u32(q.payload,value); RsService_Handle(s,&q,0,&r); return r;
}
#endif
int TestPhysicalHome(void)
{
#if APP_Y_SWITCH_HOME && Y_MOTOR_OUTPUT_ALLOWED
    test_rcc.CFGR=test_rcc.DCKCFGR=0; gpio.IDR=0; TestSetTime(0);
    YMotor_Init(); YHome_Init();
    CHECK(YHome_State()==Y_HOME_IDLE); /* no power-on movement */
    CHECK(YHome_Start());
    YMotorStatus m; YMotor_GetStatus(&m);
    CHECK(m.state==Y_MOTOR_RUNNING && m.dir_high==0 && m.frequency_hz==800);
    At(1,1); YMotor_GetStatus(&m); CHECK(m.state==Y_MOTOR_SWITCH && !m.enabled);
    At(5,0); At(9,1); At(18,1);
    YMotor_GetStatus(&m); CHECK(m.state==Y_MOTOR_SWITCH); /* true 10ms stability */
    At(19,1); YMotor_GetStatus(&m);
    CHECK(m.state==Y_MOTOR_RUNNING && m.dir_high==1 && m.frequency_hz==320);
    At(20,0); At(30,0); YMotor_GetStatus(&m);
    CHECK(m.state==Y_MOTOR_RUNNING && m.dir_high==0 && m.frequency_hz==160);
    At(31,1); At(41,1); CHECK(YHome_State()==Y_HOME_OK);
    YMotor_GetStatus(&m); CHECK(!m.enabled);
    /* At left home, rightward travel is allowed. */
    CHECK(YMotor_Start(1,10,160)==Y_MOTOR_OK);
    gpio.IDR=2; YMotor_Poll(); YMotor_GetStatus(&m); CHECK(m.state==Y_MOTOR_LIMIT);
    /* Departure from the right switch is allowed during seek. */
    TestSetTime(100); CHECK(YHome_Start());
    YMotor_GetStatus(&m); CHECK(m.dir_high==0);
    At(101,3); CHECK(YHome_State()==Y_HOME_LIMIT);
    CHECK(!YHome_Start());
    /* Stuck active left switch must be released before claiming home. */
    gpio.IDR=1; TestSetTime(200); CHECK(YHome_Start());
    YMotor_GetStatus(&m); CHECK(m.dir_high==1);
    for (unsigned i=0;i<APP_Y_HOME_RELEASE_PULSES;++i) CompletePulse();
    At(201,1); CHECK(YHome_State()==Y_HOME_FAULT);
    /* Missing contact is bounded by pulse count. */
    gpio.IDR=0; TestSetTime(300); CHECK(YHome_Start());
    for (unsigned i=0;i<APP_Y_HOME_SEEK_PULSES;++i) CompletePulse();
    At(301,0); CHECK(YHome_State()==Y_HOME_FAULT);
    /* Timeout also covers stalled timer IRQ and uint32 tick rollover. */
    TestSetTime(0xffffff00U); CHECK(YHome_Start());
    At(0xffffff00U+APP_Y_HOME_TIMEOUT_MS,0); CHECK(YHome_State()==Y_HOME_FAULT);
    /* HOME protocol: ACK, HOMING, STOP/DONE and successful zero assignment. */
    RsService s; RsService_Init(&s); TestSetTime(1000);
    RsFrame r=Send(&s,RS_HOME,1,1,1); CHECK(r.type==RS_ACK && !s.homed);
    r=Send(&s,RS_GET_STATUS,2,0,0); CHECK(r.payload[0]==RS_STATE_HOMING);
    r=Send(&s,RS_STOP,3,0,0); CHECK(r.type==RS_ACK);
    for (unsigned i=0;i<700 && (TIM3->CR1&TIM_CR1_CEN);++i) CompletePulse();
    CHECK(RsService_Poll(&s,&r) && r.type==RS_DONE && r.seq==1 && r.payload[0]==RS_STOPPED && !s.homed);
    TestSetTime(2000); gpio.IDR=0;
    r=Send(&s,RS_HOME,4,1,1); CHECK(r.type==RS_ACK);
    At(2001,1); At(2011,1); At(2012,0); At(2022,0); At(2023,1); At(2033,1);
    CHECK(RsService_Poll(&s,&r) && r.seq==4 && r.payload[0]==RS_OK && s.homed && !s.fault);
    CHECK(s.steps==0 && s.commanded_um==0 && !RsService_Poll(&s,&r));
    r=Send(&s,RS_HOME,4,1,1); CHECK(r.type==RS_DONE && YHome_State()==Y_HOME_OK);
    r=Send(&s,RS_MOVE_REL,5,8,1000); CHECK(r.type==RS_ACK);
    gpio.IDR=0;
    for (unsigned i=0;i<1600;++i) CompletePulse();
    CHECK(RsService_Poll(&s,&r) && r.payload[0]==RS_OK && s.steps==1600);
    /* HOME is interruptible during switch settling as well. */
    r=Send(&s,RS_HOME,6,1,1); CHECK(r.type==RS_ACK);
    At(3000,1);
    r=Send(&s,RS_STOP,7,0,0); CHECK(r.type==RS_ACK);
    CHECK(RsService_Poll(&s,&r) && r.payload[0]==RS_STOPPED && !s.homed);
#endif
    return 0;
}
