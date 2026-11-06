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
static void Pulses(unsigned count) { while (count--) CompletePulse(); }
static void At(uint32_t time, uint32_t inputs)
{ TestSetTime(time); gpio.IDR=inputs; YHome_Poll(); }
static void Reset(uint32_t bits)
{
    YMotor_Abort(); test_rcc.CFGR=test_rcc.DCKCFGR=0;
    YMotor_Init(); YHome_Init(); gpio.IDR=bits; TestSetTime(0);
}
static RsFrame Send(RsService *s, uint8_t type, uint8_t seq, uint8_t len, uint32_t value)
{
    RsFrame q={.type=type,.seq=seq,.len=len}, r;
    rs_put_u32(q.payload,value); RsService_Handle(s,&q,(uint8_t)gpio.IDR,&r); return r;
}
static uint32_t Get32(const uint8_t *p)
{ return (uint32_t)p[0] | (uint32_t)p[1]<<8 | (uint32_t)p[2]<<16 | (uint32_t)p[3]<<24; }
#endif

int TestPhysicalHome(void)
{
#if APP_Y_SWITCH_HOME && Y_MOTOR_OUTPUT_ALLOWED
    Reset(0);
    CHECK(YHome_State()==Y_HOME_IDLE && !(TIM3->CR1 & TIM_CR1_CEN));
    CHECK(YHome_Start());
    YMotorStatus m; YMotor_GetStatus(&m);
    CHECK(m.state==Y_MOTOR_RUNNING && m.dir_high==0 && m.frequency_hz==800);
    CHECK(m.requested_pulses==83200);
    At(1,1); YMotor_GetStatus(&m); CHECK(m.state==Y_MOTOR_SWITCH && !m.enabled);
    At(5,0); At(9,1); At(18,1);
    CHECK(y_home_debug.phase==Y_HOME_PRESS_STABLE && !(TIM3->CR1 & TIM_CR1_CEN));
    At(19,1); YMotor_GetStatus(&m);
    CHECK(m.state==Y_MOTOR_RUNNING && m.dir_high==1 && m.frequency_hz==160);
    CHECK(m.requested_pulses==3200);
    At(20,0); At(24,1); At(25,0); At(34,0);
    CHECK(YHome_State()==Y_HOME_RUNNING && y_home_debug.phase==Y_HOME_RELEASE_STABLE);
    CHECK(!(TIM3->CR1 & TIM_CR1_CEN));
    At(35,0); CHECK(YHome_State()==Y_HOME_OK);
    YMotor_GetStatus(&m); CHECK(!m.enabled && m.dir_high==1 && m.state==Y_MOTOR_SWITCH);
    CHECK(y_home_debug.reason==Y_HOME_REASON_NONE && y_home_debug.switch_bits==0);
    At(100,0); CHECK(YHome_State()==Y_HOME_OK && !(TIM3->CR1 & TIM_CR1_CEN));

    /* Initially pressed: no seek pulse; qualify first, release once, finish. */
    Reset(1); CHECK(YHome_Start());
    CHECK(y_home_debug.phase==Y_HOME_PRESS_STABLE && !(TIM3->CR1 & TIM_CR1_CEN));
    At(9,1); CHECK(!(TIM3->CR1 & TIM_CR1_CEN));
    At(10,1); CHECK(y_home_debug.phase==Y_HOME_RELEASE && y_home_debug.dir_high==1);
    Pulses(7); At(54,0); At(64,0);
    CHECK(YHome_State()==Y_HOME_OK && y_home_debug.completed_pulses==7);
    CHECK(!(TIM3->CR1 & TIM_CR1_CEN));

    /* Both limits rejected before any pulse; right-end departure is allowed. */
    Reset(3); CHECK(!YHome_Start() && y_home_debug.reason==Y_HOME_REASON_BOTH_LIMITS);
    Reset(2); CHECK(YHome_Start() && y_home_debug.dir_high==0);
    At(1,3); CHECK(YHome_State()==Y_HOME_LIMIT);
    CHECK(y_home_debug.reason==Y_HOME_REASON_BOTH_LIMITS);
    Reset(1); CHECK(YHome_Start()); At(10,1);
    At(11,2); CHECK(YHome_State()==Y_HOME_LIMIT && y_home_debug.reason==Y_HOME_REASON_RIGHT_LIMIT);
    /* Right switch still guards release qualification after the pulse stop. */
    Reset(1); CHECK(YHome_Start()); At(10,1); At(11,0); At(12,2);
    CHECK(YHome_State()==Y_HOME_LIMIT);
#endif
    return 0;
}

int TestHomeBoundaries(void)
{
#if APP_Y_SWITCH_HOME && Y_MOTOR_OUTPUT_ALLOWED
    /* Contact on the final allowed seek pulse must enter release, not fail. */
    Reset(0); CHECK(YHome_Start());
    Pulses(APP_Y_HOME_SEEK_PULSES-1); gpio.IDR=1; CompletePulse();
    At(104000,1); At(104010,1);
    CHECK(YHome_State()==Y_HOME_RUNNING && y_home_debug.phase==Y_HOME_RELEASE);
    At(104100,0); At(104110,0); CHECK(YHome_State()==Y_HOME_OK);
    /* No contact: never output a pulse beyond the budget. */
    Reset(0); CHECK(YHome_Start()); Pulses(APP_Y_HOME_SEEK_PULSES); At(104000,0);
    CHECK(YHome_State()==Y_HOME_FAULT && y_home_debug.reason==Y_HOME_REASON_TRAVEL_EXHAUSTED);
    CHECK(y_home_debug.completed_pulses==83200 && !(TIM3->CR1 & TIM_CR1_CEN));

    /* No obsolete 10s timeout; release at the last of 3200 pulses (~20s) succeeds. */
    Reset(1); CHECK(YHome_Start()); At(10,1);
    for (unsigned i=1;i<APP_Y_HOME_RELEASE_PULSES;++i) {
        CompletePulse(); At(10+(i*1000U)/160U,1);
        CHECK(YHome_State()==Y_HOME_RUNNING);
    }
    gpio.IDR=0; CompletePulse(); At(20010,0); At(20020,0);
    CHECK(YHome_State()==Y_HOME_OK && y_home_debug.completed_pulses==3200);
    Reset(1); CHECK(YHome_Start()); At(10,1);
    Pulses(APP_Y_HOME_RELEASE_PULSES); At(20010,1);
    CHECK(YHome_State()==Y_HOME_FAULT && y_home_debug.reason==Y_HOME_REASON_TRAVEL_EXHAUSTED);

    /* Stability succeeds at exactly 200ms; bounce must restart the 10ms timer. */
    Reset(1); CHECK(YHome_Start()); At(10,1); At(20,0);
    At(21,1); At(210,0); At(219,0);
    CHECK(YHome_State()==Y_HOME_RUNNING); At(220,0); CHECK(YHome_State()==Y_HOME_OK);
    Reset(1); CHECK(YHome_Start()); At(10,1); At(20,0);
    At(21,1); At(211,0); At(221,0);
    CHECK(YHome_State()==Y_HOME_FAULT && y_home_debug.reason==Y_HOME_REASON_UNSTABLE);
    Reset(1); CHECK(YHome_Start()); At(1,0); At(201,0);
    CHECK(YHome_State()==Y_HOME_FAULT && y_home_debug.phase==Y_HOME_PRESS_STABLE);
    CHECK(y_home_debug.reason==Y_HOME_REASON_UNSTABLE);

    /* Global 120s remains strict; wraparound and stalled IRQ do not defeat it. */
    Reset(0); TestSetTime(0xffffff00U); CHECK(YHome_Start());
    At(0xffffff00U+APP_Y_HOME_TIMEOUT_MS,0);
    CHECK(YHome_State()==Y_HOME_FAULT && y_home_debug.reason==Y_HOME_REASON_TIMEOUT);
    Reset(0); CHECK(YHome_Start()); At(119978,1); At(119988,1);
    At(119989,0); At(119999,0); CHECK(YHome_State()==Y_HOME_OK);
    Reset(0); CHECK(YHome_Start()); At(119979,1); At(119989,1);
    At(119990,0); At(120000,0);
    CHECK(YHome_State()==Y_HOME_FAULT && y_home_debug.reason==Y_HOME_REASON_TIMEOUT);

    /* Inject readiness loss while stopped to exercise a failed phase start. */
    Reset(1); CHECK(YHome_Start());
    test_rcc.CFGR=RCC_CFGR_PPRE1; YMotor_Init(); At(10,1);
    CHECK(YHome_State()==Y_HOME_FAULT && y_home_debug.reason==Y_HOME_REASON_START_FAILED);
    test_rcc.CFGR=0;
#endif
    return 0;
}

int TestHomeStop(void)
{
#if APP_Y_SWITCH_HOME && Y_MOTOR_OUTPUT_ALLOWED
    for (unsigned stage=0;stage<4;++stage) {
        Reset(stage ? 1 : 0);
        RsService s; RsService_Init(&s); s.homed=1; s.steps=1234; s.commanded_um=771;
        RsFrame r=Send(&s,RS_HOME,1,1,1);
        CHECK(r.type==RS_ACK && !s.homed);
        if (stage>=2) At(10,1);
        if (stage==3) At(11,0);
        r=Send(&s,RS_STOP,2,0,0); CHECK(r.type==RS_ACK);
        r=Send(&s,RS_STOP,2,0,0); CHECK(r.type==RS_ACK); /* repeated STOP */
        for (unsigned i=0;i<700 && (TIM3->CR1&TIM_CR1_CEN);++i) CompletePulse();
        CHECK(RsService_Poll(&s,&r) && r.type==RS_DONE && r.seq==1 && r.payload[0]==RS_STOPPED);
        CHECK(!s.homed && s.steps==1234 && !s.pending && !RsService_Poll(&s,&r));
        CHECK(!(TIM3->CR1&TIM_CR1_CEN) && y_home_debug.reason==Y_HOME_REASON_STOP);
    }
    /* Limit detected while stopping still wins over STOPPED. */
    Reset(1); RsService s; RsService_Init(&s);
    RsFrame r=Send(&s,RS_HOME,1,1,1); CHECK(r.type==RS_ACK); At(10,1);
    r=Send(&s,RS_STOP,2,0,0); CHECK(r.type==RS_ACK); At(11,2);
    CHECK(RsService_Poll(&s,&r) && r.payload[0]==RS_LIMIT && !s.homed && s.fault);
#endif
    return 0;
}

int TestHomeService(void)
{
#if APP_Y_SWITCH_HOME && Y_MOTOR_OUTPUT_ALLOWED
    Reset(3); RsService s; RsService_Init(&s); s.fault=1;
    RsFrame r=Send(&s,RS_HOME,1,1,1);
    CHECK(r.type==RS_NAK && r.payload[1]==RS_FAULT && !s.pending && s.fault);
    gpio.IDR=1; /* Previous LIMIT state must permit a new HOME after clearing inputs. */
    s.steps=1600; s.commanded_um=1000; s.start_steps=800; s.target_um=2000;
    r=Send(&s,RS_HOME,2,1,1); CHECK(r.type==RS_ACK && !s.homed && s.fault);
    r=Send(&s,RS_GET_STATUS,3,0,0); CHECK(r.payload[0]==RS_STATE_HOMING);
    r=Send(&s,RS_HOME,4,1,1); CHECK(r.type==RS_NAK && r.payload[1]==RS_BUSY);
    At(10,1); Pulses(8); At(60,0); At(70,0);
    CHECK(RsService_Poll(&s,&r) && r.type==RS_DONE && r.seq==2 && r.len==12);
    CHECK(r.payload[0]==RS_OK && r.payload[1]==RS_HOME && Get32(r.payload+4)==0);
    CHECK(s.homed && !s.fault && !s.pending && !s.steps && !s.commanded_um && !s.start_steps && !s.target_um);
    CHECK(!RsService_Poll(&s,&r)); /* exactly one unsolicited completion */
    r=Send(&s,RS_GET_STATUS,5,0,0);
    CHECK(r.payload[0]==RS_STATE_IDLE && r.payload[1]==1 && Get32(r.payload+4)==0 && Get32(r.payload+12)==0);
    r=Send(&s,RS_HOME,2,1,1); CHECK(r.type==RS_DONE); /* cached response, no new motion */
    CHECK(!(TIM3->CR1&TIM_CR1_CEN));
    /* STATUS switch bits reflect actual input even after successful HOME. */
    gpio.IDR=2; r=Send(&s,RS_GET_STATUS,6,0,0); CHECK((r.payload[1]&12U)==8U);
    gpio.IDR=0;
    r=Send(&s,RS_MOVE_REL,7,8,1000); CHECK(r.type==RS_ACK);
    Pulses(1600); CHECK(RsService_Poll(&s,&r) && r.payload[0]==RS_OK && s.steps==1600);
    r=Send(&s,RS_MOVE_ABS,8,8,0); CHECK(r.type==RS_ACK);
    Pulses(1600); CHECK(RsService_Poll(&s,&r) && r.payload[0]==RS_OK && s.steps==0);
    /* If the real switch is encountered on return, preserve the normal limit check. */
    r=Send(&s,RS_MOVE_REL,9,8,1000); CHECK(r.type==RS_ACK);
    Pulses(1600); CHECK(RsService_Poll(&s,&r));
    r=Send(&s,RS_MOVE_ABS,10,8,0); CHECK(r.type==RS_ACK);
    Pulses(1599); gpio.IDR=1; CompletePulse();
    CHECK(RsService_Poll(&s,&r) && r.payload[0]==RS_LIMIT && !s.homed && s.fault);
    /* Failed recovery does not erase the previous fault. */
    r=Send(&s,RS_HOME,11,1,1); CHECK(r.type==RS_ACK);
    At(80,1); Pulses(3200); At(20080,1);
    CHECK(RsService_Poll(&s,&r) && r.payload[0]==RS_RESULT_FAULT && s.fault && !s.homed);
    r=Send(&s,RS_HOME,12,1,3); CHECK(r.type==RS_NAK && r.payload[1]==RS_BAD_PARAM);
    test_rcc.CFGR=RCC_CFGR_PPRE1; YMotor_Init();
    r=Send(&s,RS_HOME,13,1,1); CHECK(r.type==RS_NAK && r.payload[1]==RS_FAULT);
    test_rcc.CFGR=0;
#endif
    return 0;
}
