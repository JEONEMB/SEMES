#include "robot_stage_service.h"
#include "y_motor.h"
#include "app_config.h"
#include "mock_hal.h"
#define CHECK(x) do { if (!(x)) return __LINE__; } while (0)

#if Y_MOTOR_OUTPUT_ALLOWED
static uint32_t Get32(const uint8_t *p)
{ return (uint32_t)p[0] | ((uint32_t)p[1]<<8) | ((uint32_t)p[2]<<16) | ((uint32_t)p[3]<<24); }
static void Pulses(unsigned n)
{
    for (unsigned i=0; i<n; ++i) {
        TIM3->CR1 &= ~TIM_CR1_CEN;
        TIM3->SR |= TIM_SR_UIF;
        YMotor_IRQHandler();
    }
}
static RsFrame Request(RsService *s, uint8_t type, uint8_t seq, int32_t y, int32_t t)
{
    RsFrame q = {.type=type, .seq=seq}, r;
    q.len = type == RS_HOME ? 1 :
        type == RS_MOVE_REL || type == RS_MOVE_ABS || type == RS_SET_SPEED ? 8 : 0;
    rs_put_u32(q.payload, (uint32_t)y); rs_put_u32(q.payload+4, (uint32_t)t);
    RsService_Handle(s, &q, 0, &r);
    return r;
}
#endif

int TestStandardMotion(void)
{
#if Y_MOTOR_OUTPUT_ALLOWED
    gpio.IDR=0; test_rcc.CFGR=test_rcc.DCKCFGR=0;
    YMotor_Init();
    RsService s; RsService_Init(&s);
    RsFrame r;
    r=Request(&s,RS_MOVE_REL,1,1000,0);
    CHECK(r.type==RS_NAK && r.payload[1]==RS_NOT_HOMED);
    r=Request(&s,RS_HOME,2,1,0);
    CHECK(r.type==RS_ACK && s.pending);
    r=Request(&s,RS_GET_STATUS,3,0,0);
    CHECK(r.payload[0]==RS_STATE_HOMING && !r.payload[1]);
    CHECK(RsService_Poll(&s,&r) && r.type==RS_DONE && r.seq==2 && r.payload[0]==RS_OK);
    CHECK(s.homed && !s.pending && s.steps==0);
    r=Request(&s,RS_SET_SPEED,4,1000,0); CHECK(r.type==RS_ACK);
    r=Request(&s,RS_SET_SPEED,5,0,0); CHECK(r.type==RS_ACK && s.speed_um_s==1000);
    r=Request(&s,RS_SET_SPEED,6,2001,0); CHECK(r.payload[1]==RS_OUT_OF_RANGE);
    r=Request(&s,RS_SET_SPEED,7,100,1); CHECK(r.payload[1]==RS_BAD_PARAM);
    r=Request(&s,RS_MOVE_REL,8,1000,0); CHECK(r.type==RS_ACK);
    YMotorStatus m; YMotor_GetStatus(&m);
    CHECK(m.requested_pulses==1600 && m.frequency_hz==1600 && m.dir_high==1);
    Pulses(800);
    r=Request(&s,RS_GET_STATUS,9,0,0);
    CHECK(r.payload[0]==RS_STATE_MOVING && (r.payload[1]&1));
    CHECK(Get32(r.payload+4)==500 && Get32(r.payload+12)==800);
    r=Request(&s,RS_MOVE_REL,8,1000,0); CHECK(r.type==RS_ACK);
    YMotor_GetStatus(&m); CHECK(m.completed_pulses==800); /* retry after query */
    r=Request(&s,RS_MOVE_REL,8,1001,0); CHECK(r.type==RS_ACK); /* same TYPE/SEQ replays */
    r=Request(&s,RS_MOVE_ABS,10,2000,0); CHECK(r.payload[1]==RS_BUSY);
    r=Request(&s,RS_SET_SPEED,11,100,0); CHECK(r.payload[1]==RS_BUSY);
    r=Request(&s,RS_GET_CONFIG,12,0,0); CHECK(r.type==RS_CONFIG && r.len==16);
    Pulses(800);
    CHECK(RsService_Poll(&s,&r) && r.seq==8 && r.payload[0]==RS_OK && Get32(r.payload+4)==1000);
    CHECK(!RsService_Poll(&s,&r));
    r=Request(&s,RS_MOVE_REL,8,1000,0); CHECK(r.type==RS_DONE); /* no second motion */
    r=Request(&s,RS_MOVE_ABS,13,0,0); CHECK(r.type==RS_ACK);
    YMotor_GetStatus(&m); CHECK(m.dir_high==0 && m.requested_pulses==1600);
    Pulses(1600); CHECK(RsService_Poll(&s,&r) && Get32(r.payload+4)==0);
    /* Five successive 1um targets must total 8 steps, not 10. */
    for (unsigned i=0;i<5;++i) {
        r=Request(&s,RS_MOVE_REL,(uint8_t)(20+i),1,0); CHECK(r.type==RS_ACK);
        YMotor_GetStatus(&m); Pulses(m.requested_pulses);
        CHECK(RsService_Poll(&s,&r));
    }
    CHECK(s.steps==8 && s.commanded_um==5);
    r=Request(&s,RS_MOVE_REL,30,-6,0); CHECK(r.payload[1]==RS_OUT_OF_RANGE);
    r=Request(&s,RS_MOVE_REL,31,INT32_MAX,0); CHECK(r.payload[1]==RS_OUT_OF_RANGE);
    r=Request(&s,RS_MOVE_ABS,32,APP_Y_MAX_UM+1,0); CHECK(r.payload[1]==RS_OUT_OF_RANGE);
    r=Request(&s,RS_MOVE_ABS,33,5,1); CHECK(r.payload[1]==RS_BAD_PARAM);
    r=Request(&s,RS_MOVE_ABS,34,5,0); CHECK(r.type==RS_ACK);
    CHECK(RsService_Poll(&s,&r) && r.payload[0]==RS_OK); /* zero-step completion */
    r=Request(&s,RS_MOVE_ABS,35,APP_Y_MAX_UM,0); CHECK(r.type==RS_ACK);
    Pulses(10);
    r=Request(&s,RS_STOP,36,0,0); CHECK(r.type==RS_ACK && r.payload[0]==RS_STOP);
    CHECK(!RsService_Poll(&s,&r));
    for (unsigned i=0;i<700 && (TIM3->CR1 & TIM_CR1_CEN);++i) {
        uint32_t period=TIM3->ARR;
        Pulses(1);
        CHECK(!(TIM3->CR1 & TIM_CR1_CEN) || TIM3->ARR>=period);
    }
    CHECK(RsService_Poll(&s,&r) && r.seq==35 && r.payload[0]==RS_STOPPED);
    CHECK(s.homed && !s.fault && s.steps>18 && s.steps<1000);
    r=Request(&s,RS_GET_CONFIG,37,0,0); CHECK(r.type==RS_CONFIG);
    r=Request(&s,RS_MOVE_REL,60,1000,0); CHECK(r.type==RS_ACK);
    YMotor_Abort(); CHECK(RsService_Poll(&s,&r) && r.payload[0]==RS_RESULT_FAULT);
    CHECK(!s.homed && s.fault); /* external immediate abort invalidates origin */
    r=Request(&s,RS_MOVE_REL,35,1,0); CHECK(r.payload[1]==RS_FAULT);
    r=Request(&s,RS_HOME,38,1,0); CHECK(r.type==RS_ACK);
    CHECK(RsService_Poll(&s,&r) && s.homed && !s.fault);
    r=Request(&s,RS_MOVE_REL,39,1000,0); CHECK(r.type==RS_ACK);
    Pulses(1600); /* STOP after physical completion must not change success */
    r=Request(&s,RS_STOP,40,0,0); CHECK(r.type==RS_ACK);
    CHECK(RsService_Poll(&s,&r) && r.payload[0]==RS_OK);
    /* A SEQ reused after a full query cycle must be a new move, not stale DONE. */
    for (unsigned i=0;i<256;++i) {
        r=Request(&s,RS_GET_STATUS,(uint8_t)(41+i),0,0);
        CHECK(r.type==RS_STATUS);
    }
    r=Request(&s,RS_MOVE_REL,39,1000,0); CHECK(r.type==RS_ACK);
    Pulses(1600); CHECK(RsService_Poll(&s,&r) && Get32(r.payload+4)==2000);
#if Y_MOTOR_USE_LIMITS
    r=Request(&s,RS_MOVE_REL,41,1000,0); CHECK(r.type==RS_ACK);
    r=Request(&s,RS_STOP,42,0,0); CHECK(r.type==RS_ACK);
    gpio.IDR=2; YMotor_Poll();
    CHECK(RsService_Poll(&s,&r) && r.payload[0]==RS_LIMIT && !s.homed);
    CHECK(!RsService_Poll(&s,&r)); /* no extra EVENT for an in-motion fault */
#endif
    RsService_InvalidatePosition(&s); CHECK(!s.homed);
    gpio.IDR=0;
    RsService_Init(&s);
    test_rcc.CFGR=RCC_CFGR_PPRE1; YMotor_Init();
    CHECK(RsService_Poll(&s,&r) && r.type==RS_EVENT && r.payload[0]==RS_EVENT_FAULT);
    CHECK(!RsService_Poll(&s,&r)); /* idle fault event only once */
    r=Request(&s,RS_GET_CONFIG,100,0,0); CHECK(r.type==RS_CONFIG);
    r=Request(&s,RS_GET_STATUS,101,0,0); CHECK(r.payload[0]==RS_STATE_FAULT);
    test_rcc.CFGR=0;
#endif
    return 0;
}

int TestUnconfigured(void)
{
#if Y_MOTOR_OUTPUT_ALLOWED
    gpio.IDR=0; test_rcc.CFGR=test_rcc.DCKCFGR=0;
    YMotor_Init();
    RsService s; RsService_Init(&s);
    RsFrame r=Request(&s,RS_GET_CONFIG,1,0,0);
    CHECK(r.type==RS_CONFIG && r.len==16);
    CHECK(Get32(r.payload)==(uint32_t)APP_Y_MIN_UM);
    CHECK(Get32(r.payload+4)==(uint32_t)APP_Y_MAX_UM);
    for (unsigned i=8;i<16;++i) CHECK(r.payload[i]==0);
    r=Request(&s,RS_HOME,2,1,0);
    CHECK(r.type==RS_NAK && r.payload[1]==RS_FAULT && !s.pending && !s.homed);
    r=Request(&s,RS_MOVE_REL,3,1000,0);
    CHECK(r.type==RS_NAK && r.payload[1]==RS_NOT_HOMED);
    YMotorStatus m; YMotor_GetStatus(&m);
    CHECK(m.state==Y_MOTOR_IDLE && !m.enabled);
    s.fault=1;
    r=Request(&s,RS_GET_CONFIG,4,0,0); CHECK(r.type==RS_CONFIG);
    r=Request(&s,RS_PING,5,0,0); CHECK(r.type==RS_PONG);
    r=Request(&s,RS_GET_STATUS,6,0,0); CHECK(r.type==RS_STATUS && r.payload[0]==RS_STATE_FAULT);
#endif
    return 0;
}
