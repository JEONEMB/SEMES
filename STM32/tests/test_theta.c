#include "robot_stage_service.h"
#include "theta_motor.h"
#include "theta_motor_port.h"
#include "theta_home.h"
#include "theta_units.h"
#include "y_motor.h"
#include "y_home.h"
#include "app_config.h"
#include "mock_hal.h"
#define CHECK(x) do { if (!(x)) return __LINE__; } while (0)
extern void TestSetTime(uint32_t now);
static void TPulse(void) { TIM1->CR1&=~TIM_CR1_CEN; TIM1->SR|=TIM_SR_UIF; ThetaMotor_IRQHandler(); }
static void YPulse(void) { TIM3->CR1&=~TIM_CR1_CEN; TIM3->SR|=TIM_SR_UIF; YMotor_IRQHandler(); }
static void TFinish(void) { for(unsigned i=0;i<20000 && (TIM1->CR1&TIM_CR1_CEN);++i) TPulse(); }
static void Reset(uint32_t bits) {
    ThetaMotor_Abort(); YMotor_Abort(); test_rcc.CFGR=test_rcc.DCKCFGR=0;
    ThetaMotor_Init(); YMotor_Init(); ThetaHome_Init(); YHome_Init();
    gpio.IDR=bits; TestSetTime(0);
}
static void At(uint32_t time,uint32_t bits) { TestSetTime(time); gpio.IDR=bits; ThetaHome_Poll(); }
static int32_t Get(const uint8_t *p) {
    return (int32_t)((uint32_t)p[0]|(uint32_t)p[1]<<8|(uint32_t)p[2]<<16|(uint32_t)p[3]<<24);
}
static RsFrame Send(RsService *s,uint8_t kind,uint8_t seq,int32_t y,int32_t t) {
    RsFrame q={.type=kind,.seq=seq},r;
    q.len=kind==RS_HOME?1:kind==RS_SET_SPEED||kind==RS_MOVE_REL||kind==RS_MOVE_ABS?8:0;
    rs_put_u32(q.payload,(uint32_t)y); rs_put_u32(q.payload+4,(uint32_t)t);
    uint8_t inputs=(uint8_t)((gpio.IDR&3U)|((gpio.IDR&THETA_HOME_Pin)?4:0));
    RsService_Handle(s,&q,inputs,&r); return r;
}
static void HomeEdges(uint32_t base,uint8_t mask) {
    uint32_t bits=(mask&1?1:0)|(mask&2?THETA_HOME_Pin:0);
    TestSetTime(base); gpio.IDR=bits; YHome_Poll(); ThetaHome_Poll();
    TestSetTime(base+10); YHome_Poll(); ThetaHome_Poll();
    TestSetTime(base+11); gpio.IDR=0; YHome_Poll(); ThetaHome_Poll();
    TestSetTime(base+21); YHome_Poll(); ThetaHome_Poll();
}
int TestThetaMotor(void) {
    Reset(0); ThetaMotorStatus m;
    CHECK(ThetaUnits_ToSteps(-5000)==-4444 && ThetaUnits_ToSteps(-10000)==-8889);
    CHECK(ThetaUnits_FromSteps(-4444)==-5000 && ThetaUnits_ToHz(1000)==889);
    CHECK(ThetaUnits_ValidSpeed(3600) && !ThetaUnits_ValidSpeed(3601));
    CHECK(!ThetaUnits_ValidSpeed(1) && !ThetaUnits_ValidSpeed(UINT32_MAX));
    CHECK(ThetaMotor_Start(0,3,889)==THETA_MOTOR_OK);
    CHECK(TIM1->RCR==0 && (TIM1->BDTR&TIM_BDTR_MOE));
    CHECK(TIM1->ARR+1-TIM1->CCR1==5 && TIM1->CCR1>=2000);
    CHECK(!(TIM3->CR1&TIM_CR1_CEN));
    CHECK(ThetaMotor_Start(1,1,100)==THETA_MOTOR_BUSY);
    /* Model TIM1 PWM2/OPM including MOE; delayed IRQ cannot emit extra pulses. */
    unsigned rises=0,falls=0,high=0,level=0;
    for(unsigned pulse=0;pulse<2;++pulse) {
        for(unsigned i=0;i<65536 && (TIM1->CR1&TIM_CR1_CEN);++i) {
            if(++TIM1->CNT>TIM1->ARR) {
                TIM1->CNT=0; TIM1->SR|=TIM_SR_UIF;
                if(TIM1->CR1&TIM_CR1_OPM) TIM1->CR1&=~TIM_CR1_CEN;
            }
            unsigned next=(TIM1->BDTR&TIM_BDTR_MOE) && (TIM1->CCER&TIM_CCER_CC1E) &&
                (TIM1->CCMR1&TIM_CCMR1_OC1M)==TIM_OCMODE_PWM2 && TIM1->CNT>=TIM1->CCR1;
            if(next && !level) ++rises;
            if(!next && level) ++falls;
            high+=next; level=next;
        }
        CHECK(rises==pulse+1 && falls==pulse+1 && high==(pulse+1)*5);
        CHECK(!(TIM1->CR1&TIM_CR1_CEN));
        ThetaMotor_IRQHandler();
    }
    ThetaMotor_GetStatus(&m); CHECK(m.completed_pulses==2);
    gpio.IDR=THETA_HOME_Pin; TPulse(); /* index does not stop normal motion */
    ThetaMotor_GetStatus(&m); CHECK(m.state==THETA_MOTOR_DONE && m.completed_pulses==3 && !m.enabled);
    CHECK(!(TIM1->CR1&TIM_CR1_CEN) && !TIM1->DIER);
    ThetaMotor_IRQHandler(); ThetaMotor_GetStatus(&m); CHECK(m.completed_pulses==3);
    CHECK(ThetaMotor_Start(0,0,889)==THETA_MOTOR_BAD_ARGUMENT);
    CHECK(ThetaMotor_Start(0,1000,3201)==THETA_MOTOR_BAD_ARGUMENT);
    CHECK(ThetaMotor_Start(0,1000,3200)==THETA_MOTOR_OK);
    TPulse(); ThetaMotor_Decelerate(); uint32_t prev=0;
    for(unsigned i=0;i<1000 && (TIM1->CR1&TIM_CR1_CEN);++i) {
        TPulse(); CHECK(!(TIM1->CR1&TIM_CR1_CEN) || TIM1->ARR>=prev); prev=TIM1->ARR;
        ThetaMotor_Decelerate(); /* retries cannot restart deceleration */
    }
    ThetaMotor_GetStatus(&m); CHECK(m.state==THETA_MOTOR_STOPPED && m.count_exact && !m.enabled);
    CHECK(ThetaMotor_Start(0,2,100)==THETA_MOTOR_OK);
    ThetaMotor_Abort(); ThetaMotor_GetStatus(&m); CHECK(!m.count_exact && m.state==THETA_MOTOR_ABORTED);
    test_rcc.CFGR=RCC_CFGR_PPRE2; ThetaMotor_Init(); ThetaMotor_GetStatus(&m); CHECK(!m.ready);
    return 0;
}
int TestThetaHome(void) {
    if (APP_THETA_MANUAL_HOME) {
        for (unsigned pressed=0;pressed<2;++pressed) {
            Reset(pressed?THETA_HOME_Pin:0);
            CHECK(ThetaHome_Start());
            CHECK(ThetaHome_State()==THETA_HOME_OK);
            CHECK(!(TIM1->CR1&TIM_CR1_CEN) && !TIM1->DIER);
            CHECK(!theta_home_debug.requested_pulses && !theta_home_debug.enabled);
            CHECK(theta_home_debug.switch_bits==pressed);
        }
        test_rcc.CFGR=RCC_CFGR_PPRE2; ThetaMotor_Init();
        CHECK(!ThetaHome_Start() && ThetaHome_State()==THETA_HOME_FAULT);
        return 0;
    }
    Reset(0); CHECK(ThetaHome_Start());
    ThetaMotorStatus m; ThetaMotor_GetStatus(&m);
    CHECK(m.dir_high==APP_THETA_CCW_DIR_HIGH && m.requested_pulses==10667 && m.frequency_hz==889);
    TPulse(); At(1,THETA_HOME_Pin); At(5,0); At(6,THETA_HOME_Pin); At(15,THETA_HOME_Pin);
    CHECK(theta_home_debug.phase==THETA_HOME_PRESS_STABLE && theta_home_debug.enabled);
    At(16,THETA_HOME_Pin); ThetaMotor_GetStatus(&m);
    CHECK(m.dir_high!=APP_THETA_CCW_DIR_HIGH && m.requested_pulses==1778 && m.frequency_hz==444);
    At(17,0); At(20,THETA_HOME_Pin); At(21,0); At(30,0);
    CHECK(ThetaHome_State()==THETA_HOME_RUNNING && !theta_home_debug.enabled);
    At(31,0); CHECK(ThetaHome_State()==THETA_HOME_OK && !(TIM1->CR1&TIM_CR1_CEN));
    Reset(THETA_HOME_Pin); CHECK(ThetaHome_Start()); At(10,THETA_HOME_Pin);
    CHECK(theta_home_debug.phase==THETA_HOME_RELEASE);
    TFinish(); At(5000,THETA_HOME_Pin);
    CHECK(ThetaHome_State()==THETA_HOME_FAULT && theta_home_debug.reason==THETA_HOME_REASON_TRAVEL_EXHAUSTED);
    Reset(0); CHECK(ThetaHome_Start()); TFinish(); At(13000,0);
    CHECK(ThetaHome_State()==THETA_HOME_FAULT);
    /* Switch edge on final permitted pulse wins over budget exhaustion. */
    Reset(0); CHECK(ThetaHome_Start());
    for(unsigned i=1;i<10667;++i) TPulse();
    gpio.IDR=THETA_HOME_Pin; TPulse(); At(12000,THETA_HOME_Pin); At(12010,THETA_HOME_Pin);
    for(unsigned i=1;i<1778;++i) TPulse();
    gpio.IDR=0; TPulse(); At(16000,0); At(16010,0); CHECK(ThetaHome_State()==THETA_HOME_OK);
    /* STOP in all four stages never reports successful HOME. */
    for(unsigned stage=0;stage<4;++stage) {
        Reset(stage?THETA_HOME_Pin:0); CHECK(ThetaHome_Start());
        if(stage>=2) At(10,THETA_HOME_Pin);
        if(stage==3) At(11,0);
        ThetaHome_Stop(); TFinish(); At(20,stage==1||stage==2?THETA_HOME_Pin:0);
        CHECK(ThetaHome_State()==THETA_HOME_STOPPED && !theta_home_debug.enabled);
    }
    Reset(0); CHECK(ThetaHome_Start()); At(APP_THETA_HOME_TIMEOUT_MS,0);
    CHECK(ThetaHome_State()==THETA_HOME_FAULT && theta_home_debug.reason==THETA_HOME_REASON_TIMEOUT);
    Reset(THETA_HOME_Pin); CHECK(ThetaHome_Start()); At(1,0); At(201,0);
    CHECK(ThetaHome_State()==THETA_HOME_FAULT && theta_home_debug.reason==THETA_HOME_REASON_UNSTABLE);
    Reset(THETA_HOME_Pin); CHECK(ThetaHome_Start()); At(10,THETA_HOME_Pin); At(11,0); At(12,THETA_HOME_Pin); At(212,THETA_HOME_Pin);
    CHECK(ThetaHome_State()==THETA_HOME_FAULT && theta_home_debug.reason==THETA_HOME_REASON_UNSTABLE);
    return 0;
}
int TestThetaService(void) {
    Reset(0); RsService s; RsService_Init(&s); RsFrame r;
    r=Send(&s,RS_MOVE_REL,1,0,-1000); CHECK(r.type==RS_NAK && r.payload[1]==RS_NOT_HOMED);
    r=Send(&s,RS_HOME,2,2,0); CHECK(r.type==RS_ACK && !s.theta_homed && !s.homed);
    r=Send(&s,RS_GET_STATUS,3,0,0); CHECK(r.payload[0]==RS_STATE_HOMING);
    HomeEdges(1,2); CHECK(RsService_Poll(&s,&r) && r.seq==2 && r.payload[0]==RS_OK);
    CHECK(s.theta_homed && !s.homed && !s.theta_steps && !s.theta_commanded);
    CHECK(!RsService_Poll(&s,&r));
    r=Send(&s,RS_HOME,2,2,0); CHECK(r.type==RS_DONE);
    r=Send(&s,RS_GET_STATUS,4,0,0); CHECK(r.payload[0]==RS_STATE_IDLE && r.payload[1]==2);
    gpio.IDR=THETA_HOME_Pin; r=Send(&s,RS_GET_STATUS,5,0,0); CHECK(r.payload[1]==18); gpio.IDR=0;
    r=Send(&s,RS_GET_CONFIG,6,0,0); CHECK(Get(r.payload+8)==-10000 && Get(r.payload+12)==10000);
    r=Send(&s,RS_SET_SPEED,7,1000,3601); CHECK(r.type==RS_NAK && s.speed_um_s==100 && s.theta_speed==1000);
    r=Send(&s,RS_SET_SPEED,8,0,1000); CHECK(r.type==RS_ACK);
    r=Send(&s,RS_MOVE_REL,9,0,-5000); CHECK(r.type==RS_ACK && !s.motor_started && s.theta_started);
    ThetaMotorStatus m; ThetaMotor_GetStatus(&m); CHECK(m.requested_pulses==4444 && m.frequency_hz==889);
    for(unsigned i=0;i<100;++i) TPulse();
    r=Send(&s,RS_GET_STATUS,10,0,0); CHECK(Get(r.payload+16)==-100 && Get(r.payload+8)==-113);
    r=Send(&s,RS_MOVE_REL,9,0,-5000); CHECK(r.type==RS_ACK);
    ThetaMotor_GetStatus(&m); CHECK(m.completed_pulses==100);
    r=Send(&s,RS_MOVE_REL,11,0,-1); CHECK(r.type==RS_NAK && r.payload[1]==RS_BUSY);
    TFinish(); CHECK(RsService_Poll(&s,&r) && r.seq==9 && r.payload[0]==RS_OK && Get(r.payload+8)==-5000);
    r=Send(&s,RS_MOVE_REL,9,0,-5000); CHECK(r.type==RS_DONE);
    r=Send(&s,RS_MOVE_REL,12,0,-5001); CHECK(r.type==RS_NAK && r.payload[1]==RS_OUT_OF_RANGE);
    r=Send(&s,RS_MOVE_ABS,13,0,0); CHECK(r.type==RS_ACK); TFinish(); CHECK(RsService_Poll(&s,&r) && !s.theta_steps);
    for(unsigned i=0;i<9;++i) {
        r=Send(&s,RS_MOVE_REL,(uint8_t)(20+i),0,-1); CHECK(r.type==RS_ACK);
        TFinish(); CHECK(RsService_Poll(&s,&r));
    }
    CHECK(s.theta_commanded==-9 && s.theta_steps==-8); /* no accumulated rounding */
    r=Send(&s,RS_MOVE_REL,40,0,-5000); CHECK(r.type==RS_ACK); TPulse();
    r=Send(&s,RS_STOP,41,0,0); CHECK(r.type==RS_ACK); TFinish();
    CHECK(RsService_Poll(&s,&r) && r.seq==40 && r.payload[0]==RS_STOPPED && s.theta_homed);
    r=Send(&s,RS_MOVE_REL,42,0,-1000); CHECK(r.type==RS_ACK); ThetaMotor_Abort();
    CHECK(RsService_Poll(&s,&r) && r.payload[0]==RS_RESULT_FAULT && !s.theta_homed);
    r=Send(&s,RS_SET_SPEED,43,0,1000); CHECK(r.type==RS_NAK && r.payload[1]==RS_FAULT);
    r=Send(&s,RS_HOME,44,2,0); CHECK(r.type==RS_ACK); HomeEdges(100,2);
    CHECK(RsService_Poll(&s,&r) && r.payload[0]==RS_OK && !s.theta_fault);
    r=Send(&s,RS_MOVE_REL,45,0,INT32_MIN); CHECK(r.type==RS_NAK && r.payload[1]==RS_OUT_OF_RANGE);
    r=Send(&s,RS_MOVE_REL,46,0,INT32_MAX); CHECK(r.type==RS_NAK && r.payload[1]==RS_OUT_OF_RANGE);
    r=Send(&s,RS_MOVE_ABS,47,0,10000); CHECK(r.type==RS_ACK);
    TFinish(); CHECK(RsService_Poll(&s,&r) && s.theta_commanded==10000);
    r=Send(&s,RS_MOVE_REL,48,0,1); CHECK(r.type==RS_NAK && r.payload[1]==RS_OUT_OF_RANGE);
    r=Send(&s,RS_MOVE_ABS,49,0,-10000); CHECK(r.type==RS_ACK);
    TFinish(); CHECK(RsService_Poll(&s,&r) && s.theta_commanded==-10000);
    r=Send(&s,RS_MOVE_REL,50,0,-1); CHECK(r.type==RS_NAK && r.payload[1]==RS_OUT_OF_RANGE);
    if (APP_THETA_MANUAL_HOME) {
        gpio.IDR=THETA_HOME_Pin;
        r=Send(&s,RS_HOME,51,2,0); CHECK(r.type==RS_ACK);
        CHECK(!(TIM1->CR1&TIM_CR1_CEN));
        CHECK(RsService_Poll(&s,&r) && r.seq==51 && r.payload[0]==RS_OK);
        CHECK(s.theta_homed && !s.theta_steps && !s.theta_commanded);
        r=Send(&s,RS_HOME,52,2,0); CHECK(r.type==RS_ACK);
        r=Send(&s,RS_STOP,53,0,0); CHECK(r.type==RS_ACK);
        CHECK(RsService_Poll(&s,&r) && r.seq==52 && r.payload[0]==RS_STOPPED && !s.theta_homed);
    }
    return 0;
}
int TestDualAxes(void) {
    Reset(0); RsService s; RsService_Init(&s); RsFrame r;
    r=Send(&s,RS_HOME,1,3,0); CHECK(r.type==RS_ACK); HomeEdges(1,3);
    CHECK(RsService_Poll(&s,&r) && r.payload[0]==RS_OK && s.homed && s.theta_homed);
    r=Send(&s,RS_SET_SPEED,2,1000,1000); CHECK(r.type==RS_ACK);
    r=Send(&s,RS_MOVE_REL,3,1000,-1000); CHECK(r.type==RS_ACK);
    CHECK((TIM1->CR1&TIM_CR1_CEN) && (TIM3->CR1&TIM_CR1_CEN));
    TFinish(); CHECK(!RsService_Poll(&s,&r)); /* Theta done, Y must still run */
    for(unsigned i=0;i<1600;++i) YPulse();
    CHECK(RsService_Poll(&s,&r) && r.seq==3 && Get(r.payload+4)==1000 && Get(r.payload+8)==-1000);
    CHECK(!RsService_Poll(&s,&r));
    r=Send(&s,RS_MOVE_REL,4,1000,-1000); CHECK(r.type==RS_ACK);
    YPulse(); TPulse(); r=Send(&s,RS_STOP,5,0,0); CHECK(r.type==RS_ACK);
    for(unsigned i=0;i<1000;++i) {
        if(TIM3->CR1&TIM_CR1_CEN) YPulse();
        if(TIM1->CR1&TIM_CR1_CEN) TPulse();
    }
    CHECK(RsService_Poll(&s,&r) && r.seq==4 && r.payload[0]==RS_STOPPED && s.homed && s.theta_homed);
    r=Send(&s,RS_MOVE_REL,6,1000,-1000); CHECK(r.type==RS_ACK);
    gpio.IDR=2; YMotor_Poll(); RsService_Maintain(&s);
    CHECK(!(TIM1->CR1&TIM_CR1_CEN) && !(TIM3->CR1&TIM_CR1_CEN));
    CHECK(RsService_Poll(&s,&r) && r.seq==6 && r.payload[0]==RS_LIMIT && !s.theta_homed && !s.homed);
    gpio.IDR=0; r=Send(&s,RS_HOME,7,3,0); CHECK(r.type==RS_ACK); HomeEdges(100,3);
    CHECK(RsService_Poll(&s,&r) && r.payload[0]==RS_OK && !s.fault && !s.theta_fault);
    /* All-axis preflight: invalid theta target never starts Y. */
    r=Send(&s,RS_MOVE_REL,8,1000,-10001); CHECK(r.type==RS_NAK && !(TIM3->CR1&TIM_CR1_CEN));
    r=Send(&s,RS_HOME,9,3,0); CHECK(r.type==RS_ACK);
    r=Send(&s,RS_STOP,10,0,0); CHECK(r.type==RS_ACK);
    for(unsigned i=0;i<1000;++i) {
        if(TIM3->CR1&TIM_CR1_CEN) YPulse();
        if(TIM1->CR1&TIM_CR1_CEN) TPulse();
    }
    CHECK(RsService_Poll(&s,&r) && r.seq==9 && r.payload[0]==RS_STOPPED && !s.homed && !s.theta_homed);
    /* One HOME failure stops partner without needing TX capacity. */
    r=Send(&s,RS_HOME,11,3,0); CHECK(r.type==RS_ACK);
    if (APP_THETA_MANUAL_HOME) { gpio.IDR=3; YHome_Poll(); }
    else TFinish();
    RsService_Maintain(&s);
    for(unsigned i=0;i<1000 && (TIM3->CR1&TIM_CR1_CEN);++i) YPulse();
    CHECK(RsService_Poll(&s,&r) && r.payload[0]==(APP_THETA_MANUAL_HOME?RS_LIMIT:RS_RESULT_FAULT) && !s.homed && !s.theta_homed);
    return 0;
}
