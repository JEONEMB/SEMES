#include "robot_stage_service.h"
#include "app_config.h"
#include "y_motor.h"
#include "y_home.h"
#include "theta_motor.h"
#include "theta_home.h"
#include "theta_units.h"
#include "y_motor_port.h"

static rs_u32 Get32(const rs_u8 *p)
{
    return (rs_u32)p[0] | ((rs_u32)p[1] << 8) |
           ((rs_u32)p[2] << 16) | ((rs_u32)p[3] << 24);
}
static int64_t Signed32(const rs_u8 *p)
{
    rs_u32 value = Get32(p);
    return value & 0x80000000U ? (int64_t)value - 4294967296LL : value;
}
static rs_i32 RoundDiv(int64_t value, int divisor)
{
    return (rs_i32)(value >= 0 ? (value + divisor / 2) / divisor :
                                   -((-value + divisor / 2) / divisor));
}
static rs_i32 StepsToUm(rs_i32 steps) { return RoundDiv((int64_t)steps * 5, 8); }
static rs_i32 PositionSteps(const RsService *s, const YMotorStatus *motor)
{
    return s->pending && s->motor_started ?
        s->start_steps + s->direction * (rs_i32)motor->completed_pulses : s->steps;
}
static rs_i32 ThetaPosition(const RsService *s, const ThetaMotorStatus *m)
{
    return s->pending && s->theta_started ?
        s->theta_start_steps + s->theta_direction * (rs_i32)m->completed_pulses : s->theta_steps;
}
static void CacheMotion(RsService *s, const RsFrame *q, const RsFrame *r)
{
    s->motion_request=*q; s->motion_reply=*r;
    s->motion_cached=s->pending=1; s->stop_requested=0; s->motion_age=0;
}
static int EqualRequest(const RsFrame *a, const RsFrame *b)
{
    if (a->type != b->type || a->seq != b->seq || a->len != b->len) return 0;
    for (unsigned i = 0; i < a->len; ++i) if (a->payload[i] != b->payload[i]) return 0;
    return 1;
}
static void Nak(RsFrame *reply, rs_u8 type, rs_u8 reason)
{
    reply->type = RS_NAK; reply->len = 2;
    reply->payload[0] = type; reply->payload[1] = reason;
}
static void Ack(RsFrame *reply, rs_u8 type)
{
    reply->type = RS_ACK; reply->len = 1; reply->payload[0] = type;
}
void RsService_Init(RsService *s)
{
    *s = (RsService){0};
    s->speed_um_s = APP_Y_DEFAULT_SPEED_UM_S;
    s->theta_speed=APP_THETA_DEFAULT_SPEED_MDEG_S;
    YHome_Init();
    ThetaHome_Init();
}
void RsService_InvalidatePosition(RsService *s)
{
    s->homed = 0;
    s->motion_cached = 0;
}
void RsService_Handle(RsService *s, const RsFrame *request,
                      rs_u8 switch_bits, RsFrame *reply)
{
    *reply = (RsFrame){0}; reply->seq = request->seq;
    /* A full SEQ cycle expires the completed-motion cache; otherwise a
     * long stream of status queries could make a new move look like a retry. */
    if (s->motion_cached && s->cached && !EqualRequest(request, &s->last_request)) {
        if (s->motion_age < 256U) ++s->motion_age;
        if (!s->pending && s->motion_age >= 256U) s->motion_cached = 0;
    }
    /* Keep accepted motion independent of queries and STOP retransmissions. */
    if (s->motion_cached && request->seq == s->motion_request.seq &&
        request->type == s->motion_request.type) {
        *reply = s->motion_reply;
        return;
    }
    if (s->cached && request->type != RS_STOP &&
        request->seq == s->last_request.seq && request->type == s->last_request.type) {
        *reply = s->reply;
        return;
    }
    int expected = -1;
    switch (request->type) {
    case RS_PING: case RS_STOP: case RS_GET_STATUS: case RS_GET_CONFIG: expected = 0; break;
    case RS_HOME: expected = 1; break;
    case RS_MOVE_REL: case RS_MOVE_ABS: case RS_SET_SPEED: expected = 8; break;
    default: break;
    }
    YMotorStatus motor;
    YMotor_GetStatus(&motor);
    ThetaMotorStatus tm; ThetaMotor_GetStatus(&tm);
    if (expected < 0) Nak(reply, request->type, RS_UNKNOWN_TYPE);
    else if (request->len != (unsigned)expected) Nak(reply, request->type, RS_BAD_LEN);
    else switch (request->type) {
    case RS_PING:
        reply->type = RS_PONG; reply->len = 4;
        reply->payload[0] = RS_VERSION; reply->payload[1] = APP_FW_MAJOR;
        reply->payload[2] = APP_FW_MINOR; reply->payload[3] = APP_FW_PATCH;
        break;
    case RS_GET_CONFIG:
        reply->type = RS_CONFIG; reply->len = 16;
        rs_put_u32(reply->payload, (rs_u32)APP_Y_MIN_UM);
        rs_put_u32(reply->payload + 4, (rs_u32)APP_Y_MAX_UM);
        if (APP_THETA_ENABLED) {
            rs_put_u32(reply->payload+8,(rs_u32)APP_THETA_MIN_MDEG);
            rs_put_u32(reply->payload+12,(rs_u32)APP_THETA_MAX_MDEG);
        }
        break;
    case RS_GET_STATUS: {
        rs_i32 steps = PositionSteps(s, &motor);
        reply->type = RS_STATUS; reply->len = 20;
        reply->payload[0] = s->pending ?
            (s->motion_request.type == RS_HOME ? RS_STATE_HOMING : RS_STATE_MOVING) :
            (s->fault || s->theta_fault || !motor.ready || !Y_MOTOR_OUTPUT_ALLOWED ||
             (APP_THETA_ENABLED && !tm.ready) ? RS_STATE_FAULT :
             motor.state == Y_MOTOR_RUNNING ? RS_STATE_MOVING : RS_STATE_IDLE);
        /* Actual switch inputs; rotary home is not an end limit. */
        reply->payload[1] = s->homed |
            (Y_MOTOR_USE_LIMITS ? (rs_u8)((switch_bits & 3U) << 2) : 0U) |
            (APP_THETA_ENABLED ? (rs_u8)((s->theta_homed<<1) | ((switch_bits&4U)<<2)) : 0U);
        rs_put_u32(reply->payload + 4, (rs_u32)StepsToUm(steps));
        rs_put_u32(reply->payload + 12, (rs_u32)steps);
        rs_i32 ts=ThetaPosition(s,&tm);
        rs_put_u32(reply->payload+8,(rs_u32)ThetaUnits_FromSteps(ts));
        rs_put_u32(reply->payload+16,(rs_u32)ts);
        break;
    }
    case RS_STOP:
        if (s->pending && (s->motion_request.type==RS_HOME ||
            (!s->motor_started && !s->theta_started) ||
            (s->motor_started && (motor.state==Y_MOTOR_RUNNING || motor.state==Y_MOTOR_ABORTED)) ||
            (s->theta_started && (tm.state==THETA_MOTOR_RUNNING || tm.state==THETA_MOTOR_ABORTED))))
            s->stop_requested=1;
        if (s->pending && s->motion_request.type==RS_HOME) {
            if (s->home_mask&1U) YHome_Stop();
            if (s->home_mask&2U) ThetaHome_Stop();
        } else {
            if (s->pending && s->motor_started) YMotor_Decelerate(); else YMotor_Abort();
            if (s->pending && s->theta_started) ThetaMotor_Decelerate(); else ThetaMotor_Abort();
        }
        Ack(reply,RS_STOP); break;
    default: {
        if (s->pending || motor.state==Y_MOTOR_RUNNING || tm.state==THETA_MOTOR_RUNNING) {
            Nak(reply,request->type,RS_BUSY); break;
        }
        if (request->type==RS_HOME) {
            rs_u8 mask=request->payload[0];
            if (!mask || (mask&~3U) || (!APP_THETA_ENABLED && (mask&2U))) {
                Nak(reply,RS_HOME,RS_BAD_PARAM); break;
            }
            /* Validate both axes before starting either timer. */
            if (((mask&1U) && ((!APP_Y_MANUAL_HOME && !APP_Y_SWITCH_HOME) ||
                 !motor.ready || !Y_MOTOR_OUTPUT_ALLOWED ||
                 (APP_Y_SWITCH_HOME && YMotorPort_RawLimits()==3U))) ||
                ((mask&2U) && (!tm.ready || !ThetaUnits_ValidSpeed(APP_THETA_HOME_SEEK_MDEG_S) ||
                 !ThetaUnits_ValidSpeed(APP_THETA_HOME_RELEASE_MDEG_S)))) {
                Nak(reply,RS_HOME,RS_FAULT); break;
            }
            /* A late start failure is an accepted operation ending in DONE FAULT. */
            s->home_mask=mask; s->active_mask=mask;
            s->motor_started=s->theta_started=0;
            if (mask&1U) { s->homed=0; if (APP_Y_SWITCH_HOME) (void)YHome_Start(); }
            if (mask&2U) { s->theta_homed=0; (void)ThetaHome_Start(); }
            Ack(reply,RS_HOME); CacheMotion(s,request,reply); break;
        }
        if (!APP_THETA_ENABLED && Get32(request->payload+4)) {
            Nak(reply,request->type,RS_BAD_PARAM); break;
        }
        if (s->fault || s->theta_fault) { Nak(reply,request->type,RS_FAULT); break; }
        if (request->type==RS_SET_SPEED) {
            rs_u32 y=Get32(request->payload), t=Get32(request->payload+4);
            if ((!APP_THETA_ENABLED && (!motor.ready || !Y_MOTOR_OUTPUT_ALLOWED)) ||
                (y && (!motor.ready || !Y_MOTOR_OUTPUT_ALLOWED)) || (t && !tm.ready)) {
                Nak(reply,RS_SET_SPEED,RS_FAULT); break;
            }
            if ((y && (y<10U || y>2000U)) || (t && !ThetaUnits_ValidSpeed(t))) {
                Nak(reply,RS_SET_SPEED,RS_OUT_OF_RANGE); break;
            }
            if (y) s->speed_um_s=y;
            if (t) s->theta_speed=t;
            Ack(reply,RS_SET_SPEED); break;
        }
        int64_t y=Signed32(request->payload), t=Signed32(request->payload+4);
        if (request->type==RS_MOVE_REL) { y+=s->commanded_um; t+=s->theta_commanded; }
        /* ABS zero is a real target. A zero-delta axis may stay unhomed while
         * the other axis is commissioned. Quantized zero still requires HOME. */
        rs_u8 mask=(y!=s->commanded_um ? 1U:0U) | (t!=s->theta_commanded ? 2U:0U);
        if (!APP_THETA_ENABLED) mask|=1U;
        if (((mask&1U) && (!motor.ready || !Y_MOTOR_OUTPUT_ALLOWED)) ||
            ((mask&2U) && (!APP_THETA_ENABLED || !tm.ready))) {
            Nak(reply,request->type,RS_FAULT); break;
        }
        if (((mask&1U) && !s->homed) || ((mask&2U) && !s->theta_homed) ||
            (!mask && !s->homed && !s->theta_homed)) {
            Nak(reply,request->type,RS_NOT_HOMED); break;
        }
        if (y<APP_Y_MIN_UM || y>APP_Y_MAX_UM ||
            t<(APP_THETA_ENABLED ? APP_THETA_MIN_MDEG:0) ||
            t>(APP_THETA_ENABLED ? APP_THETA_MAX_MDEG:0)) {
            Nak(reply,request->type,RS_OUT_OF_RANGE); break;
        }
        rs_i32 ys=RoundDiv(y*8,5), ts=ThetaUnits_ToSteps(t);
        rs_i32 yd=ys-s->steps, td=ts-s->theta_steps;
        rs_u32 yp=(rs_u32)(yd>=0?yd:-yd), tp=(rs_u32)(td>=0?td:-td);
        uint8_t ydir=yd>=0?APP_Y_POSITIVE_DIR_LEVEL:!APP_Y_POSITIVE_DIR_LEVEL;
        uint8_t tdir=td>=0?APP_THETA_CCW_DIR_HIGH:!APP_THETA_CCW_DIR_HIGH;
        uint8_t limits=YMotorPort_Limits();
        if (yp && (limits==3U || (limits & (ydir?2U:1U)))) {
            Nak(reply,request->type,RS_FAULT); break;
        }
        s->direction=yd>=0?1:-1; s->theta_direction=td>=0?1:-1;
        s->start_steps=s->steps; s->theta_start_steps=s->theta_steps;
        s->target_um=(rs_i32)y; s->theta_target=(rs_i32)t;
        s->active_mask=mask; s->motor_started=s->theta_started=0;
        uint32_t key=YMotorPort_Lock();
        YMotorResult yr=Y_MOTOR_OK; ThetaMotorResult tr=THETA_MOTOR_OK;
        if (yp) { yr=YMotor_Start(ydir,yp,(s->speed_um_s*8U+2U)/5U); s->motor_started=yr==Y_MOTOR_OK; }
        if (yr==Y_MOTOR_OK && tp) { tr=ThetaMotor_Start(tdir,tp,ThetaUnits_ToHz(s->theta_speed)); s->theta_started=tr==THETA_MOTOR_OK; }
        if (yr!=Y_MOTOR_OK || tr!=THETA_MOTOR_OK) {
            if (s->motor_started) YMotor_Abort();
            if (s->theta_started) ThetaMotor_Abort();
            s->fault|=(mask&1U)!=0; s->theta_fault|=(mask&2U)!=0;
            if (mask&1U) s->homed=0;
            if (mask&2U) s->theta_homed=0;
        }
        YMotorPort_Unlock(key);
        Ack(reply,request->type); CacheMotion(s,request,reply); break;
    }
    }

    s->cached = 1; s->last_request = *request; s->reply = *reply;
}
static rs_u8 YResult(const RsService *s, const YMotorStatus *m)
{
    if (!s->motor_started) return RS_OK;
    if (m->state==Y_MOTOR_LIMIT) return RS_LIMIT;
    if (m->state==Y_MOTOR_ABORTED || !m->ready) return RS_RESULT_FAULT;
    if (m->state==Y_MOTOR_STOPPED) return RS_STOPPED;
    return m->state==Y_MOTOR_RUNNING || m->state==Y_MOTOR_DONE ? RS_OK:RS_RESULT_FAULT;
}
static rs_u8 TResult(const RsService *s, const ThetaMotorStatus *m)
{
    if (!s->theta_started) return RS_OK;
    if (m->state==THETA_MOTOR_LIMIT) return RS_LIMIT;
    if (m->state==THETA_MOTOR_ABORTED || !m->ready) return RS_RESULT_FAULT;
    if (m->state==THETA_MOTOR_STOPPED) return RS_STOPPED;
    return m->state==THETA_MOTOR_RUNNING || m->state==THETA_MOTOR_DONE ? RS_OK:RS_RESULT_FAULT;
}
static rs_u8 Combine(rs_u8 y, rs_u8 t)
{
    if (y==RS_LIMIT || t==RS_LIMIT) return RS_LIMIT;
    if (y==RS_RESULT_FAULT || t==RS_RESULT_FAULT) return RS_RESULT_FAULT;
    if (y==RS_STOPPED || t==RS_STOPPED) return RS_STOPPED;
    return RS_OK;
}
/* Independent of TX space: axis failure stops its partner even with TX full. */
void RsService_Maintain(RsService *s)
{
    if (!s->pending) return;
    if (s->motion_request.type==RS_HOME) {
        if (s->home_mask&1U) YHome_Poll();
        if (s->home_mask&2U) ThetaHome_Poll();
        int yf=(s->home_mask&1U) && APP_Y_SWITCH_HOME &&
            (YHome_State()==Y_HOME_FAULT || YHome_State()==Y_HOME_LIMIT);
        int tf=(s->home_mask&2U) &&
            (ThetaHome_State()==THETA_HOME_FAULT || ThetaHome_State()==THETA_HOME_LIMIT);
        if (yf && (s->home_mask&2U)) ThetaHome_Stop();
        if (tf && (s->home_mask&1U)) YHome_Stop();
    } else {
        YMotorStatus y; ThetaMotorStatus t; YMotor_GetStatus(&y); ThetaMotor_GetStatus(&t);
        rs_u8 yr=YResult(s,&y), tr=TResult(s,&t);
        if (s->fault || s->theta_fault || yr>=RS_LIMIT || tr>=RS_LIMIT) {
            if (s->motor_started && y.state==Y_MOTOR_RUNNING) YMotor_Abort();
            if (s->theta_started && t.state==THETA_MOTOR_RUNNING) ThetaMotor_Abort();
        }
    }
}
int RsService_Poll(RsService *s, RsFrame *reply)
{
    RsService_Maintain(s);
    YMotorStatus motor; ThetaMotorStatus tm;
    YMotor_GetStatus(&motor); ThetaMotor_GetStatus(&tm);
    if (!s->pending) {
        if ((!motor.ready && !s->fault) || (APP_THETA_ENABLED && !tm.ready && !s->theta_fault)) {
            if (!motor.ready) { s->fault=1; s->homed=0; }
            if (APP_THETA_ENABLED && !tm.ready) { s->theta_fault=1; s->theta_homed=0; }
            RsService_Event(s,RS_EVENT_FAULT,reply); return 1;
        }
        return 0;
    }
    rs_u8 result=RS_OK;
    if (s->motion_request.type==RS_HOME) {
        rs_u8 yr=RS_OK,tr=RS_OK;
        YHomeState yh=YHome_State(); ThetaHomeState th=ThetaHome_State();
        if (((s->home_mask&1U) && APP_Y_SWITCH_HOME && yh==Y_HOME_RUNNING) ||
            ((s->home_mask&2U) && th==THETA_HOME_RUNNING)) return 0;
        if ((s->home_mask&1U) && APP_Y_SWITCH_HOME)
            yr=yh==Y_HOME_OK?RS_OK:yh==Y_HOME_STOPPED?RS_STOPPED:yh==Y_HOME_LIMIT?RS_LIMIT:RS_RESULT_FAULT;
        if (s->home_mask&2U)
            tr=th==THETA_HOME_OK?RS_OK:th==THETA_HOME_STOPPED?RS_STOPPED:th==THETA_HOME_LIMIT?RS_LIMIT:RS_RESULT_FAULT;
        result=Combine(yr,tr);
        if (s->stop_requested && result==RS_OK) result=RS_STOPPED;
        if (result==RS_OK) {
            if (s->home_mask&1U) {
                s->steps=s->commanded_um=s->start_steps=s->target_um=0;
                s->homed=1; s->fault=0;
            }
            if (s->home_mask&2U) {
                s->theta_steps=s->theta_commanded=s->theta_start_steps=s->theta_target=0;
                s->theta_homed=1; s->theta_fault=0;
            }
        } else {
            if (s->home_mask&1U) { s->homed=0; if (yr>=RS_LIMIT) s->fault=1; }
            if (s->home_mask&2U) { s->theta_homed=0; if (tr>=RS_LIMIT) s->theta_fault=1; }
        }
    } else {
        if ((s->motor_started && motor.state==Y_MOTOR_RUNNING) ||
            (s->theta_started && tm.state==THETA_MOTOR_RUNNING)) return 0;
        result=Combine(YResult(s,&motor),TResult(s,&tm));
        if ((s->fault || s->theta_fault) && result!=RS_LIMIT) result=RS_RESULT_FAULT;
        if (s->stop_requested && result==RS_OK) result=RS_STOPPED;
        s->steps=PositionSteps(s,&motor); s->theta_steps=ThetaPosition(s,&tm);
        if (result==RS_OK) {
            s->commanded_um=s->target_um; s->theta_commanded=s->theta_target;
        } else {
            s->commanded_um=StepsToUm(s->steps); s->theta_commanded=ThetaUnits_FromSteps(s->theta_steps);
            if ((s->motor_started && !motor.count_exact) ||
                ((s->active_mask&1U) && result>=RS_LIMIT)) { s->homed=0; s->fault=1; }
            if ((s->theta_started && !tm.count_exact) ||
                ((s->active_mask&2U) && result>=RS_LIMIT)) { s->theta_homed=0; s->theta_fault=1; }
        }
    }
    *reply=(RsFrame){0}; reply->type=RS_DONE; reply->len=12;
    reply->seq=s->motion_request.seq; reply->payload[0]=result; reply->payload[1]=s->motion_request.type;
    rs_put_u32(reply->payload+4,(rs_u32)StepsToUm(s->steps));
    rs_put_u32(reply->payload+8,(rs_u32)ThetaUnits_FromSteps(s->theta_steps));
    s->pending=0; s->motion_reply=*reply;
    if (s->cached && EqualRequest(&s->last_request,&s->motion_request)) s->reply=*reply;
    return 1;
}
void RsService_Event(RsService *s, rs_u8 event, RsFrame *reply)
{
    YMotorStatus motor; ThetaMotorStatus tm; YMotor_GetStatus(&motor); ThetaMotor_GetStatus(&tm);
    *reply=(RsFrame){0}; reply->type=RS_EVENT; reply->seq=s->event_seq++;
    reply->len=9; reply->payload[0]=event;
    rs_put_u32(reply->payload+1,(rs_u32)StepsToUm(PositionSteps(s,&motor)));
    rs_put_u32(reply->payload+5,(rs_u32)ThetaUnits_FromSteps(ThetaPosition(s,&tm)));
}
