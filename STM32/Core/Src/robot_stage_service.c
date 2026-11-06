#include "robot_stage_service.h"
#include "app_config.h"
#include "y_motor.h"
#include "y_home.h"

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
    YHome_Init();
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
        /* Theta disabled: zero-only range, never claims Theta homed. */
        break;
    case RS_GET_STATUS: {
        rs_i32 steps = PositionSteps(s, &motor);
        reply->type = RS_STATUS; reply->len = 20;
        reply->payload[0] = s->pending ?
            (s->motion_request.type == RS_HOME ? RS_STATE_HOMING : RS_STATE_MOVING) :
            (s->fault || !motor.ready || !Y_MOTOR_OUTPUT_ALLOWED ? RS_STATE_FAULT :
             motor.state == Y_MOTOR_RUNNING ? RS_STATE_MOVING : RS_STATE_IDLE);
        /* Uninstalled/disabled inputs are not physical pressed switches.
         * Theta is disabled; report no Theta home/limit flags. */
        reply->payload[1] = s->homed |
            (Y_MOTOR_USE_LIMITS ? (rs_u8)((switch_bits & 3U) << 2) : 0U);
        rs_put_u32(reply->payload + 4, (rs_u32)StepsToUm(steps));
        rs_put_u32(reply->payload + 12, (rs_u32)steps);
        break;
    }
    case RS_STOP:
        /* Standard moves ramp down; raw PC jog retains its immediate STOP. */
        if (s->pending && (!s->motor_started || motor.state == Y_MOTOR_RUNNING ||
                           motor.state == Y_MOTOR_ABORTED)) s->stop_requested = 1;
        if (s->pending && s->motion_request.type==RS_HOME && APP_Y_SWITCH_HOME) YHome_Stop();
        else if (s->pending && s->motor_started) YMotor_Decelerate();
        else YMotor_Abort();
        Ack(reply, RS_STOP);
        break;
    default:
        if (s->pending || motor.state == Y_MOTOR_RUNNING) {
            Nak(reply, request->type, RS_BUSY); break;
        }
        if (request->type == RS_HOME) {
            if (request->payload[0] != 1) { Nak(reply, RS_HOME, RS_BAD_PARAM); break; }
            if ((!APP_Y_MANUAL_HOME && !APP_Y_SWITCH_HOME) || !motor.ready || !Y_MOTOR_OUTPUT_ALLOWED) {
                Nak(reply, RS_HOME, RS_FAULT); break;
            }
            if (APP_Y_SWITCH_HOME && !YHome_Start()) {
                Nak(reply,RS_HOME,RS_FAULT); break;
            }
            s->homed=0;
            s->motor_started = 0;
        } else {
            if (Get32(request->payload + 4) != 0) {
                Nak(reply, request->type, RS_BAD_PARAM); break;
            }
            if (s->fault || !motor.ready || !Y_MOTOR_OUTPUT_ALLOWED) {
                Nak(reply, request->type, RS_FAULT); break;
            }
            if (request->type == RS_SET_SPEED) {
                rs_u32 speed = Get32(request->payload);
                if (speed && (speed < 10U || speed > 2000U)) {
                    Nak(reply, RS_SET_SPEED, RS_OUT_OF_RANGE); break;
                }
                if (speed) s->speed_um_s = speed;
                Ack(reply, RS_SET_SPEED); break;
            }
            if (!s->homed) { Nak(reply, request->type, RS_NOT_HOMED); break; }
            int64_t target = Signed32(request->payload);
            if (request->type == RS_MOVE_REL) target += s->commanded_um;
            if (target < APP_Y_MIN_UM || target > APP_Y_MAX_UM) {
                Nak(reply, request->type, RS_OUT_OF_RANGE); break;
            }
            rs_i32 target_steps = RoundDiv(target * 8, 5);
            rs_i32 delta = target_steps - s->steps;
            s->direction = delta >= 0 ? 1 : -1;
            rs_u32 pulses = (rs_u32)(delta >= 0 ? delta : -delta);
            if (pulses) {
                rs_u32 hz = (s->speed_um_s * 8U + 2U) / 5U;
                YMotorResult result = YMotor_Start(delta >= 0 ? APP_Y_POSITIVE_DIR_LEVEL :
                    !APP_Y_POSITIVE_DIR_LEVEL, pulses, hz);
                if (result != Y_MOTOR_OK) {
                    Nak(reply, request->type, result == Y_MOTOR_BUSY ? RS_BUSY : RS_FAULT);
                    break;
                }
            }
            s->motor_started = pulses != 0;
            s->start_steps = s->steps; s->target_um = (rs_i32)target;
        }
        Ack(reply, request->type);
        s->motion_request = *request; s->motion_reply = *reply;
        s->motion_cached = s->pending = 1; s->stop_requested = 0;
        s->motion_age = 0;
        break;
    }
    s->cached = 1; s->last_request = *request; s->reply = *reply;
}
int RsService_Poll(RsService *s, RsFrame *reply)
{
    YMotorStatus motor; YMotor_GetStatus(&motor);
    if (!s->pending) {
        if (!motor.ready && !s->fault) {
            s->fault = 1; s->homed = 0;
            RsService_Event(s, RS_EVENT_FAULT, reply);
            return 1;
        }
        return 0;
    }
    if (s->motor_started && motor.state == Y_MOTOR_RUNNING) return 0;
    rs_u8 result = RS_OK;
    /* A hardware limit/fault during deceleration takes priority over STOP. */
    if (s->motor_started && motor.state == Y_MOTOR_LIMIT) result = RS_LIMIT;
    else if (s->motor_started && motor.state == Y_MOTOR_ABORTED) result = RS_RESULT_FAULT;
    else if (s->stop_requested) result = RS_STOPPED;
    else if (s->motor_started && motor.state != Y_MOTOR_DONE)
        result = motor.state == Y_MOTOR_LIMIT ? RS_LIMIT : RS_RESULT_FAULT;
    if (s->motion_request.type==RS_HOME && APP_Y_SWITCH_HOME) {
        YHome_Poll();
        YHomeState home=YHome_State();
        if (home==Y_HOME_RUNNING) return 0;
        result=home==Y_HOME_OK ? RS_OK : home==Y_HOME_STOPPED ? RS_STOPPED :
               home==Y_HOME_LIMIT ? RS_LIMIT : RS_RESULT_FAULT;
    }
    if (s->motor_started) s->steps = PositionSteps(s, &motor);
    if (result == RS_OK) {
        if (s->motion_request.type == RS_HOME) {
            s->steps = s->commanded_um = s->start_steps = s->target_um = 0;
            s->homed = 1; s->fault = 0;
        } else s->commanded_um = s->target_um;
    } else {
        s->commanded_um = StepsToUm(s->steps);
        if ((s->motor_started && !motor.count_exact) || result == RS_LIMIT || result == RS_RESULT_FAULT) {
            s->homed = 0; s->fault = 1;
        }
    }
    *reply = (RsFrame){0}; reply->type = RS_DONE; reply->len = 12;
    reply->seq = s->motion_request.seq;
    reply->payload[0] = result; reply->payload[1] = s->motion_request.type;
    rs_put_u32(reply->payload + 4, (rs_u32)StepsToUm(s->steps));
    s->pending = 0; s->motion_reply = *reply;
    if (s->cached && EqualRequest(&s->last_request, &s->motion_request)) s->reply = *reply;
    return 1;
}
void RsService_Event(RsService *s, rs_u8 event, RsFrame *reply)
{
    YMotorStatus motor; YMotor_GetStatus(&motor);
    *reply = (RsFrame){0}; reply->type = RS_EVENT; reply->seq = s->event_seq++;
    reply->len = 9; reply->payload[0] = event;
    rs_put_u32(reply->payload + 1, (rs_u32)StepsToUm(PositionSteps(s, &motor)));
}
