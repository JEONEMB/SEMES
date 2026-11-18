// 스테이지 제어 인터페이스 + 모의 장비 (펌웨어 0.3.1 동작 흉내)
// 출처: GitLab dev STM32/robot_stage_protocol.md (11/12), STM32/Core/Inc/app_config.h·theta_config.h
//   - 단위: Y µm, Θ 0.001° 정수. 조회(PING·STATUS·CONFIG)는 바로 응답, 동작(HOME·MOVE·STOP·SET_SPEED)은 ACK/NAK, HOME·MOVE 는 DONE 으로 끝남
//   - NAK: NOT_HOMED(원점 전 MOVE), OUT_OF_RANGE, BUSY(이동 중 HOME/MOVE/SET_SPEED), FAULT, BAD_PARAM(HOME mask 0)
//   - Y: HOME 은 −Y 스위치로 물리적 원점 복귀 (찾기 약 1 mm/s, 최대 52 mm · 풀기 0.5 mm/s, 최대 2 mm). 범위 0 ~ 500 000 µm (실제 행정 50 mm, 끝 스위치 우선)
//   - Θ: 수동 원점 (HOME bit1 = 지금 자리를 Θ 0 으로, 모터 안 돎). 범위 −10 000 ~ 10 000 mdeg (수동 원점 기준)
//   - 기본 속도: Y 100 µm/s, Θ 1000 mdeg/s
// 모의 장비가 정한 것 (펌웨어에 값 없음, 실제 장비 연결 때 확인): 원점 풀린 자리를 Y 0 으로 둠, + 방향 (아래 SimFromFw)
using System;
using System.Collections.Generic;
using UnityEngine;

namespace YT
{
    public enum StState { Idle = 0, Moving = 1, Homing = 2, Fault = 3 }
    public enum DoneResult { Ok = 0, Stopped = 1, Limit = 2, Fault = 3 }
    public enum NakReason { None = 0, UnknownType = 1, BadLen = 2, NotHomed = 3, OutOfRange = 4, Busy = 5, Fault = 6, BadParam = 7 }

    [Flags]
    public enum StFlags { None = 0, YHomed = 1, THomed = 2, YMinSwitch = 4, YMaxSwitch = 8, THomeSwitch = 16 }

    public struct StStatus
    {
        public StState state; public StFlags flags; public int yUm, tMdeg, ySteps, tSteps;
    }

    public struct StConfig { public int yMinUm, yMaxUm, tMinMdeg, tMaxMdeg; }

    // DONE 페이로드 (프로토콜 5장): 결과, 완료된 명령 TYPE (2 HOME, 3 MOVE_REL, 4 MOVE_ABS), 완료 시점 위치
    public struct DoneInfo { public DoneResult result; public int cmdType, yUm, tMdeg; }

    public interface IStage
    {
        string Name { get; }
        bool Connected { get; }
        StStatus Status { get; }
        StConfig Config { get; }
        int SpeedYUmS { get; }
        int SpeedTMdegS { get; }
        event Action<string> Log;
        event Action<DoneInfo> Done;
        string Mode { get; }        // 웹과 같은 값: mock / hardware
        string Transport { get; }   // mock / serial / driver
        int[] Ping();               // proto, fw major, minor, patch
        NakReason Home(int axisMask);
        NakReason MoveAbs(int yUm, int tMdeg);
        NakReason MoveRel(int dyUm, int dtMdeg);
        NakReason Stop();
        NakReason SetSpeed(int yUmS, int tMdegS);
        void Tick(float dt);
    }

    public static class StText
    {
        public static string State(StState s) => s == StState.Idle ? "대기" : s == StState.Moving ? "이동 중" : s == StState.Homing ? "원점 복귀 중" : "오류";
        public static string Nak(NakReason r) => r switch
        {
            NakReason.UnknownType => "알 수 없는 명령", NakReason.BadLen => "잘못된 길이", NakReason.NotHomed => "원점 미설정",
            NakReason.OutOfRange => "허용 범위 초과", NakReason.Busy => "이동 중", NakReason.Fault => "장치 고장", NakReason.BadParam => "잘못된 값", _ => "",
        };
        public static string Done(DoneResult r) => r == DoneResult.Ok ? "완료" : r == DoneResult.Stopped ? "정지됨" : r == DoneResult.Limit ? "리밋" : "고장";
        public static string Mm(int um) => (um / 1000.0).ToString("0.000");
    }

    public class MockStage : IStage
    {
        public string Name => "모의 장비 (펌웨어 0.3.1 흉내)";
        public bool Connected => true;
        public event Action<string> Log;
        public event Action<DoneInfo> Done;
        public string Mode => "mock";
        public string Transport => "mock";
        public int[] Ping() { Emit("→ PING"); Emit("← PONG proto 1, fw 0.3.1 (모의)"); return new[] { 1, 0, 3, 1 }; }
        public float TimeScale = 1f;                       // 모의 장비 전용 배속

        // 도면 자세 (Rig.SetPose 에 넣는 값): Y mm (가운데 0), Θ ° (도면 정의)
        public float YSim { get; private set; }
        public float TSim { get; private set; }
        // 실제 기구 Θ (백래시 반영). 화면 자세는 이것으로 그림. 펌웨어 보고 Θ 는 명령(스텝) 기준
        public float TPhys { get; private set; }
        // GitLab dev verify_web README '장비 조건 (2026-11-13)': Θ 방향을 바꾸면 약 0.023° 빈 구간(백래시)
        public bool BacklashOn = true;
        public const float THETA_BACKLASH_DEG = 0.023f;
        // 같은 README: STM32 리셋 후 원점을 다시 잡으면 센터 좌표가 0.4 mm 쯤 바뀔 수 있음 → 켜면 부팅마다 원점 스위치 작동점이 ±0.4 mm 안에서 무작위로 달라짐 (분포는 출처 없음: 균등으로 둠)
        public bool HomeJitterOn = false;
        public const float HOME_JITTER_MM = 0.4f;
        float switchJitter;

        StState state = StState.Idle;
        bool yHomed, tHomed;
        double yUm, tMdeg;                 // 펌웨어 위치 계수 (부팅 때 0)
        double yRefSim, tRefSim;           // 계수 0 일 때의 도면 자세
        int speedY = 100, speedT = 1000;   // µm/s, mdeg/s (펌웨어 기본값)
        double tgtY, tgtT;
        string cmdName = "";
        int cmdType;
        int homeMask, homePhase;           // 0 찾기, 1 풀기
        double homeTravel;

        public StConfig Config => new StConfig { yMinUm = 0, yMaxUm = 500000, tMinMdeg = -10000, tMaxMdeg = 10000 };
        public int SpeedYUmS => speedY;
        public int SpeedTMdegS => speedT;

        // 펌웨어 계수 → 도면 자세: Y + = −Y 원점 스위치에서 멀어지는 방향 (도면 +z), Θ + = 위에서 볼 때 반시계 (두 쪽 모두 펌웨어 문서 '확정 필요')
        void SyncSim()
        {
            YSim = (float)(yRefSim + yUm / 1000.0); TSim = (float)(tRefSim + tMdeg / 1000.0);
            float h = BacklashOn ? THETA_BACKLASH_DEG / 2 : 0;
            TPhys = Mathf.Clamp(TPhys, TSim - h, TSim + h);   // 명령이 한쪽으로 밀 때만 기구가 따라옴
        }

        public MockStage(float ySim = 0, float tSim = 0) { Boot(ySim, tSim); }

        public void Boot(float ySim, float tSim)
        {
            state = StState.Idle; yHomed = tHomed = false; yUm = tMdeg = 0; yRefSim = ySim; tRefSim = tSim; TPhys = tSim; SyncSim();
            switchJitter = HomeJitterOn ? UnityEngine.Random.Range(-HOME_JITTER_MM, HOME_JITTER_MM) : 0f;
            speedY = 100; speedT = 1000;
            Emit("← EVENT BOOT (원점 플래그 0)");
        }

        public void Reboot() { float p = TPhys; Boot(YSim, TSim); TPhys = p; SyncSim(); }

        void Emit(string s) => Log?.Invoke(s);

        public StStatus Status
        {
            get
            {
                var f = StFlags.None;
                if (yHomed) f |= StFlags.YHomed;
                if (tHomed) f |= StFlags.THomed;
                if (YSim <= -Rig.Y_SWITCH_ON) f |= StFlags.YMinSwitch;
                if (YSim >= Rig.Y_SWITCH_ON) f |= StFlags.YMaxSwitch;
                if (TSim >= Rig.TH_SWITCH_ON && TSim <= 15f) f |= StFlags.THomeSwitch;
                return new StStatus
                {
                    state = state, flags = f, yUm = (int)Math.Round(yUm), tMdeg = (int)Math.Round(tMdeg),
                    ySteps = (int)Math.Round(yUm * 1.6), tSteps = (int)Math.Round(tMdeg * 3200.0 * 100 / 360000.0),
                };
            }
        }

        NakReason Gate(bool motion)
        {
            if (state == StState.Moving || state == StState.Homing) return NakReason.Busy;
            if (motion && state == StState.Fault) return NakReason.Fault;
            return NakReason.None;
        }

        NakReason Reply(string cmd, NakReason r)
        {
            Emit(r == NakReason.None ? $"← ACK {cmd}" : $"← NAK {cmd} ({StText.Nak(r)})");
            return r;
        }

        public NakReason Home(int mask)
        {
            Emit($"→ HOME mask={mask}");
            if (mask == 0 || mask > 3) return Reply("HOME", NakReason.BadParam);
            if (state == StState.Moving || state == StState.Homing) return Reply("HOME", NakReason.Busy);
            Reply("HOME", NakReason.None);
            homeMask = mask; cmdName = "HOME"; cmdType = 2;
            if ((mask & 1) != 0) { state = StState.Homing; homePhase = 0; homeTravel = 0; yHomed = false; }
            else FinishHome();
            return NakReason.None;
        }

        void FinishHome()
        {
            if ((homeMask & 2) != 0) { tRefSim = TSim; tMdeg = 0; tHomed = true; SyncSim(); }   // Θ 수동 원점: 지금 자리를 0 으로
            state = StState.Idle;
            Finish(DoneResult.Ok);
        }

        void Finish(DoneResult r)
        {
            var s = Status;
            Emit($"← DONE {cmdName} {StText.Done(r)} · Y {StText.Mm(s.yUm)} mm / Θ {StText.Mm(s.tMdeg)}°");
            Done?.Invoke(new DoneInfo { result = r, cmdType = cmdType, yUm = s.yUm, tMdeg = s.tMdeg });
        }

        public NakReason MoveAbs(int y, int t) => Move("MOVE_ABS", y, t, false);
        public NakReason MoveRel(int dy, int dt) => Move("MOVE_REL", dy, dt, true);

        NakReason Move(string name, int a, int b, bool rel)
        {
            Emit($"→ {name} Y {StText.Mm(a)} mm, Θ {StText.Mm(b)}°");
            var g = Gate(true);
            if (g != NakReason.None) return Reply(name, g);
            double ty = rel ? Math.Round(yUm) + a : a, tt = rel ? Math.Round(tMdeg) + b : b;
            bool dyMove = Math.Abs(ty - yUm) > 0.5, dtMove = Math.Abs(tt - tMdeg) > 0.5;
            if ((dyMove && !yHomed) || (dtMove && !tHomed)) return Reply(name, NakReason.NotHomed);
            var c = Config;
            if (ty < c.yMinUm || ty > c.yMaxUm || tt < c.tMinMdeg || tt > c.tMaxMdeg) return Reply(name, NakReason.OutOfRange);
            Reply(name, NakReason.None);
            tgtY = ty; tgtT = tt; cmdName = name; cmdType = rel ? 3 : 4; state = StState.Moving;
            return NakReason.None;
        }

        public NakReason Stop()
        {
            Emit("→ STOP");
            Reply("STOP", NakReason.None);
            if (state == StState.Moving || state == StState.Homing)
            {
                bool homing = state == StState.Homing;
                state = StState.Idle;
                if (homing) yHomed = false;
                Finish(DoneResult.Stopped);
            }
            return NakReason.None;
        }

        public NakReason SetSpeed(int y, int t)
        {
            Emit($"→ SET_SPEED Y {y} µm/s, Θ {t} mdeg/s");
            var g = Gate(false);
            if (g != NakReason.None) return Reply("SET_SPEED", g);
            if (y > 0) speedY = y;
            if (t > 0) speedT = t;
            return Reply("SET_SPEED", NakReason.None);
        }

        static double Step(double cur, double tgt, double maxStep) => Math.Abs(tgt - cur) <= maxStep ? tgt : cur + Math.Sign(tgt - cur) * maxStep;

        public void Tick(float dt)
        {
            double t = dt * TimeScale;
            if (state == StState.Moving)
            {
                double y0 = yUm;
                yUm = Step(yUm, tgtY, speedY * t);
                tMdeg = Step(tMdeg, tgtT, speedT * t);
                SyncSim();
                bool up = yUm > y0, down = yUm < y0;
                if ((up && YSim >= Rig.Y_SWITCH_ON) || (down && YSim <= -Rig.Y_SWITCH_ON))
                {
                    state = StState.Idle; Finish(DoneResult.Limit); return;
                }
                if (Math.Abs(yUm - tgtY) < 1e-6 && Math.Abs(tMdeg - tgtT) < 1e-6) { state = StState.Idle; Finish(DoneResult.Ok); }
            }
            else if (state == StState.Homing)
            {
                if (homePhase == 0)   // −Y 스위치 찾기 (약 1 mm/s, 최대 52 mm)
                {
                    double d = 1000 * t;
                    yUm -= d; homeTravel += d; SyncSim();
                    if (YSim <= -Rig.Y_SWITCH_ON + switchJitter) { homePhase = 1; homeTravel = 0; }
                    else if (homeTravel > 52000) { state = StState.Fault; Finish(DoneResult.Fault); }
                }
                else                   // 풀기 (0.5 mm/s, 최대 2 mm) → 풀린 자리를 Y 0 으로
                {
                    double d = 500 * t;
                    yUm += d; homeTravel += d; SyncSim();
                    if (YSim > -Rig.Y_SWITCH_ON + switchJitter + 0.05f)
                    {
                        yRefSim = YSim; yUm = 0; yHomed = true; SyncSim();
                        FinishHome();
                    }
                    else if (homeTravel > 2000) { state = StState.Fault; Finish(DoneResult.Fault); }
                }
            }
            // 기구 행정 끝 (±25)
            if (YSim > Rig.Y_STROKE || YSim < -Rig.Y_STROKE)
            {
                double lim = YSim > 0 ? Rig.Y_STROKE : -Rig.Y_STROKE;
                yUm = (lim - yRefSim) * 1000; SyncSim();
            }
        }
    }

    // 중앙 기준 (웹 demo_web 과 같은 값: center_y_um, center_t_mdeg, tolerance_y_um, tolerance_t_mdeg, 기본 0/0/10/10)
    public class CenterRef
    {
        public int yUm, tMdeg, tolYUm = 10, tolTMdeg = 10;

        public static CenterRef Load()
        {
            return new CenterRef
            {
                yUm = PlayerPrefs.GetInt("center_y_um", 0), tMdeg = PlayerPrefs.GetInt("center_t_mdeg", 0),
                tolYUm = PlayerPrefs.GetInt("tolerance_y_um", 10), tolTMdeg = PlayerPrefs.GetInt("tolerance_t_mdeg", 10),
            };
        }

        public void Save()
        {
            PlayerPrefs.SetInt("center_y_um", yUm); PlayerPrefs.SetInt("center_t_mdeg", tMdeg);
            PlayerPrefs.SetInt("tolerance_y_um", tolYUm); PlayerPrefs.SetInt("tolerance_t_mdeg", tolTMdeg);
            PlayerPrefs.Save();
        }

        public bool At(int y, int t) => Math.Abs(y - yUm) <= tolYUm && Math.Abs(t - tMdeg) <= tolTMdeg;
    }
}
