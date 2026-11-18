// 웹 3D 장비 화면 (verify_web_3d 의 iframe 안, Viewer 장면)
//   장비에는 아무 명령도 보내지 않는다. 페이지 JS 가 verify_web /api/events 에서 받은 위치를 OnState 로 넘겨 주면 그 자세로 그린다
//   이동 중에는 verify_web 이 STATUS 를 묻지 않으므로(runner.py _axis_plan) 축별 이동량·예상 시간으로 자세를 추정하고, 끝나면 보고 위치로 맞춘다
//   글씨는 모두 페이지 HTML 에서 그림 (웹 빌드에는 한글 글꼴이 없음)
// 움직임이 눈에 띄게 (보정 이동은 Y 1~2 mm, Θ 1° 안팎이라 장비 전체 보기에서는 몇 px 밖에 안 됨)
//   · 근접 보기: 스테이지를 가까이 보는 보는 방향 2개 (이 화면 전용)
//   · 과장: 켜면 센터 대비 움직임을 k 배로 그림 (페이지가 "실제 크기 아님" 표시). Y 행정 ±25 mm 안으로 제한
using System.Collections;
using UnityEngine;

namespace YT
{
    public class ViewerApp : MonoBehaviour
    {
        Rig rig; OrbitCam orbit; Camera cam; SemesView semes;
        St st; double serverOffset;   // 서버 시계 − 이 화면 시계 (초)
        float lastY = float.NaN, lastT = float.NaN;
        bool debug; float nextLog;
        float exag = 1;               // 움직임 과장 배율 (1 = 실제 크기)
        // simcam 배치 (verify_web_3d 카메라 칸의 "3D 시뮬레이터" 탭): 화면 전체 = 3D 장비 안의 가상 카메라(SEMES 화각),
        // 오른쪽 아래 작은 창 = 3D 장비 전체 (드래그·휠로 돌리고 확대). 작은 창 자리는 페이지 HTML 의 테두리와 같은 비율
        public static readonly Rect PIP = new Rect(1f - 0.015f - 0.28f, 0.03f, 0.28f, 0.28f);
        string pendingLayout, pendingOrbit; bool simcam;
        const float EXAG_MAX_T = 20;  // 과장했을 때 Θ 그림 한계 (°)

        // 페이지 JS 가 verify_web 상태에서 골라 넘기는 값 (null 이 없게 평평하게)
        [System.Serializable]
        class St
        {
            public bool hasPos; public float y, t;          // 스테이지 위치 (mm, °) = verify_web state.position
            public bool moving;
            public bool hasAxes; public double started, now; // target.axes.started, state.now (서버 시각, 초)
            public float dy, dt, ys, ts;                     // 축별 이동량과 예상 시간 (모르면 ys/ts < 0)
            public bool hasCenter; public float cy, ct;      // 센터 (스테이지 좌표) = settings.center_y_mm / center_t_deg
        }

        IEnumerator Start()
        {
            Application.targetFrameRate = 60;
            debug = Application.absoluteURL.Contains("debug=1");
            yield return Files.Preload();
            if (Files.Error != null) Debug.LogError("3D 파일 받기 실패: " + Files.Error);
            App.BuildWorld(transform, out rig, out cam, out orbit, out semes);
            semes.cam.enabled = false;           // SEMES 가상 카메라 화면은 쓰지 않음 (실제 영상은 verify_web 에 있음)
            rig.SetPose(-Rig.Y_SWITCH_ON, 0);   // 위치를 받기 전: 원점 자리
            if (pendingLayout != null) SetLayout(pendingLayout);
            // 3D 카메라(OrbitCam)가 첫 프레임에 기본 시점을 잡으므로, 페이지가 정한 시점은 그 뒤에 넣음
            yield return null; yield return null;
            ready = true;
            if (pendingOrbit != null) SetOrbit(pendingOrbit);
        }
        bool ready;

        // "simcam" = 가상 카메라 전체 화면 + 3D 장비 작은 창, 그 밖 = 3D 장비 전체 화면 (예전 3D 칸)
        public void SetLayout(string mode)
        {
            if (rig == null) { pendingLayout = mode; return; }
            simcam = mode == "simcam";
            if (simcam)
            {
                semes.cam.targetTexture = null; semes.cam.rect = new Rect(0, 0, 1, 1); semes.cam.depth = 0; semes.cam.enabled = true;
                cam.rect = PIP; cam.depth = 1;
                orbit.pointerBlocked = () => !cam.pixelRect.Contains(Input.mousePosition);   // 돌리기·확대는 작은 창 안에서만
            }
            else
            {
                semes.cam.enabled = false; cam.rect = new Rect(0, 0, 1, 1); cam.depth = 0; orbit.pointerBlocked = null;
            }
        }

        // "px,py,pz,tx,ty,tz" (도면 좌표 mm): 카메라 위치와 보는 곳. 작은 창 처음 시점을 페이지가 정함
        public void SetOrbit(string csv)
        {
            if (!ready) { pendingOrbit = csv; return; }
            var p = csv.Split(',');
            if (p.Length != 6) return;
            var v = new float[6];
            for (int i = 0; i < 6; i++) if (!float.TryParse(p[i], System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out v[i])) return;
            orbit.Set(Geo.P(v[0], v[1], v[2]), Geo.P(v[3], v[4], v[5]));
        }

        public void OnState(string json)
        {
            st = JsonUtility.FromJson<St>(json);
            if (st.now > 0) serverOffset = st.now - Time.realtimeSinceStartupAsDouble;
        }

        // "1" = 실제 크기, "10" = 10배
        public void SetExaggerate(string k)
        {
            if (float.TryParse(k, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var v)) exag = Mathf.Clamp(v, 1, 50);
            lastY = float.NaN;
        }

        public void SetView(string key)
        {
            if (orbit == null) return;
            foreach (var v in Rig.Views) if (v.key == key) { orbit.TweenTo(v.pos, v.target); return; }
            // 이 화면 전용 근접 보기: Θ 부분(스테이지 윗판) 둘레를 가까이
            Vector3 c = rig.tG.position, o = Geo.P(0, 0, 0);
            if (key == "near") orbit.TweenTo(c + (Geo.P(-300, 300, -380) - o), c + (Geo.P(20, 40, 0) - o));
            else if (key == "near_top") orbit.TweenTo(c + (Geo.P(0, 560, 0.5f) - o), c + (Geo.P(0, 0, 0) - o));
        }

        static float Part(float d, float sec, double elapsed)
        {
            if (sec < 0) return 0;                   // 속도를 모름: 끝날 때까지 출발 위치
            if (sec == 0) return d;                  // 0.5 µm 미만 이동
            return d * Mathf.Clamp01((float)(elapsed / sec));
        }

        // 펌웨어 위치 → 도면 자세 (원점 = −Y 스위치 작동점 −22.4 라고 보고 그림, Windows 대시보드와 같은 가정)
        static float SimY(float yMm) => -Rig.Y_SWITCH_ON + yMm;

        void Update()
        {
            if (rig == null || st == null || !st.hasPos) return;
            float y = st.y, t = st.t;
            if (st.moving && st.hasAxes)
            {   // 두 축은 동시에 출발 (runner.py _axis_plan). 가감속은 빼고 일정 속도로 봄
                double e = Time.realtimeSinceStartupAsDouble + serverOffset - st.started;
                y += Part(st.dy, st.ys, e); t += Part(st.dt, st.ts, e);
            }
            float dy = y, dt = t;   // 그릴 자세
            if (st.hasCenter && exag > 1)
            {
                dy = st.cy + exag * (y - st.cy); dt = st.ct + exag * (t - st.ct);
                dy = Mathf.Clamp(SimY(dy), -Rig.Y_STROKE, Rig.Y_STROKE) + Rig.Y_SWITCH_ON;
                dt = Mathf.Clamp(dt, st.ct - EXAG_MAX_T, st.ct + EXAG_MAX_T);
            }
            float yy = SimY(dy);
            if (!Mathf.Approximately(yy, lastY) || !Mathf.Approximately(dt, lastT)) { rig.SetPose(yy, dt); lastY = yy; lastT = dt; }
            if (debug && Time.unscaledTime > nextLog)
            {   // 확인용: 주소에 ?debug=1 이면 0.5초마다 그린 자세를 브라우저 콘솔에
                nextLog = Time.unscaledTime + 0.5f;
                Debug.Log($"[viewer] 보고 Y {st.y:0.000} Θ {st.t:0.000} · 추정 Y {y:0.000} Θ {t:0.000} · 그림 Y {dy:0.000} Θ {dt:0.000} (×{exag}) · 이동 {st.moving}");
            }
        }
    }
}
