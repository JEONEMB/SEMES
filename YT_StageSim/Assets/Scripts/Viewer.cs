// 웹 3D 장비 화면 (verify_web_3d 의 iframe 안, Viewer 장면)
//   장비에는 아무 명령도 보내지 않는다. 페이지 JS 가 verify_web /api/events 에서 받은 위치를 OnState 로 넘겨 주면 그 자세로 그린다
//   이동 중에는 verify_web 이 STATUS 를 묻지 않으므로(runner.py _axis_plan) 축별 이동량·예상 시간으로 자세를 추정하고, 끝나면 보고 위치로 맞춘다
//   글씨는 모두 페이지 HTML 에서 그림 (웹 빌드에는 한글 글꼴이 없음)
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

        // 페이지 JS 가 verify_web 상태에서 골라 넘기는 값 (null 이 없게 평평하게)
        [System.Serializable]
        class St
        {
            public bool hasPos; public float y, t;          // 스테이지 위치 (mm, °) = verify_web state.position
            public bool moving;
            public bool hasAxes; public double started, now; // target.axes.started, state.now (서버 시각, 초)
            public float dy, dt, ys, ts;                     // 축별 이동량과 예상 시간 (모르면 ys/ts < 0)
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
        }

        public void OnState(string json)
        {
            st = JsonUtility.FromJson<St>(json);
            if (st.now > 0) serverOffset = st.now - Time.realtimeSinceStartupAsDouble;
        }

        public void SetView(string key)
        {
            if (orbit == null) return;
            foreach (var v in Rig.Views) if (v.key == key) orbit.TweenTo(v.pos, v.target);
        }

        static float Part(float d, float sec, double elapsed)
        {
            if (sec < 0) return 0;                   // 속도를 모름: 끝날 때까지 출발 위치
            if (sec == 0) return d;                  // 0.5 µm 미만 이동
            return d * Mathf.Clamp01((float)(elapsed / sec));
        }

        void Update()
        {
            if (rig == null || st == null || !st.hasPos) return;
            float y = st.y, t = st.t;
            if (st.moving && st.hasAxes)
            {   // 두 축은 동시에 출발 (runner.py _axis_plan). 가감속은 빼고 일정 속도로 봄
                double e = Time.realtimeSinceStartupAsDouble + serverOffset - st.started;
                y += Part(st.dy, st.ys, e); t += Part(st.dt, st.ts, e);
            }
            // 펌웨어 위치 → 도면 자세 (원점 = −Y 스위치 작동점 −22.4 라고 보고 그림, Windows 대시보드와 같은 가정)
            float yy = -Rig.Y_SWITCH_ON + y;
            if (!Mathf.Approximately(yy, lastY) || !Mathf.Approximately(t, lastT)) { rig.SetPose(yy, t); lastY = yy; lastT = t; }
            if (debug && Time.unscaledTime > nextLog)
            {   // 확인용: 주소에 ?debug=1 이면 0.5초마다 그린 자세를 브라우저 콘솔에
                nextLog = Time.unscaledTime + 0.5f;
                Debug.Log($"[viewer] 보고 Y {st.y:0.000} Θ {st.t:0.000} · 그림 Y {y:0.000} Θ {t:0.000} · 이동 {st.moving}");
            }
        }
    }
}
