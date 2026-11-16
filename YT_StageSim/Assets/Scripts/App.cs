// 시작점: 장면에 아무것도 없어도 실행하면 장비·카메라·화면을 만듦
// 화면: GitLab dev verify_web 기준 (보정 확인 / 연결·설정). demo_web 전용 기능(시나리오·Random·검증·로그인)은 뺌 (Legacy_demo_web 폴더에 보관)
// 좌표 표시: 펌웨어 위치 (원점 복귀한 자리 = 0, mm·° 소수 셋째 자리)
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.UI;

namespace YT
{
    public class App : MonoBehaviour
    {
        public Rig rig;
        public MockStage mock;
        public IStage stage;
        public SemesView semes;
        public OrbitCam orbit;
        public Camera mainCam;
        public Shell shell;
        public VerifyRunner verify;
        public ICam cam;
        public bool showFov = true;
        float explode;
        bool iso, autoView; float autoT0;
        readonly Dictionary<Renderer, Material> origMat = new Dictionary<Renderer, Material>();
        readonly Dictionary<Material, Material> ghostOf = new Dictionary<Material, Material>();
        float lastY = float.NaN, lastT = float.NaN;
        string selected;
        Vector3 downPos;

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        static void Boot()
        {
            if (FindAnyObjectByType<App>() == null) new GameObject("App").AddComponent<App>();
        }

        void Awake()
        {
            Application.targetFrameRate = 60;
            var sw = System.Diagnostics.Stopwatch.StartNew();
            BuildWorld(transform, out rig, out mainCam, out orbit, out semes);
            double tWorld = sw.Elapsed.TotalMilliseconds;
            mock = new MockStage(0, 0);
            stage = mock;
            var args = System.Environment.GetCommandLineArgs();
            int i = System.Array.IndexOf(args, "-ytselftest");
            string dataDir = Application.persistentDataPath;
            if (i >= 0 && i + 1 < args.Length)
            {   // 자동 점검은 따로 폴더 사용 (평소 기록을 건드리지 않음)
                dataDir = args[i + 1]; System.IO.Directory.CreateDirectory(dataDir);
                var old = System.IO.Path.Combine(dataDir, "verify_settings.json"); if (System.IO.File.Exists(old)) System.IO.File.Delete(old);
            }
            cam = new SimCam(this);
            verify = new VerifyRunner(this, dataDir);   // GitLab dev verify_web 옮김
            // 화면 밖 영역은 흰 바탕 (웹 배경), 3D 카메라는 시연 화면의 3D 칸에만 그림
            var bg = new GameObject("BackgroundCamera").AddComponent<Camera>();
            bg.transform.SetParent(transform, false);
            bg.depth = -10; bg.cullingMask = 0; bg.clearFlags = CameraClearFlags.SolidColor; bg.backgroundColor = Color.white;
            mainCam.depth = 0;
            shell = new Shell(this);
            orbit.pointerBlocked = () => !shell.InHole(Input.mousePosition);
            if (i >= 0 && i + 1 < args.Length) StartCoroutine(SelfTest(args[i + 1]));
            int j = System.Array.IndexOf(args, "-ytperf");
            if (j >= 0 && j + 1 < args.Length) StartCoroutine(Perf(args[j + 1]));
            startupNote = $"시작 측정: 장비 만들기 {tWorld:0} ms (STL 읽기 {Stl.LoadMs:0} ms, 충돌 모양 {Stl.ColliderMs:0} ms), 대시보드·주석 {sw.Elapsed.TotalMilliseconds - tWorld:0} ms, 엔진 시작부터 {Time.realtimeSinceStartup:0.00} s";
            Debug.Log(startupNote);
        }

        string startupNote = "";

        // 성능 측정 (확인용): 시작 시간, 대기 중·자동 움직이기 중 프레임 시간
        IEnumerator Perf(string file)
        {
            var lines = new List<string> { $"GPU {SystemInfo.graphicsDeviceName} ({SystemInfo.graphicsDeviceType}), CPU {SystemInfo.processorType}, 화면 {Screen.width}×{Screen.height}",
                                           startupNote };
            yield return new WaitForSeconds(2f);
            IEnumerator Measure(string name, float sec)
            {
                var ft = new List<float>(); float t0 = Time.realtimeSinceStartup;
                while (Time.realtimeSinceStartup - t0 < sec) { yield return null; ft.Add(Time.unscaledDeltaTime * 1000); }
                ft.Sort();
                lines.Add($"{name}: 프레임 {ft.Count}개, 평균 {ft.Average():0.0} ms ({1000 / ft.Average():0} fps), 95% {ft[(int)(ft.Count * 0.95)]:0.0} ms, 최대 {ft[ft.Count - 1]:0.0} ms");
            }
            yield return Measure("대기", 3f);
            ToggleAutoView(); yield return Measure("자동 움직이기", 3f); ToggleAutoView();
            semes.cam.enabled = false; yield return Measure("대기 (카메라 화면 끔)", 3f); semes.cam.enabled = true;
            System.IO.File.WriteAllLines(file, lines);
            Application.Quit();
        }

        // 자동 점검 (확인용): 통신 규약 자기 검사 → 원점 → 센터 → 랜덤 시작 → 자동 보정 → 기록·연결 화면을 찍고 종료
        IEnumerator SelfTest(string dir)
        {
            var lines = new List<string>();
            stage.Log += s => lines.Add(s);
            System.Func<string, IEnumerator> shot = name =>
            {
                IEnumerator Co() { yield return null; yield return new WaitForEndOfFrame(); ScreenCapture.CaptureScreenshot(System.IO.Path.Combine(dir, name)); yield return null; yield return null; }
                return Co();
            };
            System.Action<string> note = s =>
            {
                var st = stage.Status;
                lines.Add($"## {s}");
                lines.Add($"   단계 {verify.phase} ({verify.message}) 회차 {verify.run?.trial}, 장비 {StText.State(st.state)} 플래그 {st.flags} Y {StText.Mm(st.yUm)} Θ {StText.Mm(st.tMdeg)}");
            };
            IEnumerator Until(System.Func<bool> f, float max = 60) { float t0 = Time.realtimeSinceStartup; while (!f() && Time.realtimeSinceStartup - t0 < max) yield return null; }
            yield return new WaitForSeconds(1.5f);
            lines.Add($"## 통신 규약 예시 프레임 검사 (robot_stage_protocol.md 9장): {(RsProto.SelfCheck() ? "일치" : "불일치")}");
            note("시작"); yield return shot("01_boot.png");
            shell.verify.TestShowHistory(); yield return null; yield return shot("01b_history_empty.png"); shell.verify.TestShowNow();
            W.Act(verify.Run); yield return Until(() => !verify.busy, 10); note("원점 전 자동 보정 (거부 기대)"); yield return shot("02_error.png");
            mock.TimeScale = 50;
            W.Act(verify.Home); yield return Until(() => !verify.busy); note("Y 원점 복귀");
            W.Act(verify.ThetaZero); yield return Until(() => !verify.busy);
            var vs = verify.settings.Clone(); vs.center_y_mm = 5; vs.center_t_deg = 0; verify.UpdateSettings(vs);
            W.Act(() => verify.Goto("random")); yield return Until(() => !verify.busy); note("랜덤 시작 위치");
            yield return shot("03_random.png");
            W.Act(verify.Run); yield return Until(() => verify.phase == "inferring" || verify.phase == "moving" || !verify.busy);
            note("자동 보정 중"); yield return shot("04_running.png");
            yield return Until(() => !verify.busy, 120); note("자동 보정 끝");
            var lv = verify.last;
            lines.Add($"   마지막 실제 오차 Y {lv?.truY:0.0000} Θ {lv?.truT:0.0000}, 기준 {(verify.reference != null)}, 알고리즘 {verify.algo.Kind}, Θ 명령 {mock.TSim:0.0000} 기구 {mock.TPhys:0.0000}");
            lines.Add($"   기록 폴더 파일 {(System.IO.Directory.Exists(verify.RunsDir) ? System.IO.Directory.GetFiles(verify.RunsDir, "*", System.IO.SearchOption.AllDirectories).Length : 0)}개");
            yield return shot("05_done.png");
            shell.verify.TestShowHistory(); yield return null; yield return shot("06_history.png");
            Select("motor"); yield return null; yield return shot("07_part.png"); Select(null);
            shell.Show(1); yield return null; yield return null; yield return shot("08_connect.png");
            {   // 연결 실패 처리 (없는 COM 포트, 없는 파이 주소) — 장치 없이 오류 문구만 확인
                try { LinkStage.OpenSerial("COM250"); lines.Add("   COM250: 열림 (예상 밖)"); } catch (RunError e) { lines.Add("   COM250: " + e.Message); }
                try { LinkStage.OpenTcp("127.0.0.1", 59999); lines.Add("   tcp 59999: 열림 (예상 밖)"); } catch (RunError e) { lines.Add("   tcp 59999: " + e.Message); }
                try { new PiCam("http://127.0.0.1:59998"); lines.Add("   pi cam: 열림 (예상 밖)"); } catch (RunError e) { lines.Add("   pi cam: " + e.Message); }
            }
            shell.Show(0); yield return null;
            if (System.Array.IndexOf(System.Environment.GetCommandLineArgs(), "-ytalgos") >= 0)
            {   // 알고리즘 고르기 + 영상 윤곽 (가짜 실행기: READY ref roi, 위치를 돌려줌)
                lines.Add($"## 알고리즘 목록 시험: {verify.algos.file}, {verify.algos.entries.Count}개, 오류 {verify.algos.loadError ?? "없음"}");
                try { verify.SelectAlgorithm("missing"); lines.Add("   [실패] 실행기 없는 알고리즘이 골라짐"); } catch (RunError e) { lines.Add("   실행기 없는 알고리즘 거부: " + e.Message); }
                try { verify.SelectAlgorithm("fake"); lines.Add($"   고름: {verify.algoId} → {verify.algo.Detail}, 준비 오류 {verify.algoInitError ?? "없음"}, 기준 {(verify.reference != null)}"); }
                catch (RunError e) { lines.Add("   [실패] " + e.Message); }
                yield return null; yield return null; yield return shot("12_algo_pick.png");
                shell.verify.TestZoom(2, true);
                W.Act(verify.Measure); yield return Until(() => !verify.busy, 60); note("가짜 실행기: 센터 기준 + 측정 1 (벗어남 기대)");
                lines.Add($"   기준 ROI {verify.refRois?.Count}, 지금 ROI {verify.roisNow?.Count}, 정렬 {verify.aligned}, 영점 고정 {verify.refFixed}, 실행기 오류 {verify.algo.Error ?? "없음"}");
                yield return null; yield return null; yield return shot("13_overlay_bad.png");
                W.Act(verify.Measure); yield return Until(() => !verify.busy, 60); note("가짜 실행기: 측정 2 (센터 OK 기대)");
                lines.Add($"   지금 ROI {verify.roisNow?.Count}, 정렬 {verify.aligned}");
                yield return null; yield return null; yield return shot("14_overlay_ok.png");
                shell.verify.TestZoom(1, false); yield return null; yield return null; yield return shot("15_overlay_1x.png");
                W.Act(() => verify.Goto((0.5, 0.1))); yield return Until(() => !verify.busy, 60); note("이동 뒤 (윤곽: 지금 지워짐, 측정 전 기대)");
                lines.Add($"   지금 ROI {verify.roisNow?.Count ?? 0}, 정렬 {verify.aligned?.ToString() ?? "모름"}");
                yield return null; yield return null; yield return shot("16_overlay_moved.png");
                try { verify.SelectAlgorithm("mock"); lines.Add($"   모의로 되돌림: {verify.algoId}, 기준 {(verify.reference != null)}, 기록 {verify.history.Count}"); } catch (RunError e) { lines.Add("   [실패] " + e.Message); }
                yield return null;
            }
            if (System.Array.IndexOf(System.Environment.GetCommandLineArgs(), "-ytfakepi") >= 0)
            {   // 가짜 파이(시험용 프로그램)로 '라즈베리파이 경유' 연결 경로 시험: 프로토콜 프레임·ACK·DONE·영상·사진 받기
                lines.Add("## 가짜 파이 연결 시험");
                IStage ls = null;
                try { ls = LinkStage.OpenTcp("127.0.0.1", 5555); UseStage(ls); lines.Add($"   스테이지 연결: {ls.Name}, 펌웨어 {string.Join(".", ((LinkStage)ls).Firmware)}"); }
                catch (RunError e) { lines.Add("   스테이지 연결 실패: " + e.Message); }
                if (ls != null) { try { verify.Measure(); lines.Add("   [실패] 실물 스테이지 + 시뮬레이터 카메라 보정이 막히지 않음"); } catch (RunError e) { lines.Add("   실물 스테이지 + 시뮬레이터 카메라 보정 차단: " + e.Message); } }
                try { UseCam(new PiCam("http://127.0.0.1:8091")); lines.Add("   카메라 연결: " + cam.Label); }
                catch (RunError e) { lines.Add("   카메라 연결 실패: " + e.Message); }
                if (ls != null)
                {
                    ls.Log += s => lines.Add(s);
                    if (System.Array.IndexOf(System.Environment.GetCommandLineArgs(), "-ytalgos") >= 0)
                        try { verify.SelectAlgorithm("fake"); lines.Add("   파이 경로 알고리즘: " + verify.algo.Detail); } catch (RunError e) { lines.Add("   [실패] " + e.Message); }
                    W.Act(verify.Home); yield return Until(() => !verify.busy, 20); note("실물 경로: Y 원점");
                    W.Act(verify.ThetaZero); yield return Until(() => !verify.busy, 20); note("실물 경로: Θ 지금 자리를 0으로");
                    W.Act(() => verify.Goto("random")); yield return Until(() => !verify.busy, 30); note("실물 경로: 랜덤 시작");
                    W.Act(verify.Run); yield return Until(() => !verify.busy, 200); note("실물 경로: 자동 보정 (가짜 파이 사진은 검은 화면, 추론은 모의)");
                    yield return shot("09_fakepi.png");
                    shell.Show(1); yield return null; yield return null; yield return shot("10_fakepi_connect.png"); shell.Show(0);
                    System.IO.File.WriteAllText(System.IO.Path.Combine(dir, "fakepi_done.flag"), "");   // 시험 도구가 이걸 보고 가짜 파이를 끔
                    yield return Until(() => !cam.Live && !stage.Connected, 40);
                    yield return new WaitForSeconds(1.2f);
                    lines.Add($"## 가짜 파이 끔 → 영상 {(cam.Live ? "있음" : "없음")}, 스테이지 연결 {stage.Connected}, 오류 {((LinkStage)ls).Error}");
                    yield return shot("11_link_lost.png");
                }
            }
            System.IO.File.WriteAllLines(System.IO.Path.Combine(dir, "selftest_log.txt"), lines);
            yield return new WaitForSeconds(0.5f);
            Application.Quit();
        }

        // ---------------- 장치 바꾸기 (연결·설정 탭)
        public void UseStage(IStage s)
        {
            if (stage == s) return;
            if (stage is LinkStage old) old.Close();
            stage = s; verify.Attach(s); verify.RefreshStatus(); lastY = float.NaN;
            if (s != mock) verify.Claim();   // verify_web devices.py: 장치를 잡을 때 Y 속도 1 mm/s
        }
        public void UseMock() => UseStage(mock);
        public void UseCam(ICam c) { if (cam != null && cam != c) cam.Close(); cam = c; }

        // 편집기 캡처에서도 쓰려고 정적 함수로 둠
        public static void BuildWorld(Transform parent, out Rig rig, out Camera cam, out OrbitCam orbit, out SemesView semes)
        {
            RenderSettings.ambientMode = AmbientMode.Trilight;
            RenderSettings.ambientSkyColor = Color.white * 0.85f;
            RenderSettings.ambientEquatorColor = Color.white * 0.75f;
            RenderSettings.ambientGroundColor = Geo.Hex(0x7d837f) * 0.85f;
            QualitySettings.shadowDistance = 3000;
            Light(parent, "Sun", Geo.P(-400, 700, -300), 0.75f, true);
            Light(parent, "Fill", Geo.P(500, 300, 400), 0.3f, false);

            rig = Rig.Build(parent);
            rig.SetPose(0, 0);

            var cgo = new GameObject("MainCamera");
            cgo.transform.SetParent(parent, false);
            cgo.tag = "MainCamera";
            cam = cgo.AddComponent<Camera>();
            cam.fieldOfView = 35; cam.nearClipPlane = 5; cam.farClipPlane = 8000;
            cam.clearFlags = CameraClearFlags.SolidColor;
            cam.backgroundColor = new Color(0.906f, 0.914f, 0.898f);
            cam.allowMSAA = true;
            orbit = cgo.AddComponent<OrbitCam>();
            var v = Rig.Views[0];
            orbit.Set(v.pos, v.target);
            semes = SemesView.Make(rig.camMount);
        }

        static void Light(Transform parent, string name, Vector3 pos, float intensity, bool shadow)
        {
            var go = new GameObject(name);
            go.transform.SetParent(parent, false);
            var l = go.AddComponent<Light>();
            l.type = LightType.Directional; l.intensity = intensity;
            l.shadows = shadow ? LightShadows.Soft : LightShadows.None;
            go.transform.rotation = Quaternion.LookRotation(-pos.normalized);
        }

        void Update()
        {
            stage.Tick(Time.deltaTime);
            if (autoView)
            {
                float t = (Time.time - autoT0) % 12f, s = Mathf.Sin((t % 6f) / 6f * Mathf.PI * 2);   // 도면 playStep: 0~6초 Y, 6~12초 Θ
                if (t < 6) rig.SetPose(25 * s, 0); else rig.SetPose(0, 10 * s);
                lastY = float.NaN;
            }
            else
            {
                float y, t;
                if (stage == mock) { y = mock.YSim; t = mock.TPhys; }   // 기구 자세 (Θ 백래시 반영)
                else
                {   // 실물: 보고 위치를 도면 자세로 (원점 = −Y 스위치 작동점 −22.4 라고 보고 그림, 실제 장착 위치와 다를 수 있음)
                    var st = verify.posY; y = st == null ? 0 : -Rig.Y_SWITCH_ON + (float)st.Value; t = (float)(verify.posT ?? 0);
                }
                if (!Mathf.Approximately(y, lastY) || !Mathf.Approximately(t, lastT)) { rig.SetPose(y, t); lastY = y; lastT = t; }
            }
            cam.Tick();
            shell.Tick();
            HandlePick();
        }

        public void SetExplode(float k)
        {
            explode = k; rig.SetExplode(k);
            ApplyFov();
        }

        public void ApplyFov() { rig.fovLines.enabled = showFov && explode < 0.02f; }

        public void ToggleAutoView()
        {
            autoView = !autoView; autoT0 = Time.time; lastY = float.NaN;
            shell.connect.SetAutoLabel(autoView);
        }

        // 선택한 부품만 또렷하게: 나머지는 반투명 (도면 cIso)
        public void SetIso(bool on) { iso = on; ApplyIso(); }

        // 반투명: Standard 의 Fade 변형은 빌드에서 빠지므로 전용 투명 셰이더 사용 (빌드 '항상 포함' 목록에 넣음)
        Material Ghost(Material m)
        {
            if (ghostOf.TryGetValue(m, out var g)) return g;
            var c = m.HasProperty(ColorId) ? m.color : Color.gray; c.a = 0.12f;
            g = new Material(Shader.Find("Legacy Shaders/Transparent/Diffuse")) { name = m.name + "_ghost", color = c };
            ghostOf[m] = g;
            return g;
        }

        void ApplyIso()
        {
            foreach (var kv in origMat) if (kv.Key != null) kv.Key.sharedMaterial = kv.Value;
            origMat.Clear();
            if (!iso || selected == null) return;
            foreach (var p in rig.parts.Values)
            {
                if (p.id == selected || p.id == "room") continue;
                foreach (var r in p.renderers)
                {
                    if (r.sharedMaterial == null || !r.sharedMaterial.HasProperty(ColorId)) continue;
                    origMat[r] = r.sharedMaterial; r.sharedMaterial = Ghost(r.sharedMaterial);
                }
            }
        }

        // 3D 화면에서 부품 누르기 (끌지 않고 누른 경우만)
        void HandlePick()
        {
            if (Input.GetMouseButtonDown(0)) downPos = Input.mousePosition;
            if (!Input.GetMouseButtonUp(0) || (Input.mousePosition - downPos).magnitude > 5 || !shell.InHole(Input.mousePosition) || (UnityEngine.EventSystems.EventSystem.current != null && UnityEngine.EventSystems.EventSystem.current.IsPointerOverGameObject())) return;
            var ray = mainCam.ScreenPointToRay(Input.mousePosition);
            string id = null;
            foreach (var h in SortedHits(ray))
            {
                var rd = h.collider.GetComponent<Renderer>();
                var pid = rig.PartOf(rd);
                if (pid == null || !rd.enabled) continue;
                if (pid == "room" && id == null) { id = "room"; continue; }
                id = pid; break;
            }
            Select(id);
        }

        static IEnumerable<RaycastHit> SortedHits(Ray ray)
        {
            var hits = Physics.RaycastAll(ray, 10000);
            System.Array.Sort(hits, (a, b) => a.distance.CompareTo(b.distance));
            return hits;
        }

        static readonly int ColorId = Shader.PropertyToID("_Color");

        public void Select(string id)
        {
            if (selected != null) foreach (var r in rig.parts[selected].renderers) r.SetPropertyBlock(null);
            selected = id;
            if (id != null)
            {
                var mpb = new MaterialPropertyBlock();
                foreach (var r in rig.parts[id].renderers)
                {
                    if (r.sharedMaterial == null || !r.sharedMaterial.HasProperty(ColorId)) continue;
                    mpb.SetColor(ColorId, Color.Lerp(r.sharedMaterial.color, new Color(0.18f, 0.44f, 1f), 0.6f));
                    r.SetPropertyBlock(mpb);
                }
            }
            shell.verify.ShowPart(id == null ? null : rig.parts[id]);
            ApplyIso();
        }

        public void ToggleHidden(string id)
        {
            foreach (var r in rig.parts[id].renderers) r.enabled = !r.enabled;
        }

        public void ShowAll()
        {
            foreach (var p in rig.parts.Values) foreach (var r in p.renderers) r.enabled = true;
        }

        public void Focus(string id)
        {
            var b = new Bounds(); bool init = false;
            foreach (var r in rig.parts[id].renderers) { if (!init) { b = r.bounds; init = true; } else b.Encapsulate(r.bounds); }
            if (!init) return;
            var dir = (orbit.transform.position - orbit.target).normalized;
            float d = id == "room" ? 1100 : (id == "elec" || id == "base" ? 520 : 300);
            orbit.TweenTo(b.center + dir * d, b.center);
        }
    }
}
