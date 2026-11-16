// 연결·설정 탭 (디자인 캔버스 'verify 기준 화면 배치안' 연결·설정 확정본)
//   ① 스테이지: 모의 / USB 직접 (COM) / 라즈베리파이 경유 (pi_bridge.py TCP)
//   ② 카메라: 시뮬레이터 / 라즈베리파이 영상 (pi_bridge.py HTTP)
//   ③ 알고리즘: 실행기 경로 직접 지정 (자주 쓰는 것은 algorithms.json 에 적고 위 막대에서 고름)
//   보정 설정 (verify_web 설정과 같은 항목) · 시뮬레이터 (모의 장비·3D 보기)
using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.UI;

namespace YT
{
    public class ConnectPage
    {
        readonly Shell shell;
        public readonly RectTransform root;
        App app => shell.app;
        VerifyRunner rn => app.verify;

        List<Button> segStage, segCam, segOut; int stageSel, camSel, outSel;
        InputField fCom, fPiStage, fPiCam, fAlgo, fAlgoCfg;
        Text stageInfo, camInfo, algoInfo, comList, tExplode;
        Toggle tGray;
        readonly Dictionary<string, InputField> sf = new Dictionary<string, InputField>();
        readonly Dictionary<string, Toggle> sc = new Dictionary<string, Toggle>();
        Button bAuto;

        static readonly (string key, string label)[] NUM_FIELDS =
        {
            ("center_y_mm", "센터 Y (mm)"), ("center_t_deg", "센터 Θ (°)"), ("tol_y_mm", "허용 Y (mm)"), ("tol_t_deg", "허용 Θ (°)"), ("max_trials", "최대 회차"),
            ("max_step_y_mm", "한 번 이동 한도 Y (mm)"), ("max_step_t_deg", "한 번 이동 한도 Θ (°)"), ("random_y_mm", "랜덤 범위 Y ± (mm)"), ("random_t_deg", "랜덤 범위 Θ ± (°)"), ("approach_y_mm", "접근 거리 Y (mm)"),
            ("approach_t_deg", "접근 거리 Θ (°)"), ("average", "평균 장수"), ("settle_px", "안정 판정 (px)"), ("settle_timeout_s", "안정 대기 한도 (s)"), ("reference_max_age_min", "센터 기준 유효 시간 (분)"),
        };
        static readonly (string key, string label)[] BOOL_FIELDS = { ("enable_t", "Θ 명령 보내기"), ("save_images", "회차 사진 저장"), ("auto_reference", "시작할 때 센터 기준 자동") };

        public ConnectPage(Shell s, RectTransform parent)
        {
            shell = s;
            root = Ui.Anchor(Ui.NewRect("connect", parent), Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero);
            Ui.Bg(root, Color.white);
            var page = Ui.Anchor(Ui.NewRect("page", root), Vector2.zero, Vector2.one, new Vector2(56, 24), new Vector2(-56, -20));
            var pv = page.gameObject.AddComponent<VerticalLayoutGroup>(); pv.spacing = 20; pv.childControlWidth = pv.childControlHeight = true; pv.childForceExpandWidth = true; pv.childForceExpandHeight = false;

            var top = Row3(page);
            BuildStage(Col(top)); BuildCamera(Col(top)); BuildAlgo(Col(top));
            var bot = Row3(page); Ui.Pref(bot, h: 0).flexibleHeight = 1;
            var c2 = Col(bot, 2); BuildSettings(c2);
            BuildSim(Col(bot));
            Fill();
        }

        static RectTransform Row3(Transform p)
        {
            var r = Ui.NewRect("row", p); var g = r.gameObject.AddComponent<HorizontalLayoutGroup>(); g.spacing = 20;
            g.childControlWidth = g.childControlHeight = true; g.childForceExpandWidth = false; g.childForceExpandHeight = true;
            return r;
        }
        static RectTransform Col(RectTransform row, float flex = 1)
        {
            var c = Ui.VBox(row, "col", 0, 0); var le = c.gameObject.AddComponent<LayoutElement>(); le.preferredWidth = 0; le.flexibleWidth = flex; return c;
        }
        static RectTransform Card(Transform col, string title, string sub, out RectTransform head)
        {
            var c = W.Panel(col, "card"); Ui.Pref(c, h: 0).flexibleHeight = 1;
            head = W.Row(c, 10, 62, false); head.GetComponent<HorizontalLayoutGroup>().padding = new RectOffset(20, 20, 0, 0);
            W.Fill(head, W.HEAD, false);
            var tv = Ui.VBox(head, "t", 0, 2); Ui.Pref(tv, 0, -1, 1);
            W.T(tv, title, 15, W.ACCENT, true, TextAnchor.MiddleLeft, 22); W.T(tv, sub, 12, W.MUTED, false, TextAnchor.MiddleLeft, 18);
            W.Line(c);
            var b = Ui.VBox(c, "body", 0, 10); b.GetComponent<VerticalLayoutGroup>().padding = new RectOffset(20, 20, 14, 14);
            return b;
        }
        static Text Info(Transform p)
        {
            var box = Ui.VBox(p, "info", 0, 0); box.GetComponent<VerticalLayoutGroup>().padding = new RectOffset(12, 12, 8, 8);
            W.Fill(box, W.SURFACE2); var t = W.T(box, "", 13, W.INK); t.alignment = TextAnchor.UpperLeft; t.lineSpacing = 1.25f; return t;
        }
        static void Label(Transform p, string s) => W.T(p, s, 12, W.MUTED, false, TextAnchor.MiddleLeft, 18);

        // ---------------- ① 스테이지
        void BuildStage(RectTransform col)
        {
            var b = Card(col, "① 스테이지 (STM32)", "연결 전에는 모의 장비로 동작합니다", out _);
            Label(b, "연결 방식");
            segStage = W.Segmented(b, new[] { "모의", "USB 직접", "라즈베리파이 경유" }, 0, k => stageSel = k, 40, 13);
            var r = W.Row(b, 10, 66);
            fCom = W.Field(r, "COM 포트 (USB 직접)", "COM4"); fPiStage = W.Field(r, "파이 주소:포트 (경유)", "192.168.0.10:5555");
            comList = W.T(b, "", 11, W.MUTED, false, TextAnchor.MiddleLeft, 16);
            var br = W.Row(b, 8, 42, false);
            var bc = W.Btn(br, "연결", () => W.Act(ConnectStage), W.B.Primary, 42, 14); Ui.Pref((RectTransform)bc.transform, 0, 42, 1);
            W.Btn(br, "끊기 (모의로)", () => W.Act(() => { app.UseMock(); W.Toast("모의 장비로 돌아왔습니다."); }), W.B.Secondary, 42, 14, 130);
            stageInfo = Info(b);
        }

        void ConnectStage()
        {
            if (rn.busy) throw new RunError("작업 중에는 연결을 바꿀 수 없습니다. 정지 후 다시 시도하세요.");
            if (stageSel == 0) { app.UseMock(); W.Toast("모의 장비로 동작합니다."); return; }
            IStage st;
            if (stageSel == 1) st = LinkStage.OpenSerial(fCom.text.Trim().ToUpperInvariant());
            else
            {
                var hp = fPiStage.text.Trim().Split(':');
                if (hp.Length != 2 || !int.TryParse(hp[1], out var port)) throw new RunError("파이 주소는 IP:포트 형식으로 입력하세요 (예: 192.168.0.10:5555).");
                st = LinkStage.OpenTcp(hp[0], port);
            }
            app.UseStage(st); SyncSegments();
            W.Toast($"연결됨: {st.Name}");
        }

        // ---------------- ② 카메라
        void BuildCamera(RectTransform col)
        {
            var b = Card(col, "② 카메라", "IMX477 은 라즈베리파이 전용 → 파이(pi_bridge.py)에서 영상을 받습니다", out _);
            Label(b, "영상 출처");
            segCam = W.Segmented(b, new[] { "시뮬레이터 가상 카메라", "라즈베리파이 영상" }, 0, k => camSel = k, 40, 13);
            fPiCam = W.Field(Ui.VBox(b, "f", 0, 0), "파이 영상 주소", "http://192.168.0.10:8091");
            var br = W.Row(b, 8, 42, false);
            var bc = W.Btn(br, "적용", () => W.Act(ApplyCamera), W.B.Primary, 42, 14); Ui.Pref((RectTransform)bc.transform, 0, 42, 1);
            camInfo = Info(b);
        }

        void ApplyCamera()
        {
            if (rn.busy) throw new RunError("작업 중에는 카메라를 바꿀 수 없습니다. 정지 후 다시 시도하세요.");
            if (camSel == 0) { app.UseCam(new SimCam(app)); W.Toast("시뮬레이터 가상 카메라를 씁니다."); return; }
            app.UseCam(new PiCam(fPiCam.text.Trim())); SyncSegments();
            W.Toast("파이 영상을 받습니다.");
        }

        // ---------------- ③ 알고리즘
        void BuildAlgo(RectTransform col)
        {
            var b = Card(col, "③ 알고리즘 직접 지정", "목록(algorithms.json)에 있는 알고리즘은 위 막대에서 바로 고름", out var head);
            W.Btn(head, "목록 파일 폴더 ↗", () => W.Act(OpenAlgoList), W.B.Text, 30, 12, 130);
            fAlgo = W.Field(Ui.VBox(b, "f", 0, 0), "실행기 경로 (비우면 모의 추론)", "");
            fAlgoCfg = W.Field(Ui.VBox(b, "f", 0, 0), "알고리즘 설정 파일 (alignment.yml)", "");
            Label(b, "출력 형식");
            segOut = W.Segmented(b, new[] { "오프셋 (센터 대비 위치)", "명령값 (보낼 이동)" }, 0, k => outSel = k, 40, 13);
            var r = W.Row(b, 8, 40, false);
            tGray = W.Check(r, "흑백 1채널 그대로 넘김", _ => { }); Ui.Pref((RectTransform)tGray.transform, 200, 26);
            var ba = W.Btn(r, "적용", () => W.Act(ApplyAlgo), W.B.Primary, 40, 14); Ui.Pref((RectTransform)ba.transform, 0, 40, 1);
            algoInfo = Info(b);
        }

        void ApplyAlgo()
        {
            var n = rn.settings.Clone();
            n.algo_path = fAlgo.text.Trim(); n.algo_config = fAlgoCfg.text.Trim(); n.algo_output = outSel == 1 ? "command" : "offset"; n.algo_gray = tGray.isOn;
            if (rn.busy) throw new RunError("작업 중에는 알고리즘을 바꿀 수 없습니다.");
            n.algorithm = n.algo_path.Length == 0 ? "mock" : "custom";
            rn.UpdateSettings(n);
            W.Toast(rn.algoInitError ?? (rn.algo.Kind == "mock" ? "모의 추론을 씁니다." : "알고리즘 실행기를 씁니다: " + rn.algo.Detail), rn.algoInitError != null);
        }

        // algorithms.json 이 있는 폴더를 열고, 파일이 없으면 빈 목록을 만들어 둠
        void OpenAlgoList()
        {
            var f = rn.algos.file;
            if (!System.IO.File.Exists(f))
                System.IO.File.WriteAllText(f, "{\n \"algorithms\": []\n}\n");
            Application.OpenURL("file:///" + System.IO.Path.GetDirectoryName(f).Replace('\\', '/'));
        }

        // ---------------- 보정 설정
        void BuildSettings(RectTransform col)
        {
            var b = Card(col, "보정 설정", "verify_web 설정과 같은 항목 · 저장하면 다음 작업부터 적용", out var head);
            W.Btn(head, "저장", () => W.Act(SaveSettings), W.B.Primary, 36, 14, 100);
            for (int i = 0; i < NUM_FIELDS.Length; i += 5)
            {
                var row = W.Row(b, 12, 62);
                for (int k = i; k < i + 5; k++) sf[NUM_FIELDS[k].key] = W.Field(row, NUM_FIELDS[k].label, "", -1, 38);
            }
            var cr = W.Row(b, 24, 28, false);
            foreach (var (key, label) in BOOL_FIELDS) { var t = W.Check(cr, label, _ => { }); Ui.Pref((RectTransform)t.transform, 230, 26); sc[key] = t; }
        }

        void SaveSettings()
        {
            var n = rn.settings.Clone();
            foreach (var (k, label) in NUM_FIELDS)
            {
                var f = typeof(VerifySettings).GetField(k);
                if (!double.TryParse(sf[k].text.Trim(), System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var v)) throw new RunError($"{label}: 숫자가 필요함");
                if (f.FieldType == typeof(int)) { if (v != Math.Floor(v)) throw new RunError($"{label}: 정수가 필요함"); f.SetValue(n, (int)v); }
                else f.SetValue(n, (float)v);
            }
            foreach (var (k, _) in BOOL_FIELDS) typeof(VerifySettings).GetField(k).SetValue(n, sc[k].isOn);
            rn.UpdateSettings(n);
            W.Toast("설정 저장");
        }

        // ---------------- 시뮬레이터 (Unity 전용)
        void BuildSim(RectTransform col)
        {
            var b = Card(col, "시뮬레이터", "모의 장비 · 3D 보기 (Unity 전용)", out _);
            var r0 = W.Row(b, 10, 36, false);
            var l0 = W.T(r0, "배속", 13, W.MUTED); Ui.Pref(l0.rectTransform, 40);
            var scales = new[] { 1, 10, 50 };
            var box = Ui.NewRect("seg", r0); Ui.Pref(box, 200, 34); var g = box.gameObject.AddComponent<VerticalLayoutGroup>(); g.childControlWidth = g.childControlHeight = true; g.childForceExpandWidth = true; g.childForceExpandHeight = false;
            W.Segmented(box, scales.Select(k => "×" + k).ToArray(), 0, k => app.mock.TimeScale = scales[k], 34, 13);
            var r1 = W.Row(b, 8, 36);
            W.Btn(r1, "모의 장비 다시 켜기", () => W.Act(() => { if (rn.busy) throw new RunError("작업 중입니다."); app.mock.Reboot(); W.Toast("모의 장비를 다시 켰습니다. 원점이 해제됩니다."); }), W.B.Secondary, 36, 13);
            W.Check(b, "Θ 백래시 0.023° (모의 장비)", v => app.mock.BacklashOn = v, app.mock.BacklashOn);
            W.Check(b, "다시 켤 때 원점 ±0.4 mm 흔들림 (모의 장비)", v => app.mock.HomeJitterOn = v, app.mock.HomeJitterOn);
            W.Line(b);
            W.Check(b, "카메라 시야 선 (분해 중에는 숨김)", v => { app.showFov = v; app.ApplyFov(); }, true);
            W.Check(b, "선택한 부품만 또렷하게", v => app.SetIso(v), false);
            var ex = W.Row(b, 10, 30, false);
            var lx = W.T(ex, "분해", 13, W.MUTED); Ui.Pref(lx.rectTransform, 40);
            var sl = Ui.NewSlider(ex, 0, 1, 0, v => { app.SetExplode(v); tExplode.text = Mathf.RoundToInt(v * 100) + "%"; });
            sl.fillRect.GetComponent<Image>().color = W.ACCENT;
            tExplode = W.T(ex, "0%", 13, W.INK, true, TextAnchor.MiddleRight); Ui.Pref(tExplode.rectTransform, 44);
            bAuto = W.Btn(b, "움직임 미리보기 (화면만)", app.ToggleAutoView, W.B.Secondary, 36, 13);
        }

        public void SetAutoLabel(bool on) { if (bAuto) bAuto.GetComponentInChildren<Text>().text = on ? "미리보기 멈추기" : "움직임 미리보기 (화면만)"; }

        void Fill()
        {
            var st = rn.settings;
            foreach (var (k, _) in NUM_FIELDS) sf[k].text = Convert.ToString(typeof(VerifySettings).GetField(k).GetValue(st), System.Globalization.CultureInfo.InvariantCulture);
            foreach (var (k, _) in BOOL_FIELDS) sc[k].SetIsOnWithoutNotify((bool)typeof(VerifySettings).GetField(k).GetValue(st));
            fAlgo.text = st.algo_path; fAlgoCfg.text = st.algo_config; outSel = st.algo_output == "command" ? 1 : 0; W.SetSegment(segOut, outSel); tGray.SetIsOnWithoutNotify(st.algo_gray);
        }

        public void OnShow() { Fill(); var ports = Win32Com.Ports(); comList.text = ports.Count == 0 ? "이 PC 의 COM 포트: 없음" : "이 PC 의 COM 포트: " + string.Join(", ", ports); SyncSegments(); Render(); }

        float next;
        public void Tick() { if (Time.unscaledTime > next) { next = Time.unscaledTime + 0.5f; Render(); } }

        void Render()
        {
            var st = app.stage;
            string ver = st is LinkStage ls ? $"{ls.Firmware[1]}.{ls.Firmware[2]}.{ls.Firmware[3]}" : "0.3.1 (모의)";
            var c = st.Config;
            string status;
            try { var s = st.Status; status = $"{StText.State(s.state)} · Y {StText.Mm(s.yUm)} mm · Θ {StText.Mm(s.tMdeg)}°"; } catch (Exception e) { status = "오류: " + e.Message; }
            stageInfo.text = $"상태  <b>{st.Name}</b>\n{status}" + (st is LinkStage l2 && l2.Error != null ? $"\n<color=#df3544>{l2.Error}</color>" : "") +
                             $"\n펌웨어 {ver} · 범위 Y {c.yMinUm / 1000.0:0.###}~{c.yMaxUm / 1000.0:0.###} mm, Θ {c.tMinMdeg / 1000.0:0.###}~{c.tMaxMdeg / 1000.0:0.###}°" +
                             $"\n속도 Y {st.SpeedYUmS / 1000.0:0.###} mm/s · Θ {st.SpeedTMdegS / 1000.0:0.###} °/s";
            var cam = app.cam;
            camInfo.text = $"상태  <b>{cam.Label}</b> · {(cam.Live ? $"{cam.Fps:0} fps" : "<color=#df3544>영상 없음</color>")}" + (cam is PiCam pc && pc.Error != null ? $"\n<color=#df3544>{pc.Error}</color>" : "") +
                           "\n알고리즘 입력 3840 × 2160 흑백" + (cam.Kind == "pi" ? " (파이: 왜곡 보정 + SEMES 화각, verify_web 과 같은 변환)" : " (가상 카메라 화면)") +
                           (cam.Kind == "pi" ? "\n촬영 조건은 파이 쪽 (기본 노출 8333 µs · 게인 3.0)" : "");
            algoInfo.text = $"상태  <b>{(rn.algo.Kind == "mock" ? "모의 추론" : "실행기 " + rn.algo.Detail)}</b> · {(rn.algo.Output == "command" ? "명령값" : "오프셋")}" +
                            (rn.algoInitError != null ? $"\n<color=#df3544>{rn.algoInitError}</color>" : "") + (rn.algo.Error != null ? $"\n<color=#df3544>{rn.algo.Error}</color>" : "") +
                            "\n통신 READY → ALGN / ALGR (verify_web 과 같음)" +
                            $"\n목록 파일 {rn.algos.file}" + (rn.algos.loadError != null ? $"\n<color=#df3544>{rn.algos.loadError}</color>" : $" · {rn.algos.entries.Count}개");
        }

        // 버튼 선택 표시를 지금 연결 상태에 맞춤 (화면에 들어올 때·연결 뒤에만)
        void SyncSegments()
        {
            stageSel = app.stage is MockStage ? 0 : app.stage.Transport == "serial" ? 1 : 2; W.SetSegment(segStage, stageSel);
            camSel = app.cam.Kind == "pi" ? 1 : 0; W.SetSegment(segCam, camSel);
        }
    }
}
