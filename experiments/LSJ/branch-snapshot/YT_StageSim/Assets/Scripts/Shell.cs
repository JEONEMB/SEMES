// 화면 틀: 위 막대 (SSAFY X SEMES = A507 · 보정 확인 / 연결·설정 탭 · 스테이지·카메라·알고리즘 상태) + 알림 + 확인 창
// verify_web 기준 (demo_web 전용 화면은 뺌). 로그인 없음 (verify_web 도 없음)
// 알고리즘 고르기: verify_web index.html algo-pick (위 막대 단추 묶음) 과 같은 자리. 목록 = 모의 + algorithms.json + 직접 지정
using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.UI;

namespace YT
{
    public class Shell
    {
        public readonly App app;
        public readonly Canvas canvas;
        public readonly RectTransform root;
        public VerifyPage verify;
        public ConnectPage connect;
        public int tab = -1;
        readonly Button navVerify, navConnect;
        readonly Text pillStage, pillCam, pillAlgo;
        readonly Image pillStageBg, pillCamBg, pillAlgoBg;
        float nextPill;
        readonly RectTransform algoBox; readonly List<(string id, Button b)> algoBtns = new List<(string, Button)>(); string algoKey;

        public Shell(App a)
        {
            app = a;
            canvas = Ui.NewCanvas("Shell");
            root = (RectTransform)canvas.transform;

            var top = Ui.Anchor(Ui.NewRect("topbar", root), new Vector2(0, 1), Vector2.one, new Vector2(0, -76), Vector2.zero);
            W.Fill(top, Color.white, false);
            var line = Ui.Anchor(Ui.NewRect("line", top), Vector2.zero, new Vector2(1, 0), Vector2.zero, new Vector2(0, 1));
            W.Fill(line, W.LINE, false); line.gameObject.AddComponent<LayoutElement>().ignoreLayout = true;
            var hg = top.gameObject.AddComponent<HorizontalLayoutGroup>(); hg.padding = new RectOffset(56, 56, 0, 0); hg.spacing = 8; hg.childAlignment = TextAnchor.MiddleLeft;
            hg.childControlWidth = hg.childControlHeight = true; hg.childForceExpandWidth = hg.childForceExpandHeight = false;
            var brand = W.T(top, "SSAFY X SEMES = A507", 28, W.ACCENT, true); Ui.Pref(brand.rectTransform, 330, 40);
            var gap = Ui.NewRect("gap", top); Ui.Pref(gap, 24, 10);
            navVerify = Tab(top, "보정 확인", () => Show(0));
            navConnect = Tab(top, "연결·설정", () => Show(1));
            var sp = Ui.NewRect("sp", top); Ui.Pref(sp, 0, 10, 1);
            (pillStage, pillStageBg) = Pill(top); (pillCam, pillCamBg) = Pill(top);
            var al = W.T(top, "알고리즘", 13, W.MUTED, true, TextAnchor.MiddleRight); Ui.Pref(al.rectTransform, 64, 32);
            algoBox = Ui.HBox(top, "algo-pick", 0, 34); algoBox.GetComponent<HorizontalLayoutGroup>().childForceExpandWidth = false;
            (pillAlgo, pillAlgoBg) = Pill(top);
            W.Btn(top, "전체 화면 ↗", () => Screen.fullScreen = !Screen.fullScreen, W.B.Text, 40, 14, 100);

            var body = Ui.Anchor(Ui.NewRect("pages", root), Vector2.zero, Vector2.one, Vector2.zero, new Vector2(0, -76));
            verify = new VerifyPage(this, body);
            connect = new ConnectPage(this, body);
            W.InitToast(root);
            W.InitModal(root);
            Show(0);
            RebuildAlgo();
        }

        // 고를 수 있는 알고리즘: 모의 + algorithms.json 순서 + 직접 지정 (경로가 있을 때)
        List<(string id, string label)> AlgoOptions()
        {
            var v = app.verify;
            var o = new List<(string, string)> { ("mock", "모의") };
            o.AddRange(v.algos.entries.Select(e => (e.id, e.label)));
            if (!string.IsNullOrWhiteSpace(v.settings.algo_path) || v.algoId == "custom") o.Add(("custom", "직접 지정"));
            return o;
        }

        void RebuildAlgo()
        {
            var opts = AlgoOptions();
            string k = string.Join("|", opts.Select(x => x.id + "=" + x.label));
            if (k == algoKey) return;
            algoKey = k;
            foreach (Transform c in algoBox) UnityEngine.Object.Destroy(c.gameObject);
            algoBtns.Clear();
            foreach (var (id, label) in opts)
            {
                var idc = id;
                var b = W.Btn(algoBox, label, () => W.Act(() => { app.verify.SelectAlgorithm(idc); W.Toast("알고리즘을 바꿨습니다: " + app.verify.AlgoLabel(idc)); }), W.B.Secondary, 34, 13, label.Length * 14 + 30);
                algoBtns.Add((id, b));
            }
            StyleAlgo();
        }

        // 고른 것 = 파랑, 실행기·설정이 없는 것 = 흐린 글씨 (누르면 이유를 알림)
        void StyleAlgo()
        {
            var v = app.verify;
            foreach (var (id, b) in algoBtns)
            {
                bool on = id == v.algoId;
                var e = v.algos.Find(id);
                bool missing = e != null && v.algos.Missing(e) != null;
                b.GetComponent<Image>().color = on ? W.PILL_BG : Color.white;
                var t = b.GetComponentInChildren<Text>(); t.color = on ? W.ACCENT : missing ? W.Hex(0xb4bdc9) : W.MUTED;
                var bd = b.transform.Find("border")?.GetComponent<Image>(); if (bd) bd.color = on ? W.ACCENT : W.INPUT_LINE;
            }
        }

        static Button Tab(Transform p, string s, Action a)
        {
            var b = W.Btn(p, s, a, W.B.Text, 76, 16, s.Length * 16 + 40);
            b.GetComponentInChildren<Text>().alignment = TextAnchor.MiddleCenter;
            var bar = Ui.NewRect("bar", b.transform); bar.anchorMin = Vector2.zero; bar.anchorMax = new Vector2(1, 0); bar.pivot = new Vector2(.5f, 0); bar.sizeDelta = new Vector2(0, 3);
            W.Fill(bar, W.ACCENT, false).raycastTarget = false;
            return b;
        }

        static (Text, Image) Pill(Transform p)
        {
            var r = Ui.NewRect("pill", p); var hg = r.gameObject.AddComponent<HorizontalLayoutGroup>(); hg.padding = new RectOffset(12, 12, 6, 6);
            hg.childControlWidth = hg.childControlHeight = true; hg.childForceExpandWidth = hg.childForceExpandHeight = false;
            var bg = W.Fill(r, W.WARN_BG); Ui.Pref(r, -1, 32).flexibleHeight = 0;
            return (W.T(r, "—", 13, W.WARN_INK), bg);
        }

        public void Show(int t)
        {
            if (t == tab) return;
            tab = t;
            verify.root.gameObject.SetActive(t == 0);
            connect.root.gameObject.SetActive(t == 1);
            app.mainCam.enabled = t == 0;
            foreach (var (b, on) in new[] { (navVerify, t == 0), (navConnect, t == 1) })
            {
                var tx = b.GetComponentInChildren<Text>(); tx.color = on ? W.ACCENT : W.MUTED; tx.fontStyle = on ? FontStyle.Bold : FontStyle.Normal;
                b.transform.Find("bar").gameObject.SetActive(on);
            }
            if (t == 0) verify.OnShow(); else connect.OnShow();
        }

        // 3D 화면 칸 안에 마우스가 있는지 (회전·부품 선택은 여기서만)
        public bool InHole(Vector2 mouse) => tab == 0 && verify.InHole(mouse);

        public void Tick()
        {
            W.TickToast();
            if (tab == 0) verify.Tick(); else connect.Tick();
            if (Time.unscaledTime < nextPill) return;
            nextPill = Time.unscaledTime + 0.5f;
            bool real = app.stage.Mode == "hardware";
            SetPill(pillStage, pillStageBg, real ? $"● 스테이지 실물 ({(app.stage.Transport == "serial" ? "USB" : "파이 경유")})" : "● 스테이지 모의", real && app.stage.Connected, real && !app.stage.Connected);
            bool pi = app.cam.Kind == "pi";
            SetPill(pillCam, pillCamBg, pi ? (app.cam.Live ? "● 카메라 파이 영상" : "● 카메라 영상 없음") : "● 카메라 시뮬레이터", pi && app.cam.Live, pi && !app.cam.Live);
            var v = app.verify;
            if (v.algos.Load()) algoKey = null;      // algorithms.json 을 고치면 바로 반영
            RebuildAlgo(); StyleAlgo();
            bool worker = v.algo.Kind != "mock", failed = v.algoInitError != null;
            SetPill(pillAlgo, pillAlgoBg, failed ? "● 실행기 못 띄움 (모의 사용)" : worker ? (v.algo.Error != null ? "● 실행기 오류" : $"● 실행기 · {(v.algo.Output == "command" ? "명령값" : "오프셋")}") : "● 모의 추론",
                    worker && v.algo.Error == null, failed || (worker && v.algo.Error != null));
        }

        // 노랑 = 모의/시뮬레이터, 파랑 = 실물 정상, 빨강 = 실물 오류
        static void SetPill(Text t, Image bg, string s, bool ok, bool bad)
        {
            t.text = s;
            bg.color = bad ? W.Hex(0xfff0f1) : ok ? W.PILL_BG : W.WARN_BG;
            t.color = bad ? W.Hex(0x9d2633) : ok ? W.ACCENT : W.WARN_INK;
        }
    }
}
