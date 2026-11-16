// 보정 확인 화면 (GitLab dev verify_web 기능, 배치는 디자인 캔버스 'verify 기준 화면 배치안' B안 확정본)
//   왼쪽 큰 칸 = 3D 장비 시뮬레이터, 아래 = 조작 버튼
//   오른쪽 위 = 카메라 영상 (시뮬레이터 / 라즈베리파이), 아래 = '현재 회차 / 회차별 기록' 탭
// 보정 설정은 연결·설정 탭으로 옮김 (verify_web 은 같은 화면 아래 접힌 칸)
using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.UI;

namespace YT
{
    public class VerifyPage
    {
        readonly Shell shell;
        public readonly RectTransform root, hole, partCard;
        App app => shell.app;
        VerifyRunner rn => app.verify;
        int lastVersion = -1;
        bool claimed;
        static readonly Vector3[] C = new Vector3[4];

        Text camInfo, trial, trialMax, result, phaseTitle, phaseMsg, note, posY, posT, centerTxt, runId, badgeTxt, btnMaxTxt, partTitle, partBody, histTabTxt, viewNote;
        Image resultBg, phaseLight, badgeDot, badgeBg; RectTransform videoEmpty, badge; Text videoEmptyMsg; RectTransform progressFill, resultBox, histBody, plateMark, cross, tabNow, tabHist;
        readonly List<(string phase, Image bg, Text t)> steps = new List<(string, Image, Text)>();
        readonly Dictionary<string, Text> cells = new Dictionary<string, Text>();
        readonly List<(Text t, int bit, bool alert)> flagTxt = new List<(Text, int, bool)>();
        RefOverlay refOv;
        RawImage view; List<Button> segZoom, segCenter, segTab; int zoom = 1; bool plateCenter;
        InputField startY, startT;
        Button bHome, bTZero, bRef, bRandom, bGoto, bRun, bStep, bMeasure, bStop, bCenter, bHide;
        RawImage chart; Texture2D chartTex;
        Text axisTop, axisTol, axisZero, chartEmpty, chartLegend; RectTransform axisLine;
        const float CHART_L = 78, CHART_BOX_H = 96;

        static Text AxisLabel(RectTransform box)
        {
            var t = W.T(box, "", 10, W.MUTED, false, TextAnchor.MiddleRight);
            var r = t.rectTransform; r.anchorMin = r.anchorMax = new Vector2(0, 0); r.pivot = new Vector2(1, .5f); r.sizeDelta = new Vector2(CHART_L - 8, 14);
            return t;
        }

        // 그래프 값 v 의 세로 위치 (칸 아래에서 px) — DrawChart 의 Y() 와 같은 식
        static void PlaceAxis(Text t, string s, float yTex, float texH)
        {
            t.text = s;
            float y = 3 + yTex / texH * (CHART_BOX_H - 6);
            t.rectTransform.anchoredPosition = new Vector2(CHART_L - 6, Mathf.Clamp(y, 7, CHART_BOX_H - 7));
        }
        PartInfo shownPart;

        // verify_web/app.py PLATE_CENTER: A507 센터 사진의 가운데 QR (알고리즘 좌표 3840×2160)
        static readonly Vector2 PLATE = new Vector2(1906.3f / 3840f, 1f - 1130.8f / 2160f);

        public VerifyPage(Shell s, RectTransform parent)
        {
            shell = s;
            root = Ui.Anchor(Ui.NewRect("verify", parent), Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero);
            const float PAD = 56, TOP = 20, BOT = 24, RIGHT = 600, GAP = 24;

            // ================= 왼쪽: 3D 장비 + 조작 =================
            var left = Ui.Anchor(Ui.NewRect("left", root), Vector2.zero, Vector2.one, new Vector2(PAD, BOT), new Vector2(-(PAD + RIGHT + GAP), -TOP));
            var lv = left.gameObject.AddComponent<VerticalLayoutGroup>(); lv.spacing = 16; lv.childControlWidth = lv.childControlHeight = true; lv.childForceExpandWidth = true; lv.childForceExpandHeight = false;
            var p3 = W.Panel(left, "panel3d", false); Ui.Pref(p3, h: 0).flexibleHeight = 1;
            var (bar3, _, _) = W.Heading(p3, "3D 장비 · 도면 기준", null, false, 48, 18);
            bar3.GetComponent<HorizontalLayoutGroup>().spacing = 4;
            foreach (var v in Rig.Views)
            {
                var vv = v;
                var vb = W.Btn(bar3, v.label, () => app.orbit.TweenTo(vv.pos, vv.target), W.B.Text, 30, 12, v.label.Length * 12 + 20);
                vb.GetComponentInChildren<Text>().alignment = TextAnchor.MiddleCenter;
            }
            hole = Ui.NewRect("hole", p3); Ui.Pref(hole, h: 0).flexibleHeight = 1;
            var hint = W.T(hole, "드래그 회전 · 오른쪽 드래그 이동 · 휠 확대 · 부품을 누르면 설명", 13, W.MUTED, false, TextAnchor.LowerRight);
            Ui.Anchor(hint.rectTransform, Vector2.zero, Vector2.one, new Vector2(16, 12), new Vector2(-16, -12));
            // 선택한 부품
            partCard = Ui.VBox(hole, "part", 18, 8);
            partCard.anchorMin = partCard.anchorMax = partCard.pivot = new Vector2(1, 1); partCard.anchoredPosition = new Vector2(-16, -16); partCard.sizeDelta = new Vector2(400, 0);
            partCard.gameObject.AddComponent<ContentSizeFitter>().verticalFit = ContentSizeFitter.FitMode.PreferredSize;
            W.Fill(partCard, Color.white); W.Border(partCard, W.LINE);
            partTitle = W.T(partCard, "", 17, W.INK, true);
            partBody = W.T(partCard, "", 14, W.INK);
            var pb = W.Row(partCard, 8, 38);
            W.Btn(pb, "가까이", () => { if (shownPart != null) app.Focus(shownPart.id); }, W.B.Secondary, 38, 13);
            bHide = W.Btn(pb, "숨기기", () => { if (shownPart != null) { app.ToggleHidden(shownPart.id); ShowPart(shownPart); } }, W.B.Secondary, 38, 13);
            W.Btn(pb, "모두 보이기", () => { app.ShowAll(); if (shownPart != null) ShowPart(shownPart); }, W.B.Secondary, 38, 13);
            W.Btn(pb, "닫기", () => app.Select(null), W.B.Secondary, 38, 13);
            partCard.gameObject.SetActive(false);

            var ctl = W.Panel(left, "controls");
            var cb = Ui.VBox(ctl, "body", 0, 10); cb.GetComponent<VerticalLayoutGroup>().padding = new RectOffset(18, 18, 14, 14);
            var r1 = W.Row(cb, 8, 42, false);
            bHome = W.Btn(r1, "⌂ Y 원점 복귀", () => W.Act(rn.Home), W.B.Secondary, 42, 14, 140);
            bTZero = W.Btn(r1, "Θ 지금 자리를 0으로", () => W.Act(rn.ThetaZero), W.B.Secondary, 42, 14, 160);
            bRef = W.Btn(r1, "⊙ 센터 기준 잡기", () => W.Act(rn.Reference), W.B.Secondary, 42, 14, 150);
            bRandom = W.Btn(r1, "↝ 랜덤 시작 위치", () => W.Act(() => rn.Goto("random")), W.B.Secondary, 42, 14, 150);
            Lbl(r1, "Y", 18); startY = W.Input(r1, "0.50", 42); Ui.Pref((RectTransform)startY.transform, 86, 42, 0); Lbl(r1, "mm", 30);
            Lbl(r1, "Θ", 18); startT = W.Input(r1, "0.00", 42); Ui.Pref((RectTransform)startT.transform, 86, 42, 0); Lbl(r1, "°", 14);
            bGoto = W.Btn(r1, "→ 지정 위치로", () => W.Act(() => rn.Goto((Num(startY.text), Num(startT.text)))), W.B.Secondary, 42, 14, 130);
            var r2 = W.Row(cb, 8, 50, false);
            bRun = W.Btn(r2, "⊕ 자동 보정 (최대 20회)", () => W.Act(rn.Run), W.B.Primary, 50, 16, 270); btnMaxTxt = bRun.GetComponentInChildren<Text>();
            bStep = W.Btn(r2, "한 단계", () => W.Act(rn.Step), W.B.Secondary, 50, 15, 120);
            bMeasure = W.Btn(r2, "측정만", () => W.Act(rn.Measure), W.B.Secondary, 50, 15, 120);
            var sp2 = Ui.NewRect("sp", r2); Ui.Pref(sp2, 0, 10, 1);
            bStop = W.Btn(r2, "■ 정지", () => W.Act(rn.Stop), W.B.Console, 50, 16, 130); bStop.GetComponent<Image>().color = W.Hex(0xfff0f1);
            var r3 = W.Row(cb, 10, 36, false);
            Lbl(r3, "센터 (스테이지 좌표)", 130, W.MUTED); centerTxt = W.T(r3, "—", 12, W.INK); Ui.Pref(centerTxt.rectTransform, 0, 34, 1); centerTxt.alignment = TextAnchor.MiddleLeft;
            bCenter = W.Btn(r3, "현재 위치를 센터로", () => W.Act(rn.SetCenterHere), W.B.Text, 26, 13, 130); bCenter.GetComponentInChildren<Text>().color = W.ACCENT;

            // ================= 오른쪽 위: 카메라 영상 =================
            var right = Ui.Anchor(Ui.NewRect("right", root), new Vector2(1, 0), Vector2.one, new Vector2(-(PAD + RIGHT), BOT), new Vector2(-PAD, -TOP));
            var rv = right.gameObject.AddComponent<VerticalLayoutGroup>(); rv.spacing = 16; rv.childControlWidth = rv.childControlHeight = true; rv.childForceExpandWidth = true; rv.childForceExpandHeight = false;
            var cp = W.Panel(right, "camera");
            var (cbar, _, _) = W.Heading(cp, "실시간 영상", null, false, 40, 12);
            cbar.GetComponent<HorizontalLayoutGroup>().spacing = 6;
            camInfo = W.T(cbar, "", 11, W.MUTED); Ui.Pref(camInfo.rectTransform, 120, 26);
            segZoom = W.Segmented(Box(cbar, 108), new[] { "1×", "2×", "4×" }, 0, k => { zoom = new[] { 1, 2, 4 }[k]; ApplyView(); }, 26, 11);
            segCenter = W.Segmented(Box(cbar, 140), new[] { "화면 중심", "플레이트" }, 0, k => { plateCenter = k == 1; ApplyView(); }, 26, 11);
            var vbox = Ui.NewRect("video", cp); Ui.Pref(vbox, h: 600 * 9f / 16f);
            vbox.gameObject.AddComponent<RectMask2D>();
            W.Fill(vbox, W.Hex(0x1b1f24), false);
            view = Ui.Anchor(Ui.NewRect("feed", vbox), Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero).gameObject.AddComponent<RawImage>();
            view.raycastTarget = false;
            refOv = new RefOverlay(vbox);   // QR·나사 윤곽 (verify_web ref-overlay)
            cross = Ui.NewRect("cross", vbox); cross.sizeDelta = new Vector2(44, 44); W.Crosshair(cross, 40);
            plateMark = Ui.NewRect("plate", vbox); plateMark.sizeDelta = new Vector2(24, 24);
            var pm = plateMark.gameObject.AddComponent<Image>(); pm.sprite = W.Ring; pm.type = Image.Type.Sliced; pm.color = W.Hex(0xe2a000); pm.raycastTarget = false;
            // 영상 없음 (verify_web .video-empty: 영상을 숨기고 안내)
            videoEmpty = Ui.VBox(vbox, "empty", 0, 8); Ui.Anchor(videoEmpty, Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero);
            videoEmpty.GetComponent<VerticalLayoutGroup>().childAlignment = TextAnchor.MiddleCenter; videoEmpty.GetComponent<VerticalLayoutGroup>().padding = new RectOffset(20, 20, 20, 20);
            W.T(videoEmpty, "영상 없음", 16, W.Hex(0xc9d3e2), true, TextAnchor.MiddleCenter, 24);
            videoEmptyMsg = W.T(videoEmpty, "", 12, W.Hex(0xc9d3e2), false, TextAnchor.MiddleCenter, 40);
            videoEmpty.gameObject.SetActive(false);
            // 이동 표시 (verify_web .badge: 영상 왼쪽 위)
            badge = Ui.HBox(vbox, "badge", 8, 30); badge.anchorMin = badge.anchorMax = badge.pivot = new Vector2(0, 1); badge.anchoredPosition = new Vector2(12, -12); badge.sizeDelta = new Vector2(140, 30);
            var bhg = badge.GetComponent<HorizontalLayoutGroup>(); bhg.padding = new RectOffset(12, 12, 0, 0); bhg.childForceExpandWidth = false; bhg.childForceExpandHeight = false;
            badgeBg = W.Fill(badge, new Color(1, 1, 1, .9f)); badgeBg.raycastTarget = false;
            var bd = Ui.NewRect("dot", badge); badgeDot = bd.gameObject.AddComponent<Image>(); badgeDot.sprite = W.Dot; Ui.Pref(bd, 10, 10);
            badgeTxt = W.T(badge, "정지", 13, W.INK, true); Ui.Pref(badgeTxt.rectTransform, 100, 24);
            // 범례 (verify_web .legend)
            var lg = W.Row(cp, 16, 28, false); lg.GetComponent<HorizontalLayoutGroup>().padding = new RectOffset(12, 12, 0, 0);
            var l1 = W.T(lg, "<color=#df3544>━</color> 화면 중심", 11, W.MUTED); Ui.Pref(l1.rectTransform, 80, 24);
            var l2 = W.T(lg, "<color=#e2a000>○</color> SEMES 기준 QR 위치", 11, W.MUTED); Ui.Pref(l2.rectTransform, 130, 24);
            viewNote = W.T(lg, "", 11, W.MUTED, false, TextAnchor.MiddleRight); Ui.Pref(viewNote.rectTransform, 0, 24, 1);
            var lg2 = W.Row(cp, 16, 24, false); lg2.GetComponent<HorizontalLayoutGroup>().padding = new RectOffset(12, 12, 0, 4);
            var l3 = W.T(lg2, "<color=#3fa9d8>┅</color> 목표 = 센터 기준 사진의 QR·나사   <color=#1b7f4b>━</color> 지금 = 마지막 측정 (초록 센터 OK · 빨강 벗어남 · 노랑 측정 전)", 11, W.MUTED);
            Ui.Pref(l3.rectTransform, 0, 20, 1);

            // ================= 오른쪽 아래: 현재 회차 / 회차별 기록 =================
            var tp = W.Panel(right, "tabs"); Ui.Pref(tp, h: 0).flexibleHeight = 1;
            var tb = Ui.HBox(tp, "tabbar", 4, 44); var tbg = tb.GetComponent<HorizontalLayoutGroup>(); tbg.padding = new RectOffset(8, 8, 0, 0); tbg.childForceExpandWidth = false; tbg.childAlignment = TextAnchor.LowerLeft;
            W.Fill(tb, W.HEAD, false);
            var t1 = W.Btn(tb, "현재 회차", () => ShowTab(0), W.B.Text, 44, 14, 110);
            var t2 = W.Btn(tb, "회차별 기록", () => ShowTab(1), W.B.Text, 44, 14, 140); histTabTxt = t2.GetComponentInChildren<Text>();
            segTab = new List<Button> { t1, t2 };
            foreach (var b in segTab) { b.GetComponentInChildren<Text>().alignment = TextAnchor.MiddleCenter; var bar = Ui.NewRect("bar", b.transform); bar.anchorMin = Vector2.zero; bar.anchorMax = new Vector2(1, 0); bar.pivot = new Vector2(.5f, 0); bar.sizeDelta = new Vector2(0, 3); W.Fill(bar, W.ACCENT, false).raycastTarget = false; }
            runId = W.T(tb, "", 11, W.MUTED, false, TextAnchor.MiddleRight); Ui.Pref(runId.rectTransform, 0, 30, 1);
            W.Line(tp);
            var area = Ui.NewRect("area", tp); Ui.Pref(area, h: 0).flexibleHeight = 1;

            // 현재 회차
            tabNow = Ui.VBox(area, "now", 0, 10); Ui.Anchor(tabNow, Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero);
            tabNow.GetComponent<VerticalLayoutGroup>().padding = new RectOffset(20, 20, 14, 14);
            var tr = W.Row(tabNow, 12, 72, false); tr.GetComponent<HorizontalLayoutGroup>().childAlignment = TextAnchor.LowerLeft;
            var tv = Ui.VBox(tr, "t", 0, 0); Ui.Pref(tv, 200, 72);
            W.T(tv, "회차", 12, W.MUTED, true, TextAnchor.MiddleLeft, 16);
            var tn = W.Row(tv, 8, 56, false); tn.GetComponent<HorizontalLayoutGroup>().childAlignment = TextAnchor.LowerLeft;
            trial = W.T(tn, "00", 52, W.ACCENT, false, TextAnchor.LowerLeft); Ui.Pref(trial.rectTransform, 76, 56);
            trialMax = W.T(tn, "/ 20", 18, W.Hex(0x7d8da4), false, TextAnchor.LowerLeft); Ui.Pref(trialMax.rectTransform, 70, 30);
            resultBox = Ui.HBox(tr, "result", 0, 40); resultBox.GetComponent<HorizontalLayoutGroup>().padding = new RectOffset(12, 12, 0, 0);
            Ui.Pref(resultBox, 0, 40, 1); resultBg = W.Fill(resultBox, W.PILL_BG);
            result = W.T(resultBox, "", 13, W.ACCENT, true, TextAnchor.MiddleCenter);
            var track = Ui.NewRect("progress", tabNow); W.Fill(track, W.TRACK, false); Ui.Pref(track, h: 4);
            progressFill = Ui.NewRect("fill", track); W.Fill(progressFill, W.ACCENT, false); Ui.Anchor(progressFill, Vector2.zero, new Vector2(0, 1), Vector2.zero, Vector2.zero);
            var st = W.Row(tabNow, 6, 32);
            foreach (var (ph, name) in new[] { ("settling", "흔들림 대기"), ("capturing", "촬영"), ("inferring", "추론"), ("moving", "보정 이동") })
            {
                var c = Ui.HBox(st, "step", 0, 32); var bg = W.Fill(c, Color.white); W.Border(c, W.LINE);
                var t = W.T(c, name, 12, W.MUTED, false, TextAnchor.MiddleCenter); steps.Add((ph, bg, t));
            }
            var pbx = Ui.HBox(tabNow, "phase", 10, 0); var pg = pbx.GetComponent<HorizontalLayoutGroup>(); pg.childForceExpandWidth = false; pg.childForceExpandHeight = false; pg.childAlignment = TextAnchor.UpperLeft;
            pbx.GetComponent<LayoutElement>().preferredHeight = -1;
            var pl = Ui.NewRect("light", pbx); phaseLight = pl.gameObject.AddComponent<Image>(); phaseLight.sprite = W.Dot; Ui.Pref(pl, 9, 9);
            var pv = Ui.VBox(pbx, "t", 0, 2); Ui.Pref(pv, 0, -1, 1);
            phaseTitle = W.T(pv, "대기", 16, W.INK, true, TextAnchor.MiddleLeft, 22);
            phaseMsg = W.T(pv, "", 13, W.MUTED); phaseMsg.alignment = TextAnchor.UpperLeft;
            W.Line(tabNow);
            var th = W.Row(tabNow, 0, 22, false); Cell(th, "", 70, W.MUTED); foreach (var h in new[] { "추론", "실제", "차이", "명령" }) Cell(th, h, 0, W.MUTED, 1);
            foreach (var ax in new[] { "y", "t" })
            {
                var row = W.Row(tabNow, 0, 30, false); Cell(row, ax == "y" ? "Y  mm" : "Θ  °", 70, W.INK, 0, true);
                foreach (var k in new[] { "inf", "tru", "err", "cmd" }) cells[$"{k}-{ax}"] = Cell(row, "—", 0, W.INK, 1, false, 17);
            }
            note = W.T(tabNow, "추론 = 알고리즘이 본 센터 대비 위치 · 실제 = STATUS 위치 − 센터", 11, W.MUTED); note.alignment = TextAnchor.UpperLeft;
            W.Line(tabNow);
            var pr = W.Row(tabNow, 8, 30, false);
            Lbl(pr, "스테이지", 54, W.MUTED); posY = W.T(pr, "—", 17, W.INK, true); Ui.Pref(posY.rectTransform, 80, 30); Lbl(pr, "mm", 26, W.MUTED);
            posT = W.T(pr, "—", 17, W.INK, true); Ui.Pref(posT.rectTransform, 70, 30); Lbl(pr, "°", 12, W.MUTED);
            foreach (var (n, bit, al) in new[] { ("Y 원점", 1, false), ("Θ 원점", 2, false), ("Y− 리밋", 4, true), ("Y+ 리밋", 8, true) })
            { var t = W.T(pr, n, 12, W.MUTED, false, TextAnchor.MiddleCenter); Ui.Pref(t.rectTransform, 58, 24); flagTxt.Add((t, bit, al)); }

            // 회차별 기록
            tabHist = Ui.VBox(area, "hist", 0, 0); Ui.Anchor(tabHist, Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero);
            tabHist.GetComponent<VerticalLayoutGroup>().padding = new RectOffset(12, 12, 10, 0);
            chartTex = new Texture2D(500, 90, TextureFormat.RGBA32, false) { filterMode = FilterMode.Bilinear, wrapMode = TextureWrapMode.Clamp };
            // 그래프 칸: 테두리 + 왼쪽 축 값 (위 = 최댓값, 가운데 = 허용, 아래 = 0). 기록이 없으면 안내 문구만
            var cbox = Ui.NewRect("chartbox", tabHist); Ui.Pref(cbox, h: 96); W.Border(cbox, W.LINE);
            var cr = Ui.Anchor(Ui.NewRect("chart", cbox), Vector2.zero, Vector2.one, new Vector2(CHART_L, 3), new Vector2(-3, -3));
            chart = cr.gameObject.AddComponent<RawImage>(); chart.texture = chartTex; chart.raycastTarget = false;
            axisLine = Ui.Anchor(Ui.NewRect("axis", cbox), Vector2.zero, new Vector2(0, 1), new Vector2(CHART_L - 1, 3), new Vector2(CHART_L, -3));
            W.Fill(axisLine, W.LINE, false).raycastTarget = false;
            axisTop = AxisLabel(cbox); axisTol = AxisLabel(cbox); axisZero = AxisLabel(cbox);
            chartEmpty = W.T(cbox, "기록이 없습니다", 13, W.MUTED, false, TextAnchor.MiddleCenter);
            Ui.Anchor(chartEmpty.rectTransform, Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero);
            chartLegend = W.T(tabHist, "<color=#0A3D91>━</color> |추론 Y|   <color=#b06f0c>━</color> |실제 Y|   <color=#1f7a45>■</color> 허용", 11, W.MUTED, false, TextAnchor.MiddleRight, 20);
            var hh = W.Row(tabHist, 0, 28, false);
            HistCols(hh, new[] { "회차", "추론 Y", "실제 Y", "차이 Y", "추론 Θ", "실제 Θ", "차이 Θ", "명령 Y", "ms", "대기", "판정" }, W.MUTED, 11);
            W.Line(tabHist);
            histBody = Scroll(tabHist);
            var fr = W.Row(tabHist, 0, 32, false);
            W.Btn(fr, "기록 폴더 열기 ↗", () => { System.IO.Directory.CreateDirectory(rn.RunsDir); Application.OpenURL("file:///" + rn.RunsDir.Replace('\\', '/')); }, W.B.Text, 30, 12, 140);

            ShowTab(0);
            ApplyView();
        }

        void ShowTab(int k)
        {
            tabNow.gameObject.SetActive(k == 0); tabHist.gameObject.SetActive(k == 1);
            for (int i = 0; i < segTab.Count; i++)
            {
                var t = segTab[i].GetComponentInChildren<Text>(); t.color = i == k ? W.ACCENT : W.MUTED; t.fontStyle = i == k ? FontStyle.Bold : FontStyle.Normal;
                segTab[i].transform.Find("bar").gameObject.SetActive(i == k);
            }
        }

        // ---------- 도우미
        static RectTransform Box(Transform p, float w) { var b = Ui.NewRect("box", p); Ui.Pref(b, w, 26); var g = b.gameObject.AddComponent<VerticalLayoutGroup>(); g.childControlWidth = g.childControlHeight = true; g.childForceExpandWidth = true; g.childForceExpandHeight = false; return b; }
        static Text Lbl(Transform p, string s, float w, Color? c = null) { var t = W.T(p, s, 13, c ?? W.INK, false, TextAnchor.MiddleCenter); var le = Ui.Pref(t.rectTransform, w, 30); le.minWidth = w; return t; }
        static Text Cell(Transform p, string s, float w, Color c, float flex = 0, bool bold = false, int size = 12)
        { var t = W.T(p, s, size, c, bold, w > 0 ? TextAnchor.MiddleLeft : TextAnchor.MiddleRight); var le = Ui.Pref(t.rectTransform, w, -1, flex); le.minWidth = w; return t; }
        static readonly float[] HW = { 48, 0, 0, 0, 0, 0, 0, 0, 36, 34, 54 };
        static void HistCols(Transform row, string[] v, Color c, int size = 11)
        {
            for (int i = 0; i < v.Length; i++) { var t = W.T(row, v[i], size, c, false, i == 0 ? TextAnchor.MiddleLeft : TextAnchor.MiddleRight); var le = Ui.Pref(t.rectTransform, HW[i], -1, HW[i] == 0 ? 1 : 0); le.minWidth = HW[i]; }
        }
        static RectTransform Scroll(Transform p)
        {
            var box = Ui.NewRect("scroll", p); var le = Ui.Pref(box, h: 80); le.flexibleHeight = 1;
            box.gameObject.AddComponent<Image>().color = new Color(1, 1, 1, 0);
            var vp = Ui.Anchor(Ui.NewRect("vp", box), Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero); vp.gameObject.AddComponent<RectMask2D>();
            var c = Ui.VBox(vp, "content", 0, 0); c.anchorMin = new Vector2(0, 1); c.anchorMax = Vector2.one; c.pivot = new Vector2(.5f, 1); c.offsetMin = c.offsetMax = Vector2.zero;
            c.gameObject.AddComponent<ContentSizeFitter>().verticalFit = ContentSizeFitter.FitMode.PreferredSize;
            var sr = box.gameObject.AddComponent<ScrollRect>(); sr.viewport = vp; sr.content = c; sr.horizontal = false; sr.movementType = ScrollRect.MovementType.Clamped; sr.scrollSensitivity = 40;
            return c;
        }
        static double Num(string s)
        {
            if (!double.TryParse((s ?? "").Trim(), System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var v) || double.IsNaN(v) || double.IsInfinity(v))
                throw new RunError("숫자를 입력하세요.");
            return v;
        }
        static string Fmt(double? v, int d = 3) => v == null || double.IsNaN(v.Value) ? "—" : (v.Value >= 0 ? "+" : "") + v.Value.ToString("F" + d);

        // 확대: 시뮬레이터는 화면에서 잘라 보고, 파이는 verify_web 과 같이 파이 쪽 변환표를 바꿈
        void ApplyView()
        {
            app.cam.SetView(zoom, plateCenter);
            camInfo.text = $"{zoom}× · " + (plateCenter ? "플레이트" : "화면 중심");
        }

        void PlaceOverlays()
        {
            if (app.cam is PiCam pc)
            {
                view.uvRect = new Rect(0, 0, 1, 1);
                Place(cross, new Vector2(.5f, .5f));
                Place(plateMark, new Vector2(pc.Plate[0] / 1280f, 1 - pc.Plate[1] / 720f));
                return;
            }
            float s = 1f / zoom;
            Vector2 c = plateCenter ? PLATE : new Vector2(.5f, .5f);
            float x = Mathf.Clamp(c.x - s / 2, 0, 1 - s), y = Mathf.Clamp(c.y - s / 2, 0, 1 - s);
            view.uvRect = new Rect(x, y, s, s);
            Place(cross, PlaceUv(new Vector2(.5f, .5f)));
            Place(plateMark, PlaceUv(PLATE));
        }

        // 알고리즘 좌표(3840×2160) → 윤곽 그림 좌표(1280×720, 왼쪽 아래 0)
        //   시뮬레이터: 가상 카메라 화면 전체 = 알고리즘 사진 전체, 확대는 화면에서 잘라 봄 (uvRect)
        //   파이: pi_bridge /health view_map (verify_web ViewMapper.to_view 와 같은 1차식)
        void UpdateRefOverlay(bool live, bool moving)
        {
            Func<double, double, Vector2?> map; string mk, why = null;
            if (app.cam is PiCam pcam)
            {
                var vm = pcam.ViewMap;
                if (vm == null) { map = (x, y) => null; mk = "pi-none"; why = "윤곽 위치를 모름 · 파이에서 새 pi_bridge.py 를 실행하세요"; }
                else { map = (x, y) => new Vector2((float)(vm[0] * x + vm[1]), 720f - (float)(vm[2] * y + vm[3])); mk = $"pi{vm[0]},{vm[1]},{vm[2]},{vm[3]}"; }
            }
            else
            {
                var u = view.uvRect;
                map = (x, y) => { var p = PlaceUv(new Vector2((float)(x / 3840), 1f - (float)(y / 2160))); return new Vector2(p.x * 1280, p.y * 720); };
                mk = $"sim{u.x:0.0000},{u.y:0.0000},{u.width:0.0000}";
            }
            refOv.Update(rn, live, moving, map, mk, why);
        }

        // 자동 점검용: 확대 단추를 누른 것과 같게
        public void TestZoom(int z, bool plate)
        {
            zoom = z; plateCenter = plate;
            W.SetSegment(segZoom, z == 4 ? 2 : z == 2 ? 1 : 0); W.SetSegment(segCenter, plate ? 1 : 0);
            ApplyView();
        }

        Vector2 PlaceUv(Vector2 uv) { var u = view.uvRect; return new Vector2((uv.x - u.x) / u.width, (uv.y - u.y) / u.height); }
        static void Place(RectTransform r, Vector2 p) { r.gameObject.SetActive(p.x >= 0 && p.x <= 1 && p.y >= 0 && p.y <= 1); r.anchorMin = r.anchorMax = p; r.anchoredPosition = Vector2.zero; }

        public bool InHole(Vector2 m) { hole.GetWorldCorners(C); return m.x >= C[0].x && m.x <= C[2].x && m.y >= C[0].y && m.y <= C[2].y; }

        public void OnShow()
        {
            if (!claimed) { claimed = true; W.Act(rn.Claim); }
            rn.RefreshStatus(); lastVersion = -1;
        }

        float nextPoll;
        public void Tick()
        {
            hole.GetWorldCorners(C);
            float sw = Screen.width, sh = Screen.height;
            app.mainCam.rect = new Rect(C[0].x / sw, C[0].y / sh, Mathf.Max(1, C[2].x - C[0].x) / sw, Mathf.Max(1, C[2].y - C[0].y) / sh);
            if (Time.unscaledTime > nextPoll) { nextPoll = Time.unscaledTime + 1f; W.Act(rn.Poll); }   // verify_web: 대기 중 1초마다 위치 갱신
            bool moving = rn.moving || rn.stageState == "moving" || rn.stageState == "homing";
            var amber = W.AMBER;
            badgeDot.color = moving ? new Color(amber.r, amber.g, amber.b, Mathf.Lerp(.3f, 1f, Mathf.Abs(Mathf.Cos(Time.unscaledTime * Mathf.PI * 1.25f)))) : W.Hex(0x99a7bb);
            badgeBg.color = moving ? W.Hex(0xfff3df) : new Color(1, 1, 1, .9f);
            badgeTxt.color = moving ? amber : W.INK;
            badgeTxt.text = moving ? (rn.phase == "homing" ? "원점 복귀 중" : "모터 이동 중") : "정지";
            view.texture = app.cam.View;
            bool live = app.cam.Live;
            view.enabled = live; videoEmpty.gameObject.SetActive(!live);
            videoEmptyMsg.text = app.cam is PiCam pc && pc.Error != null ? pc.Error : "카메라 연결을 기다리는 중";
            PlaceOverlays();
            UpdateRefOverlay(live, moving);
            viewNote.text = (app.cam.Kind == "pi" ? "왜곡 보정 · SEMES 화각 · 1280×720" : "시뮬레이터 가상 카메라 · SEMES 화각") + (live ? $" · {app.cam.Fps:0} fps" : "");
            if (rn.version != lastVersion) { lastVersion = rn.version; Render(); }
        }

        // verify.js render()
        public void Render()
        {
            var s = rn.settings; var run = rn.run; var last = rn.last ?? rn.measure;
            int max = run != null ? run.maxTrials : s.max_trials;
            trial.text = (run != null ? run.trial : 0).ToString("00"); trialMax.text = "/ " + max; btnMaxTxt.text = $"⊕ 자동 보정 (최대 {s.max_trials}회)";
            progressFill.anchorMax = new Vector2(Mathf.Clamp01((run != null ? run.trial : 0) / (float)max), 1);
            if (run != null && run.result != null)
            {
                resultBox.gameObject.SetActive(true);
                bool ok = run.result == "converged", warn = ok && last?.truthWithin == false;
                result.text = run.resultLabel + (warn ? " · 실제는 허용 밖!" : "");
                resultBg.color = warn ? W.WARN_BG : ok ? W.Hex(0xe8f6ee) : W.Hex(0xfff0f1);
                result.color = warn ? W.WARN_INK : ok ? W.Hex(0x1f7a45) : W.Hex(0x9d2633);
            }
            else resultBox.gameObject.SetActive(false);
            foreach (var (ph, bg, t) in steps) { bool on = ph == rn.phase; bg.color = on ? W.PILL_BG : Color.white; t.color = on ? W.ACCENT : W.MUTED; t.fontStyle = on ? FontStyle.Bold : FontStyle.Normal; }
            phaseTitle.text = VerifyRunner.PHASES.TryGetValue(rn.phase, out var pn) ? pn : rn.phase;
            phaseMsg.text = !string.IsNullOrEmpty(rn.message) ? rn.message : rn.stageError != null ? "스테이지: " + rn.stageError : "";
            phaseLight.color = rn.phase == "error" ? W.RED : rn.phase == "done" ? W.ACCENT : rn.busy ? W.AMBER : W.MUTED;
            if (last != null)
            {
                Color Ok(double? v, double tol) => v != null && Math.Abs(v.Value) <= tol ? W.Hex(0x1f7a45) : W.INK;
                SetCell("inf-y", last.offY, Ok(last.offY, s.tol_y_mm)); SetCell("inf-t", last.offT, s.enable_t ? Ok(last.offT, s.tol_t_deg) : W.INK);
                SetCell("tru-y", last.truY, Ok(last.truY, s.tol_y_mm)); SetCell("tru-t", last.truT, W.INK);
                SetCell("err-y", last.errY, last.errY != null && Math.Abs(last.errY.Value) > s.tol_y_mm ? W.RED : W.INK);
                SetCell("err-t", last.errT, last.errT != null && Math.Abs(last.errT.Value) > s.tol_t_deg ? W.RED : W.INK);
                SetCell("cmd-y", last.cmdY, W.INK); SetCell("cmd-t", s.enable_t ? last.cmdT : 0, W.INK);
                note.text = $"{last.label} · 추론 {last.inferMs} ms · 흔들림 대기 {last.settleS} s" + (last.settled ? "" : " (안정 판정 실패, 시간 초과)") + (s.enable_t ? "" : " · Θ 명령 꺼짐(0 전송)");
            }
            posY.text = rn.posY?.ToString("0.000") ?? "—"; posT.text = rn.posT?.ToString("0.000") ?? "—";
            foreach (var (t, bit, al) in flagTxt) { bool on = rn.flags >= 0 && (rn.flags & bit) != 0; t.color = on ? (al ? W.RED : W.ACCENT) : W.Hex(0xb4c0d0); t.fontStyle = on ? FontStyle.Bold : FontStyle.Normal; }
            // verify.js: '시각 등록 (학습 영점 대비 Y ±µm, Θ ±°)' 또는 '없음 — 학습 영점 사용'
            string refs = "센터 기준 " + (rn.reference == null ? "없음 — 학습 영점 사용" : $"{rn.reference.Value.ts:HH:mm:ss} 등록 (학습 영점 대비 Y {Fmt(rn.reference.Value.y * 1000, 1)} µm, Θ {Fmt(rn.reference.Value.t, 4)}°)");
            centerTxt.text = $"Y {s.center_y_mm:0.000} mm · Θ {s.center_t_deg:0.000}°  ·  허용 ±{s.tol_y_mm} mm" + (s.enable_t ? $" · ±{s.tol_t_deg}°" : "") + "  ·  " + refs;
            // verify.js: 장치가 준비되지 않았으면 (스테이지 끊김) 모든 버튼, 작업 중이면 정지 외 버튼을 잠금
            bool ready = app.stage.Connected;
            foreach (var b in new[] { bHome, bTZero, bRef, bRandom, bGoto, bRun, bStep, bMeasure, bCenter }) b.interactable = ready && !rn.busy;
            bStop.interactable = ready;
            runId.text = run == null ? "" : $"실행 {run.id}" + (run.startOffset != null ? $" · 시작 센터+({Fmt(run.startOffset.Value.Item1)} mm, {Fmt(run.startOffset.Value.Item2)}°)" : "");
            histTabTxt.text = $"회차별 기록 {rn.history.Count}";
            RenderHistory(s);
        }

        void SetCell(string k, double? v, Color c) { cells[k].text = Fmt(v); cells[k].color = c; }

        void RenderHistory(VerifySettings s)
        {
            for (int i = histBody.childCount - 1; i >= 0; i--) { var c = histBody.GetChild(i); c.SetParent(null, false); UnityEngine.Object.Destroy(c.gameObject); }
            var h = rn.history;
            if (h.Count == 0) { var r0 = W.Row(histBody, 0, 36, false); W.T(r0, "아직 기록이 없습니다.", 13, W.MUTED); DrawChart(h, s); return; }
            for (int i = h.Count - 1; i >= 0; i--)
            {
                var e = h[i];
                var r = W.Row(histBody, 0, 28, false);
                if (i == h.Count - 1) W.Fill(r, W.PILL_BG, false);
                string verdict = e.within ? (e.truthWithin == false ? "<color=#8c5509>추론만</color>" : "<color=#1f7a45>완료</color>") : "<color=#637187>이동</color>";
                HistCols(r, new[] { e.label, Fmt(e.offY), Fmt(e.truY), Fmt(e.errY), Fmt(e.offT), Fmt(e.truT), Fmt(e.errT), Fmt(e.cmdY), e.inferMs.ToString("0"), e.settleS + (e.settled ? "" : "!"), verdict }, W.INK, 11);
                W.Line(histBody, 1, W.ROW_LINE);
            }
            DrawChart(h, s);
        }

        // verify.js drawChart: |추론 Y| (파랑), |실제 Y| (주황), 허용 띠 (연두)
        void DrawChart(List<VEntry> h, VerifySettings s)
        {
            int Wd = chartTex.width, Ht = chartTex.height;
            bool empty = h.Count == 0;
            chart.enabled = !empty; axisLine.gameObject.SetActive(!empty); chartEmpty.gameObject.SetActive(empty); chartLegend.gameObject.SetActive(!empty);
            foreach (var t in new[] { axisTop, axisTol, axisZero }) t.gameObject.SetActive(!empty);
            if (empty) return;
            var px = new Color32[Wd * Ht];
            var vals = h.SelectMany(e => new[] { Math.Abs(e.offY), e.truY == null ? 0 : Math.Abs(e.truY.Value) });
            double top = Math.Max(s.tol_y_mm * 3, vals.DefaultIfEmpty(0).Max()); top = Math.Max(top, 1e-6);
            float X(int i) => h.Count < 2 ? Wd / 2f : 6 + i * (Wd - 12f) / (h.Count - 1);
            float Y(double v) => 4 + (float)(Math.Min(v, top) / top) * (Ht - 8);
            int tolY = (int)Y(s.tol_y_mm);
            PlaceAxis(axisTop, $"{top:0.###} mm", Y(top), Ht);
            PlaceAxis(axisTol, $"허용 {s.tol_y_mm:0.###}", Y(s.tol_y_mm), Ht);
            PlaceAxis(axisZero, "0", Y(0), Ht);
            axisZero.gameObject.SetActive((Y(s.tol_y_mm) - Y(0)) / Ht * (CHART_BOX_H - 6) >= 12);   // '허용' 과 겹치면 0 은 숨김
            for (int y = 0; y < Ht; y++) for (int x = 0; x < Wd; x++) px[y * Wd + x] = y <= tolY ? new Color32(0xe8, 0xf6, 0xee, 255) : new Color32(255, 255, 255, 255);
            void Line(Func<VEntry, double> get, Color32 c)
            {
                for (int i = 0; i < h.Count; i++)
                {
                    float x0 = X(i), y0 = Y(get(h[i])), x1 = i + 1 < h.Count ? X(i + 1) : x0, y1 = i + 1 < h.Count ? Y(get(h[i + 1])) : y0;
                    int n = Mathf.CeilToInt(Mathf.Max(Mathf.Abs(x1 - x0), Mathf.Abs(y1 - y0))) + 1;
                    for (int k = 0; k <= n; k++)
                    {
                        float t = k / (float)n; int xx = Mathf.RoundToInt(Mathf.Lerp(x0, x1, t)), yy = Mathf.RoundToInt(Mathf.Lerp(y0, y1, t));
                        for (int a = -1; a <= 0; a++) for (int b2 = -1; b2 <= 0; b2++) { int X2 = xx + a, Y2 = yy + b2; if (X2 >= 0 && X2 < Wd && Y2 >= 0 && Y2 < Ht) px[Y2 * Wd + X2] = c; }
                    }
                }
            }
            Line(e => Math.Abs(e.offY), new Color32(0x0A, 0x3D, 0x91, 255));
            Line(e => e.truY == null ? 0 : Math.Abs(e.truY.Value), new Color32(0xb0, 0x6f, 0x0c, 255));
            chartTex.SetPixels32(px); chartTex.Apply(false);
        }

        public void ShowPart(PartInfo p)
        {
            shownPart = p;
            partCard.gameObject.SetActive(p != null);
            if (p == null) return;
            partTitle.text = (string.IsNullOrEmpty(p.no) ? "" : p.no + "  ") + p.name;
            bool hidden = p.renderers.Count > 0 && !p.renderers[0].enabled;
            partBody.text = $"<color=#0A3D91>{(Rig.KIND.TryGetValue(p.kind, out var k) ? k : p.kind)}</color>\n<b>크기</b>  {p.size}"
                            + (string.IsNullOrEmpty(p.where) ? "" : $"\n<b>위치</b>  {p.where}") + (string.IsNullOrEmpty(p.how) ? "" : $"\n<b>고정</b>  {p.how}");
            bHide.GetComponentInChildren<Text>().text = hidden ? "다시 보이기" : "숨기기";
        }

        public void TestShowHistory() => ShowTab(1);
        public void TestShowNow() => ShowTab(0);
    }
}
