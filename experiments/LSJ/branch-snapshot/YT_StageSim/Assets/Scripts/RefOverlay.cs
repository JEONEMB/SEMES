// 영상 위 QR·나사 윤곽 (GitLab dev verify_web app.py make_overlay · verify.js drawOverlay, 11-13 617a474 · 45990aa · 11-16 cfb6483)
//   목표(점선, 하늘색) = 센터 기준 사진에서 알고리즘이 찾은 QR 사각형 + 나사 원 + 가운데 QR 십자
//   지금(실선, 색 = 정렬 여부) = 마지막 측정. QR 은 기준 사각형을 이번 측정의 QR 중심에 맞춰 회전·이동, 나사는 찾은 중심에 원 + 목표→지금 선
//   오른쪽 위 글씨 = 센터 OK / 센터에서 벗어남 / 측정 전 / 이동 중
// 웹과 다른 점: 웹은 기준 사진에서 OpenCV 로 QR 네 꼭짓점을 찾고 못 찾으면 알고리즘 ROI 사각형을 씀. Unity 에는 OpenCV 가 없어 ROI 사각형만 씀
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using UnityEngine;
using UnityEngine.UI;

namespace YT
{
    public class RefOverlay
    {
        const int TW = 1280, TH = 720;                 // verify_web 화면 좌표 (viewBox 1280×720)
        readonly RawImage img; readonly Texture2D tex; readonly Color32[] buf = new Color32[TW * TH];
        readonly RectTransform badge; readonly Image badgeBg; readonly Text badgeTxt;
        string key;
        float thick = 4;                               // 화면에서 약 2 px 이 되도록 (영상 칸 폭에 맞춰 바꿈)

        static readonly Color32 TARGET = new Color32(0x7f, 0xd8, 0xff, 255), OK = new Color32(0x2b, 0xd4, 0x6b, 255),
                                BAD = new Color32(0xff, 0x4d, 0x5e, 255), UNKNOWN = new Color32(0xff, 0xd1, 0x66, 255);

        public RefOverlay(RectTransform vbox)
        {
            tex = new Texture2D(TW, TH, TextureFormat.RGBA32, false) { filterMode = FilterMode.Bilinear, wrapMode = TextureWrapMode.Clamp, name = "RefOverlay" };
            tex.SetPixels32(buf); tex.Apply(false);
            img = Ui.Anchor(Ui.NewRect("ref-overlay", vbox), Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero).gameObject.AddComponent<RawImage>();
            img.texture = tex; img.raycastTarget = false;
            badge = Ui.NewRect("center-badge", vbox);
            badge.anchorMin = badge.anchorMax = badge.pivot = new Vector2(1, 1); badge.anchoredPosition = new Vector2(-12, -12);
            var hg = badge.gameObject.AddComponent<HorizontalLayoutGroup>(); hg.padding = new RectOffset(14, 14, 6, 6);
            hg.childControlWidth = hg.childControlHeight = true; hg.childForceExpandWidth = hg.childForceExpandHeight = false;
            badge.gameObject.AddComponent<ContentSizeFitter>().horizontalFit = ContentSizeFitter.FitMode.PreferredSize;
            badge.GetComponent<ContentSizeFitter>().verticalFit = ContentSizeFitter.FitMode.PreferredSize;
            badgeBg = W.Fill(badge, Color.white); badgeBg.raycastTarget = false;
            badgeTxt = W.T(badge, "", 13, W.INK, true); badgeTxt.horizontalOverflow = HorizontalWrapMode.Overflow;
        }

        // map: 알고리즘 좌표(3840×2160) → 윤곽 그림 좌표 (왼쪽 아래 0, 1280×720). 모르면 null
        public void Update(VerifyRunner rn, bool live, bool moving, Func<double, double, Vector2?> map, string mapKey, string noMapWhy)
        {
            var rt = img.rectTransform.rect;
            if (rt.width > 1) thick = Mathf.Clamp(2f * TW / rt.width, 2, 8);
            string state = rn.aligned == true ? "ok" : rn.aligned == false ? "bad" : "unknown";
            string k = $"{live}|{moving}|{state}|{Id(rn.refRois)}|{Id(rn.roisNow)}|{rn.refFixed}|{mapKey}|{noMapWhy}|{thick:0.0}";
            if (k == key) return;
            key = k;

            img.enabled = live; badge.gameObject.SetActive(live);
            Array.Clear(buf, 0, buf.Length);
            string text; Color bg, fg;
            var refR = rn.refRois;
            if (rn.reference == null) { text = "센터 기준 없음 · 보정·측정을 시작하면 먼저 찍습니다"; bg = W.Hex(0xfff3df); fg = W.AMBER; }
            else if (refR == null || refR.Count == 0)
            {
                text = rn.refFixed ? "윤곽 없음 · 이 알고리즘은 센터 기준에서 위치를 돌려주지 않음" : rn.algo.Kind == "mock" ? "윤곽 없음 · 모의 추론은 위치를 돌려주지 않음" : "윤곽 없음 · 알고리즘이 QR·나사 위치를 돌려주지 않음";
                bg = new Color(1, 1, 1, .9f); fg = W.INK;
            }
            else if (noMapWhy != null) { text = noMapWhy; bg = new Color(1, 1, 1, .9f); fg = W.INK; }
            else
            {
                bool hasNow = rn.roisNow != null && rn.roisNow.Count > 0;
                text = state == "ok" ? "센터 OK" : state == "bad" ? "센터에서 벗어남" : moving ? "이동 중…" : "측정 전";
                bg = state == "ok" ? W.Hex(0x1b7f4b) : state == "bad" ? W.Hex(0xb12c39) : W.Hex(0xfff3df);
                fg = state == "ok" || state == "bad" ? Color.white : W.AMBER;
                Draw(refR, hasNow ? rn.roisNow : null, state, map);
            }
            badgeTxt.text = text; badgeTxt.color = fg; badgeBg.color = bg;
            tex.SetPixels32(buf); tex.Apply(false);
        }

        static string Id(object o) => o == null ? "-" : RuntimeHelpers.GetHashCode(o).ToString();

        // app.py make_overlay + verify.js drawOverlay
        void Draw(List<Roi> refR, List<Roi> nowR, string state, Func<double, double, Vector2?> map)
        {
            var rois = new Dictionary<string, Roi>();
            foreach (var r in refR) rois[r.name] = r;
            var qrNames = rois.Keys.Where(n => n.ToUpperInvariant().StartsWith("QR")).OrderBy(n => rois[n].cx).ToList();
            var screws = rois.Keys.Where(n => !qrNames.Contains(n)).ToList();
            var quads = qrNames.Select(n => { var r = rois[n]; return new[] { new Vector2d(r.x, r.y), new Vector2d(r.x + r.w, r.y), new Vector2d(r.x + r.w, r.y + r.h), new Vector2d(r.x, r.y + r.h) }; }).ToList();
            var o0 = map(0, 0); var o1 = map(1000, 0);
            if (o0 == null || o1 == null) return;
            float scale = Vector2.Distance(o0.Value, o1.Value) / 1000f;     // 알고리즘 1 px → 그림 px
            Color32 now = state == "ok" ? OK : state == "bad" ? BAD : UNKNOWN;
            float tT = thick, tN = thick * 1.5f;

            // 목표 (점선)
            foreach (var q in quads) Poly(q.Select(p => map(p.x, p.y)).ToArray(), TARGET, tT, true);
            var tgtScrew = new Dictionary<string, (Vector2 c, float r)>();
            foreach (var n in screws)
            {
                var r = rois[n]; var c = map(r.cx, r.cy); if (c == null) continue;
                float rad = Mathf.Max(Mathf.Min(r.w, r.h) * 0.3f * scale, 6);
                tgtScrew[n] = (c.Value, rad); Circle(c.Value, rad, TARGET, tT, true);
            }
            if (quads.Count == 3)
            {   // 가운데 QR 중심 십자
                var q = quads[1].Select(p => map(p.x, p.y)).ToArray();
                if (q.All(p => p != null))
                {
                    var m = (q[0].Value + q[1].Value + q[2].Value + q[3].Value) / 4; float arm = Vector2.Distance(q[0].Value, q[1].Value) * 0.25f;
                    Line(m - new Vector2(arm, 0), m + new Vector2(arm, 0), TARGET, tT, true);
                    Line(m - new Vector2(0, arm), m + new Vector2(0, arm), TARGET, tT, true);
                }
            }
            if (nowR == null) return;

            // 지금 (실선)
            var nowD = new Dictionary<string, Roi>();
            foreach (var r in nowR) nowD[r.name] = r;
            var pairs = qrNames.Where(nowD.ContainsKey).ToList();
            if (quads.Count > 0 && pairs.Count >= 2)
            {
                var f = Rigid(pairs.Select(n => new Vector2d(rois[n].cx, rois[n].cy)).ToList(), pairs.Select(n => new Vector2d(nowD[n].cx, nowD[n].cy)).ToList());
                foreach (var q in quads) Poly(q.Select(p => { var t = f(p); return map(t.x, t.y); }).ToArray(), now, tN, false);
            }
            foreach (var n in screws)
            {
                if (!nowD.TryGetValue(n, out var r)) continue;
                var c = map(r.cx, r.cy); if (c == null) continue;
                float rad = Mathf.Max(Mathf.Min(rois[n].w, rois[n].h) * 0.3f * scale, 6);
                Circle(c.Value, rad, now, tN, false);
                if (tgtScrew.TryGetValue(n, out var tg)) Line(tg.c, c.Value, now, tT, false);   // 목표 → 지금 (어긋난 방향)
            }
            foreach (var r in nowR) { var c = map(r.cx, r.cy); if (c != null) Disc(c.Value, 3 * thick / 2, now); }
        }

        struct Vector2d { public double x, y; public Vector2d(double x, double y) { this.x = x; this.y = y; } }

        // app.py _rigid: src→dst 회전+이동 (최소제곱, 점 2개 이상)
        static Func<Vector2d, Vector2d> Rigid(List<Vector2d> src, List<Vector2d> dst)
        {
            int n = src.Count;
            double sx = src.Average(p => p.x), sy = src.Average(p => p.y), dx = dst.Average(p => p.x), dy = dst.Average(p => p.y), a = 0, b = 0;
            for (int i = 0; i < n; i++)
            {
                double x1 = src[i].x - sx, y1 = src[i].y - sy, x2 = dst[i].x - dx, y2 = dst[i].y - dy;
                a += x1 * x2 + y1 * y2; b += x1 * y2 - y1 * x2;
            }
            double th = Math.Atan2(b, a), c = Math.Cos(th), s = Math.Sin(th);
            return p => new Vector2d(c * (p.x - sx) - s * (p.y - sy) + dx, s * (p.x - sx) + c * (p.y - sy) + dy);
        }

        // ---------------- 그리기 (점선: 화면 기준 7 px 선 / 5 px 틈, verify.css stroke-dasharray 7 5)
        float dashPos;
        void Poly(Vector2?[] pts, Color32 c, float w, bool dash)
        {
            if (pts.Any(p => p == null)) return;
            dashPos = 0;
            for (int i = 0; i < pts.Length; i++) Seg(pts[i].Value, pts[(i + 1) % pts.Length].Value, c, w, dash);
        }

        void Circle(Vector2 m, float r, Color32 c, float w, bool dash)
        {
            const int N = 64; dashPos = 0;
            for (int i = 0; i < N; i++)
            {
                float a0 = i * 2 * Mathf.PI / N, a1 = (i + 1) * 2 * Mathf.PI / N;
                Seg(m + r * new Vector2(Mathf.Cos(a0), Mathf.Sin(a0)), m + r * new Vector2(Mathf.Cos(a1), Mathf.Sin(a1)), c, w, dash);
            }
        }

        void Line(Vector2 a, Vector2 b, Color32 c, float w, bool dash) { dashPos = 0; Seg(a, b, c, w, dash); }

        void Seg(Vector2 a, Vector2 b, Color32 c, float w, bool dash)
        {
            float len = Vector2.Distance(a, b), on = 7 * thick / 2, off = 5 * thick / 2;
            int steps = Mathf.Max(1, Mathf.CeilToInt(len));
            for (int i = 0; i <= steps; i++)
            {
                float d = dashPos + len * i / steps;
                if (dash && d % (on + off) >= on) continue;
                Disc(Vector2.Lerp(a, b, (float)i / steps), w / 2, c);
            }
            dashPos += len;
        }

        void Disc(Vector2 p, float r, Color32 c)
        {
            int x0 = Mathf.FloorToInt(p.x - r), x1 = Mathf.CeilToInt(p.x + r), y0 = Mathf.FloorToInt(p.y - r), y1 = Mathf.CeilToInt(p.y + r);
            float rr = r * r;
            for (int y = Math.Max(0, y0); y <= Math.Min(TH - 1, y1); y++)
                for (int x = Math.Max(0, x0); x <= Math.Min(TW - 1, x1); x++)
                {
                    float ddx = x + .5f - p.x, ddy = y + .5f - p.y;
                    if (ddx * ddx + ddy * ddy <= rr + r) buf[y * TW + x] = c;
                }
        }
    }
}
